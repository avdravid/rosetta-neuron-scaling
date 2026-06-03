#!/usr/bin/env python3
"""
compute_cross_activations.py
MULTI-GPU + tokenizer-mismatch safe cross-activation computation.

FIX (visualization correctness):
- When a Model-1 span overlaps multiple Model-2 tokens, we now SAVE the per-token
  activations in that span as `cross_span`.
- Also save `cross_span_max_activation` and `cross_span_max_position`.

Distributed behavior (torchrun):
- Shard buddy pairs by index across ranks
- Each rank writes partial output: output_path.rank{r}.json
- Rank 0 merges into output_path
"""

import argparse
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# =========================
# Distributed helpers
# =========================

def dist_is_on() -> bool:
    return dist.is_available() and dist.is_initialized()

def dist_init_if_needed() -> Tuple[int, int, int, str]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = f"cuda:{local_rank}"
        else:
            device = "cpu"
        return rank, world, local_rank, device
    return 0, 1, 0, "cuda" if torch.cuda.is_available() else "cpu"

def dist_barrier():
    if dist_is_on():
        dist.barrier()

def is_rank0(rank: int) -> bool:
    return rank == 0

def rprint(rank: int, *args, **kwargs):
    if is_rank0(rank):
        print(*args, **kwargs)


# =========================
# Special-token mapping
# =========================

def build_special_token_maps(tok1, tok2) -> Tuple[Dict[int, int], Dict[int, str]]:
    names = ["bos", "eos", "pad", "cls", "sep", "unk", "mask"]
    id_map: Dict[int, int] = {}
    id_to_name: Dict[int, str] = {}
    for name in names:
        id1 = getattr(tok1, f"{name}_token_id", None)
        id2 = getattr(tok2, f"{name}_token_id", None)
        if isinstance(id1, int) and isinstance(id2, int):
            id_map[int(id1)] = int(id2)
            id_to_name[int(id1)] = name
    return id_map, id_to_name


# =========================
# Activation hook
# =========================

class ActivationCache:
    def __init__(self, model: nn.Module, layer_names: List[str], capture_input: bool = False):
        self.model = model
        self.layer_names = layer_names
        self.capture_input = capture_input
        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks = []
        self._register_hooks()

    def _get_module(self, name: str) -> nn.Module:
        module = self.model
        for part in name.split("."):
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module

    def _make_hook(self, name: str):
        def hook(module, inp, output):
            if self.capture_input:
                if isinstance(inp, tuple) and len(inp) > 0:
                    self.activations[name] = inp[0].detach()
                else:
                    self.activations[name] = inp.detach()
            else:
                if isinstance(output, tuple):
                    self.activations[name] = output[0].detach()
                else:
                    self.activations[name] = output.detach()
        return hook

    def _register_hooks(self):
        for name in self.layer_names:
            try:
                module = self._get_module(name)
                h = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(h)
            except (AttributeError, IndexError) as e:
                print(f"WARNING: Could not hook '{name}': {e}")

    def get(self, name: str) -> Optional[torch.Tensor]:
        return self.activations.get(name)

    def clear(self):
        self.activations = {}

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def find_mlp_hook_names(model, model_name: str, layer_indices: List[int]) -> Tuple[List[str], bool]:
    model_name_lower = model_name.lower()
    test_layer = layer_indices[0] if layer_indices else 0

    if "pythia" in model_name_lower or "neox" in model_name_lower:
        try:
            _ = model.gpt_neox.layers[test_layer].mlp.act
            return [f"gpt_neox.layers.{i}.mlp.act" for i in layer_indices], False
        except AttributeError:
            pass
        _ = model.gpt_neox.layers[test_layer].mlp.dense_4h_to_h
        return [f"gpt_neox.layers.{i}.mlp.dense_4h_to_h" for i in layer_indices], True

    if "gpt2" in model_name_lower:
        _ = model.transformer.h[test_layer].mlp.c_proj
        return [f"transformer.h.{i}.mlp.c_proj" for i in layer_indices], True

    if "opt" in model_name_lower:
        # OPT MLP is fc1 -> ReLU -> fc2; hook fc2 and capture its input (post-ReLU).
        _ = model.model.decoder.layers[test_layer].fc2
        return [f"model.decoder.layers.{i}.fc2" for i in layer_indices], True

    if ("llama" in model_name_lower
            or "gemma" in model_name_lower or "qwen" in model_name_lower):
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            _ = model.model.layers[test_layer].mlp.down_proj
            return [f"model.layers.{i}.mlp.down_proj" for i in layer_indices], True
        if hasattr(model, "layers"):
            _ = model.layers[test_layer].mlp.down_proj
            return [f"layers.{i}.mlp.down_proj" for i in layer_indices], True
        raise ValueError(f"Unsupported model structure for {model_name}")

    raise ValueError(f"Could not detect hook points for {model_name}")


# =========================
# I/O helpers
# =========================

def load_best_buddies(path: str) -> List[dict]:
    with open(path, "r") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        out = []
        for v in obj.values():
            if isinstance(v, list):
                out.extend(v)
        return out
    return []


def load_activations(output_dir: str) -> Dict[int, Dict[int, dict]]:
    result = {}
    if not os.path.isdir(output_dir):
        return result
    for fname in os.listdir(output_dir):
        if fname.startswith("layer_") and fname.endswith("_activations.json"):
            try:
                layer_idx = int(fname.split("_")[1])
                with open(os.path.join(output_dir, fname), "r") as f:
                    data = json.load(f)
                result[layer_idx] = {int(k): v for k, v in data.items()}
            except Exception:
                continue
    return result


def load_docs_subset(cache_dir: str, wanted_doc_ids: set) -> Dict[str, str]:
    docs_path = os.path.join(cache_dir, "docs.jsonl")
    out: Dict[str, str] = {}
    if not os.path.exists(docs_path):
        return out
    with open(docs_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            did = rec.get("doc_id")
            if did in wanted_doc_ids:
                out[did] = rec.get("text", "")
                if len(out) >= len(wanted_doc_ids):
                    break
    return out


def safe_span_text(text: str, cs: int, ce: int) -> str:
    n = len(text)
    cs2 = max(0, min(n, int(cs)))
    ce2 = max(0, min(n, int(ce)))
    if ce2 <= cs2:
        return ""
    return text[cs2:ce2]


def overlap_len(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def token_text_from_offset(tokenizer, text: str, token_id: int, off: Optional[Tuple[int, int]]) -> str:
    if off is not None:
        cs, ce = off
        if isinstance(cs, int) and isinstance(ce, int) and ce > cs:
            s = safe_span_text(text, cs, ce)
            if s:
                return s
    return tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)


# =========================
# Core
# =========================

@torch.inference_mode()
def compute_cross_activations_for_neuron(
    model,
    tokenizer,
    hook_name: str,
    capture_input: bool,
    neuron_idx: int,
    examples: List[dict],
    docs_map: Dict[str, str],
    device: str,
    seq_length: int,
    pool: str,
    context_size: int,
    add_special_tokens: bool,
    allow_position_fallback: bool,
    special_token_id_map: Optional[Dict[int, int]] = None,
    special_id_to_name: Optional[Dict[int, str]] = None,
) -> List[dict]:
    cache = ActivationCache(model, [hook_name], capture_input=capture_input)
    results = []

    by_doc = defaultdict(list)
    for i, ex in enumerate(examples):
        by_doc[ex.get("doc_id", "")].append((i, ex))

    orig_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "right"

    try:
        for doc_id, doc_examples in by_doc.items():
            text = docs_map.get(doc_id)
            if text is None:
                for i, ex in doc_examples:
                    r = dict(ex)
                    r["cross_activation"] = None
                    r["cross_context_before"] = []
                    r["cross_context_after"] = []
                    r["cross_span"] = []
                    results.append((i, r))
                continue

            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=seq_length,
                add_special_tokens=add_special_tokens,
                return_offsets_mapping=True if tokenizer.is_fast else False,
            )

            input_ids = enc["input_ids"].to(device)
            attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

            offsets = None
            if "offset_mapping" in enc:
                offsets = enc["offset_mapping"][0].tolist()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            acts = cache.get(hook_name)
            cache.clear()

            if acts is None:
                for i, ex in doc_examples:
                    r = dict(ex)
                    r["cross_activation"] = None
                    r["cross_context_before"] = []
                    r["cross_context_after"] = []
                    r["cross_span"] = []
                    results.append((i, r))
                continue

            # Normalize activations to (S, D) for single-item batch
            B, S = int(input_ids.shape[0]), int(input_ids.shape[1])
            if acts.ndim == 3:
                acts = acts[0].to(torch.float32)
            elif acts.ndim == 2:
                if acts.shape[0] == B * S:
                    acts = acts.view(B, S, -1)[0].to(torch.float32)
                elif acts.shape[0] == S:
                    acts = acts.to(torch.float32)
                else:
                    # Unexpected shape; skip this doc
                    for i, ex in doc_examples:
                        r = dict(ex)
                        r["cross_activation"] = None
                        r["cross_context_before"] = []
                        r["cross_context_after"] = []
                        r["cross_span"] = []
                        results.append((i, r))
                    continue
            else:
                for i, ex in doc_examples:
                    r = dict(ex)
                    r["cross_activation"] = None
                    r["cross_context_before"] = []
                    r["cross_context_after"] = []
                    r["cross_span"] = []
                    results.append((i, r))
                continue

            S, D = acts.shape
            logits = outputs.logits[0]
            max_abs_per_pos = logits.abs().max(dim=-1).values.to(torch.float32)
            max_logprob_per_pos = logits.log_softmax(dim=-1).max(dim=-1).values.to(torch.float32)
            tokens = input_ids[0].tolist()

            for i, ex in doc_examples:
                r = dict(ex)

                cs = ex.get("char_start")
                ce = ex.get("char_end")

                matched_positions: List[int] = []
                weights: List[float] = []

                if offsets is not None and cs is not None and ce is not None:
                    cs = int(cs); ce = int(ce)
                    for pos, (tcs, tce) in enumerate(offsets):
                        if pos >= S:
                            break
                        if not (isinstance(tcs, int) and isinstance(tce, int) and tce > tcs):
                            continue
                        if tcs < ce and tce > cs:
                            matched_positions.append(pos)
                            if pool == "wmean":
                                weights.append(float(overlap_len(cs, ce, tcs, tce)))

                if not matched_positions and special_token_id_map and special_id_to_name:
                    ex_tid = ex.get("token_id")
                    if isinstance(ex_tid, int) and ex_tid in special_id_to_name:
                        mapped_tid = special_token_id_map.get(ex_tid)
                        if mapped_tid is not None:
                            candidates = [p for p, tid in enumerate(tokens) if int(tid) == int(mapped_tid)]
                            if candidates:
                                pos0 = ex.get("position")
                                if isinstance(pos0, int) and pos0 in candidates:
                                    matched_positions = [pos0]
                                elif isinstance(pos0, int):
                                    matched_positions = [min(candidates, key=lambda p: abs(p - pos0))]
                                else:
                                    matched_positions = [candidates[0]]
                                weights = [1.0]

                if not matched_positions and allow_position_fallback:
                    pos0 = int(ex.get("position", 0))
                    if 0 <= pos0 < S:
                        matched_positions = [pos0]
                        weights = [1.0]

                if not matched_positions or neuron_idx >= D:
                    r["cross_activation"] = None
                    r["cross_context_before"] = []
                    r["cross_context_after"] = []
                    r["cross_span"] = []
                    results.append((i, r))
                    continue

                matched_positions = sorted(set(matched_positions))

                # --- NEW: capture per-token span activations ---
                span = []
                span_raws = []
                for pos in matched_positions:
                    tid = int(tokens[pos])
                    off = None
                    if offsets is not None and pos < len(offsets):
                        off = tuple(offsets[pos])
                    tok_txt = token_text_from_offset(tokenizer, text, tid, off)
                    raw = float(acts[pos, neuron_idx].item())
                    span.append({
                        "position": int(pos),
                        "token_id": tid,
                        "token": tok_txt,
                        "activation": raw,
                        "offset": [int(off[0]), int(off[1])] if off is not None else None,
                    })
                    span_raws.append(raw)

                # pool activation over matched tokens
                vals = acts[matched_positions, neuron_idx]

                if pool == "mean":
                    pool_act = float(vals.mean().item())
                elif pool == "max":
                    pool_act = float(vals.max().item())
                elif pool == "median":
                    pool_act = float(vals.median().item())
                elif pool == "wmean":
                    w = torch.tensor(weights, device=vals.device, dtype=torch.float32)
                    if float(w.sum().item()) <= 0:
                        pool_act = float(vals.mean().item())
                    else:
                        w = w / w.sum()
                        pool_act = float((vals * w).sum().item())
                else:
                    pool_act = float(vals.mean().item())

                # max inside span (raw)
                max_idx = int(max(range(len(span_raws)), key=lambda k: span_raws[k]))
                max_pos = int(span[max_idx]["position"])
                max_act = float(span_raws[max_idx])

                left = min(matched_positions)
                right = max(matched_positions)

                # token display string for "main" span
                if cs is not None and ce is not None:
                    r["cross_token"] = safe_span_text(text, int(cs), int(ce))
                else:
                    r["cross_token"] = span[max_idx]["token"]

                # Backward-compatible fields
                r["cross_activation"] = pool_act
                r["cross_positions"] = matched_positions
                r["cross_pool"] = pool
                r["cross_position"] = left

                # NEW fields for correct visualization
                r["cross_span"] = span
                r["cross_span_pool_activation"] = pool_act
                r["cross_span_max_activation"] = max_act
                r["cross_span_max_position"] = max_pos
                r["logit_max_abs"] = float(max_abs_per_pos[left].item())
                r["logprob_max"] = float(max_logprob_per_pos[left].item())

                # Context before/after around span
                before = []
                after = []

                b0 = max(0, left - context_size)
                for p in range(b0, left):
                    off = None
                    if offsets is not None and p < len(offsets):
                        off = tuple(offsets[p])
                    before.append({
                        "token": token_text_from_offset(tokenizer, text, tokens[p], off),
                        "activation": float(acts[p, neuron_idx].item()),
                    })

                a1 = min(S, right + context_size + 1)
                for p in range(right + 1, a1):
                    off = None
                    if offsets is not None and p < len(offsets):
                        off = tuple(offsets[p])
                    after.append({
                        "token": token_text_from_offset(tokenizer, text, tokens[p], off),
                        "activation": float(acts[p, neuron_idx].item()),
                    })

                r["cross_context_before"] = before
                r["cross_context_after"] = after

                results.append((i, r))

    finally:
        tokenizer.padding_side = orig_padding_side
        cache.remove_hooks()

    results.sort(key=lambda x: x[0])
    return [x[1] for x in results]


def main():
    rank, world_size, local_rank, auto_device = dist_init_if_needed()

    parser = argparse.ArgumentParser()
    parser.add_argument("--buddies_path", type=str, required=True)
    parser.add_argument("--act1_dir", type=str, required=True)
    parser.add_argument("--model2", type=str, required=True)
    parser.add_argument("--tokenizer2", type=str, default="", help="Optional tokenizer override for model2")
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")

    parser.add_argument("--max_pairs", type=int, default=100)
    parser.add_argument("--max_examples", type=int, default=10)
    parser.add_argument("--seq_length", type=int, default=512)

    parser.add_argument("--pool", type=str, default="wmean", choices=["mean", "max", "median", "wmean"])
    parser.add_argument("--context_size", type=int, default=10)
    parser.add_argument("--add_special_tokens", action="store_true")
    parser.add_argument("--allow_position_fallback", action="store_true")
    parser.add_argument("--map_special_tokens", action="store_true",
                        help="Align special tokens across models by type (BOS/EOS/PAD/CLS/SEP/UNK/MASK)")

    args = parser.parse_args()

    if args.device.startswith("cuda") and auto_device.startswith("cuda"):
        args.device = auto_device

    rprint(rank, f"Loading best buddies from {args.buddies_path}")
    buddies = load_best_buddies(args.buddies_path)
    buddies = sorted(buddies, key=lambda x: -float(x.get("correlation", 0.0)))[:args.max_pairs]
    rprint(rank, f"Total pairs: {len(buddies)}")

    my_buddies = [b for i, b in enumerate(buddies) if (i % world_size) == rank]
    rprint(rank, f"Shard: rank {rank}/{world_size} => {len(my_buddies)} pairs")

    rprint(rank, f"Loading Model 1 activations from {args.act1_dir}")
    act1 = load_activations(args.act1_dir)

    meta1_path = os.path.join(args.act1_dir, "metadata.json")
    model1_name = "Model 1"
    if os.path.exists(meta1_path):
        with open(meta1_path) as f:
            meta1 = json.load(f)
            model1_name = meta1.get("model_name", "Model 1")

    rprint(rank, f"Loading Model 2: {args.model2} on {args.device}")
    torch_dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    tok2_id = args.tokenizer2.strip() or args.model2
    tokenizer2 = AutoTokenizer.from_pretrained(tok2_id, use_fast=True)
    tokenizer2.pad_token = tokenizer2.pad_token or tokenizer2.eos_token

    special_token_id_map = None
    special_id_to_name = None
    if args.map_special_tokens:
        if model1_name == "Model 1":
            rprint(rank, "Warning: model1_name missing in act1 metadata; skipping special-token mapping.")
        else:
            rprint(rank, f"Loading Model 1 tokenizer for special-token mapping: {model1_name}")
            tokenizer1 = AutoTokenizer.from_pretrained(model1_name, use_fast=True)
            tokenizer1.pad_token = tokenizer1.pad_token or tokenizer1.eos_token
            special_token_id_map, special_id_to_name = build_special_token_maps(tokenizer1, tokenizer2)
            if not special_token_id_map:
                rprint(rank, "Warning: no overlapping special tokens found for mapping.")

    model2 = AutoModelForCausalLM.from_pretrained(args.model2, torch_dtype=torch_dtype)
    model2.to(args.device)
    model2.eval()

    layers_needed = sorted({int(b.get("model2_layer", 0)) for b in my_buddies})
    hook_names, capture_input = find_mlp_hook_names(model2, args.model2, layers_needed)
    layer_to_hook = {l: h for l, h in zip(layers_needed, hook_names)}

    wanted_doc_ids = set()
    for buddy in my_buddies:
        l1 = int(buddy.get("model1_layer", 0))
        n1 = int(buddy.get("model1_neuron", 0))
        if l1 not in act1 or n1 not in act1[l1]:
            continue
        exs = act1[l1][n1].get("examples", [])[:args.max_examples]
        for ex in exs:
            did = ex.get("doc_id")
            if did:
                wanted_doc_ids.add(did)

    docs_map = load_docs_subset(args.cache_dir, wanted_doc_ids)

    results = []
    for buddy in tqdm(my_buddies, desc=f"[rank{rank}] Cross-activations", disable=(rank != 0)):
        layer1 = int(buddy.get("model1_layer", 0))
        neuron1 = int(buddy.get("model1_neuron", 0))
        layer2 = int(buddy.get("model2_layer", 0))
        neuron2 = int(buddy.get("model2_neuron", 0))
        correlation = float(buddy.get("correlation", 0.0))

        if layer1 not in act1 or neuron1 not in act1[layer1]:
            continue

        neuron_data = act1[layer1][neuron1]
        examples = neuron_data.get("examples", [])[:args.max_examples]
        if not examples:
            continue

        hook_name = layer_to_hook.get(layer2)
        if not hook_name:
            continue

        cross_examples = compute_cross_activations_for_neuron(
            model=model2,
            tokenizer=tokenizer2,
            hook_name=hook_name,
            capture_input=capture_input,
            neuron_idx=neuron2,
            examples=examples,
            docs_map=docs_map,
            device=args.device,
            seq_length=args.seq_length,
            pool=args.pool,
            context_size=args.context_size,
            add_special_tokens=args.add_special_tokens,
            allow_position_fallback=args.allow_position_fallback,
            special_token_id_map=special_token_id_map,
            special_id_to_name=special_id_to_name,
        )

        results.append({
            "model1_layer": layer1,
            "model1_neuron": neuron1,
            "model2_layer": layer2,
            "model2_neuron": neuron2,
            "correlation": correlation,
            "examples": cross_examples,
            "model1_stats": {
                "mean": neuron_data.get("mean_activation", 0),
                "std": neuron_data.get("std_activation", 0),
                "max": neuron_data.get("max_activation", 0),
                "min": neuron_data.get("min_activation", 0),
            }
        })

    part_path = f"{args.output_path}.rank{rank}.json"
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(part_path, "w") as f:
        json.dump({
            "model1_name": model1_name,
            "model2_name": args.model2,
            "tokenizer2_name": tok2_id,
            "pairs": results,
            "cross_pool": args.pool,
            "cross_context_size": args.context_size,
            "seq_length_model2": args.seq_length,
            "rank": rank,
            "world_size": world_size,
        }, f)

    dist_barrier()

    if is_rank0(rank):
        all_pairs = []
        for r in range(world_size):
            p = f"{args.output_path}.rank{r}.json"
            if not os.path.exists(p):
                continue
            with open(p, "r") as f:
                obj = json.load(f)
            all_pairs.extend(obj.get("pairs", []))

        all_pairs.sort(key=lambda x: -float(x.get("correlation", 0.0)))

        out = {
            "model1_name": model1_name,
            "model2_name": args.model2,
            "tokenizer2_name": tok2_id,
            "pairs": all_pairs,
            "cross_pool": args.pool,
            "cross_context_size": args.context_size,
            "seq_length_model2": args.seq_length,
            "distributed": {"world_size": world_size, "sharded_pairs": True},
        }
        # Compute logprob max stats and add z-scores
        total = 0
        s1 = 0.0
        s2 = 0.0
        for p in all_pairs:
            for ex in p.get("examples", []):
                v = ex.get("logprob_max")
                if v is None:
                    continue
                total += 1
                fv = float(v)
                s1 += fv
                s2 += fv * fv
        if total > 0:
            mean = s1 / total
            var = max(0.0, (s2 / total) - mean * mean)
            std = var ** 0.5
            if std <= 1e-8:
                std = 1e-8
            for p in all_pairs:
                for ex in p.get("examples", []):
                    v = ex.get("logprob_max")
                    if v is None:
                        continue
                    ex["logprob_max_z"] = (float(v) - mean) / std
            out["logprob_max_stats"] = {"mean": mean, "std": std, "count": total}
        with open(args.output_path, "w") as f:
            json.dump(out, f)

        rprint(rank, f"[rank0] Saved merged {len(all_pairs)} pairs to {args.output_path}")

    dist_barrier()

    if dist_is_on():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
