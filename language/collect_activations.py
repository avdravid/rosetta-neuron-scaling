#!/usr/bin/env python3
"""
collect_activations.py — Deterministic cached corpus + activation collection
MULTI-GPU + cross-tokenizer safe (char spans).

Distributed behavior (torchrun):
- Shard cached docs by global doc_index: each rank takes doc_index % world_size == rank
- All-reduce stats (sum/sumsq/min/max/count) so mean/std are global and exact
- Each rank writes layer_{L}_activations.rank{r}.json
- Rank 0 merges top-K examples per neuron across ranks into layer_{L}_activations.json
- Rank 0 writes metadata.json (others write metadata.rank{r}.json)

Tokenization safety:
- Use full sequence indices (no attention-mask "compressed" indexing)
- Prefer offset_mapping; optionally require non-empty char span for center token
- Render tokens via raw substring when offsets exist (more stable across tokenizers)
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import heapq
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.pile_sampler import (
    is_pile_dataset_name,
    iter_pile_text_windows_proportional_budget,
    parse_pile_subsets,
)

# =========================
# Distributed helpers
# =========================

def dist_is_on() -> bool:
    return dist.is_available() and dist.is_initialized()

def dist_init_if_needed() -> Tuple[int, int, int, str]:
    """
    Returns (rank, world_size, local_rank, device_str).
    If not running under torchrun, returns (0,1,0,args.device-ish later).
    """
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
# Data structures
# =========================

@dataclass
class TokenActivation:
    example_id: str
    doc_id: str
    token: str
    token_id: int
    activation: float
    context_before: List[Dict[str, Any]]
    context_after: List[Dict[str, Any]]
    position: int
    sample_idx: int
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    logit_max_abs: Optional[float] = None
    logprob_max: Optional[float] = None


@dataclass
class NeuronData:
    layer: int
    neuron_idx: int
    examples: List[dict]
    mean_activation: float
    std_activation: float
    max_activation: float
    min_activation: float


# =========================
# Activation Cache
# =========================

class ActivationCache:
    def __init__(self, model: nn.Module, layer_names: List[str], capture_input: bool = False):
        self.model = model
        self.layer_names = layer_names
        self.capture_input = capture_input
        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
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
        def hook(module: nn.Module, inp: tuple, output):
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
                print(f"WARNING: Could not hook layer '{name}': {e}")

    def get_activations(self) -> Dict[str, torch.Tensor]:
        return self.activations

    def clear(self):
        self.activations = {}

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def normalize_acts(acts: torch.Tensor, batch_size: int, seq_len: int) -> Optional[torch.Tensor]:
    if acts is None:
        return None
    if acts.ndim == 3:
        return acts
    if acts.ndim == 2:
        if acts.shape[0] == batch_size * seq_len:
            return acts.view(batch_size, seq_len, -1)
        if acts.shape[0] == seq_len:
            return acts.unsqueeze(0)
        if acts.shape[0] == batch_size:
            return acts.unsqueeze(1)
    return None


# =========================
# Model inspection utilities
# =========================

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

    raise ValueError(f"Could not auto-detect MLP hook points for model: {model_name}")


# =========================
# Hashing / IDs
# =========================

def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def safe_span_text(text: str, cs: int, ce: int) -> str:
    n = len(text)
    cs2 = max(0, min(n, int(cs)))
    ce2 = max(0, min(n, int(ce)))
    if ce2 <= cs2:
        return ""
    return text[cs2:ce2]

def decode_single_token(tokenizer, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)

def build_example_id(doc_id: str, tok_pos: int, offsets_row: Optional[List[Tuple[int, int]]]) -> Tuple[str, Optional[int], Optional[int]]:
    if offsets_row is not None and 0 <= tok_pos < len(offsets_row):
        cs, ce = offsets_row[tok_pos]
        if isinstance(cs, int) and isinstance(ce, int) and ce > cs:
            return f"{doc_id}:{cs}:{ce}", cs, ce
    return f"{doc_id}:tok{tok_pos}", None, None

def token_text_from_pos(tokenizer, text: str, token_id: int, pos: int, offsets_row: Optional[List[Tuple[int, int]]]) -> str:
    if offsets_row is not None and 0 <= pos < len(offsets_row):
        cs, ce = offsets_row[pos]
        if isinstance(cs, int) and isinstance(ce, int) and ce > cs:
            s = safe_span_text(text, cs, ce)
            if s:
                return s
    return decode_single_token(tokenizer, token_id)


def build_token_texts(
    tokenizer,
    text: str,
    tokens_full: List[int],
    offsets_row: Optional[List[Tuple[int, int]]],
) -> List[str]:
    out: List[str] = []
    if offsets_row is not None and len(offsets_row) == len(tokens_full):
        for i, tid in enumerate(tokens_full):
            cs, ce = offsets_row[i]
            if isinstance(cs, int) and isinstance(ce, int) and ce > cs:
                s = safe_span_text(text, cs, ce)
                out.append(s if s else decode_single_token(tokenizer, tid))
            else:
                out.append(decode_single_token(tokenizer, tid))
        return out
    for tid in tokens_full:
        out.append(decode_single_token(tokenizer, tid))
    return out


# =========================
# Best buddies loading
# =========================

def load_best_buddies(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        out = []
        for v in obj.values():
            if isinstance(v, list):
                out.extend(v)
        return out
    raise ValueError(f"Unknown best buddies format: {type(obj)}")

def buddies_to_neurons_by_layer(
    best_buddies: List[dict],
    which_model: int,
    min_correlation: Optional[float] = None,
    top_pairs: Optional[int] = None,
) -> Dict[int, List[int]]:
    layer_key = "model1_layer" if which_model == 1 else "model2_layer"
    neuron_key = "model1_neuron" if which_model == 1 else "model2_neuron"

    pairs = [b for b in best_buddies if isinstance(b, dict)]
    if min_correlation is not None:
        thr = float(min_correlation)
        pairs = [b for b in pairs if float(b.get("correlation", -1e9)) >= thr]

    pairs.sort(key=lambda x: float(x.get("correlation", 0.0)), reverse=True)
    if top_pairs is not None:
        pairs = pairs[:int(top_pairs)]

    out: Dict[int, Set[int]] = {}
    for b in pairs:
        try:
            L = int(b[layer_key])
            n = int(b[neuron_key])
            out.setdefault(L, set()).add(n)
        except Exception:
            continue
    return {L: sorted(list(ns)) for L, ns in out.items()}


# =========================
# Cache building + reading
# =========================

def cache_paths(cache_dir: str) -> Tuple[str, str]:
    return os.path.join(cache_dir, "docs.jsonl"), os.path.join(cache_dir, "cache_meta.json")

def build_cache(
    cache_dir: str,
    dataset_name: str,
    split: str,
    text_field: str,
    cache_size: int,
    min_chars: int,
    context_length: int = 256,
    tokenizer_id: Optional[str] = None,
    pile_subsets: Optional[List[str]] = None,
    seed: int = 42,
    rank: int = 0,
) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    docs_path, meta_path = cache_paths(cache_dir)

    n_written = 0
    total_tokens = 0
    tokens_by_subset: Dict[str, int] = {}

    try:
        with open(docs_path, "w", encoding="utf-8") as f:
            if is_pile_dataset_name(dataset_name):
                if not tokenizer_id:
                    raise ValueError("Pile cache build requires --tokenizer to count tokens.")
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
                tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
                for text, subset, token_count in iter_pile_text_windows_proportional_budget(
                    tokenizer,
                    total_token_budget=cache_size,
                    context_length=context_length,
                    dataset_name=dataset_name,
                    split=split,
                    min_chars=min_chars,
                    seed=seed,
                    buffer_size=10000,
                    pile_subsets=pile_subsets,
                ):
                    doc_id = sha1_text(text)
                    f.write(
                        json.dumps({"doc_id": doc_id, "text": text, "subset": subset}, ensure_ascii=False) + "\n"
                    )
                    n_written += 1
                    total_tokens += token_count
                    tokens_by_subset[subset] = tokens_by_subset.get(subset, 0) + token_count
                    if total_tokens >= cache_size:
                        break
            else:
                ds = load_dataset(dataset_name, split=split, streaming=True)
                for ex in tqdm(ds, desc=f"Building cache ({cache_size})", disable=(rank != 0)):
                    t = ex.get(text_field, "")
                    if not isinstance(t, str) or len(t) < min_chars:
                        continue
                    doc_id = sha1_text(t)
                    f.write(json.dumps({"doc_id": doc_id, "text": t}, ensure_ascii=False) + "\n")
                    n_written += 1
                    if n_written >= cache_size:
                        break
    finally:
        # Explicitly clean up streaming dataset to avoid GIL thread state errors on exit
        if "ds" in locals():
            del ds
        gc.collect()

    meta = {
        "dataset": dataset_name,
        "split": split,
        "text_field": text_field,
        "cache_size": cache_size,
        "min_chars": min_chars,
        "docs_path": os.path.abspath(docs_path),
        "num_docs_written": n_written,
    }
    if is_pile_dataset_name(dataset_name):
        meta.update(
            {
                "token_budget": cache_size,
                "tokens_written_total": total_tokens,
                "tokens_written_by_subset": tokens_by_subset,
                "pile_subsets": list(tokens_by_subset.keys()),
                "tokenizer": tokenizer_id,
            }
        )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def iter_cached_batches_sharded(
    cache_dir: str,
    batch_size: int,
    max_docs: Optional[int],
    rank: int,
    world_size: int,
) -> Iterable[Tuple[List[int], List[str], List[str]]]:
    """
    Yields (doc_indices, texts, doc_ids) where doc_indices are GLOBAL doc indices in docs.jsonl.
    Sharding rule: idx % world_size == rank
    """
    docs_path, _ = cache_paths(cache_dir)
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Missing cache file: {docs_path}")

    batch_texts: List[str] = []
    batch_ids: List[str] = []
    batch_idxs: List[int] = []

    idx = 0
    with open(docs_path, "r", encoding="utf-8") as f:
        for line in f:
            if max_docs is not None and idx >= max_docs:
                break

            if (idx % world_size) != rank:
                idx += 1
                continue

            rec = json.loads(line)
            batch_texts.append(rec["text"])
            batch_ids.append(rec["doc_id"])
            batch_idxs.append(idx)

            if len(batch_texts) >= batch_size:
                yield batch_idxs, batch_texts, batch_ids
                batch_texts, batch_ids, batch_idxs = [], [], []

            idx += 1

    if batch_texts:
        yield batch_idxs, batch_texts, batch_ids


# =========================
# Heap with dedup
# =========================

HeapItem = Tuple[float, int, str, dict]  # (act, tie, example_id, payload)

def heap_add_dedup(heap: List[HeapItem], best: Dict[str, float], item: HeapItem, k: int) -> None:
    act, _, exid, _ = item
    prev = best.get(exid)
    if prev is not None and act <= prev:
        return
    best[exid] = act
    heapq.heappush(heap, item)

    limit = 3 * k
    if len(heap) > limit:
        for _ in range(len(heap) - limit):
            heapq.heappop(heap)
        keep = {exid2 for _, _, exid2, _ in heap}
        for key in list(best.keys()):
            if key not in keep:
                del best[key]

def heap_finalize_topk(heap: List[HeapItem], k: int) -> List[dict]:
    best: Dict[str, HeapItem] = {}
    for act, tie, exid, payload in heap:
        cur = best.get(exid)
        if cur is None or act > cur[0]:
            best[exid] = (act, tie, exid, payload)
    items = list(best.values())
    items.sort(key=lambda x: x[0], reverse=True)
    return [it[3] for it in items[:k]]


# =========================
# Model utilities
# =========================

def get_num_layers(model) -> int:
    cfg = getattr(model, "config", None)
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        if cfg is not None and hasattr(cfg, attr):
            return int(getattr(cfg, attr))
    raise ValueError("Cannot infer num layers from model.config")

def parse_layers_arg(layers_arg: str, num_layers: int) -> List[int]:
    s = layers_arg.strip().lower()
    if s == "all":
        return list(range(num_layers))
    out: List[int] = []
    for part in layers_arg.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return [l for l in out if 0 <= l < num_layers]


# =========================
# Build TokenActivation
# =========================

def build_token_activation(
    token_texts: List[str],
    text: str,
    tokens_full: List[int],
    attn_full: List[bool],
    offsets_row: Optional[List[Tuple[int, int]]],
    acts_window: List[float],
    center_pos: int,
    context_size: int,
    doc_id: str,
    sample_idx: int,
    logit_max_abs: Optional[float],
    logprob_max: Optional[float],
) -> TokenActivation:
    S = len(tokens_full)
    ctx_start = max(0, center_pos - context_size)
    ctx_end = min(S, center_pos + context_size + 1)
    center_in_window = center_pos - ctx_start

    context_before: List[Dict[str, Any]] = []
    for i in range(ctx_start, center_pos):
        if not attn_full[i]:
            continue
        wi = i - ctx_start
        tid = int(tokens_full[i])
        context_before.append({
            "token": token_texts[i],
            "token_id": tid,
            "activation": float(acts_window[wi]) if wi < len(acts_window) else 0.0,
        })

    context_after: List[Dict[str, Any]] = []
    for i in range(center_pos + 1, ctx_end):
        if not attn_full[i]:
            continue
        wi = i - ctx_start
        tid = int(tokens_full[i])
        context_after.append({
            "token": token_texts[i],
            "token_id": tid,
            "activation": float(acts_window[wi]) if wi < len(acts_window) else 0.0,
        })

    example_id, cs, ce = build_example_id(doc_id, center_pos, offsets_row)
    token_id = int(tokens_full[center_pos])
    token_str = token_texts[center_pos] if center_pos < len(token_texts) else token_text_from_pos(tokenizer, text, token_id, center_pos, offsets_row)

    return TokenActivation(
        example_id=example_id,
        doc_id=doc_id,
        token=token_str,
        token_id=token_id,
        activation=float(acts_window[center_in_window]) if center_in_window < len(acts_window) else 0.0,
        context_before=context_before,
        context_after=context_after,
        position=int(center_pos),
        sample_idx=int(sample_idx),
        char_start=cs,
        char_end=ce,
        logit_max_abs=logit_max_abs,
        logprob_max=logprob_max,
    )


# =========================
# Core collection - ALL LAYERS AT ONCE
# =========================

@torch.inference_mode()
def collect_all_layers_activations(
    model,
    tokenizer,
    layer_to_hook: Dict[int, str],
    neurons_by_layer: Dict[int, List[int]],
    capture_input: bool,
    cache_dir: str,
    max_docs: int,
    seq_length: int,
    batch_size: int,
    top_k: int,
    per_sample_top: int,
    context_size: int,
    device: str,
    add_special_tokens: bool,
    require_char_span: bool,
    allow_special_tokens_without_char_span: bool,
    rank: int,
    world_size: int,
) -> Dict[int, Dict[int, NeuronData]]:
    """
    Collect activations for ALL layers in a single pass through the data.
    Returns: {layer_idx: {neuron_idx: NeuronData}}
    """
    layers_to_collect = sorted(layer_to_hook.keys())
    if not layers_to_collect:
        return {}

    # Build hook names list and layer index mapping
    hook_names = [layer_to_hook[li] for li in layers_to_collect]
    hook_to_layer = {hn: li for li, hn in layer_to_hook.items()}

    # Per-layer neuron indices on device
    layer_local_cols: Dict[int, torch.Tensor] = {}
    layer_neurons: Dict[int, List[int]] = {}
    for li in layers_to_collect:
        neurons = neurons_by_layer.get(li, [])
        if neurons:
            layer_local_cols[li] = torch.tensor(neurons, device=device, dtype=torch.long)
            layer_neurons[li] = neurons

    # Per-layer heaps and stats
    heaps: Dict[int, Dict[int, List[HeapItem]]] = {}  # layer -> neuron -> heap
    best_by_id: Dict[int, Dict[int, Dict[str, float]]] = {}  # layer -> neuron -> exid -> best_val

    sum_vecs: Dict[int, torch.Tensor] = {}
    sumsq_vecs: Dict[int, torch.Tensor] = {}
    max_vecs: Dict[int, torch.Tensor] = {}
    min_vecs: Dict[int, torch.Tensor] = {}

    for li in layers_to_collect:
        neurons = layer_neurons.get(li, [])
        nN = len(neurons)
        if nN == 0:
            continue
        heaps[li] = {n: [] for n in neurons}
        best_by_id[li] = {n: {} for n in neurons}
        sum_vecs[li] = torch.zeros((nN,), device=device, dtype=torch.float64)
        sumsq_vecs[li] = torch.zeros((nN,), device=device, dtype=torch.float64)
        max_vecs[li] = torch.full((nN,), float("-inf"), device=device, dtype=torch.float32)
        min_vecs[li] = torch.full((nN,), float("inf"), device=device, dtype=torch.float32)

    count_tokens = 0
    tie = 0

    # Hook ALL layers at once
    cache = ActivationCache(model, hook_names, capture_input=capture_input)

    local_total = max(0, (max_docs - rank + world_size - 1) // world_size)
    pbar = tqdm(total=local_total, desc=f"[rank{rank}] All layers", unit="docs", disable=(rank != 0))

    orig_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "right"
    special_ids = set(tokenizer.all_special_ids or []) if allow_special_tokens_without_char_span else set()

    try:
        for doc_idxs, batch_texts, batch_doc_ids in iter_cached_batches_sharded(
            cache_dir, batch_size=batch_size, max_docs=max_docs, rank=rank, world_size=world_size
        ):
            tok_kwargs = dict(
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=seq_length,
                add_special_tokens=add_special_tokens,
            )
            if tokenizer.is_fast:
                tok_kwargs["return_offsets_mapping"] = True

            enc = tokenizer(batch_texts, **tok_kwargs)

            offsets_cpu: Optional[List[List[Tuple[int, int]]]] = None
            if "offset_mapping" in enc:
                om = enc["offset_mapping"]
                try:
                    offsets_cpu = [[(int(a), int(b)) for (a, b) in row] for row in om.cpu().tolist()]
                except Exception:
                    offsets_cpu = None

            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            # Single forward pass for ALL layers (and logits)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits
            max_abs_per_pos = logits.abs().max(dim=-1).values
            max_logprob_per_pos = logits.log_softmax(dim=-1).max(dim=-1).values
            max_abs_per_pos_cpu = max_abs_per_pos.detach().float().cpu().numpy()
            max_logprob_per_pos_cpu = max_logprob_per_pos.detach().float().cpu().numpy()
            del outputs, logits, max_abs_per_pos, max_logprob_per_pos

            all_acts = cache.get_activations()
            cache.clear()

            B = input_ids.shape[0]
            S = input_ids.shape[1]
            mask = attention_mask.bool()

            input_ids_cpu = input_ids.detach().float().cpu().numpy()
            attn_cpu = attention_mask.detach().float().cpu().numpy()

            tokens_full_list = [input_ids_cpu[b].tolist() for b in range(B)]
            attn_full_list = [attn_cpu[b].astype(bool).tolist() for b in range(B)]
            if offsets_cpu is not None:
                offsets_list = [offsets_cpu[b] if b < len(offsets_cpu) else None for b in range(B)]
            else:
                offsets_list = [None for _ in range(B)]
            token_texts_list = [
                build_token_texts(tokenizer, batch_texts[b], tokens_full_list[b], offsets_list[b])
                for b in range(B)
            ]

            # Update token count (same for all layers)
            if count_tokens == 0 and mask.any():
                # Only count once per batch
                count_tokens += int(mask.sum().item())
            elif mask.any():
                count_tokens += int(mask.sum().item())

            # Process each layer's activations
            for hook_name in hook_names:
                acts = all_acts.get(hook_name)
                acts = normalize_acts(acts, B, S)
                if acts is None or acts.ndim != 3:
                    continue

                li = hook_to_layer[hook_name]
                neurons = layer_neurons.get(li, [])
                if not neurons:
                    continue

                local_cols = layer_local_cols[li]
                acts_sub = acts.index_select(dim=2, index=local_cols).float()  # (B,S,nN)
                acts_sub_cpu = acts_sub.detach().float().cpu().numpy()

                # Update stats
                if mask.any():
                    flat = acts_sub[mask]  # (Ntok, nN)
                    sum_vecs[li] += flat.sum(dim=0).double()
                    sumsq_vecs[li] += (flat * flat).sum(dim=0).double()
                    max_vecs[li] = torch.maximum(max_vecs[li], flat.max(dim=0).values)
                    min_vecs[li] = torch.minimum(min_vecs[li], flat.min(dim=0).values)

                # Per sample top positions
                k_local = max(1, min(per_sample_top, S))
                neg_inf = torch.tensor(float("-inf"), device=device, dtype=acts_sub.dtype)
                acts_masked = torch.where(mask.unsqueeze(-1), acts_sub, neg_inf)
                top_vals, top_pos = torch.topk(acts_masked, k=k_local, dim=1)

                top_vals_cpu = top_vals.detach().float().cpu().numpy()
                top_pos_cpu = top_pos.detach().float().cpu().numpy()

                layer_heaps = heaps[li]
                layer_best = best_by_id[li]

                for b in range(B):
                    doc_id = batch_doc_ids[b]
                    sample_idx = doc_idxs[b]
                    text = batch_texts[b]

                    attn_full = attn_full_list[b]
                    tokens_full = tokens_full_list[b]
                    offsets_row = offsets_list[b]
                    token_texts = token_texts_list[b]
                    acts_sub_b = acts_sub_cpu[b]

                    for n_local, n_global in enumerate(neurons):
                        heap = layer_heaps[n_global]
                        best_map = layer_best[n_global]

                        for kk in range(k_local):
                            val = float(top_vals_cpu[b, kk, n_local])
                            if not np.isfinite(val):
                                continue

                            pos = int(top_pos_cpu[b, kk, n_local])
                            if pos < 0 or pos >= S:
                                continue
                            if not attn_full[pos]:
                                continue

                            if require_char_span and offsets_row is not None:
                                cs, ce = offsets_row[pos]
                                if not (isinstance(cs, int) and isinstance(ce, int) and ce > cs):
                                    token_id = int(tokens_full[pos])
                                    if token_id not in special_ids:
                                        continue

                            exid, _, _ = build_example_id(doc_id, pos, offsets_row)
                            prev = best_map.get(exid)
                            if prev is not None and val <= prev:
                                continue

                            ctx_start = max(0, pos - context_size)
                            ctx_end = min(S, pos + context_size + 1)
                            acts_window = acts_sub_b[ctx_start:ctx_end, n_local].tolist()

                            logit_max_abs = None
                            logprob_max = None
                            try:
                                logit_max_abs = float(max_abs_per_pos_cpu[b][pos])
                            except Exception:
                                logit_max_abs = None
                            try:
                                logprob_max = float(max_logprob_per_pos_cpu[b][pos])
                            except Exception:
                                logprob_max = None
                            tok_act = build_token_activation(
                                token_texts=token_texts,
                                text=text,
                                tokens_full=tokens_full,
                                attn_full=attn_full,
                                offsets_row=offsets_row,
                                acts_window=acts_window,
                                center_pos=pos,
                                context_size=context_size,
                                doc_id=doc_id,
                                sample_idx=sample_idx,
                                logit_max_abs=logit_max_abs,
                                logprob_max=logprob_max,
                            )

                            tie += 1
                            payload = asdict(tok_act)
                            heap_add_dedup(heap, best_map, (val, tie, tok_act.example_id, payload), top_k)

                del acts_sub, acts_sub_cpu, acts_masked, top_vals, top_pos

            pbar.update(len(batch_texts))

            del enc, input_ids, attention_mask, all_acts

    finally:
        pbar.close()
        tokenizer.padding_side = orig_padding_side
        cache.remove_hooks()

    # All-reduce stats for exact global mean/std/min/max
    if dist_is_on():
        for li in layers_to_collect:
            if li not in sum_vecs:
                continue
            dist.all_reduce(sum_vecs[li], op=dist.ReduceOp.SUM)
            dist.all_reduce(sumsq_vecs[li], op=dist.ReduceOp.SUM)
            dist.all_reduce(max_vecs[li], op=dist.ReduceOp.MAX)
            dist.all_reduce(min_vecs[li], op=dist.ReduceOp.MIN)
        ct = torch.tensor([count_tokens], device=device, dtype=torch.long)
        dist.all_reduce(ct, op=dist.ReduceOp.SUM)
        count_tokens = int(ct.item())

    # Build output
    out: Dict[int, Dict[int, NeuronData]] = {}
    for li in layers_to_collect:
        neurons = layer_neurons.get(li, [])
        if not neurons:
            out[li] = {}
            continue

        nN = len(neurons)
        if count_tokens == 0:
            mean = np.zeros((nN,), dtype=np.float64)
            std = np.zeros((nN,), dtype=np.float64)
            max_cpu = np.zeros((nN,), dtype=np.float64)
            min_cpu = np.zeros((nN,), dtype=np.float64)
        else:
            denom = float(count_tokens)
            mean = (sum_vecs[li] / denom).detach().float().cpu().numpy()
            var = (sumsq_vecs[li] / denom).detach().float().cpu().numpy() - mean ** 2
            std = np.sqrt(np.maximum(var, 0.0))
            max_cpu = max_vecs[li].detach().float().cpu().numpy()
            min_cpu = min_vecs[li].detach().float().cpu().numpy()
            max_cpu = np.where(np.isfinite(max_cpu), max_cpu, 0.0)
            min_cpu = np.where(np.isfinite(min_cpu), min_cpu, 0.0)

        layer_out: Dict[int, NeuronData] = {}
        for i, n in enumerate(neurons):
            examples = heap_finalize_topk(heaps[li][n], top_k)
            layer_out[n] = NeuronData(
                layer=li,
                neuron_idx=n,
                examples=examples,
                mean_activation=float(mean[i]),
                std_activation=float(std[i]),
                max_activation=float(max_cpu[i]),
                min_activation=float(min_cpu[i]),
            )
        out[li] = layer_out

    return out


# =========================
# Saving + merge
# =========================

def save_layer_rank_file(output_dir: str, layer_idx: int, layer_data: Dict[int, NeuronData], rank: int) -> str:
    os.makedirs(output_dir, exist_ok=True)
    out: Dict[str, Any] = {}
    for n, nd in layer_data.items():
        out[str(n)] = {
            "layer": nd.layer,
            "neuron_idx": nd.neuron_idx,
            "examples": nd.examples,
            "mean_activation": nd.mean_activation,
            "std_activation": nd.std_activation,
            "max_activation": nd.max_activation,
            "min_activation": nd.min_activation,
        }
    path = os.path.join(output_dir, f"layer_{layer_idx}_activations.rank{rank}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    return path

def merge_examples_topk(ex_lists: List[List[dict]], top_k: int) -> List[dict]:
    best: Dict[str, dict] = {}
    for lst in ex_lists:
        for ex in lst:
            exid = ex.get("example_id")
            if not exid:
                continue
            act = float(ex.get("activation", float("-inf")))
            prev = best.get(exid)
            if prev is None or act > float(prev.get("activation", float("-inf"))):
                best[exid] = ex
    items = list(best.values())
    items.sort(key=lambda e: float(e.get("activation", 0.0)), reverse=True)
    return items[:top_k]

def merge_layer_files(output_dir: str, layer_idx: int, world_size: int, top_k: int) -> None:
    merged: Dict[str, Any] = {}

    # gather all rank files
    rank_paths = [
        os.path.join(output_dir, f"layer_{layer_idx}_activations.rank{r}.json")
        for r in range(world_size)
    ]
    rank_dicts = []
    for p in rank_paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                rank_dicts.append(json.load(f))

    if not rank_dicts:
        return

    # keys = neuron ids as strings
    neuron_keys = set()
    for d in rank_dicts:
        neuron_keys.update(d.keys())

    for nk in neuron_keys:
        # collect examples from each rank file
        ex_lists = []
        base_rec = None
        for d in rank_dicts:
            if nk in d:
                rec = d[nk]
                base_rec = base_rec or rec
                ex_lists.append(rec.get("examples", []))

        if base_rec is None:
            continue

        merged_examples = merge_examples_topk(ex_lists, top_k)

        # stats should already be global (from all_reduce), so take from base_rec
        merged[nk] = dict(base_rec)
        merged[nk]["examples"] = merged_examples

    out_path = os.path.join(output_dir, f"layer_{layer_idx}_activations.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f)

def write_metadata(output_dir: str, meta: dict, rank: int) -> str:
    path = os.path.join(output_dir, f"metadata.rank{rank}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return path

def merge_metadata(output_dir: str, world_size: int) -> None:
    path0 = os.path.join(output_dir, "metadata.rank0.json")
    if not os.path.exists(path0):
        return
    with open(path0, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["distributed"] = {
        "world_size": world_size,
        "note": "Docs sharded by doc_index % world_size; stats all-reduced; examples merged on rank0",
    }
    out_path = os.path.join(output_dir, "metadata.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# =========================
# Main
# =========================

def main():
    rank, world_size, local_rank, auto_device = dist_init_if_needed()

    p = argparse.ArgumentParser()

    # Back-compat: ignore --mode if someone passes it
    p.add_argument("--mode", type=str, default="select", help="(ignored)")

    p.add_argument("--build_cache", action="store_true")
    p.add_argument("--cache_dir", type=str, required=True)

    # Cache build options
    p.add_argument("--dataset", type=str, default="/datasets/pile/current")
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--text_field", type=str, default="text")
    p.add_argument("--cache_size", type=int, default=10000)
    p.add_argument("--min_chars", type=int, default=50)
    p.add_argument("--pile_subsets", type=str, default="", help="Comma-separated list of Pile subsets (optional).")
    p.add_argument("--seed", type=int, default=42)

    # Collection options
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--tokenizer", type=str, default="", help="Optional tokenizer override (HF id/path)")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--max_docs", type=int, default=None)

    p.add_argument("--best_buddies_path", type=str, default=None)
    p.add_argument("--which_model", type=int, choices=[1, 2], default=1)
    p.add_argument("--buddy_min_correlation", type=float, default=None)
    p.add_argument("--buddy_top_pairs", type=int, default=None)

    p.add_argument("--layers", type=str, default="all")
    p.add_argument("--neurons", type=str, default=None)
    p.add_argument("--neuron_layer_pairs", type=str, default=None,
                   help="Path to JSON file mapping {layer_idx (str|int): [neuron_idx, ...]}. "
                        "When set, --layers and --neurons are ignored.")

    p.add_argument("--seq_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--per_sample_top", type=int, default=3)
    p.add_argument("--context_size", type=int, default=10)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "float32", "bfloat16"])

    # Tokenizer-alignment knobs
    p.add_argument("--add_special_tokens", action="store_true")
    p.add_argument("--allow_special_tokens_without_char_span", action="store_true",
                   help="Allow special tokens to pass require_char_span even without offsets.")
    p.add_argument("--require_char_span", action="store_true",
                   help="Recommended for cross-tokenizer: only keep examples with non-empty (char_start,char_end).")

    args = p.parse_args()

    # If distributed + cuda requested, override to local device
    if args.device.startswith("cuda") and auto_device.startswith("cuda"):
        args.device = auto_device

    # Build cache mode: only rank0 builds; others wait
    if args.build_cache:
        if is_rank0(rank):
            pile_subsets = parse_pile_subsets(args.pile_subsets)
            tokenizer_id = args.tokenizer.strip() or None
            build_cache(
                cache_dir=args.cache_dir,
                dataset_name=args.dataset,
                split=args.split,
                text_field=args.text_field,
                cache_size=args.cache_size,
                min_chars=args.min_chars,
                context_length=args.seq_length,
                tokenizer_id=tokenizer_id,
                pile_subsets=pile_subsets,
                seed=args.seed,
                rank=rank,
            )
        dist_barrier()
        # Avoid rare shutdown crashes from native thread pools after streaming datasets.
        os._exit(0)

    if args.model is None or args.output_dir is None:
        raise ValueError("Provide --model and --output_dir (or use --build_cache)")

    docs_path, meta_path = cache_paths(args.cache_dir)
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Cache not found: {docs_path}. Run with --build_cache first.")
    cache_meta = json.load(open(meta_path, "r", encoding="utf-8"))
    cache_count = int(cache_meta.get("num_docs_written", 0))
    max_docs = args.max_docs if args.max_docs is not None else cache_count
    max_docs = min(max_docs, cache_count)

    torch_dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]

    tok_id = args.tokenizer.strip() or args.model
    rprint(rank, f"Loading tokenizer: {tok_id}")
    tokenizer = AutoTokenizer.from_pretrained(tok_id, use_fast=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    rprint(rank, f"Loading model: {args.model} on {args.device}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype)
    model.to(args.device)
    model.eval()

    num_layers = get_num_layers(model)
    layers_to_collect = parse_layers_arg(args.layers, num_layers)

    # Determine neurons to collect
    if args.neuron_layer_pairs is not None:
        with open(args.neuron_layer_pairs, "r", encoding="utf-8") as f:
            raw = json.load(f)
        neurons_by_layer = {int(k): [int(n) for n in v] for k, v in raw.items()}
        layers_to_collect = sorted(neurons_by_layer.keys())
    elif args.best_buddies_path is not None:
        bb = load_best_buddies(args.best_buddies_path)
        neurons_by_layer = buddies_to_neurons_by_layer(
            bb,
            which_model=args.which_model,
            min_correlation=args.buddy_min_correlation,
            top_pairs=args.buddy_top_pairs,
        )
        layers_to_collect = [l for l in layers_to_collect if l in neurons_by_layer]
    elif args.neurons is not None:
        neuron_list = [int(n.strip()) for n in args.neurons.split(",") if n.strip()]
        neurons_by_layer = {l: neuron_list for l in layers_to_collect}
    else:
        raise ValueError("Provide --neuron_layer_pairs, --best_buddies_path, or --neurons")

    if not layers_to_collect:
        raise ValueError("No layers to collect.")

    rprint(rank, "Auto-detecting hook points...")
    hook_names, capture_input = find_mlp_hook_names(model, args.model, layers_to_collect)
    layer_to_hook = {li: hn for li, hn in zip(layers_to_collect, hook_names)}

    # Collect ALL layers in a single pass through the data (much faster!)
    os.makedirs(args.output_dir, exist_ok=True)

    rprint(rank, f"Collecting activations for {len(layers_to_collect)} layers in single pass...")
    all_layer_data = collect_all_layers_activations(
        model=model,
        tokenizer=tokenizer,
        layer_to_hook=layer_to_hook,
        neurons_by_layer=neurons_by_layer,
        capture_input=capture_input,
        cache_dir=args.cache_dir,
        max_docs=max_docs,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        top_k=args.top_k,
        per_sample_top=args.per_sample_top,
        context_size=args.context_size,
        device=args.device,
        add_special_tokens=args.add_special_tokens,
        require_char_span=args.require_char_span,
        allow_special_tokens_without_char_span=args.allow_special_tokens_without_char_span,
        rank=rank,
        world_size=world_size,
    )

    # Save per-rank files for each layer
    for layer_idx in layers_to_collect:
        layer_data = all_layer_data.get(layer_idx, {})
        save_layer_rank_file(args.output_dir, layer_idx, layer_data, rank)

    gc.collect()

    # Write rank metadata and merge on rank0
    extra_meta = {
        "model_name": args.model,
        "tokenizer_name": tok_id,
        "layers": layers_to_collect,
        "format": "cached_corpus_aligned_v2_charspan",
        "cache_dir": os.path.abspath(args.cache_dir),
        "cache_meta": cache_meta,
        "max_docs_scanned": max_docs,
        "best_buddies_path": args.best_buddies_path,
        "which_model": args.which_model,
        "buddy_min_correlation": args.buddy_min_correlation,
        "buddy_top_pairs": args.buddy_top_pairs,
        "hook_names": hook_names,
        "capture_input": capture_input,
        "add_special_tokens": bool(args.add_special_tokens),
        "require_char_span": bool(args.require_char_span),
        "allow_special_tokens_without_char_span": bool(args.allow_special_tokens_without_char_span),
        "rank": rank,
        "world_size": world_size,
    }
    write_metadata(args.output_dir, extra_meta, rank)

    dist_barrier()

    if is_rank0(rank):
        for layer_idx in layers_to_collect:
            merge_layer_files(args.output_dir, layer_idx, world_size, args.top_k)
        merge_metadata(args.output_dir, world_size)
        rprint(rank, f"[rank0] Merged activations written to {args.output_dir}")

    dist_barrier()

    if dist_is_on():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
