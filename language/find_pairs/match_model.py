"""Model helpers for match_lm."""

from __future__ import annotations

import functools
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM


def get_mlp_post_activation_modules(model, model_name: str):
    """
    Returns a list of (layer_name, module, capture_output) tuples.

    For most models (GELU/SiLU), we hook the down projection and capture INPUT[0].
    For OPT, we also hook the down projection (fc2) and capture INPUT[0], which
    is post-ReLU.
    """
    layers = []
    name = model_name.lower()

    if "pythia" in name:
        for i, layer in enumerate(model.gpt_neox.layers):
            layers.append((f"layer_{i}", layer.mlp.dense_4h_to_h, False))
    elif "gpt2" in name:
        for i, layer in enumerate(model.transformer.h):
            layers.append((f"layer_{i}", layer.mlp.c_proj, False))
    elif "opt" in name:
        for i, layer in enumerate(model.model.decoder.layers):
            layers.append((f"layer_{i}", layer.fc2, False))
    elif "llama" in name or "gemma" in name or "qwen" in name:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layer_list = model.model.layers
        elif hasattr(model, "layers"):
            layer_list = model.layers
        else:
            raise ValueError(f"Unsupported model structure for {model_name}")
        for i, layer in enumerate(layer_list):
            layers.append((f"layer_{i}", layer.mlp.down_proj, False))
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return layers


class MultiActivationCapture:
    """Captures activations for multiple modules in one forward.

    By default captures INPUT[0] (for down projection layers).
    If capture_output[i] is True, captures OUTPUT instead.
    """
    def __init__(self, modules: List[torch.nn.Module], capture_output: Optional[List[bool]] = None):
        self.modules = modules
        self.capture_output = capture_output or [False] * len(modules)
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.activations: List[Optional[torch.Tensor]] = [None] * len(modules)

    def _make_hook(self, idx: int, use_output: bool):
        def hook(module, input, output):
            if use_output:
                self.activations[idx] = output.detach()
            else:
                self.activations[idx] = input[0].detach()
        return hook

    def register(self):
        for i, m in enumerate(self.modules):
            self.handles.append(m.register_forward_hook(self._make_hook(i, self.capture_output[i])))
        return self

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        self.activations = [None] * len(self.modules)

    def get_and_clear(self, expected_batch: Optional[int] = None, expected_seq: Optional[int] = None) -> List[torch.Tensor]:
        acts = self.activations
        self.activations = [None] * len(self.modules)
        for i, a in enumerate(acts):
            if a is None:
                raise RuntimeError(f"Missing activation for hooked module idx={i}.")
            # Normalize to (batch, seq, hidden)
            if a.ndim == 2:
                if expected_batch is not None and expected_seq is not None:
                    if a.shape[0] == expected_batch * expected_seq:
                        a = a.view(expected_batch, expected_seq, -1)
                    elif a.shape[0] == expected_seq:
                        a = a.unsqueeze(0)
                    elif a.shape[0] == expected_batch:
                        a = a.unsqueeze(1)
                    else:
                        raise RuntimeError(
                            f"Unexpected 2D activation shape for idx={i}: {tuple(a.shape)}, "
                            f"expected batch={expected_batch} seq={expected_seq}"
                        )
                else:
                    a = a.unsqueeze(0)
                acts[i] = a
            elif a.ndim != 3:
                raise RuntimeError(f"Unexpected activation shape for idx={i}: {tuple(a.shape)}")
        return acts  # type: ignore


# ----------------------------
# Backbone-only forward (skip logits)
# ----------------------------

def make_backbone_forward(model):
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as _FSDP
        is_fsdp = isinstance(model, _FSDP)
    except Exception:
        is_fsdp = False

    if is_fsdp:
        # Under FSDP the top-level unit owns sharded params (embeddings, norms,
        # lm_head). Bypassing it via .model/.transformer would use un-gathered
        # shards. Route through the full forward so FSDP gathers correctly;
        # return values are discarded by callers, so the extra lm_head matmul
        # is the only overhead.
        def fwd(*, input_ids, attention_mask):
            return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return fwd

    if hasattr(model, "gpt_neox"):
        backbone = model.gpt_neox
    elif hasattr(model, "transformer"):
        backbone = model.transformer
    elif hasattr(model, "model"):
        backbone = model.model
    else:
        backbone = model  # fallback

    def fwd(*, input_ids, attention_mask):
        return backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    return fwd


def _resolve_transformer_layer_classes(model, model_id: str):
    """Return the set of nn.Module classes that FSDP's auto-wrap policy should
    wrap (one FSDP unit per transformer block)."""
    name = model_id.lower()
    classes: set = set()
    if "pythia" in name:
        from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer
        classes.add(GPTNeoXLayer)
    elif "gpt2" in name:
        from transformers.models.gpt2.modeling_gpt2 import GPT2Block
        classes.add(GPT2Block)
    elif "opt" in name:
        from transformers.models.opt.modeling_opt import OPTDecoderLayer
        classes.add(OPTDecoderLayer)
    elif "qwen" in name:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
        classes.add(Qwen2DecoderLayer)
    elif "llama" in name:
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        classes.add(LlamaDecoderLayer)
    elif "gemma" in name:
        try:
            from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer
            classes.add(GemmaDecoderLayer)
        except Exception:
            pass
    if not classes:
        # Fallback: find any module whose class name ends with "DecoderLayer"
        for m in model.modules():
            cls_name = m.__class__.__name__
            if cls_name.endswith("DecoderLayer") or cls_name.endswith("Block"):
                classes.add(m.__class__)
    if not classes:
        raise ValueError(f"Could not resolve transformer layer class for {model_id}")
    return classes


def _wrap_fsdp(model, device: torch.device, dtype: torch.dtype, model_id: str):
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
        MixedPrecision,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    layer_classes = _resolve_transformer_layer_classes(model, model_id)
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy, transformer_layer_cls=layer_classes
    )
    # Already loaded in `dtype`; keep compute in the same dtype to avoid casts.
    mp = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)
    wrapped = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device,
        use_orig_params=True,
        forward_prefetch=True,
        limit_all_gathers=True,
        mixed_precision=mp,
        sync_module_states=False,
    )
    return wrapped


def _load_model(
    model_id: str,
    dtype: torch.dtype,
    device: torch.device,
    *,
    use_fsdp: bool = False,
    revision: Optional[str] = None,
):
    kwargs = dict(dtype=dtype) if not use_fsdp else dict(dtype=dtype, low_cpu_mem_usage=True)
    if revision:
        kwargs["revision"] = revision
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except TypeError:
        kwargs.pop("dtype", None)
        kwargs["torch_dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if use_fsdp:
        # FSDP moves shards to `device` itself; skip the .to(device) that would
        # transiently materialize the full model on GPU per rank.
        model.eval()
        model.requires_grad_(False)
        model = _wrap_fsdp(model, device, dtype, model_id)
        model.eval()
    else:
        model.to(device)
        model.eval()
    return model
