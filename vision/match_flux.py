#!/usr/bin/env python3
"""
Distributed activation matching between FLUX.2-klein-4B and a vision tower
(OpenCLIP, PixIO, or InternViT).

What this script does
---------------------
- Builds one prompt per ImageNet-1k class (or more by cycling over the 1000 classes).
- Generates images with FLUX.2-klein using the same prompt schedule on every pass.
- Captures FLUX feed-forward activations from the *explicit* image-stream FFNs in the
  double-stream transformer blocks.
- Captures vision-tower MLP activations from either an OpenCLIP visual tower, a PixIO ViT, or an InternViT vision backbone.
- Resamples both activation maps onto a selectable canonical grid (FLUX or the vision tower).
- Computes standardized correlations and mutual top-k matches across neurons/channels.

Design note on FLUX hook placement
----------------------------------
FLUX.2 has two transformer block families:
- `transformer_blocks`: explicit double-stream blocks with image-stream FFNs (`ff`) and
  text-stream FFNs (`ff_context`).
- `single_transformer_blocks`: single-stream blocks whose MLP path is fused with attention.

This script intentionally hooks the explicit image-stream FFNs in `transformer_blocks`
only, because these are the cleanest analogue to the original "match FFN neurons"
setting. In the diffusers source, those FFNs are `Flux2FeedForward` modules with
`linear_in -> act_fn (SwiGLU) -> linear_out`.

Hook kinds
----------
- point_input / input:
    input to `ff.linear_out` (post-activation hidden FFN state; most neuron-like)
- point_output / output:
    output of `ff.linear_out`
- inverted_output:
    output of `ff.linear_in`

Examples
--------
Single GPU smoke test:

  python match_flux2_klein_openclip_imagenet_multigpu.py \
    --save-dir ./debug_flux_match \
    --num-images 16 \
    --batch-size 2 \
    --height 256 \
    --width 256 \
    --disc-arch ViT-L-14 \
    --disc-pretrained datacomp_xl_s13b_b90k \
    --disc-input-size 224

InternViT smoke test:

  python match_flux2_klein_openclip_imagenet_multigpu.py \
    --save-dir ./debug_flux_match_internvit \
    --num-images 16 \
    --batch-size 1 \
    --height 256 \
    --width 256 \
    --disc-family internvit \
    --internvit-model-id OpenGVLab/InternViT-300M-448px \
    --disc-input-size 448

Multi-GPU:

  torchrun --standalone --nproc_per_node=8 match_flux2_klein_openclip_imagenet_multigpu.py \
    --save-dir ./flux_match \
    --num-images 1000 \
    --batch-size 2 \
    --height 256 \
    --width 256 \
    --disc-arch ViT-L-14 \
    --disc-pretrained datacomp_xl_s13b_b90k \
    --disc-input-size 224
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image

try:
    from tqdm.auto import tqdm
except Exception:
    class _TqdmNoop:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable
        def __iter__(self):
            return iter(self.iterable) if self.iterable is not None else iter(())
        def update(self, n=1):
            return None
        def set_postfix(self, *args, **kwargs):
            return None
        def close(self):
            return None
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
    def tqdm(iterable=None, **kwargs):
        return _TqdmNoop(iterable=iterable, **kwargs)
    tqdm.write = print


OFFICIAL_IMAGENET_CLASSES_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

INTERNVIT_MODEL_ALIASES: Dict[str, str] = {
    "internvit_6b_224": "OpenGVLab/InternViT-6B-224px",
    "internvit_6b_224px": "OpenGVLab/InternViT-6B-224px",
    "internvit_6b_448_v1_2": "OpenGVLab/InternViT-6B-448px-V1-2",
    "internvit_6b_448px_v1_2": "OpenGVLab/InternViT-6B-448px-V1-2",
    "internvit_6b_448_v1_5": "OpenGVLab/InternViT-6B-448px-V1-5",
    "internvit_6b_448px_v1_5": "OpenGVLab/InternViT-6B-448px-V1-5",
    "internvit_300m_448": "OpenGVLab/InternViT-300M-448px",
    "internvit_300m_448px": "OpenGVLab/InternViT-300M-448px",
    "internvit_300m_448_v2_5": "OpenGVLab/InternViT-300M-448px-V2_5",
    "internvit_300m_448px_v2_5": "OpenGVLab/InternViT-300M-448px-V2_5",
    "internvit_6b_448_v2_5": "OpenGVLab/InternViT-6B-448px-V2_5",
    "internvit_6b_448px_v2_5": "OpenGVLab/InternViT-6B-448px-V2_5",
    "opengvlab_internvit_6b_224px": "OpenGVLab/InternViT-6B-224px",
    "opengvlab_internvit_6b_448px_v1_2": "OpenGVLab/InternViT-6B-448px-V1-2",
    "opengvlab_internvit_6b_448px_v1_5": "OpenGVLab/InternViT-6B-448px-V1-5",
    "opengvlab_internvit_300m_448px": "OpenGVLab/InternViT-300M-448px",
    "opengvlab_internvit_300m_448px_v2_5": "OpenGVLab/InternViT-300M-448px-V2_5",
    "opengvlab_internvit_6b_448px_v2_5": "OpenGVLab/InternViT-6B-448px-V2_5",
}


# -----------------------------------------------------------------------------
# Distributed helpers
# -----------------------------------------------------------------------------


@dataclass
class DistEnv:
    enabled: bool
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed() -> DistEnv:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistEnv(enabled=False, rank=0, world_size=1, local_rank=0)

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed multi-GPU run requires CUDA.")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return DistEnv(enabled=True, rank=rank, world_size=world_size, local_rank=local_rank)


def cleanup_distributed(dist_env: DistEnv) -> None:
    if dist_env.enabled and dist.is_initialized():
        barrier(dist_env)
        dist.destroy_process_group()


def barrier(dist_env: DistEnv) -> None:
    if not dist_env.enabled:
        return
    if torch.cuda.is_available():
        dist.barrier(device_ids=[dist_env.local_rank])
    else:
        dist.barrier()


def print0(dist_env: DistEnv, *args, **kwargs) -> None:
    if dist_env.is_main:
        print(*args, **kwargs)


def _iter_tensors(tree):
    if tree is None:
        return
    if isinstance(tree, torch.Tensor):
        yield tree
    elif isinstance(tree, (list, tuple)):
        for x in tree:
            yield from _iter_tensors(x)
    elif isinstance(tree, dict):
        for x in tree.values():
            yield from _iter_tensors(x)


def all_reduce_inplace(tree, op=dist.ReduceOp.SUM, dist_env: Optional[DistEnv] = None) -> None:
    if dist_env is None or not dist_env.enabled:
        return
    for t in _iter_tensors(tree):
        dist.all_reduce(t, op=op)


def reduce_inplace_to_root(tree, dist_env: DistEnv, dst: int = 0) -> None:
    if not dist_env.enabled:
        return
    for t in _iter_tensors(tree):
        dist.reduce(t, dst=dst, op=dist.ReduceOp.SUM)


def rank_shard_bounds(num_items: int, world_size: int, rank: int) -> Tuple[int, int]:
    base = num_items // world_size
    rem = num_items % world_size
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return start, end


def tqdm_kwargs(dist_env: DistEnv) -> Dict[str, object]:
    return {"disable": not dist_env.is_main, "dynamic_ncols": True}


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------


def batch_seed(base_seed: int, global_batch_start: int) -> int:
    return int(
        (int(base_seed) * 6364136223846793005 + int(global_batch_start) * 1442695040888963407 + 1)
        % (2**63 - 1)
    )


def torch_dtype_from_name(name: str) -> torch.dtype:
    name = name.lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype name: {name}")
    return mapping[name]


def normalize_hook_kind(hook_kind: str) -> str:
    aliases = {"input": "point_input", "output": "point_output"}
    return aliases.get(hook_kind, hook_kind)


def parse_int_tuple(text: str) -> Tuple[int, ...]:
    items = [x.strip() for x in text.split(",") if x.strip()]
    if not items:
        raise ValueError(f"Expected a comma-separated int list, got: {text!r}")
    return tuple(int(x) for x in items)


def resolve_capture_step(capture_step: str, num_inference_steps: int) -> int:
    if num_inference_steps <= 0:
        raise ValueError(f"num_inference_steps must be > 0, got {num_inference_steps}")

    spec = str(capture_step).strip().lower()
    if spec == "last":
        return num_inference_steps - 1
    if spec == "first":
        return 0

    try:
        step = int(spec)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported capture step spec: {capture_step!r}. "
            "Use 'first', 'last', or an integer step index (negative indices allowed)."
        ) from exc

    if step < 0:
        step += num_inference_steps
    if not (0 <= step < num_inference_steps):
        raise ValueError(
            f"Resolved capture step {step} is out of range for num_inference_steps={num_inference_steps}. "
            f"Requested value: {capture_step!r}"
        )
    return step


def sanitize_filename_component(text: str, max_len: int = 48) -> str:
    text = text.strip().lower().replace(" ", "_")
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "item"
    return text[:max_len]


def add_indefinite_article(label: str) -> str:
    stripped = label.strip()
    if not stripped:
        return label
    lower = stripped.lower()
    if lower.startswith(("a ", "an ", "the ")):
        return stripped
    if lower[0] in "aeiou":
        return f"an {stripped}"
    return f"a {stripped}"


# -----------------------------------------------------------------------------
# Prompt schedule
# -----------------------------------------------------------------------------


@dataclass
class PromptEntry:
    class_idx: int
    raw_label: str
    prompt_label: str
    prompt: str


def _normalize_label_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def choose_prompt_friendly_label(label: str, alias_mode: str = "longest") -> str:
    raw = _normalize_label_text(label)
    if alias_mode == "raw":
        return raw

    aliases = [_normalize_label_text(part) for part in raw.split(",") if _normalize_label_text(part)]
    if not aliases:
        return raw
    if alias_mode == "first":
        return aliases[0]

    def score(alias: str) -> tuple[int, int]:
        words = re.findall(r"[A-Za-z']+", alias)
        lower = alias.lower()
        primary = 0
        primary += min(len(alias), 64)
        primary += 8 * max(len(words) - 1, 0)
        if re.fullmatch(r"[A-Z][a-z]+ [a-z]+", alias):
            primary -= 25
        if len(words) == 1:
            primary -= 4
        if any(token in lower for token in [
            "cat", "dog", "bird", "shark", "snake", "fish", "lizard", "frog", "toad",
            "truck", "car", "plane", "boat", "robe", "suit", "corgi", "retriever",
        ]):
            primary += 6
        return primary, len(alias)

    return max(aliases, key=score)


def load_imagenet_classes(class_list_file: Optional[str] = None) -> List[str]:
    if class_list_file is not None:
        path = Path(class_list_file)
        if not path.exists():
            raise FileNotFoundError(f"Class list file not found: {path}")
        if path.suffix.lower() == ".json":
            labels = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(labels, list):
                raise ValueError("JSON class list must be a list of strings.")
            labels = [str(x).strip() for x in labels]
        else:
            labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(labels) != 1000:
            raise ValueError(f"Expected 1000 ImageNet labels, found {len(labels)} in {path}")
        return labels

    try:
        from torchvision.models import ResNet50_Weights
        labels = list(ResNet50_Weights.IMAGENET1K_V2.meta["categories"])
        if len(labels) == 1000:
            return [str(x).strip() for x in labels]
    except Exception:
        pass

    cache_dir = Path.home() / ".cache" / "rosetta_vision"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "imagenet_classes.txt"
    if not cache_path.exists():
        urllib.request.urlretrieve(OFFICIAL_IMAGENET_CLASSES_URL, cache_path)
    labels = [line.strip() for line in cache_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(labels) != 1000:
        raise ValueError(f"Expected 1000 ImageNet labels from {cache_path}, found {len(labels)}")
    return labels


def build_prompt_schedule(
    labels: Sequence[str],
    prompt_template: str,
    start_index: int,
    num_images: int,
    label_alias_mode: str = "longest",
) -> List[PromptEntry]:
    if start_index < 0:
        raise ValueError("--start-index must be >= 0")
    if num_images <= 0:
        raise ValueError("--num-images must be > 0")
    if len(labels) == 0:
        raise ValueError("No labels are available.")

    entries: List[PromptEntry] = []
    num_labels = len(labels)
    for global_idx in range(start_index, start_index + num_images):
        class_idx = global_idx % num_labels
        raw_label = str(labels[class_idx]).strip()
        prompt_label = choose_prompt_friendly_label(raw_label, alias_mode=label_alias_mode)

        label_article = add_indefinite_article(raw_label)
        prompt_label_article = add_indefinite_article(prompt_label)
        prompt = prompt_template.format(
            label=raw_label,
            label_article=label_article,
            prompt_label=prompt_label,
            prompt_label_article=prompt_label_article,
            class_idx=class_idx,
            global_idx=global_idx,
        )
        entries.append(PromptEntry(class_idx=class_idx, raw_label=raw_label, prompt_label=prompt_label, prompt=prompt))

    return entries


def save_prompt_manifest(entries: Sequence[PromptEntry], save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    def row(e: PromptEntry) -> Dict[str, object]:
        return {
            "class_idx": e.class_idx,
            "raw_label": e.raw_label,
            "prompt_label": e.prompt_label,
            "prompt": e.prompt,
        }

    with open(save_dir / "prompt_manifest.json", "w", encoding="utf-8") as f:
        json.dump([row(e) for e in entries], f, indent=2, ensure_ascii=False)

    with open(save_dir / "prompt_manifest.jsonl", "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(row(e), ensure_ascii=False))
            f.write("\n")

    with open(save_dir / "prompt_manifest.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class_idx", "raw_label", "prompt_label", "prompt"])
        writer.writeheader()
        for e in entries:
            writer.writerow(row(e))


# -----------------------------------------------------------------------------
# Hook utilities
# -----------------------------------------------------------------------------



class MultiActivationCapture:
    """Capture module inputs or outputs for a list of modules, with optional per-hook postprocessing."""

    def __init__(
        self,
        modules: List[torch.nn.Module],
        capture_output: Optional[List[bool]] = None,
        postprocess: Optional[List[Optional[Callable[[torch.Tensor], torch.Tensor]]]] = None,
        capture_call_index: Optional[int] = None,
    ):
        self.modules = modules
        self.capture_output = capture_output or [False] * len(modules)
        self.postprocess = postprocess or [None] * len(modules)
        self.capture_call_index = capture_call_index
        if len(self.capture_output) != len(self.modules):
            raise ValueError(
                f"capture_output length {len(self.capture_output)} != number of modules {len(self.modules)}"
            )
        if len(self.postprocess) != len(self.modules):
            raise ValueError(
                f"postprocess length {len(self.postprocess)} != number of modules {len(self.modules)}"
            )
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.activations: List[Optional[torch.Tensor]] = [None] * len(modules)
        self.call_counts: List[int] = [0] * len(modules)

    def _make_hook(self, idx: int, use_output: bool):
        post_fn = self.postprocess[idx]

        def hook(_module, inputs, output):
            call_idx = self.call_counts[idx]
            self.call_counts[idx] = call_idx + 1
            if self.capture_call_index is not None and call_idx != self.capture_call_index:
                return

            act = output if use_output else inputs[0]
            if isinstance(act, (tuple, list)):
                act = act[0]
            if not isinstance(act, torch.Tensor):
                raise RuntimeError(f"Expected tensor activation from hook {idx}, got {type(act)}")
            act = act.detach()
            if post_fn is not None:
                act = post_fn(act)
                if isinstance(act, (tuple, list)):
                    act = act[0]
                if not isinstance(act, torch.Tensor):
                    raise RuntimeError(f"Expected tensor activation from postprocess {idx}, got {type(act)}")
                act = act.detach()
            self.activations[idx] = act

        return hook

    def register(self) -> "MultiActivationCapture":
        for i, module in enumerate(self.modules):
            self.handles.append(module.register_forward_hook(self._make_hook(i, self.capture_output[i])))
        return self

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.activations = [None] * len(self.modules)
        self.call_counts = [0] * len(self.modules)

    def get_and_clear(self) -> List[torch.Tensor]:
        acts = self.activations
        call_counts = self.call_counts
        self.activations = [None] * len(self.modules)
        self.call_counts = [0] * len(self.modules)
        out: List[torch.Tensor] = []
        for i, act in enumerate(acts):
            if act is None:
                if self.capture_call_index is None:
                    raise RuntimeError(
                        f"Missing activation for hook index {i}. "
                        "For FLUX with num_inference_steps > 1 this script captures the last denoising step only."
                    )
                raise RuntimeError(
                    f"Missing activation for hook index {i}. Requested capture_call_index={self.capture_call_index}, "
                    f"but this module was called {call_counts[i]} times in the forward pass."
                )
            if act.ndim not in (3, 4):
                raise RuntimeError(f"Expected 3D or 4D activation, got {tuple(act.shape)}")
            out.append(act)
        return out


# -----------------------------------------------------------------------------
# Model specs
# -----------------------------------------------------------------------------


@dataclass
class FluxSpec:
    pipeline: object
    transformer: torch.nn.Module
    modules: List[torch.nn.Module]
    layer_names: List[str]
    capture_output: List[bool]
    capture_postprocess: Optional[List[Optional[Callable[[torch.Tensor], torch.Tensor]]]] = None
    patch_token_offset: int = 0
    token_image_patch_size: int = 16
    native_grid_hw: Optional[Tuple[int, int]] = None
    hook_kind: str = "point_input"
    model_id: str = ""
    model_dtype: torch.dtype = torch.float32
    text_encoder_out_layers: Tuple[int, ...] = ()
    notes: Dict[str, object] = None


@dataclass
class TowerSpec:
    family: str
    model: torch.nn.Module
    modules: List[torch.nn.Module]
    layer_names: List[str]
    capture_output: List[bool]
    capture_postprocess: Optional[List[Optional[Callable[[torch.Tensor], torch.Tensor]]]] = None
    patch_token_offset: int = 0
    patch_size: int = 0
    expected_num_patches: Optional[int] = None
    native_grid_hw: Optional[Tuple[int, int]] = None
    preprocess: Callable[[torch.Tensor], torch.Tensor] = None
    forward: Callable[[torch.Tensor], object] = None
    notes: Dict[str, object] = None
    native_image_size_hw: Optional[Tuple[int, int]] = None
    model_dtype: torch.dtype = torch.float32


@dataclass
class LayerStats:
    """Pearson z-score normalizer for one layer's neurons."""
    mean: torch.Tensor
    invstd: torch.Tensor
    dim: int

    def normalize(self, x: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return (x.to(dtype) - self.mean.to(device=device, dtype=dtype)) * self.invstd.to(device=device, dtype=dtype)


@dataclass
class QuantileLayerStats:
    """Spearman rank normalizer: maps activations through an empirical per-neuron CDF.

    Uses midrank for tie handling (average of searchsorted_left and searchsorted_right) and
    empirical standardization from the rank-transformed reservoir. Required for sparse
    activations (ReLU/GELU produce many exact zeros that otherwise blow up the variance).
    """
    sorted_samples: torch.Tensor  # [D, K], float32, ascending
    rank_mean: torch.Tensor       # [D]
    rank_invstd: torch.Tensor     # [D]
    dim: int
    num_global_samples: int

    def normalize(self, x: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        ss = self.sorted_samples.to(device=device, dtype=torch.float32)
        v = x.to(torch.float32).transpose(0, 1).contiguous()
        idx_l = torch.searchsorted(ss, v, right=False)
        idx_r = torch.searchsorted(ss, v, right=True)
        K = ss.shape[1]
        r = 0.5 * (idx_l.to(torch.float32) + idx_r.to(torch.float32)) / float(K)
        mean = self.rank_mean.to(device=device, dtype=torch.float32).unsqueeze(1)
        invstd = self.rank_invstd.to(device=device, dtype=torch.float32).unsqueeze(1)
        z = (r - mean) * invstd
        return z.transpose(0, 1).contiguous().to(dtype)


def _rank_moments_of_sorted(sorted_samples: torch.Tensor, var_thresh: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-neuron mean and invstd of the midrank-transformed reservoir. For (nearly) constant
    neurons (variance below var_thresh) we set invstd=0 so they zero out in normalize() rather
    than amplifying numerical noise into spurious huge correlations."""
    K = sorted_samples.shape[1]
    idx_l = torch.searchsorted(sorted_samples, sorted_samples, right=False).to(torch.float32)
    idx_r = torch.searchsorted(sorted_samples, sorted_samples, right=True).to(torch.float32)
    r_self = 0.5 * (idx_l + idx_r) / float(K)
    mean = r_self.mean(dim=-1)
    var = ((r_self - mean.unsqueeze(-1)) ** 2).mean(dim=-1)
    alive = var > var_thresh
    invstd = torch.where(alive, var.clamp(min=var_thresh).rsqrt(), torch.zeros_like(var))
    return mean, invstd


Normalizer = Union[LayerStats, QuantileLayerStats]



# -----------------------------------------------------------------------------
# FLUX loader
# -----------------------------------------------------------------------------


def _infer_flux_image_grid_hw(image_size_hw: Tuple[int, int], token_image_patch_size: int) -> Tuple[int, int]:
    height, width = image_size_hw
    if height % token_image_patch_size != 0 or width % token_image_patch_size != 0:
        raise ValueError(
            f"Requested size {height}x{width} is not divisible by FLUX token_image_patch_size={token_image_patch_size}."
        )
    return height // token_image_patch_size, width // token_image_patch_size


def _make_take_last_tokens_postprocess(
    num_image_tokens: int,
    feature_start: Optional[int] = None,
    feature_end: Optional[int] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    def postprocess(act: torch.Tensor) -> torch.Tensor:
        if act.ndim != 3:
            raise RuntimeError(
                "Single-stream FLUX postprocess expects a [B, tokens, hidden] tensor, "
                f"got {tuple(act.shape)}"
            )
        if act.shape[1] < num_image_tokens:
            raise RuntimeError(
                f"Expected at least {num_image_tokens} tokens so the last tokens are image tokens, "
                f"got shape {tuple(act.shape)}"
            )
        x = act[:, -num_image_tokens:, :]
        if feature_start is not None or feature_end is not None:
            x = x[..., feature_start:feature_end]
        return x.contiguous()

    return postprocess


def load_flux2_klein(
    model_id: str,
    device: torch.device,
    model_dtype: torch.dtype,
    hook_kind: str,
    text_encoder_out_layers: Tuple[int, ...],
    enable_model_cpu_offload: bool,
    disable_progress_bar: bool,
    image_size_hw: Tuple[int, int],
    block_families: str = "double_and_single",
) -> FluxSpec:
    try:
        from diffusers import Flux2KleinPipeline
    except Exception as exc:
        raise ImportError("Loading FLUX.2-klein requires diffusers with Flux2KleinPipeline support.") from exc

    hook_kind = normalize_hook_kind(hook_kind)
    if block_families not in {"double_only", "double_and_single"}:
        raise ValueError(f"Unsupported FLUX block_families: {block_families}")

    pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=model_dtype)
    try:
        pipe.set_progress_bar_config(disable=disable_progress_bar)
    except Exception:
        pass

    if enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    transformer = pipe.transformer
    if not hasattr(transformer, "transformer_blocks"):
        raise ValueError(f"Expected FLUX transformer with .transformer_blocks, got {type(transformer)}")

    token_image_patch_size = int(getattr(getattr(pipe, "image_processor", object()), "vae_scale_factor", 16))
    image_grid_hw = _infer_flux_image_grid_hw(image_size_hw, token_image_patch_size)
    num_image_tokens = int(image_grid_hw[0] * image_grid_hw[1])

    modules: List[torch.nn.Module] = []
    layer_names: List[str] = []
    capture_output: List[bool] = []
    capture_postprocess: List[Optional[Callable[[torch.Tensor], torch.Tensor]]] = []

    # Explicit double-stream image FFNs.
    for i, block in enumerate(transformer.transformer_blocks):
        ff = getattr(block, "ff", None)
        if ff is None:
            raise ValueError(f"Expected image-stream ff in double-stream layer {i}, got {type(block)}")

        if hook_kind == "point_input":
            proj = getattr(ff, "linear_out", None)
            use_output = False
        elif hook_kind == "point_output":
            proj = getattr(ff, "linear_out", None)
            use_output = True
        elif hook_kind == "inverted_output":
            proj = getattr(ff, "linear_in", None)
            use_output = True
        else:
            raise ValueError(f"Unsupported hook_kind for FLUX: {hook_kind}")

        if proj is None:
            raise ValueError(f"Missing expected FFN projection in double-stream layer {i}, got {type(block)}")

        modules.append(proj)
        layer_names.append(f"flux_ff_{i:02d}")
        capture_output.append(use_output)
        capture_postprocess.append(None)

    # Single-stream fused attention/MLP blocks.
    if block_families == "double_and_single":
        if not hasattr(transformer, "single_transformer_blocks"):
            raise ValueError(
                "Requested single-stream FLUX hooks, but transformer has no .single_transformer_blocks"
            )

        for i, block in enumerate(transformer.single_transformer_blocks):
            attn = getattr(block, "attn", None)
            if attn is None:
                raise ValueError(f"Expected .attn in single-stream layer {i}, got {type(block)}")

            if hook_kind == "point_input":
                proj = getattr(attn, "to_out", None)
                if proj is None:
                    raise ValueError(f"Expected attn.to_out in single-stream layer {i}, got {type(block)}")
                mlp_feature_start = int(getattr(proj, "out_features", 0))
                mlp_feature_end = int(getattr(proj, "in_features", 0))
                if not (0 < mlp_feature_start < mlp_feature_end):
                    raise ValueError(
                        f"Could not infer single-stream MLP slice from to_out in layer {i}: "
                        f"in_features={getattr(proj, 'in_features', None)}, "
                        f"out_features={getattr(proj, 'out_features', None)}"
                    )
                modules.append(proj)
                layer_names.append(f"flux_single_ff_{i:02d}")
                capture_output.append(False)
                capture_postprocess.append(
                    _make_take_last_tokens_postprocess(
                        num_image_tokens=num_image_tokens,
                        feature_start=mlp_feature_start,
                        feature_end=mlp_feature_end,
                    )
                )

            elif hook_kind == "point_output":
                proj = getattr(attn, "to_out", None)
                if proj is None:
                    raise ValueError(f"Expected attn.to_out in single-stream layer {i}, got {type(block)}")
                modules.append(proj)
                layer_names.append(f"flux_single_ff_{i:02d}")
                capture_output.append(True)
                capture_postprocess.append(
                    _make_take_last_tokens_postprocess(num_image_tokens=num_image_tokens)
                )

            elif hook_kind == "inverted_output":
                proj = getattr(attn, "to_qkv_mlp_proj", None)
                if proj is None:
                    raise ValueError(
                        f"Expected attn.to_qkv_mlp_proj in single-stream layer {i}, got {type(block)}"
                    )
                qkv_dim = 3 * int(getattr(proj, "in_features", 0))
                mlp_feature_end = int(getattr(proj, "out_features", 0))
                if not (0 < qkv_dim < mlp_feature_end):
                    raise ValueError(
                        f"Could not infer single-stream MLP slice from to_qkv_mlp_proj in layer {i}: "
                        f"in_features={getattr(proj, 'in_features', None)}, "
                        f"out_features={getattr(proj, 'out_features', None)}"
                    )
                modules.append(proj)
                layer_names.append(f"flux_single_ff_{i:02d}")
                capture_output.append(True)
                capture_postprocess.append(
                    _make_take_last_tokens_postprocess(
                        num_image_tokens=num_image_tokens,
                        feature_start=qkv_dim,
                        feature_end=mlp_feature_end,
                    )
                )

            else:
                raise ValueError(f"Unsupported hook_kind for FLUX: {hook_kind}")

    return FluxSpec(
        pipeline=pipe,
        transformer=transformer,
        modules=modules,
        layer_names=layer_names,
        capture_output=capture_output,
        capture_postprocess=capture_postprocess,
        patch_token_offset=0,
        token_image_patch_size=token_image_patch_size,
        native_grid_hw=image_grid_hw,
        hook_kind=hook_kind,
        model_id=model_id,
        model_dtype=model_dtype,
        text_encoder_out_layers=text_encoder_out_layers,
        notes={
            "token_image_patch_size": token_image_patch_size,
            "image_grid_hw": list(image_grid_hw),
            "num_image_tokens": num_image_tokens,
            "hook_kind": hook_kind,
            "block_families": block_families,
            "num_double_layers_hooked": len(getattr(transformer, "transformer_blocks", [])),
            "num_single_layers_hooked": len(getattr(transformer, "single_transformer_blocks", []))
            if block_families == "double_and_single"
            else 0,
            "single_stream_blocks_hooked": block_families == "double_and_single",
            "single_stream_image_tokens_only": block_families == "double_and_single",
            "text_encoder_out_layers": list(text_encoder_out_layers),
        },
    )


# -----------------------------------------------------------------------------
# Vision-tower loaders
# -----------------------------------------------------------------------------


def _as_hw(size: Optional[int | Sequence[int]]) -> Optional[Tuple[int, int]]:
    if size is None:
        return None
    if isinstance(size, int):
        return int(size), int(size)
    seq = list(size)
    if len(seq) == 1:
        return int(seq[0]), int(seq[0])
    if len(seq) >= 2:
        return int(seq[0]), int(seq[1])
    return None


def _make_image_preprocess(
    mean: Sequence[float],
    std: Sequence[float],
    resize_hw: Optional[Tuple[int, int]] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    mean_t = torch.tensor(mean).view(1, 3, 1, 1)
    std_t = torch.tensor(std).view(1, 3, 1, 1)

    def preprocess(images_m11: torch.Tensor) -> torch.Tensor:
        x = (images_m11.clamp(-1.0, 1.0) + 1.0) / 2.0
        if resize_hw is not None and tuple(x.shape[-2:]) != tuple(resize_hw):
            x = F.interpolate(x, size=resize_hw, mode="bicubic", align_corners=False, antialias=True)
        mean = mean_t.to(device=x.device, dtype=x.dtype)
        std = std_t.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std

    return preprocess


def _extract_openclip_down_proj_module(mlp: torch.nn.Module, layer_idx: int) -> torch.nn.Module:
    if hasattr(mlp, "down_proj"):
        return getattr(mlp, "down_proj")
    if hasattr(mlp, "fc2"):
        return getattr(mlp, "fc2")
    if hasattr(mlp, "c_proj"):
        return getattr(mlp, "c_proj")
    if hasattr(mlp, "w2"):
        return getattr(mlp, "w2")
    if hasattr(mlp, "w3"):
        return getattr(mlp, "w3")
    raise ValueError(f"Unsupported OpenCLIP down projection structure in layer {layer_idx}: {type(mlp)}")


def _extract_openclip_inverted_or_hidden_module(mlp: torch.nn.Module, layer_idx: int) -> torch.nn.Module:
    for name in ("act", "activation", "gelu", "nonlinearity", "act1"):
        mod = getattr(mlp, name, None)
        if isinstance(mod, torch.nn.Module):
            return mod

    if isinstance(mlp, torch.nn.Sequential) and len(mlp) >= 2:
        return mlp[1]

    for name in ("up_proj", "fc1", "c_fc", "gate_proj", "w1", "wi_0"):
        mod = getattr(mlp, name, None)
        if isinstance(mod, torch.nn.Module):
            return mod

    raise ValueError(f"Unsupported OpenCLIP up/hidden projection structure in layer {layer_idx}: {type(mlp)}")


def _normalize_internvit_alias_key(text: str) -> str:
    key = str(text).strip().lower()
    key = key.replace("/", "_").replace("\\", "_")
    key = re.sub(r"[^a-z0-9]+", "_", key)
    key = re.sub(r"_+", "_", key).strip("_")
    return key


def resolve_internvit_model_id(model_name_or_id: str) -> str:
    text = str(model_name_or_id).strip()
    if not text:
        raise ValueError("InternViT model id must be non-empty.")
    if "/" in text or Path(text).expanduser().exists():
        return text

    key = _normalize_internvit_alias_key(text)
    if key in INTERNVIT_MODEL_ALIASES:
        return INTERNVIT_MODEL_ALIASES[key]

    known = sorted(set(INTERNVIT_MODEL_ALIASES.values()))
    raise ValueError(
        f"Unknown InternViT model alias: {model_name_or_id!r}. Pass a full Hugging Face model id/local path, "
        f"or one of the known released checkpoints: {', '.join(known)}"
    )


def _extract_hw_from_processor_value(value) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    if isinstance(value, int):
        return int(value), int(value)
    if isinstance(value, (tuple, list)):
        return _as_hw(value)
    if isinstance(value, dict):
        if "height" in value and "width" in value:
            return int(value["height"]), int(value["width"])
        for key in ("shortest_edge", "shortest_side", "size"):
            if key in value and isinstance(value[key], (int, float)):
                v = int(value[key])
                return v, v
        if len(value) == 1:
            only_value = next(iter(value.values()))
            if isinstance(only_value, (int, float)):
                v = int(only_value)
                return v, v
    return None


def _load_hf_image_processor(model_id: str):
    try:
        from transformers import AutoImageProcessor

        try:
            return AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
        except TypeError:
            return AutoImageProcessor.from_pretrained(model_id)
    except Exception:
        pass

    try:
        from transformers import CLIPImageProcessor

        return CLIPImageProcessor.from_pretrained(model_id)
    except Exception:
        return None


def _resolve_processor_stats_and_size(
    image_processor,
    fallback_image_size_hw: Optional[Tuple[int, int]] = None,
) -> Tuple[Optional[Tuple[int, int]], Optional[Sequence[float]], Optional[Sequence[float]], Optional[str]]:
    if image_processor is None:
        return fallback_image_size_hw, None, None, None

    resize_hw = None
    for attr_name in ("crop_size", "size"):
        resize_hw = _extract_hw_from_processor_value(getattr(image_processor, attr_name, None))
        if resize_hw is not None:
            break

    if resize_hw is None:
        resize_hw = fallback_image_size_hw

    mean = getattr(image_processor, "image_mean", None)
    std = getattr(image_processor, "image_std", None)
    processor_name = image_processor.__class__.__name__
    return resize_hw, mean, std, processor_name


def _infer_conv_patch_size(conv_like: torch.nn.Module, fallback: Optional[int] = None) -> Optional[int]:
    kernel = getattr(conv_like, "kernel_size", None)
    if isinstance(kernel, tuple) and len(kernel) >= 1:
        return int(kernel[0])
    if isinstance(kernel, int):
        return int(kernel)
    return fallback


def load_internvit_tower(
    model_name_or_id: str,
    device: torch.device,
    hook_kind: str,
    model_dtype: torch.dtype,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
    input_size_hw: Optional[Tuple[int, int]] = None,
) -> TowerSpec:
    try:
        from transformers import AutoModel
    except Exception as exc:
        raise ImportError(
            "Loading InternViT requires transformers with trust_remote_code support for the InternVisionModel checkpoints."
        ) from exc

    resolved_model_id = resolve_internvit_model_id(model_name_or_id)

    load_kwargs = {
    "trust_remote_code": True,
    "torch_dtype": model_dtype,
    "low_cpu_mem_usage": False,
    "device_map": None,
}
    model = AutoModel.from_pretrained(resolved_model_id, **load_kwargs)

    model = model.eval()

    vision_model = getattr(model, "vision_model", model)
    encoder = getattr(vision_model, "encoder", None)
    embeddings = getattr(vision_model, "embeddings", None)
    if encoder is None or not hasattr(encoder, "layers"):
        raise ValueError(
            f"Unsupported InternViT structure for {resolved_model_id}: expected an encoder.layers stack, got {type(model)}"
        )
    if embeddings is None:
        raise ValueError(
            f"Unsupported InternViT structure for {resolved_model_id}: expected model.embeddings or vision_model.embeddings."
        )

    blocks = list(encoder.layers)
    patch_embed = getattr(embeddings, "patch_embedding", None)
    config_patch_size = getattr(getattr(model, "config", object()), "patch_size", None)
    patch_size = _infer_conv_patch_size(patch_embed, fallback=int(config_patch_size) if config_patch_size is not None else None)
    if patch_size is None:
        raise ValueError(f"Could not determine InternViT patch size for {resolved_model_id}.")

    config_image_size_hw = _as_hw(getattr(getattr(model, "config", object()), "image_size", None))
    image_processor = _load_hf_image_processor(resolved_model_id)
    processor_image_size_hw, processor_mean, processor_std, processor_name = _resolve_processor_stats_and_size(
        image_processor=image_processor,
        fallback_image_size_hw=config_image_size_hw,
    )

    native_image_size_hw = input_size_hw or processor_image_size_hw or config_image_size_hw
    if native_image_size_hw is None:
        raise ValueError(
            f"Could not infer InternViT input size for {resolved_model_id}; pass --disc-input-size explicitly."
        )
    if native_image_size_hw[0] != native_image_size_hw[1]:
        raise ValueError(
            f"InternViT currently expects square inputs, got {native_image_size_hw[0]}x{native_image_size_hw[1]}."
        )

    if native_image_size_hw[0] % patch_size != 0 or native_image_size_hw[1] % patch_size != 0:
        raise ValueError(
            f"InternViT input size {native_image_size_hw[0]}x{native_image_size_hw[1]} must be divisible by patch size {patch_size}."
        )

    position_embeddings_resized = False
    if config_image_size_hw is not None and tuple(native_image_size_hw) != tuple(config_image_size_hw):
        if not hasattr(vision_model, "resize_pos_embeddings"):
            raise ValueError(
                f"InternViT model {resolved_model_id} does not expose resize_pos_embeddings; cannot change input size "
                f"from {config_image_size_hw[0]} to {native_image_size_hw[0]}."
            )
        if native_image_size_hw[0] != native_image_size_hw[1]:
            raise ValueError("InternViT positional embedding resize currently supports square inputs only.")
        vision_model.resize_pos_embeddings(
            old_size=int(config_image_size_hw[0]),
            new_size=int(native_image_size_hw[0]),
            patch_size=int(patch_size),
        )
        try:
            model.config.image_size = int(native_image_size_hw[0])
        except Exception:
            pass
        for attr_name, attr_value in (
            ("image_size", int(native_image_size_hw[0])),
            ("num_patches", int((native_image_size_hw[0] // patch_size) * (native_image_size_hw[1] // patch_size))),
            ("num_positions", int((native_image_size_hw[0] // patch_size) * (native_image_size_hw[1] // patch_size) + 1)),
        ):
            if hasattr(embeddings, attr_name):
                try:
                    setattr(embeddings, attr_name, attr_value)
                except Exception:
                    pass
        position_embeddings_resized = True

    model = model.to(device)
    if model_dtype != torch.float32:
        model = model.to(dtype=model_dtype)

    hook_kind = normalize_hook_kind(hook_kind)
    modules: List[torch.nn.Module] = []
    layer_names: List[str] = []
    capture_output: List[bool] = []

    for i, blk in enumerate(blocks):
        mlp = getattr(blk, "mlp", None)
        if mlp is None:
            raise ValueError(f"Expected .mlp in InternViT block {i}, got {type(blk)}")

        if hook_kind == "point_input":
            proj = getattr(mlp, "fc2", None)
            use_output = False
        elif hook_kind == "point_output":
            proj = getattr(mlp, "fc2", None)
            use_output = True
        elif hook_kind == "inverted_output":
            proj = getattr(mlp, "act", None) or getattr(mlp, "fc1", None)
            use_output = True
        else:
            raise ValueError(f"Unsupported hook_kind for InternViT: {hook_kind}")

        if proj is None:
            raise ValueError(f"Missing expected InternViT MLP projection in layer {i}, got {type(blk)}")

        modules.append(proj)
        layer_names.append(f"disc_block_{i:02d}")
        capture_output.append(use_output)

    if mean is None:
        mean = processor_mean or [0.485, 0.456, 0.406]
    if std is None:
        std = processor_std or [0.229, 0.224, 0.225]

    native_grid_hw = (
        int(native_image_size_hw[0] // patch_size),
        int(native_image_size_hw[1] // patch_size),
    )
    expected_num_patches = int(native_grid_hw[0] * native_grid_hw[1])

    preprocess = _make_image_preprocess(mean, std, resize_hw=native_image_size_hw)

    def forward(images: torch.Tensor):
        try:
            return model(pixel_values=images)
        except TypeError:
            return model(images)

    return TowerSpec(
        family="internvit",
        model=model,
        modules=modules,
        layer_names=layer_names,
        capture_output=capture_output,
        patch_token_offset=1,
        patch_size=patch_size,
        expected_num_patches=expected_num_patches,
        native_grid_hw=native_grid_hw,
        preprocess=preprocess,
        forward=forward,
        notes={
            "model_id": resolved_model_id,
            "requested_model_name_or_id": model_name_or_id,
            "processor_name": processor_name,
            "processor_image_size_hw": list(processor_image_size_hw) if processor_image_size_hw is not None else None,
            "config_image_size_hw": list(config_image_size_hw) if config_image_size_hw is not None else None,
            "patch_token_offset": 1,
            "hook_kind": hook_kind,
            "position_embeddings_resized": position_embeddings_resized,
        },
        native_image_size_hw=native_image_size_hw,
        model_dtype=model_dtype,
    )


def load_openclip_tower(
    model_name: str,
    pretrained: str,
    device: torch.device,
    hook_kind: str,
    model_dtype: torch.dtype,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
    input_size_hw: Optional[Tuple[int, int]] = None,
) -> TowerSpec:
    import open_clip

    model, _, preprocess_tf = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    model = model.eval()
    if model_dtype != torch.float32:
        model = model.to(dtype=model_dtype)

    visual = model.visual

    def _infer_patch_size_from_patch_embed(patch_embed: torch.nn.Module) -> Optional[int]:
        patch_size_obj = getattr(patch_embed, "patch_size", None)
        if isinstance(patch_size_obj, tuple):
            return int(patch_size_obj[0])
        if isinstance(patch_size_obj, int):
            return int(patch_size_obj)
        return None

    def _infer_num_patches_from_patch_embed(patch_embed: torch.nn.Module) -> Optional[int]:
        if hasattr(patch_embed, "num_patches"):
            try:
                return int(getattr(patch_embed, "num_patches"))
            except Exception:
                pass
        grid_size = getattr(patch_embed, "grid_size", None)
        if isinstance(grid_size, tuple) and len(grid_size) >= 2:
            try:
                return int(grid_size[0] * grid_size[1])
            except Exception:
                pass
        if isinstance(grid_size, int):
            try:
                return int(grid_size * grid_size)
            except Exception:
                pass
        return None

    blocks: List[torch.nn.Module]
    patch_size: Optional[int] = None
    expected_num_patches: Optional[int] = None
    patch_offset: int = 1
    visual_kind: str = "unknown"
    native_image_size_hw = input_size_hw
    native_grid_hw: Optional[Tuple[int, int]] = None

    if hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks"):
        visual_kind = "openclip_vit"
        blocks = list(visual.transformer.resblocks)
        patch_offset = int(getattr(visual, "num_prefix_tokens", 1))

        conv1 = getattr(visual, "conv1", None)
        kernel = getattr(conv1, "kernel_size", None)
        if isinstance(kernel, tuple):
            patch_size = int(kernel[0])
        elif isinstance(kernel, int):
            patch_size = int(kernel)

        grid_size = getattr(visual, "grid_size", None)
        if isinstance(grid_size, tuple):
            expected_num_patches = int(grid_size[0] * grid_size[1])
        elif isinstance(grid_size, int):
            expected_num_patches = int(grid_size * grid_size)

        if native_image_size_hw is None:
            native_image_size_hw = _as_hw(getattr(visual, "image_size", None) or getattr(visual, "input_resolution", None))

    elif hasattr(visual, "trunk") and hasattr(visual.trunk, "blocks"):
        visual_kind = "openclip_timm"
        trunk = visual.trunk

        if input_size_hw is not None:
            try:
                if hasattr(visual, "set_input_size"):
                    visual.set_input_size(input_size_hw)
                elif hasattr(trunk, "set_input_size"):
                    trunk.set_input_size(input_size_hw)
            except Exception:
                pass

        blocks = list(trunk.blocks)
        patch_offset = int(getattr(trunk, "num_prefix_tokens", getattr(visual, "num_prefix_tokens", 1)))

        patch_embed = getattr(trunk, "patch_embed", None)
        if patch_embed is None:
            raise ValueError(f"OpenCLIP timm visual tower {type(trunk)} has no patch_embed; cannot infer patch size.")

        patch_size = _infer_patch_size_from_patch_embed(patch_embed)
        expected_num_patches = _infer_num_patches_from_patch_embed(patch_embed)

        if native_image_size_hw is None:
            native_image_size_hw = _as_hw(
                getattr(visual, "image_size", None)
                or getattr(trunk, "img_size", None)
                or getattr(trunk, "image_size", None)
                or getattr(patch_embed, "img_size", None)
            )

    elif hasattr(visual, "blocks"):
        visual_kind = "generic_vit_blocks"
        blocks = list(visual.blocks)
        patch_offset = int(getattr(visual, "num_prefix_tokens", 1))

        patch_embed = getattr(visual, "patch_embed", None)
        if patch_embed is not None:
            patch_size = _infer_patch_size_from_patch_embed(patch_embed)
            expected_num_patches = _infer_num_patches_from_patch_embed(patch_embed)

        if native_image_size_hw is None:
            native_image_size_hw = _as_hw(getattr(visual, "img_size", None) or getattr(visual, "image_size", None))

    else:
        raise ValueError(
            "Unsupported OpenCLIP visual tower structure. "
            f"visual type: {type(visual)}. "
            "Expected one of: visual.transformer.resblocks, visual.trunk.blocks, or visual.blocks."
        )

    if patch_size is None:
        raise ValueError(
            f"Could not determine patch size for OpenCLIP visual tower type {type(visual)} "
            f"(kind={visual_kind})."
        )

    hook_kind = normalize_hook_kind(hook_kind)
    if hook_kind == "point_input":
        capture_output = [False] * len(blocks)
    elif hook_kind in {"point_output", "inverted_output"}:
        capture_output = [True] * len(blocks)
    else:
        raise ValueError(f"Unsupported hook_kind for OpenCLIP: {hook_kind}")

    modules: List[torch.nn.Module] = []
    layer_names: List[str] = []
    for i, blk in enumerate(blocks):
        if not hasattr(blk, "mlp"):
            raise ValueError(f"Expected .mlp in OpenCLIP block {i}, got {type(blk)} (kind={visual_kind}).")
        if hook_kind in {"point_input", "point_output"}:
            modules.append(_extract_openclip_down_proj_module(blk.mlp, i))
        else:
            modules.append(_extract_openclip_inverted_or_hidden_module(blk.mlp, i))
        layer_names.append(f"disc_block_{i:02d}")

    if mean is None or std is None:
        norm_mean = None
        norm_std = None
        for tf in getattr(preprocess_tf, "transforms", []):
            if tf.__class__.__name__ == "Normalize":
                norm_mean = list(tf.mean)
                norm_std = list(tf.std)
                break
        if norm_mean is None or norm_std is None:
            norm_mean = [0.48145466, 0.4578275, 0.40821073]
            norm_std = [0.26862954, 0.26130258, 0.27577711]
        mean = norm_mean
        std = norm_std

    if native_image_size_hw is not None:
        if native_image_size_hw[0] % patch_size != 0 or native_image_size_hw[1] % patch_size != 0:
            raise ValueError(
                f"OpenCLIP input size {native_image_size_hw[0]}x{native_image_size_hw[1]} must be divisible by patch size {patch_size}."
            )
        native_grid_hw = (
            int(native_image_size_hw[0] // patch_size),
            int(native_image_size_hw[1] // patch_size),
        )

    preprocess = _make_image_preprocess(mean, std, resize_hw=native_image_size_hw)

    def forward(images: torch.Tensor):
        return visual(images)

    return TowerSpec(
        family="openclip",
        model=model,
        modules=modules,
        layer_names=layer_names,
        capture_output=capture_output,
        patch_token_offset=patch_offset,
        patch_size=patch_size,
        expected_num_patches=expected_num_patches,
        native_grid_hw=native_grid_hw,
        preprocess=preprocess,
        forward=forward,
        notes={
            "model_name": model_name,
            "pretrained": pretrained,
            "visual_kind": visual_kind,
            "patch_token_offset": patch_offset,
            "hook_kind": hook_kind,
        },
        native_image_size_hw=native_image_size_hw,
        model_dtype=model_dtype,
    )




def _resolve_pixio_module_dir(module_dir: Optional[str]) -> Optional[Path]:
    if module_dir is None:
        return None

    path = Path(module_dir).expanduser().resolve()
    candidates = [path, path / "pixio"]
    for candidate in candidates:
        if (candidate / "pixio.py").exists() and (candidate / "layers").is_dir():
            return candidate

    raise FileNotFoundError(
        f"Could not locate a PixIO source directory under {path}. "
        "Expected either the directory itself, or a child directory, to contain "
        "'pixio.py' and a sibling 'layers/' package."
    )


def _import_pixio_module(module_dir: Optional[str] = None):
    resolved_dir = _resolve_pixio_module_dir(module_dir)
    if resolved_dir is not None:
        resolved_str = str(resolved_dir)
        if resolved_str not in sys.path:
            sys.path.insert(0, resolved_str)

        existing_pixio = sys.modules.get("pixio")
        existing_pixio_file = getattr(existing_pixio, "__file__", None)
        if existing_pixio_file is not None and Path(existing_pixio_file).expanduser().resolve().parent != resolved_dir:
            del sys.modules["pixio"]

        existing_layers = sys.modules.get("layers")
        existing_layers_file = getattr(existing_layers, "__file__", None)
        if existing_layers_file is not None and Path(existing_layers_file).expanduser().resolve().parent != resolved_dir / "layers":
            stale_layer_keys = [name for name in sys.modules if name == "layers" or name.startswith("layers.")]
            for key in stale_layer_keys:
                del sys.modules[key]

    try:
        return importlib.import_module("pixio")
    except Exception as exc:
        location_note = (
            f" from source directory {resolved_dir}"
            if resolved_dir is not None
            else ""
        )
        raise ImportError(
            "Loading PixIO requires the PixIO source code to be importable as "
            "`import pixio` (for example by passing --pixio-module-dir to the "
            "directory that contains pixio.py and layers/)."
            + location_note
        ) from exc


def _extract_pixio_mlp_inverted_or_hidden_module(mlp: torch.nn.Module, layer_idx: int) -> torch.nn.Module:
    for name in ("act", "activation", "gelu", "nonlinearity", "act1"):
        mod = getattr(mlp, name, None)
        if isinstance(mod, torch.nn.Module):
            return mod

    for name in ("fc1", "up_proj", "c_fc", "gate_proj", "w1", "wi_0"):
        mod = getattr(mlp, name, None)
        if isinstance(mod, torch.nn.Module):
            return mod

    raise ValueError(f"Unsupported PixIO up/hidden projection structure in layer {layer_idx}: {type(mlp)}")


def load_pixio_tower(
    model_name: str,
    checkpoint_path: Optional[str],
    device: torch.device,
    hook_kind: str,
    model_dtype: torch.dtype,
    module_dir: Optional[str] = None,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
    input_size_hw: Optional[Tuple[int, int]] = None,
) -> TowerSpec:
    pixio_module = _import_pixio_module(module_dir=module_dir)
    model_ctor = getattr(pixio_module, model_name, None)
    if model_ctor is None or not callable(model_ctor):
        available = sorted(name for name in dir(pixio_module) if name.startswith("pixio_vit"))
        raise ValueError(
            f"Unknown PixIO model '{model_name}'. Available constructors include: {available}"
        )

    checkpoint = None
    if checkpoint_path is not None and str(checkpoint_path).strip():
        checkpoint = str(Path(checkpoint_path).expanduser().resolve())
        if not Path(checkpoint).exists():
            raise FileNotFoundError(f"PixIO checkpoint not found: {checkpoint}")

    model = model_ctor(pretrained=checkpoint)
    model = model.eval().to(device)
    if model_dtype != torch.float32:
        model = model.to(dtype=model_dtype)

    if not hasattr(model, "blocks"):
        raise ValueError(f"Expected PixIO model with .blocks, got {type(model)}")
    if not hasattr(model, "patch_embed"):
        raise ValueError(f"Expected PixIO model with .patch_embed, got {type(model)}")

    blocks = list(model.blocks)
    patch_embed = model.patch_embed
    patch_size_obj = getattr(patch_embed, "patch_size", None)
    if isinstance(patch_size_obj, tuple):
        patch_size = int(patch_size_obj[0])
    elif isinstance(patch_size_obj, int):
        patch_size = int(patch_size_obj)
    else:
        raise ValueError(f"Could not determine PixIO patch size from patch_embed: {type(patch_embed)}")

    n_cls_tokens = int(getattr(model, "n_cls_tokens", 0))
    if n_cls_tokens <= 0:
        raise ValueError(
            f"Expected PixIO model to expose a positive n_cls_tokens, got {n_cls_tokens}"
        )

    hook_kind = normalize_hook_kind(hook_kind)
    modules: List[torch.nn.Module] = []
    layer_names: List[str] = []
    capture_output: List[bool] = []

    for i, blk in enumerate(blocks):
        mlp = getattr(blk, "mlp", None)
        if mlp is None:
            raise ValueError(f"Expected .mlp in PixIO block {i}, got {type(blk)}")

        if hook_kind == "point_input":
            proj = getattr(mlp, "fc2", None)
            use_output = False
        elif hook_kind == "point_output":
            proj = getattr(mlp, "fc2", None)
            use_output = True
        elif hook_kind == "inverted_output":
            proj = _extract_pixio_mlp_inverted_or_hidden_module(mlp, i)
            use_output = True
        else:
            raise ValueError(f"Unsupported hook_kind for PixIO: {hook_kind}")

        if proj is None:
            raise ValueError(f"Missing expected PixIO MLP projection in layer {i}, got {type(blk)}")

        modules.append(proj)
        layer_names.append(f"disc_block_{i:02d}")
        capture_output.append(use_output)

    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    preprocess = _make_image_preprocess(mean, std, resize_hw=input_size_hw)

    def forward(images: torch.Tensor):
        return model(images)

    expected_num_patches = None
    native_grid_hw = None
    if input_size_hw is not None:
        if input_size_hw[0] % patch_size != 0 or input_size_hw[1] % patch_size != 0:
            raise ValueError(
                f"PixIO input size {input_size_hw[0]}x{input_size_hw[1]} must be divisible by patch size {patch_size}."
            )
        native_grid_hw = (
            int(input_size_hw[0] // patch_size),
            int(input_size_hw[1] // patch_size),
        )
        expected_num_patches = int(native_grid_hw[0] * native_grid_hw[1])

    return TowerSpec(
        family="pixio",
        model=model,
        modules=modules,
        layer_names=layer_names,
        capture_output=capture_output,
        patch_token_offset=n_cls_tokens,
        patch_size=patch_size,
        expected_num_patches=expected_num_patches,
        native_grid_hw=native_grid_hw,
        preprocess=preprocess,
        forward=forward,
        notes={
            "model_name": model_name,
            "checkpoint_path": checkpoint,
            "patch_token_offset": n_cls_tokens,
            "n_cls_tokens": n_cls_tokens,
            "hook_kind": hook_kind,
        },
        native_image_size_hw=input_size_hw,
        model_dtype=model_dtype,
    )


# -----------------------------------------------------------------------------
# DINOv3 loader
# -----------------------------------------------------------------------------


def _extract_dinov3_down_proj_module(mlp: torch.nn.Module, layer_idx: int) -> torch.nn.Module:
    # Local SwiGLUFFN: w1/w2/w3, w3 is the down-projection.
    if hasattr(mlp, "w3") and hasattr(mlp, "w1") and hasattr(mlp, "w2"):
        return mlp.w3
    # HF DINOv3ViTGatedMLP: gate_proj/up_proj/down_proj.
    if hasattr(mlp, "down_proj"):
        return mlp.down_proj
    # Local Mlp: fc1/fc2.
    if hasattr(mlp, "fc2"):
        return mlp.fc2
    raise ValueError(
        f"Unsupported DINOv3 MLP structure in block {layer_idx}: {type(mlp)}"
    )


def _extract_dinov3_hidden_module(mlp: torch.nn.Module, layer_idx: int) -> torch.nn.Module:
    # Local Mlp: fc1 output is the hidden activation.
    has_swiglu_local = hasattr(mlp, "w1") and hasattr(mlp, "w2") and hasattr(mlp, "w3")
    has_gated_hf = hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj") and hasattr(mlp, "down_proj")
    if hasattr(mlp, "fc1") and not has_swiglu_local and not has_gated_hf:
        return mlp.fc1
    # Both gated variants have no single hidden module: act(gate) * up is a functional op.
    if has_swiglu_local or has_gated_hf:
        raise NotImplementedError(
            f"inverted_output hook is not supported for DINOv3 gated/SwiGLU MLP at block {layer_idx}; "
            "use point_input or point_output instead."
        )
    raise ValueError(
        f"Unsupported DINOv3 MLP structure in block {layer_idx}: {type(mlp)}"
    )


def _finalize_dinov3_tower(
    model: torch.nn.Module,
    device: torch.device,
    hook_kind: str,
    model_dtype: torch.dtype,
    mean: Optional[Sequence[float]],
    std: Optional[Sequence[float]],
    input_size_hw: Optional[Tuple[int, int]],
    notes_extra: Dict[str, object],
    forward_fn: Callable[[torch.Tensor], object],
) -> TowerSpec:
    if hasattr(model, "blocks"):
        blocks = list(model.blocks)
    elif hasattr(model, "layer"):
        blocks = list(model.layer)
    elif hasattr(model, "model") and hasattr(model.model, "layer"):
        blocks = list(model.model.layer)
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        blocks = list(model.encoder.layer)
    else:
        raise ValueError(
            "Expected DINOv3 model with .blocks, .layer, .model.layer, or .encoder.layer"
        )

    hook_kind = normalize_hook_kind(hook_kind)
    modules: List[torch.nn.Module] = []
    layer_names: List[str] = []
    capture_output: List[bool] = []

    for i, blk in enumerate(blocks):
        mlp = getattr(blk, "mlp", None)
        if mlp is None:
            raise ValueError(f"Expected .mlp in DINOv3 block {i}, got {type(blk)}")

        if hook_kind == "point_input":
            proj = _extract_dinov3_down_proj_module(mlp, i)
            use_output = False
        elif hook_kind == "point_output":
            proj = _extract_dinov3_down_proj_module(mlp, i)
            use_output = True
        elif hook_kind == "inverted_output":
            proj = _extract_dinov3_hidden_module(mlp, i)
            use_output = True
        else:
            raise ValueError(f"Unsupported hook_kind for DINOv3: {hook_kind}")

        modules.append(proj)
        layer_names.append(f"disc_block_{i:02d}")
        capture_output.append(use_output)

    config = getattr(model, "config", None)
    patch_size_obj = (
        getattr(model, "patch_size", None)
        or getattr(config, "patch_size", None)
    )
    if isinstance(patch_size_obj, tuple):
        patch_size = int(patch_size_obj[0])
    elif isinstance(patch_size_obj, int):
        patch_size = int(patch_size_obj)
    else:
        raise ValueError("Could not determine DINOv3 patch size.")

    n_storage_tokens = int(
        getattr(model, "n_storage_tokens", 0)
        or getattr(config, "n_storage_tokens", 0)
        or getattr(model, "num_register_tokens", 0)
        or getattr(config, "num_register_tokens", 0)
    )
    patch_offset = 1 + n_storage_tokens

    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    preprocess = _make_image_preprocess(mean, std, resize_hw=input_size_hw)

    expected_num_patches = None
    native_grid_hw = None
    if input_size_hw is not None:
        if input_size_hw[0] % patch_size != 0 or input_size_hw[1] % patch_size != 0:
            raise ValueError(
                f"DINOv3 input size {input_size_hw[0]}x{input_size_hw[1]} "
                f"must be divisible by patch size {patch_size}."
            )
        native_grid_hw = (
            int(input_size_hw[0] // patch_size),
            int(input_size_hw[1] // patch_size),
        )
        expected_num_patches = int(native_grid_hw[0] * native_grid_hw[1])

    notes = {
        "hook_kind": hook_kind,
        "n_storage_tokens": n_storage_tokens,
        "patch_token_offset": patch_offset,
    }
    notes.update(notes_extra)

    return TowerSpec(
        family="dinov3",
        model=model,
        modules=modules,
        layer_names=layer_names,
        capture_output=capture_output,
        patch_token_offset=patch_offset,
        patch_size=patch_size,
        expected_num_patches=expected_num_patches,
        native_grid_hw=native_grid_hw,
        preprocess=preprocess,
        forward=forward_fn,
        notes=notes,
        native_image_size_hw=input_size_hw,
        model_dtype=model_dtype,
    )


def load_dinov3_local_tower(
    repo: str,
    arch: str,
    weights: str,
    device: torch.device,
    hook_kind: str,
    model_dtype: torch.dtype,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
    input_size_hw: Optional[Tuple[int, int]] = None,
) -> TowerSpec:
    if not weights:
        raise ValueError("DINOv3 local loading requires --disc-weights.")
    model = torch.hub.load(str(Path(repo).resolve()), arch, source="local", weights=weights)
    model = model.to(device).eval()
    if model_dtype != torch.float32:
        model = model.to(dtype=model_dtype)

    def forward(images: torch.Tensor):
        if hasattr(model, "forward_features"):
            return model.forward_features(images)
        return model(images)

    return _finalize_dinov3_tower(
        model=model,
        device=device,
        hook_kind=hook_kind,
        model_dtype=model_dtype,
        mean=mean,
        std=std,
        input_size_hw=input_size_hw,
        notes_extra={"arch": arch, "loader": "local", "repo": str(repo), "weights": weights},
        forward_fn=forward,
    )


def load_dinov3_hf_tower(
    model_id: str,
    device: torch.device,
    hook_kind: str,
    model_dtype: torch.dtype,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
    input_size_hw: Optional[Tuple[int, int]] = None,
) -> TowerSpec:
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise ImportError(
            "Loading DINOv3 via Hugging Face requires transformers with DINOv3 support."
        ) from exc

    model = AutoModel.from_pretrained(model_id).to(device).eval()
    if model_dtype != torch.float32:
        model = model.to(dtype=model_dtype)

    def forward(images: torch.Tensor):
        model_images = images.to(device=device, dtype=model_dtype, non_blocking=True)
        try:
            return model(pixel_values=model_images)
        except TypeError:
            return model(model_images)

    return _finalize_dinov3_tower(
        model=model,
        device=device,
        hook_kind=hook_kind,
        model_dtype=model_dtype,
        mean=mean,
        std=std,
        input_size_hw=input_size_hw,
        notes_extra={"arch": model_id, "loader": "hf", "model_id": model_id},
        forward_fn=forward,
    )


# -----------------------------------------------------------------------------
# Activation slicing / resampling / stats / top-k utilities
# -----------------------------------------------------------------------------


def _infer_square_grid(num_tokens: int) -> Tuple[int, int]:
    side = int(round(math.sqrt(num_tokens)))
    if side * side != num_tokens:
        raise RuntimeError(f"Expected a square token grid, got {num_tokens} patch tokens.")
    return side, side


def _activation_to_spatial_map(
    act: torch.Tensor,
    patch_token_offset: int,
    known_grid_hw: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    if act.ndim == 3:
        if patch_token_offset >= act.shape[1]:
            raise RuntimeError(
                f"Patch token offset {patch_token_offset} >= token count {act.shape[1]}; wrong hook point or prefix size."
            )
        x = act[:, patch_token_offset:, :]
        if known_grid_hw is None:
            grid_hw = _infer_square_grid(x.shape[1])
        else:
            if int(known_grid_hw[0] * known_grid_hw[1]) != int(x.shape[1]):
                raise RuntimeError(
                    f"Known grid {known_grid_hw[0]}x{known_grid_hw[1]} implies "
                    f"{known_grid_hw[0] * known_grid_hw[1]} tokens, but activation has {x.shape[1]} patch tokens."
                )
            grid_hw = known_grid_hw
        x = x.reshape(x.shape[0], grid_hw[0], grid_hw[1], x.shape[-1])
        return x, grid_hw
    if act.ndim == 4:
        x = act.permute(0, 2, 3, 1).contiguous()
        grid_hw = (int(act.shape[2]), int(act.shape[3]))
        return x, grid_hw
    raise RuntimeError(f"Expected [B, tokens, hidden] or [B, C, H, W], got {tuple(act.shape)}")


def _flatten_activation_on_grid(
    act: torch.Tensor,
    patch_token_offset: int,
    target_grid_hw: Optional[Tuple[int, int]],
    resample_mode: str,
    native_grid_hw_hint: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    x, native_grid_hw = _activation_to_spatial_map(act, patch_token_offset, known_grid_hw=native_grid_hw_hint)
    if target_grid_hw is not None and native_grid_hw != target_grid_hw:
        x = x.permute(0, 3, 1, 2)
        if resample_mode in {"bilinear", "bicubic"}:
            x = F.interpolate(x, size=target_grid_hw, mode=resample_mode, align_corners=False)
        else:
            x = F.interpolate(x, size=target_grid_hw, mode=resample_mode)
        x = x.permute(0, 2, 3, 1).contiguous()
    x = x.reshape(-1, x.shape[-1])
    return x, native_grid_hw


def _allocate_running_stats(dims: Sequence[int], device: torch.device) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    sums = [torch.zeros(d, dtype=torch.float64, device=device) for d in dims]
    sumsqs = [torch.zeros(d, dtype=torch.float64, device=device) for d in dims]
    return sums, sumsqs


def _finalize_stats(sums: List[torch.Tensor], sumsqs: List[torch.Tensor], count: int, eps: float = 1e-6) -> List[LayerStats]:
    stats: List[LayerStats] = []
    if count <= 0:
        raise RuntimeError("No samples were accumulated.")
    for s, ssq in zip(sums, sumsqs):
        mean = (s / count).to(torch.float32)
        var = (ssq / count).to(torch.float32) - mean.square()
        var = torch.clamp(var, min=eps)
        invstd = torch.rsqrt(var)
        stats.append(LayerStats(mean=mean, invstd=invstd, dim=int(mean.numel())))
    return stats


class _ReservoirCollector:
    """Per-layer A-Res reservoir sampler. Keeps K samples per neuron on `device`."""

    def __init__(self, K: int, device: torch.device, seed: int):
        self.K = int(K)
        self.device = device
        self.values: Optional[torch.Tensor] = None
        self.keys: Optional[torch.Tensor] = None
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(int(seed))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x32 = x.to(torch.float32)
        N, D = x32.shape
        if self.values is None:
            self.values = torch.zeros((self.K, D), dtype=torch.float32, device=self.device)
            self.keys = torch.full((self.K, D), float("inf"), dtype=torch.float32, device=self.device)
        new_keys = torch.rand((N, D), generator=self.generator, device=self.device, dtype=torch.float32)
        all_keys = torch.cat([self.keys, new_keys], dim=0)
        all_values = torch.cat([self.values, x32], dim=0)
        topk_keys, topk_idx = torch.topk(all_keys, k=self.K, dim=0, largest=False)
        self.keys = topk_keys
        self.values = torch.gather(all_values, 0, topk_idx)

    @torch.no_grad()
    def finalize(self, dist_env: DistEnv) -> torch.Tensor:
        if self.values is None or self.keys is None:
            raise RuntimeError("Reservoir received no samples.")
        local_values = self.values
        local_keys = self.keys
        if dist_env.enabled and dist_env.world_size > 1:
            vals_list = [torch.zeros_like(local_values) for _ in range(dist_env.world_size)]
            keys_list = [torch.zeros_like(local_keys) for _ in range(dist_env.world_size)]
            dist.all_gather(vals_list, local_values)
            dist.all_gather(keys_list, local_keys)
            merged_values = torch.cat(vals_list, dim=0)
            merged_keys = torch.cat(keys_list, dim=0)
        else:
            merged_values = local_values
            merged_keys = local_keys
        valid = torch.isfinite(merged_keys)
        merged_values_safe = torch.where(valid, merged_values, torch.full_like(merged_values, float("inf")))
        sorted_full, _ = torch.sort(merged_values_safe.transpose(0, 1).contiguous(), dim=-1)
        valid_counts = valid.to(torch.int64).sum(dim=0)
        min_valid = int(valid_counts.min().item()) if valid_counts.numel() > 0 else 0
        if min_valid <= 0:
            raise RuntimeError("Reservoir collected zero valid samples for some neurons.")
        return sorted_full[:, :min_valid].contiguous()


class _ExactCollector:
    """Per-layer exact sample collector. Buffers activations on CPU and concatenates in finalize."""

    def __init__(self, device: torch.device):
        self.device = device
        self.chunks: List[torch.Tensor] = []

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        self.chunks.append(x.to(torch.float32).detach().cpu())

    @torch.no_grad()
    def finalize(self, dist_env: DistEnv) -> torch.Tensor:
        if not self.chunks:
            raise RuntimeError("Exact collector received no samples.")
        local = torch.cat(self.chunks, dim=0)
        D = local.shape[1]
        if dist_env.enabled and dist_env.world_size > 1:
            local_n = torch.tensor([local.shape[0]], dtype=torch.int64, device=self.device)
            ns_list = [torch.zeros(1, dtype=torch.int64, device=self.device) for _ in range(dist_env.world_size)]
            dist.all_gather(ns_list, local_n)
            ns = [int(n.item()) for n in ns_list]
            max_n = max(ns)
            padded = torch.zeros((max_n, D), dtype=torch.float32, device=self.device)
            padded[:local.shape[0]] = local.to(self.device)
            gather_list = [torch.zeros_like(padded) for _ in range(dist_env.world_size)]
            dist.all_gather(gather_list, padded)
            pieces = [g[:n].cpu() for g, n in zip(gather_list, ns)]
            merged = torch.cat(pieces, dim=0)
        else:
            merged = local
        sorted_full, _ = torch.sort(merged.transpose(0, 1).contiguous(), dim=-1)
        return sorted_full.to(self.device)


class _PearsonCollector:
    """Per-side Pearson stats collector: accumulates per-neuron sum and sum-of-squares."""

    def __init__(self, device: torch.device):
        self.device = device
        self.sums: Optional[List[torch.Tensor]] = None
        self.sumsqs: Optional[List[torch.Tensor]] = None
        self.local_samples = 0

    @torch.no_grad()
    def update_layers(self, layer_batches: List[torch.Tensor]) -> None:
        if self.sums is None:
            dims = [int(x.shape[1]) for x in layer_batches]
            self.sums, self.sumsqs = _allocate_running_stats(dims, device=self.device)
        n = int(layer_batches[0].shape[0])
        if any(x.shape[0] != n for x in layer_batches):
            raise RuntimeError("Inconsistent batch size across layers in Pearson collector.")
        self.local_samples += n
        for i, x in enumerate(layer_batches):
            x64 = x.to(torch.float64)
            self.sums[i].add_(x64.sum(dim=0))
            self.sumsqs[i].add_(x64.square().sum(dim=0))

    @torch.no_grad()
    def finalize(self, dist_env: DistEnv) -> Tuple[List[LayerStats], int]:
        if self.sums is None or self.sumsqs is None:
            raise RuntimeError("Pearson collector received no samples.")
        all_reduce_inplace(self.sums, dist_env=dist_env)
        all_reduce_inplace(self.sumsqs, dist_env=dist_env)
        count_t = torch.tensor([self.local_samples], dtype=torch.int64, device=self.device)
        all_reduce_inplace(count_t, dist_env=dist_env)
        total = int(count_t.item())
        return _finalize_stats(self.sums, self.sumsqs, total), total


class _SpearmanCollector:
    """Per-side Spearman collector: per-layer reservoir (approx) or exact sample buffer."""

    def __init__(self, device: torch.device, mode: str, reservoir_size: int, seed: int):
        if mode not in {"approx", "exact"}:
            raise ValueError(f"Unknown spearman mode: {mode!r}")
        self.device = device
        self.mode = mode
        self.reservoir_size = int(reservoir_size)
        self.seed = int(seed)
        self.collectors: Optional[List[object]] = None
        self.local_samples = 0

    def _make_layer_collector(self, layer_idx: int):
        if self.mode == "approx":
            return _ReservoirCollector(K=self.reservoir_size, device=self.device, seed=self.seed + 100003 * (layer_idx + 1))
        return _ExactCollector(device=self.device)

    @torch.no_grad()
    def update_layers(self, layer_batches: List[torch.Tensor]) -> None:
        if self.collectors is None:
            self.collectors = [self._make_layer_collector(i) for i in range(len(layer_batches))]
        n = int(layer_batches[0].shape[0])
        if any(x.shape[0] != n for x in layer_batches):
            raise RuntimeError("Inconsistent batch size across layers in Spearman collector.")
        self.local_samples += n
        for c, x in zip(self.collectors, layer_batches):
            c.update(x)

    @torch.no_grad()
    def finalize(self, dist_env: DistEnv) -> Tuple[List[QuantileLayerStats], int]:
        if self.collectors is None:
            raise RuntimeError("Spearman collector received no samples.")
        count_t = torch.tensor([self.local_samples], dtype=torch.int64, device=self.device)
        all_reduce_inplace(count_t, dist_env=dist_env)
        total = int(count_t.item())
        normalizers: List[QuantileLayerStats] = []
        for c in self.collectors:
            sorted_samples = c.finalize(dist_env)
            rank_mean, rank_invstd = _rank_moments_of_sorted(sorted_samples)
            normalizers.append(QuantileLayerStats(
                sorted_samples=sorted_samples,
                rank_mean=rank_mean,
                rank_invstd=rank_invstd,
                dim=int(sorted_samples.shape[0]),
                num_global_samples=total,
            ))
        return normalizers, total


def _make_side_collector(
    similarity: str,
    spearman_mode: str,
    spearman_reservoir_size: int,
    device: torch.device,
    seed: int,
):
    if similarity == "pearson":
        return _PearsonCollector(device=device)
    if similarity == "spearman":
        return _SpearmanCollector(device=device, mode=spearman_mode, reservoir_size=spearman_reservoir_size, seed=seed)
    raise ValueError(f"Unknown similarity: {similarity!r}")


def _init_global_topk(layer_dims: Dict[int, int], topk: int) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    scores: Dict[int, torch.Tensor] = {}
    layers: Dict[int, torch.Tensor] = {}
    neurons: Dict[int, torch.Tensor] = {}
    for layer_idx, dim in layer_dims.items():
        scores[layer_idx] = torch.full((dim, topk), float("-inf"), dtype=torch.float32)
        layers[layer_idx] = torch.full((dim, topk), -1, dtype=torch.int64)
        neurons[layer_idx] = torch.full((dim, topk), -1, dtype=torch.int64)
    return scores, layers, neurons


def _merge_rowwise_topk(existing_scores, existing_layers, existing_neurons, new_scores, new_layer_idx):
    k_old = existing_scores.shape[1]
    k_new = min(k_old, new_scores.shape[1])
    cand_scores, cand_neurons = torch.topk(new_scores, k=k_new, dim=1)
    cand_layers = torch.full_like(cand_neurons, new_layer_idx, dtype=torch.int64)
    all_scores = torch.cat([existing_scores, cand_scores.cpu()], dim=1)
    all_layers = torch.cat([existing_layers, cand_layers.cpu()], dim=1)
    all_neurons = torch.cat([existing_neurons, cand_neurons.cpu().to(torch.int64)], dim=1)
    merged_scores, pos = torch.topk(all_scores, k=k_old, dim=1)
    merged_layers = torch.gather(all_layers, 1, pos)
    merged_neurons = torch.gather(all_neurons, 1, pos)
    return merged_scores, merged_layers, merged_neurons


def _merge_colwise_topk(existing_scores, existing_layers, existing_neurons, new_scores, new_layer_idx):
    k_old = existing_scores.shape[1]
    k_new = min(k_old, new_scores.shape[0])
    cand_scores, cand_neurons = torch.topk(new_scores, k=k_new, dim=0)
    cand_scores = cand_scores.transpose(0, 1).cpu()
    cand_neurons = cand_neurons.transpose(0, 1).cpu().to(torch.int64)
    cand_layers = torch.full_like(cand_neurons, new_layer_idx, dtype=torch.int64)
    all_scores = torch.cat([existing_scores, cand_scores], dim=1)
    all_layers = torch.cat([existing_layers, cand_layers], dim=1)
    all_neurons = torch.cat([existing_neurons, cand_neurons], dim=1)
    merged_scores, pos = torch.topk(all_scores, k=k_old, dim=1)
    merged_layers = torch.gather(all_layers, 1, pos)
    merged_neurons = torch.gather(all_neurons, 1, pos)
    return merged_scores, merged_layers, merged_neurons


def _num_batches(num_items: int, batch_size: int) -> int:
    return (num_items + batch_size - 1) // batch_size


def _pil_images_to_tensor01(images: Sequence[Image.Image]) -> torch.Tensor:
    tensors: List[torch.Tensor] = []
    for img in images:
        arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
        ten = torch.from_numpy(arr).permute(2, 0, 1).contiguous().to(torch.float32) / 255.0
        tensors.append(ten)
    if not tensors:
        raise RuntimeError("Received an empty image list from Flux pipeline.")
    return torch.stack(tensors, dim=0)


def _ensure_bchw_tensor01(images) -> torch.Tensor:
    if isinstance(images, torch.Tensor):
        x = images.detach()
        if x.ndim != 4:
            raise RuntimeError(f"Expected 4D tensor from Flux pipeline, got {tuple(x.shape)}")
        if x.shape[1] not in {1, 3} and x.shape[-1] in {1, 3}:
            x = x.permute(0, 3, 1, 2).contiguous()
        return x.to(torch.float32)
    if isinstance(images, Image.Image):
        return _pil_images_to_tensor01([images])
    if isinstance(images, (list, tuple)):
        if len(images) == 0:
            raise RuntimeError("Received an empty image list from Flux pipeline.")
        if isinstance(images[0], Image.Image):
            return _pil_images_to_tensor01(images)
        raise RuntimeError(f"Unsupported Flux pipeline image list element type: {type(images[0])}")
    raise RuntimeError(f"Unsupported Flux pipeline image output type: {type(images)}")


# -----------------------------------------------------------------------------
# Image saving
# -----------------------------------------------------------------------------


@torch.inference_mode()
def _save_image_batch(images_m11: torch.Tensor, entries: Sequence[PromptEntry], out_dir: Path, start_index: int, image_format: str = "png") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = images_m11.detach().clamp(-1.0, 1.0)
    imgs = ((imgs + 1.0) / 2.0 * 255.0).round().to(torch.uint8)
    imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
    fmt = image_format.lower().lstrip(".")
    for offset, (img, entry) in enumerate(zip(imgs, entries)):
        idx = start_index + offset
        label_slug = sanitize_filename_component(entry.prompt_label)
        filename = out_dir / f"img_{idx:06d}_class_{entry.class_idx:04d}_{label_slug}.{fmt}"
        Image.fromarray(img).save(filename)


# -----------------------------------------------------------------------------
# Flux image iteration
# -----------------------------------------------------------------------------


@torch.inference_mode()
def iter_generated_batches(
    flux: FluxSpec,
    prompt_schedule: Sequence[PromptEntry],
    batch_size: int,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    max_sequence_length: int,
    device: torch.device,
    shard_start: int,
    shard_end: int,
    save_images_dir: Optional[Path] = None,
    save_image_format: str = "png",
) -> Iterable[Tuple[torch.Tensor, List[PromptEntry], int]]:
    pipe = flux.pipeline
    gen_device = "cuda" if str(device).startswith("cuda") else "cpu"

    for start in range(shard_start, shard_end, batch_size):
        end = min(start + batch_size, shard_end)
        entries = list(prompt_schedule[start:end])
        prompts = [e.prompt for e in entries]
        generator = torch.Generator(device=gen_device)
        generator.manual_seed(batch_seed(seed, start))

        out = pipe(
            prompt=prompts,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            output_type="pt",
            return_dict=True,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=flux.text_encoder_out_layers,
        )
        images01 = _ensure_bchw_tensor01(out.images)
        images_m11 = images01.mul(2.0).sub(1.0)

        if save_images_dir is not None:
            _save_image_batch(images_m11=images_m11, entries=entries, out_dir=save_images_dir, start_index=start, image_format=save_image_format)

        yield images_m11, entries, start


# -----------------------------------------------------------------------------
# Passes
# -----------------------------------------------------------------------------


@torch.inference_mode()
def compute_layer_stats(
    flux: FluxSpec,
    disc: TowerSpec,
    prompt_schedule: Sequence[PromptEntry],
    batch_size: int,
    seed: int,
    num_inference_steps: int,
    flux_capture_step_index: int,
    guidance_scale: float,
    height: int,
    width: int,
    max_sequence_length: int,
    device: torch.device,
    act_resample_mode: str,
    canonical_grid_source: str,
    dist_env: DistEnv,
    similarity: str = "pearson",
    spearman_mode: str = "approx",
    spearman_reservoir_size: int = 4096,
    save_images_dir: Optional[Path] = None,
    save_image_format: str = "png",
) -> Tuple[List[Normalizer], List[Normalizer], int, Tuple[int, int], Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    num_images = len(prompt_schedule)
    shard_start, shard_end = rank_shard_bounds(num_images, dist_env.world_size, dist_env.rank)
    local_num_images = shard_end - shard_start
    if local_num_images <= 0:
        raise ValueError(f"Rank {dist_env.rank} received no prompts. Ensure num_images >= world_size.")

    cap_flux = MultiActivationCapture(
        flux.modules,
        capture_output=flux.capture_output,
        postprocess=flux.capture_postprocess,
        capture_call_index=flux_capture_step_index,
    ).register()
    cap_disc = MultiActivationCapture(
        disc.modules,
        capture_output=disc.capture_output,
        postprocess=disc.capture_postprocess,
    ).register()
    num_batches = _num_batches(local_num_images, batch_size)

    flux_collector = _make_side_collector(
        similarity=similarity,
        spearman_mode=spearman_mode,
        spearman_reservoir_size=spearman_reservoir_size,
        device=device,
        seed=int(seed) + 1009 * (dist_env.rank + 1),
    )
    disc_collector = _make_side_collector(
        similarity=similarity,
        spearman_mode=spearman_mode,
        spearman_reservoir_size=spearman_reservoir_size,
        device=device,
        seed=int(seed) + 2017 * (dist_env.rank + 1),
    )

    try:
        local_samples = 0
        canonical_grid_hw = None
        flux_native_grid_hw = None
        disc_native_grid_hw = None

        with tqdm(total=num_batches, desc=f"stats batches [rank{dist_env.rank}]", **tqdm_kwargs(dist_env)) as pbar:
            for batch_idx, (images_m11, _entries, _global_start) in enumerate(
                iter_generated_batches(
                    flux=flux,
                    prompt_schedule=prompt_schedule,
                    batch_size=batch_size,
                    seed=seed,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    max_sequence_length=max_sequence_length,
                    device=device,
                    shard_start=shard_start,
                    shard_end=shard_end,
                    save_images_dir=save_images_dir,
                    save_image_format=save_image_format,
                ),
                start=1,
            ):
                flux_acts = cap_flux.get_and_clear()

                disc_images = disc.preprocess(images_m11).to(device=device, dtype=disc.model_dtype)
                _ = disc.forward(disc_images)
                disc_acts = cap_disc.get_and_clear()

                if canonical_grid_source == "flux":
                    flat_flux = []
                    for act in flux_acts:
                        if canonical_grid_hw is None:
                            x, native_grid_hw = _flatten_activation_on_grid(
                                act,
                                flux.patch_token_offset,
                                None,
                                act_resample_mode,
                                native_grid_hw_hint=flux.native_grid_hw,
                            )
                            canonical_grid_hw = native_grid_hw
                            flux_native_grid_hw = native_grid_hw
                        else:
                            x, native_grid_hw = _flatten_activation_on_grid(
                                act,
                                flux.patch_token_offset,
                                canonical_grid_hw,
                                act_resample_mode,
                                native_grid_hw_hint=flux.native_grid_hw,
                            )
                            flux_native_grid_hw = flux_native_grid_hw or native_grid_hw
                        flat_flux.append(x.to(torch.float32))
                    assert canonical_grid_hw is not None
                    flat_disc = []
                    for act in disc_acts:
                        y, native_grid_hw = _flatten_activation_on_grid(
                            act,
                            disc.patch_token_offset,
                            canonical_grid_hw,
                            act_resample_mode,
                            native_grid_hw_hint=disc.native_grid_hw,
                        )
                        disc_native_grid_hw = disc_native_grid_hw or native_grid_hw
                        flat_disc.append(y.to(torch.float32))
                elif canonical_grid_source == "disc":
                    flat_disc = []
                    for act in disc_acts:
                        if canonical_grid_hw is None:
                            y, native_grid_hw = _flatten_activation_on_grid(
                                act,
                                disc.patch_token_offset,
                                None,
                                act_resample_mode,
                                native_grid_hw_hint=disc.native_grid_hw,
                            )
                            canonical_grid_hw = native_grid_hw
                            disc_native_grid_hw = native_grid_hw
                        else:
                            y, native_grid_hw = _flatten_activation_on_grid(
                                act,
                                disc.patch_token_offset,
                                canonical_grid_hw,
                                act_resample_mode,
                                native_grid_hw_hint=disc.native_grid_hw,
                            )
                            disc_native_grid_hw = disc_native_grid_hw or native_grid_hw
                        flat_disc.append(y.to(torch.float32))
                    assert canonical_grid_hw is not None
                    flat_flux = []
                    for act in flux_acts:
                        x, native_grid_hw = _flatten_activation_on_grid(
                            act,
                            flux.patch_token_offset,
                            canonical_grid_hw,
                            act_resample_mode,
                            native_grid_hw_hint=flux.native_grid_hw,
                        )
                        flux_native_grid_hw = flux_native_grid_hw or native_grid_hw
                        flat_flux.append(x.to(torch.float32))
                else:
                    raise ValueError(f"Unsupported canonical_grid_source: {canonical_grid_source}")

                sample_count = flat_flux[0].shape[0]
                if any(x.shape[0] != sample_count for x in flat_flux + flat_disc):
                    raise RuntimeError("Inconsistent canonical patch sample count across layers.")
                local_samples += int(sample_count)

                flux_collector.update_layers(flat_flux)
                disc_collector.update_layers(flat_disc)

                processed_images = min(batch_idx * batch_size, local_num_images)
                pbar.update(1)
                pbar.set_postfix(local_images=f"{processed_images}/{local_num_images}", local_samples=f"{local_samples:,}")

        assert canonical_grid_hw is not None

        flux_normalizers, total_samples = flux_collector.finalize(dist_env)
        disc_normalizers, disc_total_samples = disc_collector.finalize(dist_env)
        if total_samples != disc_total_samples:
            raise RuntimeError(
                f"Inconsistent sample counts between flux ({total_samples}) and disc ({disc_total_samples}) collectors."
            )
        return flux_normalizers, disc_normalizers, total_samples, canonical_grid_hw, flux_native_grid_hw, disc_native_grid_hw
    finally:
        cap_flux.remove()
        cap_disc.remove()


@torch.inference_mode()
def accumulate_corr_for_disc_chunk(
    flux: FluxSpec,
    disc: TowerSpec,
    disc_modules_chunk: List[torch.nn.Module],
    disc_chunk_capture_output: List[bool],
    disc_chunk_indices: List[int],
    flux_normalizers: List[Normalizer],
    disc_normalizers: List[Normalizer],
    prompt_schedule: Sequence[PromptEntry],
    batch_size: int,
    seed: int,
    num_inference_steps: int,
    flux_capture_step_index: int,
    guidance_scale: float,
    height: int,
    width: int,
    max_sequence_length: int,
    device: torch.device,
    compute_dtype: torch.dtype,
    canonical_grid_hw: Tuple[int, int],
    act_resample_mode: str,
    dist_env: DistEnv,
) -> Optional[List[List[torch.Tensor]]]:
    num_images = len(prompt_schedule)
    shard_start, shard_end = rank_shard_bounds(num_images, dist_env.world_size, dist_env.rank)
    local_num_images = shard_end - shard_start
    if local_num_images <= 0:
        raise ValueError(f"Rank {dist_env.rank} received no prompts. Ensure num_images >= world_size.")

    cap_flux = MultiActivationCapture(
        flux.modules,
        capture_output=flux.capture_output,
        postprocess=flux.capture_postprocess,
        capture_call_index=flux_capture_step_index,
    ).register()
    cap_disc = MultiActivationCapture(
        disc_modules_chunk,
        capture_output=disc_chunk_capture_output,
        postprocess=None,
    ).register()
    num_batches = _num_batches(local_num_images, batch_size)

    chunk_start = disc_chunk_indices[0]
    chunk_end = disc_chunk_indices[-1] + 1
    chunk_desc = f"corr batches disc[{chunk_start}:{chunk_end}]"

    try:
        accumulators: Optional[List[List[torch.Tensor]]] = None

        with tqdm(total=num_batches, desc=f"{chunk_desc} [rank{dist_env.rank}]", leave=False, **tqdm_kwargs(dist_env)) as pbar:
            for batch_idx, (images_m11, _entries, _global_start) in enumerate(
                iter_generated_batches(
                    flux=flux,
                    prompt_schedule=prompt_schedule,
                    batch_size=batch_size,
                    seed=seed,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    max_sequence_length=max_sequence_length,
                    device=device,
                    shard_start=shard_start,
                    shard_end=shard_end,
                ),
                start=1,
            ):
                flux_acts = cap_flux.get_and_clear()

                disc_images = disc.preprocess(images_m11).to(device=device, dtype=disc.model_dtype)
                _ = disc.forward(disc_images)
                disc_acts = cap_disc.get_and_clear()

                flat_flux = []
                for i, act in enumerate(flux_acts):
                    x, _ = _flatten_activation_on_grid(
                        act,
                        flux.patch_token_offset,
                        canonical_grid_hw,
                        act_resample_mode,
                        native_grid_hw_hint=flux.native_grid_hw,
                    )
                    x = flux_normalizers[i].normalize(x, device=device, dtype=compute_dtype)
                    flat_flux.append(x)

                flat_disc = []
                for local_j, act in enumerate(disc_acts):
                    j = disc_chunk_indices[local_j]
                    y, _ = _flatten_activation_on_grid(
                        act,
                        disc.patch_token_offset,
                        canonical_grid_hw,
                        act_resample_mode,
                        native_grid_hw_hint=disc.native_grid_hw,
                    )
                    y = disc_normalizers[j].normalize(y, device=device, dtype=compute_dtype)
                    flat_disc.append(y)

                if accumulators is None:
                    accumulators = []
                    for x in flat_flux:
                        row = []
                        for y in flat_disc:
                            row.append(torch.zeros((x.shape[1], y.shape[1]), dtype=torch.float32, device=device))
                        accumulators.append(row)

                assert accumulators is not None
                for i, x in enumerate(flat_flux):
                    xt = x.to(torch.float32).transpose(0, 1)
                    for local_j, y in enumerate(flat_disc):
                        accumulators[i][local_j].addmm_(xt, y.to(torch.float32))

                processed_images = min(batch_idx * batch_size, local_num_images)
                pbar.update(1)
                pbar.set_postfix(local_images=f"{processed_images}/{local_num_images}")

        if accumulators is None:
            raise RuntimeError("No activations collected while accumulating correlations.")

        if dist_env.enabled:
            reduce_inplace_to_root(accumulators, dist_env=dist_env, dst=0)
            if not dist_env.is_main:
                return None

        return accumulators
    finally:
        cap_flux.remove()
        cap_disc.remove()


# -----------------------------------------------------------------------------
# Best-buddy extraction
# -----------------------------------------------------------------------------


def build_mutual_topk_pairs(
    flux_layer_names: List[str],
    disc_layer_names: List[str],
    topk_a_scores: Dict[int, torch.Tensor],
    topk_a_layers: Dict[int, torch.Tensor],
    topk_a_neurons: Dict[int, torch.Tensor],
    topk_b_scores: Dict[int, torch.Tensor],
    topk_b_layers: Dict[int, torch.Tensor],
    topk_b_neurons: Dict[int, torch.Tensor],
    topk: int,
) -> List[Dict[str, object]]:
    pairs: List[Dict[str, object]] = []
    seen: set[Tuple[int, int, int, int]] = set()

    for la, scores in topk_a_scores.items():
        layers_b = topk_a_layers[la]
        neurons_b = topk_a_neurons[la]
        for na in range(scores.shape[0]):
            for rank_a in range(min(topk, scores.shape[1])):
                lb = int(layers_b[na, rank_a].item())
                nb = int(neurons_b[na, rank_a].item())
                corr = float(scores[na, rank_a].item())
                if lb < 0 or nb < 0 or not math.isfinite(corr):
                    continue
                rev_layers = topk_b_layers[lb][nb]
                rev_neurons = topk_b_neurons[lb][nb]
                rank_b = None
                for idx in range(min(topk, rev_layers.shape[0])):
                    if int(rev_layers[idx].item()) == la and int(rev_neurons[idx].item()) == na:
                        rank_b = idx
                        break
                if rank_b is None:
                    continue
                key = (la, na, lb, nb)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(
                    {
                        "flux_layer_idx": la,
                        "flux_layer": flux_layer_names[la],
                        "flux_neuron": na,
                        "disc_layer_idx": lb,
                        "disc_layer": disc_layer_names[lb],
                        "disc_neuron": nb,
                        "correlation": corr,
                        "rank_in_flux": rank_a + 1,
                        "rank_in_disc": rank_b + 1,
                    }
                )

    pairs.sort(key=lambda x: float(x["correlation"]), reverse=True)
    return pairs


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # Flux generation
    parser.add_argument("--flux-model-id", type=str, default="black-forest-labs/FLUX.2-klein-4B")
    parser.add_argument("--flux-dtype", type=str, choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--num-inference-steps", type=int, default=2)
    parser.add_argument(
        "--flux-capture-step",
        type=str,
        default="last",
        help=(
            "Which FLUX denoising step to capture when num_inference_steps > 1. "
            "Use 'first', 'last', or an integer step index. Negative indices are allowed. "
            "Example: 0 = first step, 1 = second step, -1 = last step. "
            "Note: OpenCLIP activations are still computed on the final generated image."
        ),
    )
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--text-encoder-out-layers", type=str, default="9,18,27")
    parser.add_argument("--enable-model-cpu-offload", action="store_true")

    # Prompt schedule
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="a sharp photo of {prompt_label_article}",
        help=(
            "Prompt template. Available placeholders: {label}, {label_article}, {prompt_label}, "
            "{prompt_label_article}, {class_idx}."
        ),
    )
    parser.add_argument("--label-alias-mode", choices=["raw", "first", "longest"], default="longest")
    parser.add_argument("--class-list-file", type=str, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-images", type=int, default=1000)

    # Discriminative tower
    parser.add_argument("--disc-family", type=str, choices=["openclip", "pixio", "internvit", "dinov3"], default="openclip")
    parser.add_argument(
        "--disc-arch",
        type=str,
        default="ViT-L-14",
        help=(
            "For --disc-family=openclip: OpenCLIP architecture name (for example ViT-L-14). "
            "For --disc-family=pixio: PixIO constructor name (for example pixio_vith16). "
            "For --disc-family=dinov3: local torch.hub entrypoint name (for example dinov3_vitb16) "
            "when --dinov3-repo is set, otherwise a Hugging Face model id (for example facebook/dinov3-vitb16-pretrain-lvd1689m). "
            "Ignored for --disc-family=internvit unless you intentionally reuse it as a model-id-like string."
        ),
    )
    parser.add_argument(
        "--disc-pretrained",
        type=str,
        default="datacomp_xl_s13b_b90k",
        help=(
            "For --disc-family=openclip: OpenCLIP pretrained tag. "
            "For --disc-family=pixio: optional local checkpoint path (unless --pixio-checkpoint is used). "
            "Unused for --disc-family=internvit and --disc-family=dinov3 "
            "(DINOv3 local loading uses --dinov3-repo plus --disc-weights instead)."
        ),
    )
    parser.add_argument(
        "--disc-weights",
        type=str,
        default=None,
        help=(
            "For --disc-family=dinov3 with --dinov3-repo set: path / URL / name passed to "
            "torch.hub.load(..., weights=...). Required for local DINOv3 loading. Unused otherwise."
        ),
    )
    parser.add_argument(
        "--dinov3-repo",
        type=str,
        default=None,
        help=(
            "Optional local DINOv3 repo path. If set, DINOv3 is loaded locally via torch.hub "
            "(requires --disc-weights). If unset, DINOv3 is loaded from Hugging Face using "
            "--disc-arch as the model id."
        ),
    )
    parser.add_argument("--disc-model-dtype", type=str, choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument(
        "--disc-input-size",
        type=int,
        default=None,
        help=(
            "Optional square input size for the discriminator tower. "
            "Defaults to 224 for OpenCLIP and DINOv3, to the FLUX image size for PixIO, "
            "and to the model/processor image size for InternViT. "
            "For InternViT, changing this size interpolates positional embeddings when the checkpoint exposes resize_pos_embeddings."
        ),
    )
    parser.add_argument(
        "--pixio-module-dir",
        type=str,
        default=None,
        help=(
            "Optional path to the PixIO source directory. Pass either the repo root or the child "
            "directory that contains pixio.py and layers/ if `import pixio` is otherwise unavailable."
        ),
    )
    parser.add_argument(
        "--pixio-checkpoint",
        type=str,
        default=None,
        help="Optional local PixIO checkpoint path. Overrides --disc-pretrained when set.",
    )
    parser.add_argument(
        "--internvit-model-id",
        type=str,
        default="OpenGVLab/InternViT-300M-448px",
        help=(
            "For --disc-family=internvit: Hugging Face model id, local checkpoint directory, or a known alias such as "
            "internvit_6b_224, internvit_6b_448_v1_2, internvit_6b_448_v1_5, internvit_300m_448, "
            "internvit_300m_448_v2_5, or internvit_6b_448_v2_5."
        ),
    )
    parser.add_argument("--disc-mean", type=float, nargs=3, default=None)
    parser.add_argument("--disc-std", type=float, nargs=3, default=None)

    # Matching
    parser.add_argument(
        "--hook-kind",
        choices=["input", "output", "point_input", "point_output", "inverted_output"],
        default="input",
        help=(
            "Hook placement on FLUX MLP-like paths. "
            "For double-stream blocks: input/point_input = input to ff.linear_out "
            "(post-activation hidden state), output/point_output = output of ff.linear_out, "
            "inverted_output = output of ff.linear_in. "
            "For single-stream blocks: point_input captures the input to attn.to_out, "
            "restricted to image tokens and the MLP suffix; point_output captures attn.to_out "
            "output on image tokens; inverted_output captures the MLP suffix of attn.to_qkv_mlp_proj "
            "on image tokens."
        ),
    )
    parser.add_argument(
        "--flux-block-families",
        choices=["double_only", "double_and_single"],
        default="double_and_single",
        help=(
            "Which FLUX block families to hook. "
            "'double_only' preserves the original behavior; "
            "'double_and_single' also hooks single_transformer_blocks, restricted to image tokens."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Per-rank batch size in distributed mode.")
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--disc-chunk-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compute-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--act-resample-mode", choices=["nearest", "bilinear", "bicubic", "area"], default="bilinear")
    parser.add_argument("--canonical-grid-source", choices=["flux", "disc"], default="flux")
    parser.add_argument(
        "--similarity",
        choices=["pearson", "spearman"],
        default="pearson",
        help=(
            "Similarity metric for matching neurons. "
            "'pearson' (default) is z-scored Pearson correlation. "
            "'spearman' applies per-neuron rank transformation via an empirical CDF before the matmul."
        ),
    )
    parser.add_argument(
        "--spearman-mode",
        choices=["approx", "exact"],
        default="approx",
        help=(
            "Only used when --similarity spearman. 'approx' builds a per-neuron CDF via reservoir "
            "sampling of --spearman-reservoir-size activations per neuron. 'exact' stores every "
            "activation on CPU and ranks the full population at finalize time (memory grows linearly)."
        ),
    )
    parser.add_argument(
        "--spearman-reservoir-size",
        type=int,
        default=4096,
        help="Per-neuron reservoir size used in --spearman-mode approx. Larger = closer to exact Spearman.",
    )
    parser.add_argument("--warn-accumulator-gb", type=float, default=8.0)

    # Outputs
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--save-full-corr", action="store_true")
    parser.add_argument("--save-generated-images", action="store_true")
    parser.add_argument("--generated-images-subdir", type=str, default="generated_images")
    parser.add_argument("--generated-image-format", choices=["png", "jpg", "jpeg", "webp"], default="png")

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    dist_env = init_distributed()

    try:
        if dist_env.enabled:
            if not args.device.startswith("cuda"):
                raise ValueError("In distributed mode, use a CUDA device.")
            device = torch.device(f"cuda:{dist_env.local_rank}")
        else:
            device = torch.device(args.device)

        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        compute_dtype = torch_dtype_from_name(args.compute_dtype)
        flux_dtype = torch_dtype_from_name(args.flux_dtype)
        disc_model_dtype = torch_dtype_from_name(args.disc_model_dtype)
        normalized_hook_kind = normalize_hook_kind(args.hook_kind)
        text_encoder_out_layers = parse_int_tuple(args.text_encoder_out_layers)
        flux_capture_step_index = resolve_capture_step(args.flux_capture_step, args.num_inference_steps)

        if args.num_images < dist_env.world_size:
            raise ValueError(f"num_images ({args.num_images}) must be >= world_size ({dist_env.world_size}).")

        if dist_env.is_main:
            os.makedirs(args.save_dir, exist_ok=True)
        barrier(dist_env)

        save_dir = Path(args.save_dir)
        save_images_dir = save_dir / args.generated_images_subdir if args.save_generated_images else None

        labels = load_imagenet_classes(args.class_list_file)
        prompt_schedule = build_prompt_schedule(
            labels=labels,
            prompt_template=args.prompt_template,
            start_index=args.start_index,
            num_images=args.num_images,
            label_alias_mode=args.label_alias_mode,
        )
        if dist_env.is_main:
            save_prompt_manifest(prompt_schedule, save_dir)

        flux = load_flux2_klein(
            model_id=args.flux_model_id,
            device=device,
            model_dtype=flux_dtype,
            hook_kind=normalized_hook_kind,
            text_encoder_out_layers=text_encoder_out_layers,
            enable_model_cpu_offload=args.enable_model_cpu_offload,
            disable_progress_bar=True,
            image_size_hw=(args.height, args.width),
            block_families=args.flux_block_families,
        )

        if args.disc_family == "openclip":
            disc_input_hw = _as_hw(224 if args.disc_input_size is None else args.disc_input_size)
            disc = load_openclip_tower(
                model_name=args.disc_arch,
                pretrained=args.disc_pretrained,
                device=device,
                hook_kind=normalized_hook_kind,
                model_dtype=disc_model_dtype,
                mean=args.disc_mean,
                std=args.disc_std,
                input_size_hw=disc_input_hw,
            )
        elif args.disc_family == "pixio":
            disc_input_hw = _as_hw(args.disc_input_size) if args.disc_input_size is not None else (args.height, args.width)
            pixio_checkpoint = args.pixio_checkpoint or args.disc_pretrained
            disc = load_pixio_tower(
                model_name=args.disc_arch,
                checkpoint_path=pixio_checkpoint,
                device=device,
                hook_kind=normalized_hook_kind,
                model_dtype=disc_model_dtype,
                module_dir=args.pixio_module_dir,
                mean=args.disc_mean,
                std=args.disc_std,
                input_size_hw=disc_input_hw,
            )
        elif args.disc_family == "internvit":
            disc_input_hw = _as_hw(args.disc_input_size) if args.disc_input_size is not None else None
            disc = load_internvit_tower(
                model_name_or_id=args.internvit_model_id,
                device=device,
                hook_kind=normalized_hook_kind,
                model_dtype=disc_model_dtype,
                mean=args.disc_mean,
                std=args.disc_std,
                input_size_hw=disc_input_hw,
            )
        elif args.disc_family == "dinov3":
            disc_input_hw = _as_hw(224 if args.disc_input_size is None else args.disc_input_size)
            if args.dinov3_repo is not None:
                if not args.disc_weights:
                    raise ValueError(
                        "For local DINOv3 loading, provide --disc-weights together with --dinov3-repo."
                    )
                disc = load_dinov3_local_tower(
                    repo=args.dinov3_repo,
                    arch=args.disc_arch,
                    weights=args.disc_weights,
                    device=device,
                    hook_kind=normalized_hook_kind,
                    model_dtype=disc_model_dtype,
                    mean=args.disc_mean,
                    std=args.disc_std,
                    input_size_hw=disc_input_hw,
                )
            else:
                disc = load_dinov3_hf_tower(
                    model_id=args.disc_arch,
                    device=device,
                    hook_kind=normalized_hook_kind,
                    model_dtype=disc_model_dtype,
                    mean=args.disc_mean,
                    std=args.disc_std,
                    input_size_hw=disc_input_hw,
                )
        else:
            raise ValueError(f"Unsupported disc_family: {args.disc_family}")

        if dist_env.is_main:
            print(f"[setup] distributed world size: {dist_env.world_size}")
            print(f"[setup] per-rank batch size: {args.batch_size}")
            print(f"[setup] num prompts: {len(prompt_schedule)}")
            print(f"[setup] flux model: {args.flux_model_id}")
            print(f"[setup] flux hook kind: {normalized_hook_kind}")
            print(f"[setup] flux block families: {args.flux_block_families}")
            print(f"[setup] flux dtype: {flux_dtype}")
            print(f"[setup] text encoder out layers: {text_encoder_out_layers}")
            print(f"[setup] guidance scale: {args.guidance_scale}")
            print(f"[setup] num inference steps: {args.num_inference_steps}")
            print(
                f"[setup] flux capture step: {flux_capture_step_index} (0-based, requested={args.flux_capture_step!r}, "
                f"human={flux_capture_step_index + 1}/{args.num_inference_steps})"
            )
            print(f"[setup] disc family: {args.disc_family}")
            if args.disc_family == "openclip":
                print(f"[setup] disc arch: {args.disc_arch}")
                print(f"[setup] disc pretrained: {args.disc_pretrained}")
            elif args.disc_family == "pixio":
                print(f"[setup] disc arch: {args.disc_arch}")
                print(f"[setup] pixio checkpoint: {args.pixio_checkpoint or args.disc_pretrained}")
                if args.pixio_module_dir is not None:
                    print(f"[setup] pixio module dir: {args.pixio_module_dir}")
            elif args.disc_family == "internvit":
                print(f"[setup] internvit model id: {disc.notes.get('model_id', args.internvit_model_id)}")
            elif args.disc_family == "dinov3":
                print(f"[setup] dinov3 arch: {args.disc_arch}")
                if args.dinov3_repo is not None:
                    print(f"[setup] dinov3 repo: {args.dinov3_repo}")
                    print(f"[setup] dinov3 weights: {args.disc_weights}")
                else:
                    print("[setup] dinov3 loader: huggingface")
            print(f"[setup] disc dtype: {disc_model_dtype}")
            print(f"[setup] label alias mode: {args.label_alias_mode}")
            print(f"[setup] args.num_images: {args.num_images}")
            if args.enable_model_cpu_offload:
                print("[setup] model CPU offload: ENABLED")
            else:
                print("[setup] model CPU offload: DISABLED")
            if args.num_inference_steps > 1:
                print(
                    f"[setup] multi-step FLUX run: hooks capture denoising step "
                    f"{flux_capture_step_index} (0-based) / {flux_capture_step_index + 1} of {args.num_inference_steps}."
                )
                if flux_capture_step_index != args.num_inference_steps - 1:
                    print(
                        "[setup] NOTE: vision-tower activations are still computed on the final generated image, "
                        "so choosing a non-last FLUX capture step compares an earlier denoising state to the final image."
                    )
            if prompt_schedule:
                print("[setup] first prompts:")
                for entry in prompt_schedule[: min(5, len(prompt_schedule))]:
                    print(f"  - class {entry.class_idx:04d}: raw='{entry.raw_label}' | prompt_label='{entry.prompt_label}' | prompt='{entry.prompt}'")
            if disc.native_image_size_hw is not None:
                print(f"[setup] disc input size: {disc.native_image_size_hw[0]}x{disc.native_image_size_hw[1]}")
            if args.disc_family in {"pixio", "internvit", "dinov3"}:
                print(
                    f"[setup] {args.disc_family} patch size: {disc.patch_size}, class-token prefix: {disc.patch_token_offset}"
                )

        print0(dist_env, f"[setup] similarity: {args.similarity}")
        if args.similarity == "spearman":
            print0(dist_env, f"[setup] spearman mode: {args.spearman_mode} (reservoir size {args.spearman_reservoir_size})")
        print0(dist_env, "[1/3] Computing per-layer stats on aligned grid...")
        flux_normalizers, disc_normalizers, total_samples, canonical_grid_hw, flux_native_grid_hw, disc_native_grid_hw = compute_layer_stats(
            flux=flux,
            disc=disc,
            prompt_schedule=prompt_schedule,
            batch_size=args.batch_size,
            seed=args.seed,
            num_inference_steps=args.num_inference_steps,
            flux_capture_step_index=flux_capture_step_index,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            max_sequence_length=args.max_sequence_length,
            device=device,
            act_resample_mode=args.act_resample_mode,
            canonical_grid_source=args.canonical_grid_source,
            dist_env=dist_env,
            similarity=args.similarity,
            spearman_mode=args.spearman_mode,
            spearman_reservoir_size=args.spearman_reservoir_size,
            save_images_dir=save_images_dir,
            save_image_format=args.generated_image_format,
        )
        print0(dist_env, f"Collected {total_samples:,} aligned canonical patch samples.")
        print0(dist_env, f"[setup] canonical grid: {canonical_grid_hw[0]}x{canonical_grid_hw[1]} ({args.canonical_grid_source})")
        if flux_native_grid_hw is not None:
            print0(dist_env, f"[setup] flux native activation grid: {flux_native_grid_hw[0]}x{flux_native_grid_hw[1]}")
        if disc_native_grid_hw is not None:
            print0(dist_env, f"[setup] disc native activation grid: {disc_native_grid_hw[0]}x{disc_native_grid_hw[1]}")
        if args.canonical_grid_source == "flux" and disc_native_grid_hw is not None and disc_native_grid_hw != canonical_grid_hw:
            print0(dist_env, f"[setup] disc activations are resampled to the canonical grid: {disc_native_grid_hw[0]}x{disc_native_grid_hw[1]} -> {canonical_grid_hw[0]}x{canonical_grid_hw[1]}")
        if args.canonical_grid_source == "disc" and flux_native_grid_hw is not None and flux_native_grid_hw != canonical_grid_hw:
            print0(dist_env, f"[setup] flux activations are resampled to the canonical grid: {flux_native_grid_hw[0]}x{flux_native_grid_hw[1]} -> {canonical_grid_hw[0]}x{canonical_grid_hw[1]}")

        print0(dist_env, "[2/3] Accumulating correlations and global top-k neighbors...")
        flux_dims = {i: st.dim for i, st in enumerate(flux_normalizers)}
        disc_dims = {j: st.dim for j, st in enumerate(disc_normalizers)}

        topk_a_scores, topk_a_layers, topk_a_neurons = _init_global_topk(flux_dims, args.topk)
        topk_b_scores, topk_b_layers, topk_b_neurons = _init_global_topk(disc_dims, args.topk)

        corr_dir = save_dir / "corr"
        if args.save_full_corr and dist_env.is_main:
            corr_dir.mkdir(parents=True, exist_ok=True)
        barrier(dist_env)

        chunk_starts = list(range(0, len(disc.modules), args.disc_chunk_size))
        for chunk_start in tqdm(chunk_starts, desc="disc layer chunks", **tqdm_kwargs(dist_env)):
            chunk_end = min(chunk_start + args.disc_chunk_size, len(disc.modules))
            chunk_indices = list(range(chunk_start, chunk_end))
            chunk_modules = [disc.modules[idx] for idx in chunk_indices]
            chunk_capture_output = [disc.capture_output[idx] for idx in chunk_indices]

            if dist_env.is_main:
                est_bytes = 0
                for i in range(len(flux_normalizers)):
                    for j in chunk_indices:
                        est_bytes += flux_normalizers[i].dim * disc_normalizers[j].dim * 4
                est_gb = est_bytes / (1024 ** 3)
                tqdm.write(f"  - disc layers {chunk_start}:{chunk_end} | est accum memory ~ {est_gb:.2f} GB")
                if est_gb > args.warn_accumulator_gb:
                    tqdm.write("    WARNING: large accumulator estimate. Consider smaller --disc-chunk-size or a lower-dimensional hook kind.")

            accumulators = accumulate_corr_for_disc_chunk(
                flux=flux,
                disc=disc,
                disc_modules_chunk=chunk_modules,
                disc_chunk_capture_output=chunk_capture_output,
                disc_chunk_indices=chunk_indices,
                flux_normalizers=flux_normalizers,
                disc_normalizers=disc_normalizers,
                prompt_schedule=prompt_schedule,
                batch_size=args.batch_size,
                seed=args.seed,
                num_inference_steps=args.num_inference_steps,
                flux_capture_step_index=flux_capture_step_index,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                max_sequence_length=args.max_sequence_length,
                device=device,
                compute_dtype=compute_dtype,
                canonical_grid_hw=canonical_grid_hw,
                act_resample_mode=args.act_resample_mode,
                dist_env=dist_env,
            )

            if dist_env.is_main:
                assert accumulators is not None
                for i, flux_layer_name in enumerate(flux.layer_names):
                    for local_j, disc_idx in enumerate(chunk_indices):
                        corr = (accumulators[i][local_j] / float(total_samples)).cpu()
                        topk_a_scores[i], topk_a_layers[i], topk_a_neurons[i] = _merge_rowwise_topk(topk_a_scores[i], topk_a_layers[i], topk_a_neurons[i], corr, disc_idx)
                        topk_b_scores[disc_idx], topk_b_layers[disc_idx], topk_b_neurons[disc_idx] = _merge_colwise_topk(topk_b_scores[disc_idx], topk_b_layers[disc_idx], topk_b_neurons[disc_idx], corr, i)
                        if args.save_full_corr:
                            out_path = corr_dir / f"corr_{flux_layer_name}_vs_{disc.layer_names[disc_idx]}.pt"
                            torch.save(corr, out_path)

            barrier(dist_env)

        if not dist_env.is_main:
            return

        print("[3/3] Extracting mutual top-k matches...")
        best_buddies = build_mutual_topk_pairs(
            flux_layer_names=flux.layer_names,
            disc_layer_names=disc.layer_names,
            topk_a_scores=topk_a_scores,
            topk_a_layers=topk_a_layers,
            topk_a_neurons=topk_a_neurons,
            topk_b_scores=topk_b_scores,
            topk_b_layers=topk_b_layers,
            topk_b_neurons=topk_b_neurons,
            topk=args.topk,
        )

        metadata = {
            "flux_model_id": args.flux_model_id,
            "flux_notes": flux.notes,
            "flux_dtype": args.flux_dtype,
            "hook_kind": normalized_hook_kind,
            "hook_kind_requested": args.hook_kind,
            "disc_family": disc.family,
            "disc_arch": args.disc_arch,
            "disc_pretrained": args.disc_pretrained,
            "internvit_model_id": args.internvit_model_id if args.disc_family == "internvit" else None,
            "disc_model_ref": disc.notes.get("model_id") if isinstance(disc.notes, dict) else None,
            "disc_notes": disc.notes,
            "disc_model_dtype": args.disc_model_dtype,
            "disc_native_image_size_hw": list(disc.native_image_size_hw) if disc.native_image_size_hw is not None else None,
            "canonical_grid_source": args.canonical_grid_source,
            "canonical_grid_hw": list(canonical_grid_hw),
            "flux_native_grid_hw": list(flux_native_grid_hw) if flux_native_grid_hw is not None else None,
            "disc_native_grid_hw": list(disc_native_grid_hw) if disc_native_grid_hw is not None else None,
            "requested_height": args.height,
            "requested_width": args.width,
            "act_resample_mode": args.act_resample_mode,
            "similarity": args.similarity,
            "spearman_mode": args.spearman_mode if args.similarity == "spearman" else None,
            "spearman_reservoir_size": args.spearman_reservoir_size if args.similarity == "spearman" and args.spearman_mode == "approx" else None,
            "num_images": len(prompt_schedule),
            "start_index": args.start_index,
            "batch_size_per_rank": args.batch_size,
            "distributed_world_size": dist_env.world_size,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
            "topk": args.topk,
            "total_patch_samples": total_samples,
            "flux_num_layers": len(flux.layer_names),
            "disc_num_layers": len(disc.layer_names),
            "flux_token_image_patch_size": flux.token_image_patch_size,
            "disc_patch_size": disc.patch_size,
            "prompt_template": args.prompt_template,
            "label_alias_mode": args.label_alias_mode,
            "max_sequence_length": args.max_sequence_length,
            "text_encoder_out_layers": list(text_encoder_out_layers),
            "enable_model_cpu_offload": args.enable_model_cpu_offload,
            "generation_seed_mode": "batch_seed(global_batch_start_index)",
            "save_generated_images": args.save_generated_images,
            "generated_images_subdir": args.generated_images_subdir if args.save_generated_images else None,
            "generated_image_format": args.generated_image_format if args.save_generated_images else None,
            "flux_capture_step_requested": args.flux_capture_step,
            "flux_capture_step_index": flux_capture_step_index,
            "flux_capture_step_one_based": flux_capture_step_index + 1,
        }

        with open(save_dir / "run_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        with open(save_dir / "best_buddies.json", "w", encoding="utf-8") as f:
            json.dump(best_buddies, f, indent=2)

        neighbors_dir = save_dir / "neighbors"
        neighbors_dir.mkdir(parents=True, exist_ok=True)
        for i, layer_name in enumerate(flux.layer_names):
            torch.save(
                {
                    "scores": topk_a_scores[i],
                    "disc_layer_idx": topk_a_layers[i],
                    "disc_neuron": topk_a_neurons[i],
                },
                neighbors_dir / f"flux_{layer_name}_top{args.topk}.pt",
            )

        print(f"Saved run metadata to {save_dir / 'run_metadata.json'}")
        print(f"Saved prompt manifest to {save_dir / 'prompt_manifest.json'}")
        print(f"Saved {len(best_buddies):,} mutual top-k pairs to {save_dir / 'best_buddies.json'}")
        if save_images_dir is not None:
            print(f"Saved generated images to {save_images_dir}")
        if best_buddies:
            print("Top 10 pairs:")
            for row in best_buddies[:10]:
                print(f"  {row['flux_layer']}[{row['flux_neuron']}] <-> {row['disc_layer']}[{row['disc_neuron']}]: {row['correlation']:.4f}")
    finally:
        cleanup_distributed(dist_env)


if __name__ == "__main__":
    main()
