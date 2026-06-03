
#!/usr/bin/env python3
"""
Distributed activation matching between Sana Sprint and an OpenCLIP or PixIO vision tower.

What this script does
---------------------
- Builds one prompt per ImageNet-1k class (or a subset), e.g. "a photo of banana".
- Generates images with Sana Sprint using the same prompt schedule on every pass.
- Captures Sana feed-forward activations from the denoiser transformer.
- Captures OpenCLIP ViT or PixIO ViT MLP activations from the visual tower.
- Resamples both activation maps onto a selectable canonical grid (Sana or OpenCLIP).
- Computes standardized correlations and mutual top-k matches across neurons/channels.

Key design choices
------------------
- This script is aimed at one-step Sana Sprint usage by default, but it can run with
  more steps. If num_inference_steps > 1, the hooks capture the activations from the
  *last* denoising step because the same modules are called repeatedly.
- By default, the script matches projection *outputs* rather than hidden MLP inputs.
  For Sana + large EVA/OpenCLIP models this is far more memory-friendly than matching
  hidden MLP dimensions directly. To mimic the original script more closely, pass
  --hook-kind input.
- The canonical spatial grid can be chosen from either Sana or the discriminative tower via
  --canonical-grid-source. It is inferred from the observed activation grid on the
  first batch, so this also works when Sana's resolution binning is enabled.

Example
-------
Single GPU smoke test:

  python match_sana_sprint_openclip_imagenet_multigpu.py \
    --save-dir ./debug_sana_match \
    --num-images 16 \
    --batch-size 2 \
    --height 256 \
    --width 256 \
    --disc-arch EVA02-E-14-plus \
    --disc-pretrained laion2b_s9b_b144k \
    --disc-input-size 224

Multi-GPU:

  torchrun --standalone --nproc_per_node=8 match_sana_sprint_openclip_imagenet_multigpu.py \
    --save-dir ./sana_match \
    --num-images 1000 \
    --batch-size 2 \
    --height 256 \
    --width 256 \
    --disc-arch EVA02-E-14-plus \
    --disc-pretrained laion2b_s9b_b144k \
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

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

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
    aliases = {
        "input": "point_input",
        "output": "point_output",
    }
    return aliases.get(hook_kind, hook_kind)


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
        # Penalize scientific-looking genus/species aliases such as 'Felis concolor'.
        if re.fullmatch(r"[A-Z][a-z]+ [a-z]+", alias):
            primary -= 25
        # Slightly penalize very short single-word aliases like 'tabby' or 'Cardigan'.
        if len(words) == 1:
            primary -= 4
        # Prefer common descriptive phrases over taxonomic/rare aliases.
        if any(token in lower for token in ["cat", "dog", "bird", "shark", "snake", "fish", "lizard", "frog", "toad", "truck", "car", "plane", "boat", "robe", "suit", "corgi"]):
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

        # keep your existing alias/cleanup logic if you already have it
        prompt_label = raw_label
        if "choose_prompt_label" in globals():
            prompt_label = choose_prompt_label(raw_label, mode=label_alias_mode)
        elif "normalize_imagenet_label" in globals():
            prompt_label = normalize_imagenet_label(raw_label, mode=label_alias_mode)

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

        entries.append(
            PromptEntry(
                class_idx=class_idx,
                raw_label=raw_label,
                prompt_label=prompt_label,
                prompt=prompt,
            )
        )

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
    """
    Capture module inputs or outputs for a list of modules.

    - capture_output[i] = False  -> store inputs[0]
    - capture_output[i] = True   -> store output
    """

    def __init__(self, modules: List[torch.nn.Module], capture_output: Optional[List[bool]] = None):
        self.modules = modules
        self.capture_output = capture_output or [False] * len(modules)
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.activations: List[Optional[torch.Tensor]] = [None] * len(modules)

    def _make_hook(self, idx: int, use_output: bool):
        def hook(_module, inputs, output):
            act = output if use_output else inputs[0]
            if isinstance(act, (tuple, list)):
                act = act[0]
            if not isinstance(act, torch.Tensor):
                raise RuntimeError(f"Expected tensor activation from hook {idx}, got {type(act)}")
            self.activations[idx] = act.detach()
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

    def get_and_clear(self) -> List[torch.Tensor]:
        acts = self.activations
        self.activations = [None] * len(self.modules)
        out: List[torch.Tensor] = []
        for i, act in enumerate(acts):
            if act is None:
                raise RuntimeError(
                    f"Missing activation for hook index {i}. "
                    "For Sana with num_inference_steps > 1 this script captures the last denoising step only."
                )
            if act.ndim not in (3, 4):
                raise RuntimeError(f"Expected 3D or 4D activation, got {tuple(act.shape)}")
            out.append(act)
        return out


# -----------------------------------------------------------------------------
# Model specs
# -----------------------------------------------------------------------------


@dataclass
class SanaSpec:
    pipeline: object
    transformer: torch.nn.Module
    modules: List[torch.nn.Module]
    layer_names: List[str]
    capture_output: List[bool]
    patch_token_offset: int
    latent_patch_size: int
    image_patch_size: int
    vae_scale_factor: int
    hook_kind: str
    model_id: str
    model_dtype: torch.dtype
    notes: Dict[str, object]


@dataclass
class TowerSpec:
    family: str
    model: torch.nn.Module
    modules: List[torch.nn.Module]
    layer_names: List[str]
    capture_output: List[bool]
    patch_token_offset: int
    patch_size: int
    expected_num_patches: Optional[int]
    preprocess: Callable[[torch.Tensor], torch.Tensor]
    forward: Callable[[torch.Tensor], object]
    notes: Dict[str, object]
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

    Uses midrank for tie handling and empirical standardization from the rank-transformed
    reservoir. Required for sparse activations (ReLU/GELU produce many exact zeros).
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
# Sana loader
# -----------------------------------------------------------------------------


def load_sana_sprint(
    model_id: str,
    device: torch.device,
    model_dtype: torch.dtype,
    hook_kind: str,
    enable_vae_tiling: bool,
    enable_vae_slicing: bool,
    disable_progress_bar: bool,
) -> SanaSpec:
    try:
        from diffusers import SanaSprintPipeline
    except Exception as exc:
        raise ImportError("Loading Sana Sprint requires diffusers with SanaSprintPipeline support.") from exc

    hook_kind = normalize_hook_kind(hook_kind)

    pipe = SanaSprintPipeline.from_pretrained(model_id, torch_dtype=model_dtype)
    pipe = pipe.to(device)
    try:
        pipe.set_progress_bar_config(disable=disable_progress_bar)
    except Exception:
        pass

    if enable_vae_tiling:
        try:
            pipe.vae.enable_tiling()
        except Exception:
            pass
    if enable_vae_slicing:
        try:
            pipe.vae.enable_slicing()
        except Exception:
            pass

    transformer = pipe.transformer
    if not hasattr(transformer, "transformer_blocks"):
        raise ValueError(
            f"Expected Sana transformer with .transformer_blocks, got {type(transformer)}"
        )

    modules: List[torch.nn.Module] = []
    layer_names: List[str] = []

    if hook_kind == "point_input":
        capture_output = [False] * len(transformer.transformer_blocks)
        for i, block in enumerate(transformer.transformer_blocks):
            ff = getattr(block, "ff", None)
            proj = getattr(ff, "conv_point", None) if ff is not None else None
            if proj is None and ff is not None:
                proj = getattr(ff, "point_conv", None)
            if proj is None:
                raise ValueError(
                    f"Expected Sana FFN point projection in layer {i}, got {type(block)}"
                )
            modules.append(proj)
            layer_names.append(f"sana_point_{i:02d}")

    elif hook_kind == "point_output":
        capture_output = [True] * len(transformer.transformer_blocks)
        for i, block in enumerate(transformer.transformer_blocks):
            ff = getattr(block, "ff", None)
            proj = getattr(ff, "conv_point", None) if ff is not None else None
            if proj is None and ff is not None:
                proj = getattr(ff, "point_conv", None)
            if proj is None:
                raise ValueError(
                    f"Expected Sana FFN point projection in layer {i}, got {type(block)}"
                )
            modules.append(proj)
            layer_names.append(f"sana_point_{i:02d}")

    elif hook_kind == "inverted_output":
        capture_output = [True] * len(transformer.transformer_blocks)
        for i, block in enumerate(transformer.transformer_blocks):
            ff = getattr(block, "ff", None)
            proj = getattr(ff, "conv_inverted", None) if ff is not None else None
            if proj is None and ff is not None:
                proj = getattr(ff, "inverted_conv", None)
            if proj is None:
                raise ValueError(
                    f"Expected Sana FFN inverted projection in layer {i}, got {type(block)}"
                )
            modules.append(proj)
            layer_names.append(f"sana_inverted_{i:02d}")

    else:
        raise ValueError(f"Unsupported hook_kind for Sana: {hook_kind}")

    latent_patch_size = int(getattr(transformer.config, "patch_size", 1))
    vae_scale_factor = int(getattr(pipe, "vae_scale_factor", 32))
    image_patch_size = int(vae_scale_factor * latent_patch_size)

    return SanaSpec(
        pipeline=pipe,
        transformer=transformer,
        modules=modules,
        layer_names=layer_names,
        capture_output=capture_output,
        patch_token_offset=0,
        latent_patch_size=latent_patch_size,
        image_patch_size=image_patch_size,
        vae_scale_factor=vae_scale_factor,
        hook_kind=hook_kind,
        model_id=model_id,
        model_dtype=model_dtype,
        notes={
            "latent_patch_size": latent_patch_size,
            "vae_scale_factor": vae_scale_factor,
            "effective_image_patch_size": image_patch_size,
            "hook_kind": hook_kind,
            "num_layers": len(layer_names),
        },
    )


# -----------------------------------------------------------------------------
# OpenCLIP tower loader (supports classic OpenCLIP ViT and timm/EVA towers)
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
    # Prefer the post-activation hidden state when there is an explicit activation module.
    for name in ("act", "activation", "gelu", "nonlinearity", "act1"):
        mod = getattr(mlp, name, None)
        if isinstance(mod, torch.nn.Module):
            return mod

    if isinstance(mlp, torch.nn.Sequential) and len(mlp) >= 2:
        return mlp[1]

    # Fallback: use the first/up projection output. This is pre-activation in some MLPs,
    # but still provides a useful "up projection" analogue.
    for name in ("up_proj", "fc1", "c_fc", "gate_proj", "w1", "wi_0"):
        mod = getattr(mlp, name, None)
        if isinstance(mod, torch.nn.Module):
            return mod

    raise ValueError(f"Unsupported OpenCLIP up/hidden projection structure in layer {layer_idx}: {type(mlp)}")


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

    model, _, preprocess_tf = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
    )
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
            native_image_size_hw = _as_hw(
                getattr(visual, "image_size", None) or getattr(visual, "input_resolution", None)
            )

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
            raise ValueError(
                f"OpenCLIP timm visual tower {type(trunk)} has no patch_embed; cannot infer patch size."
            )

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
            native_image_size_hw = _as_hw(
                getattr(visual, "img_size", None) or getattr(visual, "image_size", None)
            )

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
            raise ValueError(
                f"Expected .mlp in OpenCLIP block {i}, got {type(blk)} (kind={visual_kind})."
            )
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



# -----------------------------------------------------------------------------
# PixIO tower loader
# -----------------------------------------------------------------------------


_PIXIO_ARCH_ALIASES = {
    "vitb16": "pixio_vitb16",
    "pixio_vitb16": "pixio_vitb16",
    "pixio-vitb16": "pixio_vitb16",
    "vitl16": "pixio_vitl16",
    "pixio_vitl16": "pixio_vitl16",
    "pixio-vitl16": "pixio_vitl16",
    "vith16": "pixio_vith16",
    "pixio_vith16": "pixio_vith16",
    "pixio-vith16": "pixio_vith16",
    "vit1b16": "pixio_vit1b16",
    "pixio_vit1b16": "pixio_vit1b16",
    "pixio-vit1b16": "pixio_vit1b16",
    "vit5b16": "pixio_vit5b16",
    "pixio_vit5b16": "pixio_vit5b16",
    "pixio-vit5b16": "pixio_vit5b16",
}


def _normalize_pixio_arch_name(name: str) -> str:
    key = str(name).strip().lower()
    return _PIXIO_ARCH_ALIASES.get(key, key)


def _maybe_add_pixio_repo_to_syspath(repo_dir: Optional[str]) -> Optional[Path]:
    if repo_dir is None:
        return None

    root = Path(repo_dir).expanduser().resolve()
    candidates = [root, root / "pixio"]

    for cand in candidates:
        if (cand / "pixio.py").exists() and (cand / "layers").exists():
            cand_str = str(cand)
            if cand_str not in sys.path:
                sys.path.insert(0, cand_str)
            return cand

    raise FileNotFoundError(
        "Could not find a PixIO source directory. Expected either "
        f"{root}/pixio.py and {root}/layers/, or {root}/pixio/pixio.py and {root}/pixio/layers/."
    )


def _import_pixio_factory(factory_name: str, repo_dir: Optional[str]) -> Tuple[Callable[..., torch.nn.Module], Optional[Path]]:
    import_dir = _maybe_add_pixio_repo_to_syspath(repo_dir)

    try:
        pixio_mod = importlib.import_module("pixio")
    except Exception as exc:
        hint = (
            "Install PixIO on your PYTHONPATH or pass --pixio-repo-dir pointing to the repo directory "
            "that contains pixio.py and layers/."
        )
        raise ImportError(f"Could not import PixIO. {hint}") from exc

    factory = getattr(pixio_mod, factory_name, None)
    if callable(factory):
        return factory, import_dir

    try:
        pixio_submod = importlib.import_module("pixio.pixio")
    except Exception:
        pixio_submod = None

    if pixio_submod is not None:
        factory = getattr(pixio_submod, factory_name, None)
        if callable(factory):
            return factory, import_dir

    raise ValueError(
        f"PixIO model factory '{factory_name}' was not found. "
        "Expected one of: pixio_vitb16, pixio_vitl16, pixio_vith16, pixio_vit1b16, pixio_vit5b16."
    )


def _extract_pixio_down_proj_module(mlp: torch.nn.Module, layer_idx: int) -> torch.nn.Module:
    proj = getattr(mlp, "fc2", None)
    if proj is None:
        raise ValueError(f"Expected PixIO MLP down projection (fc2) in layer {layer_idx}, got {type(mlp)}")
    return proj


def _extract_pixio_inverted_or_hidden_module(mlp: torch.nn.Module, layer_idx: int) -> torch.nn.Module:
    act = getattr(mlp, "act", None)
    if isinstance(act, torch.nn.Module):
        return act
    proj = getattr(mlp, "fc1", None)
    if proj is not None:
        return proj
    raise ValueError(
        f"Expected PixIO MLP activation or up projection (act/fc1) in layer {layer_idx}, got {type(mlp)}"
    )


def _load_pixio_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> None:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = state
    if isinstance(state, dict):
        for key in ("state_dict", "model", "module", "encoder", "backbone"):
            maybe = state.get(key)
            if isinstance(maybe, dict) and maybe:
                state_dict = maybe
                break

    if not isinstance(state_dict, dict) or not state_dict:
        raise RuntimeError(
            f"Unsupported PixIO checkpoint format at {checkpoint_path}. "
            "Expected a PyTorch state_dict or a wrapper dict containing one."
        )

    candidate_state_dicts = [state_dict]
    for prefix in ("module.", "model.", "backbone.", "encoder."):
        if all(isinstance(k, str) and k.startswith(prefix) for k in state_dict.keys()):
            candidate_state_dicts.append({k[len(prefix):]: v for k, v in state_dict.items()})

    last_exc: Optional[Exception] = None
    for candidate in candidate_state_dicts:
        try:
            model.load_state_dict(candidate, strict=True)
            return
        except Exception as exc:
            last_exc = exc

    raise RuntimeError(
        f"Could not load PixIO checkpoint from {checkpoint_path}. "
        f"Last load_state_dict error: {last_exc}"
    ) from last_exc


def load_pixio_tower(
    model_name: str,
    checkpoint_path: str,
    device: torch.device,
    hook_kind: str,
    model_dtype: torch.dtype,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
    input_size_hw: Optional[Tuple[int, int]] = None,
    repo_dir: Optional[str] = None,
) -> TowerSpec:
    factory_name = _normalize_pixio_arch_name(model_name)
    factory, import_dir = _import_pixio_factory(factory_name, repo_dir)

    try:
        model = factory(pretrained=None)
    except TypeError:
        model = factory()

    if checkpoint_path:
        _load_pixio_checkpoint(model, checkpoint_path)

    model = model.to(device=device).eval()
    if model_dtype != torch.float32:
        model = model.to(dtype=model_dtype)

    if not hasattr(model, "blocks") or not hasattr(model, "patch_embed"):
        raise ValueError(
            f"Unsupported PixIO model structure {type(model)}. "
            "Expected attributes .blocks and .patch_embed."
        )

    blocks = list(model.blocks)
    patch_embed = model.patch_embed

    patch_size_obj = getattr(patch_embed, "patch_size", None)
    if isinstance(patch_size_obj, tuple):
        patch_size = int(patch_size_obj[0])
    elif isinstance(patch_size_obj, int):
        patch_size = int(patch_size_obj)
    else:
        raise ValueError(f"Could not determine PixIO patch size from {type(patch_embed)}")

    hook_kind = normalize_hook_kind(hook_kind)
    if hook_kind == "point_input":
        capture_output = [False] * len(blocks)
    elif hook_kind in {"point_output", "inverted_output"}:
        capture_output = [True] * len(blocks)
    else:
        raise ValueError(f"Unsupported hook_kind for PixIO: {hook_kind}")

    modules: List[torch.nn.Module] = []
    layer_names: List[str] = []
    for i, blk in enumerate(blocks):
        mlp = getattr(blk, "mlp", None)
        if mlp is None:
            raise ValueError(f"Expected .mlp in PixIO block {i}, got {type(blk)}")

        if hook_kind in {"point_input", "point_output"}:
            modules.append(_extract_pixio_down_proj_module(mlp, i))
        else:
            modules.append(_extract_pixio_inverted_or_hidden_module(mlp, i))
        layer_names.append(f"disc_block_{i:02d}")

    patch_token_offset = int(getattr(model, "n_cls_tokens", 0))

    native_image_size_hw = input_size_hw
    if native_image_size_hw is None:
        native_image_size_hw = _as_hw(getattr(patch_embed, "img_size", None))
    if native_image_size_hw is not None and any((dim % patch_size) != 0 for dim in native_image_size_hw):
        raise ValueError(
            f"PixIO input size {native_image_size_hw} must be divisible by the patch size {patch_size}."
        )

    expected_num_patches = None
    if native_image_size_hw is not None:
        expected_num_patches = int((native_image_size_hw[0] // patch_size) * (native_image_size_hw[1] // patch_size))
    elif hasattr(patch_embed, "num_patches"):
        try:
            expected_num_patches = int(patch_embed.num_patches)
        except Exception:
            pass

    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    preprocess = _make_image_preprocess(mean, std, resize_hw=native_image_size_hw)

    def forward(images: torch.Tensor):
        try:
            return model(images, block_ids=[])
        except TypeError:
            return model(images)

    return TowerSpec(
        family="pixio",
        model=model,
        modules=modules,
        layer_names=layer_names,
        capture_output=capture_output,
        patch_token_offset=patch_token_offset,
        patch_size=patch_size,
        expected_num_patches=expected_num_patches,
        preprocess=preprocess,
        forward=forward,
        notes={
            "model_name": factory_name,
            "checkpoint_path": checkpoint_path,
            "import_dir": str(import_dir) if import_dir is not None else None,
            "patch_token_offset": patch_token_offset,
            "hook_kind": hook_kind,
            "supports_variable_resolution_via_pos_interp": True,
        },
        native_image_size_hw=native_image_size_hw,
        model_dtype=model_dtype,
    )


# -----------------------------------------------------------------------------
# Activation slicing / resampling / stats / top-k utilities
# -----------------------------------------------------------------------------


def _infer_square_grid(num_tokens: int) -> Tuple[int, int]:
    side = int(round(math.sqrt(num_tokens)))
    if side * side != num_tokens:
        raise RuntimeError(f"Expected a square token grid, got {num_tokens} patch tokens.")
    return side, side


def _activation_to_spatial_map(act: torch.Tensor, patch_token_offset: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    if act.ndim == 3:
        if patch_token_offset >= act.shape[1]:
            raise RuntimeError(
                f"Patch token offset {patch_token_offset} >= token count {act.shape[1]}; wrong hook point or prefix size."
            )
        x = act[:, patch_token_offset:, :]
        grid_hw = _infer_square_grid(x.shape[1])
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
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    x, native_grid_hw = _activation_to_spatial_map(act, patch_token_offset)
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


def _merge_rowwise_topk(
    existing_scores: torch.Tensor,
    existing_layers: torch.Tensor,
    existing_neurons: torch.Tensor,
    new_scores: torch.Tensor,
    new_layer_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def _merge_colwise_topk(
    existing_scores: torch.Tensor,
    existing_layers: torch.Tensor,
    existing_neurons: torch.Tensor,
    new_scores: torch.Tensor,
    new_layer_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        raise RuntimeError("Received an empty image list from Sana pipeline.")
    return torch.stack(tensors, dim=0)


def _ensure_bchw_tensor01(images) -> torch.Tensor:
    if isinstance(images, torch.Tensor):
        x = images.detach()
        if x.ndim != 4:
            raise RuntimeError(f"Expected 4D tensor from Sana pipeline, got {tuple(x.shape)}")
        if x.shape[1] not in {1, 3} and x.shape[-1] in {1, 3}:
            x = x.permute(0, 3, 1, 2).contiguous()
        return x.to(torch.float32)

    if isinstance(images, Image.Image):
        return _pil_images_to_tensor01([images])

    if isinstance(images, (list, tuple)):
        if len(images) == 0:
            raise RuntimeError("Received an empty image list from Sana pipeline.")
        if isinstance(images[0], Image.Image):
            return _pil_images_to_tensor01(images)
        raise RuntimeError(f"Unsupported Sana pipeline image list element type: {type(images[0])}")

    raise RuntimeError(f"Unsupported Sana pipeline image output type: {type(images)}")


# -----------------------------------------------------------------------------
# Image saving
# -----------------------------------------------------------------------------


@torch.inference_mode()
def _save_image_batch(
    images_m11: torch.Tensor,
    entries: Sequence[PromptEntry],
    out_dir: Path,
    start_index: int,
    image_format: str = "png",
) -> None:
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
# Sana image iteration
# -----------------------------------------------------------------------------


@torch.inference_mode()
def iter_generated_batches(
    sana: SanaSpec,
    prompt_schedule: Sequence[PromptEntry],
    batch_size: int,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    max_timesteps: float,
    intermediate_timestep: Optional[float],
    height: int,
    width: int,
    use_resolution_binning: bool,
    clean_caption: bool,
    max_sequence_length: int,
    prompt_enhancement: bool,
    device: torch.device,
    shard_start: int,
    shard_end: int,
    save_images_dir: Optional[Path] = None,
    save_image_format: str = "png",
) -> Iterable[Tuple[torch.Tensor, List[PromptEntry], int]]:
    pipe = sana.pipeline
    gen_device = "cuda" if str(device).startswith("cuda") else "cpu"

    for start in range(shard_start, shard_end, batch_size):
        end = min(start + batch_size, shard_end)
        entries = list(prompt_schedule[start:end])
        prompts = [e.prompt for e in entries]
        generator = torch.Generator(device=gen_device)
        generator.manual_seed(batch_seed(seed, start))

        pipe_kwargs = dict(
            prompt=prompts,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            max_timesteps=max_timesteps,
            intermediate_timesteps=intermediate_timestep,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pt",
            return_dict=True,
            clean_caption=clean_caption,
            max_sequence_length=max_sequence_length,
            use_resolution_binning=use_resolution_binning,
        )
        # To preserve the notebook's default behavior, only override
        # complex_human_instruction when prompt enhancement is explicitly disabled.
        if not prompt_enhancement:
            pipe_kwargs["complex_human_instruction"] = []

        out = pipe(**pipe_kwargs)
        images01 = _ensure_bchw_tensor01(out.images)
        images_m11 = images01.mul(2.0).sub(1.0)

        if save_images_dir is not None:
            _save_image_batch(
                images_m11=images_m11,
                entries=entries,
                out_dir=save_images_dir,
                start_index=start,
                image_format=save_image_format,
            )

        yield images_m11, entries, start


# -----------------------------------------------------------------------------
# Passes
# -----------------------------------------------------------------------------


@torch.inference_mode()
def compute_layer_stats(
    sana: SanaSpec,
    disc: TowerSpec,
    prompt_schedule: Sequence[PromptEntry],
    batch_size: int,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    max_timesteps: float,
    intermediate_timestep: Optional[float],
    height: int,
    width: int,
    use_resolution_binning: bool,
    clean_caption: bool,
    max_sequence_length: int,
    prompt_enhancement: bool,
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

    cap_sana = MultiActivationCapture(sana.modules, capture_output=sana.capture_output).register()
    cap_disc = MultiActivationCapture(disc.modules, capture_output=disc.capture_output).register()
    num_batches = _num_batches(local_num_images, batch_size)

    sana_collector = _make_side_collector(
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
        canonical_grid_hw: Optional[Tuple[int, int]] = None
        sana_native_grid_hw: Optional[Tuple[int, int]] = None
        disc_native_grid_hw: Optional[Tuple[int, int]] = None

        with tqdm(total=num_batches, desc=f"stats batches [rank{dist_env.rank}]", **tqdm_kwargs(dist_env)) as pbar:
            for batch_idx, (images_m11, _entries, _global_start) in enumerate(
                iter_generated_batches(
                    sana=sana,
                    prompt_schedule=prompt_schedule,
                    batch_size=batch_size,
                    seed=seed,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    max_timesteps=max_timesteps,
                    intermediate_timestep=intermediate_timestep,
                    height=height,
                    width=width,
                    use_resolution_binning=use_resolution_binning,
                    clean_caption=clean_caption,
                    max_sequence_length=max_sequence_length,
                    prompt_enhancement=prompt_enhancement,
                    device=device,
                    shard_start=shard_start,
                    shard_end=shard_end,
                    save_images_dir=save_images_dir,
                    save_image_format=save_image_format,
                ),
                start=1,
            ):
                sana_acts = cap_sana.get_and_clear()

                disc_images = disc.preprocess(images_m11).to(device=device, dtype=disc.model_dtype)
                _ = disc.forward(disc_images)
                disc_acts = cap_disc.get_and_clear()

                if canonical_grid_source == "sana":
                    flat_sana: List[torch.Tensor] = []
                    for act in sana_acts:
                        if canonical_grid_hw is None:
                            x, native_grid_hw = _flatten_activation_on_grid(
                                act,
                                patch_token_offset=sana.patch_token_offset,
                                target_grid_hw=None,
                                resample_mode=act_resample_mode,
                            )
                            canonical_grid_hw = native_grid_hw
                            sana_native_grid_hw = native_grid_hw
                        else:
                            x, native_grid_hw = _flatten_activation_on_grid(
                                act,
                                patch_token_offset=sana.patch_token_offset,
                                target_grid_hw=canonical_grid_hw,
                                resample_mode=act_resample_mode,
                            )
                            sana_native_grid_hw = sana_native_grid_hw or native_grid_hw
                        flat_sana.append(x.to(torch.float32))

                    assert canonical_grid_hw is not None

                    flat_disc: List[torch.Tensor] = []
                    for act in disc_acts:
                        y, native_grid_hw = _flatten_activation_on_grid(
                            act,
                            patch_token_offset=disc.patch_token_offset,
                            target_grid_hw=canonical_grid_hw,
                            resample_mode=act_resample_mode,
                        )
                        disc_native_grid_hw = disc_native_grid_hw or native_grid_hw
                        flat_disc.append(y.to(torch.float32))

                elif canonical_grid_source == "disc":
                    flat_disc: List[torch.Tensor] = []
                    for act in disc_acts:
                        if canonical_grid_hw is None:
                            y, native_grid_hw = _flatten_activation_on_grid(
                                act,
                                patch_token_offset=disc.patch_token_offset,
                                target_grid_hw=None,
                                resample_mode=act_resample_mode,
                            )
                            canonical_grid_hw = native_grid_hw
                            disc_native_grid_hw = native_grid_hw
                        else:
                            y, native_grid_hw = _flatten_activation_on_grid(
                                act,
                                patch_token_offset=disc.patch_token_offset,
                                target_grid_hw=canonical_grid_hw,
                                resample_mode=act_resample_mode,
                            )
                            disc_native_grid_hw = disc_native_grid_hw or native_grid_hw
                        flat_disc.append(y.to(torch.float32))

                    assert canonical_grid_hw is not None

                    flat_sana: List[torch.Tensor] = []
                    for act in sana_acts:
                        x, native_grid_hw = _flatten_activation_on_grid(
                            act,
                            patch_token_offset=sana.patch_token_offset,
                            target_grid_hw=canonical_grid_hw,
                            resample_mode=act_resample_mode,
                        )
                        sana_native_grid_hw = sana_native_grid_hw or native_grid_hw
                        flat_sana.append(x.to(torch.float32))

                else:
                    raise ValueError(f"Unsupported canonical_grid_source: {canonical_grid_source}")

                sample_count = flat_sana[0].shape[0]
                if any(x.shape[0] != sample_count for x in flat_sana + flat_disc):
                    raise RuntimeError("Inconsistent canonical patch sample count across layers.")
                local_samples += int(sample_count)

                sana_collector.update_layers(flat_sana)
                disc_collector.update_layers(flat_disc)

                processed_images = min(batch_idx * batch_size, local_num_images)
                pbar.update(1)
                pbar.set_postfix(local_images=f"{processed_images}/{local_num_images}", local_samples=f"{local_samples:,}")

        assert canonical_grid_hw is not None

        sana_normalizers, total_samples = sana_collector.finalize(dist_env)
        disc_normalizers, disc_total_samples = disc_collector.finalize(dist_env)
        if total_samples != disc_total_samples:
            raise RuntimeError(
                f"Inconsistent sample counts between sana ({total_samples}) and disc ({disc_total_samples}) collectors."
            )
        return sana_normalizers, disc_normalizers, total_samples, canonical_grid_hw, sana_native_grid_hw, disc_native_grid_hw
    finally:
        cap_sana.remove()
        cap_disc.remove()


@torch.inference_mode()
def accumulate_corr_for_disc_chunk(
    sana: SanaSpec,
    disc: TowerSpec,
    disc_modules_chunk: List[torch.nn.Module],
    disc_chunk_capture_output: List[bool],
    disc_chunk_indices: List[int],
    sana_normalizers: List[Normalizer],
    disc_normalizers: List[Normalizer],
    prompt_schedule: Sequence[PromptEntry],
    batch_size: int,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    max_timesteps: float,
    intermediate_timestep: Optional[float],
    height: int,
    width: int,
    use_resolution_binning: bool,
    clean_caption: bool,
    max_sequence_length: int,
    prompt_enhancement: bool,
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

    cap_sana = MultiActivationCapture(sana.modules, capture_output=sana.capture_output).register()
    cap_disc = MultiActivationCapture(disc_modules_chunk, capture_output=disc_chunk_capture_output).register()
    num_batches = _num_batches(local_num_images, batch_size)

    chunk_start = disc_chunk_indices[0]
    chunk_end = disc_chunk_indices[-1] + 1
    chunk_desc = f"corr batches disc[{chunk_start}:{chunk_end}]"

    try:
        accumulators: Optional[List[List[torch.Tensor]]] = None

        with tqdm(total=num_batches, desc=f"{chunk_desc} [rank{dist_env.rank}]", leave=False, **tqdm_kwargs(dist_env)) as pbar:
            for batch_idx, (images_m11, _entries, _global_start) in enumerate(
                iter_generated_batches(
                    sana=sana,
                    prompt_schedule=prompt_schedule,
                    batch_size=batch_size,
                    seed=seed,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    max_timesteps=max_timesteps,
                    intermediate_timestep=intermediate_timestep,
                    height=height,
                    width=width,
                    use_resolution_binning=use_resolution_binning,
                    clean_caption=clean_caption,
                    max_sequence_length=max_sequence_length,
                    prompt_enhancement=prompt_enhancement,
                    device=device,
                    shard_start=shard_start,
                    shard_end=shard_end,
                ),
                start=1,
            ):
                sana_acts = cap_sana.get_and_clear()

                disc_images = disc.preprocess(images_m11).to(device=device, dtype=disc.model_dtype)
                _ = disc.forward(disc_images)
                disc_acts = cap_disc.get_and_clear()

                flat_sana: List[torch.Tensor] = []
                for i, act in enumerate(sana_acts):
                    x, _ = _flatten_activation_on_grid(
                        act,
                        patch_token_offset=sana.patch_token_offset,
                        target_grid_hw=canonical_grid_hw,
                        resample_mode=act_resample_mode,
                    )
                    x = sana_normalizers[i].normalize(x, device=device, dtype=compute_dtype)
                    flat_sana.append(x)

                flat_disc: List[torch.Tensor] = []
                for local_j, act in enumerate(disc_acts):
                    j = disc_chunk_indices[local_j]
                    y, _ = _flatten_activation_on_grid(
                        act,
                        patch_token_offset=disc.patch_token_offset,
                        target_grid_hw=canonical_grid_hw,
                        resample_mode=act_resample_mode,
                    )
                    y = disc_normalizers[j].normalize(y, device=device, dtype=compute_dtype)
                    flat_disc.append(y)

                if accumulators is None:
                    accumulators = []
                    for x in flat_sana:
                        row: List[torch.Tensor] = []
                        for y in flat_disc:
                            row.append(torch.zeros((x.shape[1], y.shape[1]), dtype=torch.float32, device=device))
                        accumulators.append(row)

                assert accumulators is not None
                for i, x in enumerate(flat_sana):
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
        cap_sana.remove()
        cap_disc.remove()


# -----------------------------------------------------------------------------
# Best-buddy extraction
# -----------------------------------------------------------------------------


def build_mutual_topk_pairs(
    sana_layer_names: List[str],
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
                        "sana_layer_idx": la,
                        "sana_layer": sana_layer_names[la],
                        "sana_neuron": na,
                        "disc_layer_idx": lb,
                        "disc_layer": disc_layer_names[lb],
                        "disc_neuron": nb,
                        "correlation": corr,
                        "rank_in_sana": rank_a + 1,
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

    # Sana generation
    parser.add_argument("--sana-model-id", type=str, default="Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers")
    parser.add_argument("--sana-dtype", type=str, choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--num-inference-steps", type=int, default=1)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--max-timesteps", type=float, default=1.57080)
    parser.add_argument("--intermediate-timestep", type=float, default=None)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--use-resolution-binning", dest="use_resolution_binning", action="store_true")
    parser.add_argument("--disable-resolution-binning", dest="use_resolution_binning", action="store_false")
    parser.set_defaults(use_resolution_binning=True)
    parser.add_argument("--clean-caption", action="store_true")
    parser.add_argument("--max-sequence-length", type=int, default=300)
    parser.add_argument("--disable-prompt-enhancement", dest="prompt_enhancement", action="store_false")
    parser.add_argument("--enable-prompt-enhancement", dest="prompt_enhancement", action="store_true")
    parser.set_defaults(prompt_enhancement=True)
    parser.add_argument("--enable-vae-tiling", action="store_true")
    parser.add_argument("--enable-vae-slicing", action="store_true")

    # Prompt schedule
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="{prompt_label_article}",
        help=(
            "Quality-first prompt template. Available placeholders: {label}, {label_article}, "
            "{prompt_label}, {prompt_label_article}, {class_idx}. Use "
            "'a photo of {prompt_label_article}' if you want a stricter ImageNet-photo style."
        ),
    )
    parser.add_argument("--label-alias-mode", choices=["raw", "first", "longest"], default="longest")
    parser.add_argument("--class-list-file", type=str, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-images", type=int, default=1000)

    # Discriminative tower
    parser.add_argument("--disc-family", type=str, choices=["openclip", "pixio"], default="openclip")
    parser.add_argument(
        "--disc-arch",
        type=str,
        default="EVA02-E-14-plus",
        help=(
            "For --disc-family openclip: the OpenCLIP model name (for example EVA02-E-14-plus). "
            "For --disc-family pixio: the PixIO constructor name (for example pixio_vith16)."
        ),
    )
    parser.add_argument(
        "--disc-pretrained",
        type=str,
        default="laion2b_s9b_b144k",
        help="OpenCLIP pretrained tag. Ignored when --disc-family pixio is used.",
    )
    parser.add_argument(
        "--disc-checkpoint",
        type=str,
        default=None,
        help="Local checkpoint path for --disc-family pixio.",
    )
    parser.add_argument(
        "--pixio-repo-dir",
        type=str,
        default=None,
        help=(
            "Optional path to the local PixIO source directory. You can point this at either the repo root "
            "or the inner repo/pixio directory that contains pixio.py and layers/."
        ),
    )
    parser.add_argument("--disc-model-dtype", type=str, choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--disc-input-size", type=int, default=224)
    parser.add_argument("--disc-mean", type=float, nargs=3, default=None)
    parser.add_argument("--disc-std", type=float, nargs=3, default=None)

    # Matching
    parser.add_argument(
        "--hook-kind",
        choices=["input", "output", "point_input", "point_output", "inverted_output"],
        default="output",
        help=(
            "Hook placement. 'input'/'point_input' = input to the down projection (most neuron-like hidden state), "
            "'output'/'point_output' = output of the down projection, "
            "'inverted_output' = output of Sana's inverted/up projection and the closest available OpenCLIP/PixIO up/hidden projection."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Per-rank batch size in distributed mode.")
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--disc-chunk-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compute-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--act-resample-mode", choices=["nearest", "bilinear", "bicubic", "area"], default="bilinear")
    parser.add_argument("--canonical-grid-source", choices=["sana", "disc"], default="sana",
                        help="Which model defines the canonical spatial grid used for activation alignment.")
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
            "sampling. 'exact' stores every activation on CPU and ranks the full population "
            "(memory grows linearly with the data)."
        ),
    )
    parser.add_argument(
        "--spearman-reservoir-size",
        type=int,
        default=4096,
        help="Per-neuron reservoir size used in --spearman-mode approx. Larger = closer to exact Spearman.",
    )
    parser.add_argument("--warn-accumulator-gb", type=float, default=8.0,
                        help="Print a warning if estimated accumulator memory for one disc chunk exceeds this many GB.")

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
        sana_dtype = torch_dtype_from_name(args.sana_dtype)
        disc_model_dtype = torch_dtype_from_name(args.disc_model_dtype)
        normalized_hook_kind = normalize_hook_kind(args.hook_kind)

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

        sana = load_sana_sprint(
            model_id=args.sana_model_id,
            device=device,
            model_dtype=sana_dtype,
            hook_kind=normalized_hook_kind,
            enable_vae_tiling=args.enable_vae_tiling,
            enable_vae_slicing=args.enable_vae_slicing,
            disable_progress_bar=True, #not dist_env.is_main,
        )

        disc_input_hw = _as_hw(args.disc_input_size)
        if args.disc_family == "openclip":
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
            if not args.disc_checkpoint:
                raise ValueError("--disc-checkpoint is required when --disc-family pixio.")
            disc = load_pixio_tower(
                model_name=args.disc_arch,
                checkpoint_path=args.disc_checkpoint,
                device=device,
                hook_kind=normalized_hook_kind,
                model_dtype=disc_model_dtype,
                mean=args.disc_mean,
                std=args.disc_std,
                input_size_hw=disc_input_hw,
                repo_dir=args.pixio_repo_dir,
            )
        else:
            raise ValueError(f"Unsupported disc family: {args.disc_family}")

        if dist_env.is_main:
            print(f"[setup] distributed world size: {dist_env.world_size}")
            print(f"[setup] per-rank batch size: {args.batch_size}")
            print(f"[setup] num prompts: {len(prompt_schedule)}")
            print(f"[setup] sana model: {args.sana_model_id}")
            print(f"[setup] sana hook kind: {normalized_hook_kind}")
            print(f"[setup] sana dtype: {sana_dtype}")
            print(f"[setup] disc family: {args.disc_family}")
            print(f"[setup] disc arch: {args.disc_arch}")
            if args.disc_family == "openclip":
                print(f"[setup] disc pretrained: {args.disc_pretrained}")
            elif args.disc_checkpoint is not None:
                print(f"[setup] disc checkpoint: {args.disc_checkpoint}")
                if args.pixio_repo_dir is not None:
                    print(f"[setup] pixio repo dir: {args.pixio_repo_dir}")
            print(f"[setup] disc dtype: {disc_model_dtype}")
            if args.num_inference_steps > 1:
                print(
                    f"[setup] num_inference_steps={args.num_inference_steps}; "
                    "this script captures the last denoising step only."
                )
            if args.use_resolution_binning:
                print("[setup] Sana resolution binning is ENABLED (matches notebook default).")
            else:
                print("[setup] Sana resolution binning is DISABLED.")
            print(f"[setup] prompt enhancement: {'ENABLED' if args.prompt_enhancement else 'DISABLED'}")
            print(f"[setup] label alias mode: {args.label_alias_mode}")
            print(f"[setup] args.num_images: {args.num_images}")
            if prompt_schedule:
                print("[setup] first prompts:")
                for entry in prompt_schedule[: min(5, len(prompt_schedule))]:
                    print(f"  - class {entry.class_idx:04d}: raw='{entry.raw_label}' | prompt_label='{entry.prompt_label}' | prompt='{entry.prompt}'")
            if disc.native_image_size_hw is not None:
                print(f"[setup] disc input size: {disc.native_image_size_hw[0]}x{disc.native_image_size_hw[1]}")

        print0(dist_env, f"[setup] similarity: {args.similarity}")
        if args.similarity == "spearman":
            print0(dist_env, f"[setup] spearman mode: {args.spearman_mode} (reservoir size {args.spearman_reservoir_size})")
        print0(dist_env, "[1/3] Computing per-layer stats on aligned grid...")
        sana_normalizers, disc_normalizers, total_samples, canonical_grid_hw, sana_native_grid_hw, disc_native_grid_hw = compute_layer_stats(
            sana=sana,
            disc=disc,
            prompt_schedule=prompt_schedule,
            batch_size=args.batch_size,
            seed=args.seed,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            max_timesteps=args.max_timesteps,
            intermediate_timestep=args.intermediate_timestep,
            height=args.height,
            width=args.width,
            use_resolution_binning=args.use_resolution_binning,
            clean_caption=args.clean_caption,
            max_sequence_length=args.max_sequence_length,
            prompt_enhancement=args.prompt_enhancement,
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
        if sana_native_grid_hw is not None:
            print0(dist_env, f"[setup] sana native activation grid: {sana_native_grid_hw[0]}x{sana_native_grid_hw[1]}")
        if disc_native_grid_hw is not None:
            print0(dist_env, f"[setup] disc native activation grid: {disc_native_grid_hw[0]}x{disc_native_grid_hw[1]}")
        if args.canonical_grid_source == "sana" and disc_native_grid_hw is not None and disc_native_grid_hw != canonical_grid_hw:
            print0(
                dist_env,
                f"[setup] disc activations are resampled to the canonical grid: "
                f"{disc_native_grid_hw[0]}x{disc_native_grid_hw[1]} -> {canonical_grid_hw[0]}x{canonical_grid_hw[1]}",
            )
        if args.canonical_grid_source == "disc" and sana_native_grid_hw is not None and sana_native_grid_hw != canonical_grid_hw:
            print0(
                dist_env,
                f"[setup] sana activations are resampled to the canonical grid: "
                f"{sana_native_grid_hw[0]}x{sana_native_grid_hw[1]} -> {canonical_grid_hw[0]}x{canonical_grid_hw[1]}",
            )

        print0(dist_env, "[2/3] Accumulating correlations and global top-k neighbors...")
        sana_dims = {i: st.dim for i, st in enumerate(sana_normalizers)}
        disc_dims = {j: st.dim for j, st in enumerate(disc_normalizers)}

        topk_a_scores, topk_a_layers, topk_a_neurons = _init_global_topk(sana_dims, args.topk)
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
                for i in range(len(sana_normalizers)):
                    for j in chunk_indices:
                        est_bytes += sana_normalizers[i].dim * disc_normalizers[j].dim * 4
                est_gb = est_bytes / (1024 ** 3)
                tqdm.write(f"  - disc layers {chunk_start}:{chunk_end} | est accum memory ~ {est_gb:.2f} GB")
                if est_gb > args.warn_accumulator_gb:
                    tqdm.write(
                        "    WARNING: large accumulator estimate. Consider smaller --disc-chunk-size "
                        "or --hook-kind output."
                    )

            accumulators = accumulate_corr_for_disc_chunk(
                sana=sana,
                disc=disc,
                disc_modules_chunk=chunk_modules,
                disc_chunk_capture_output=chunk_capture_output,
                disc_chunk_indices=chunk_indices,
                sana_normalizers=sana_normalizers,
                disc_normalizers=disc_normalizers,
                prompt_schedule=prompt_schedule,
                batch_size=args.batch_size,
                seed=args.seed,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                max_timesteps=args.max_timesteps,
                intermediate_timestep=args.intermediate_timestep,
                height=args.height,
                width=args.width,
                use_resolution_binning=args.use_resolution_binning,
                clean_caption=args.clean_caption,
                max_sequence_length=args.max_sequence_length,
                prompt_enhancement=args.prompt_enhancement,
                device=device,
                compute_dtype=compute_dtype,
                canonical_grid_hw=canonical_grid_hw,
                act_resample_mode=args.act_resample_mode,
                dist_env=dist_env,
            )

            if dist_env.is_main:
                assert accumulators is not None
                for i, sana_layer_name in enumerate(sana.layer_names):
                    for local_j, disc_idx in enumerate(chunk_indices):
                        corr = (accumulators[i][local_j] / float(total_samples)).cpu()

                        topk_a_scores[i], topk_a_layers[i], topk_a_neurons[i] = _merge_rowwise_topk(
                            topk_a_scores[i], topk_a_layers[i], topk_a_neurons[i], corr, disc_idx
                        )
                        topk_b_scores[disc_idx], topk_b_layers[disc_idx], topk_b_neurons[disc_idx] = _merge_colwise_topk(
                            topk_b_scores[disc_idx], topk_b_layers[disc_idx], topk_b_neurons[disc_idx], corr, i
                        )

                        if args.save_full_corr:
                            out_path = corr_dir / f"corr_{sana_layer_name}_vs_{disc.layer_names[disc_idx]}.pt"
                            torch.save(corr, out_path)

            barrier(dist_env)

        if not dist_env.is_main:
            return

        print("[3/3] Extracting mutual top-k matches...")
        best_buddies = build_mutual_topk_pairs(
            sana_layer_names=sana.layer_names,
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
            "sana_model_id": args.sana_model_id,
            "sana_notes": sana.notes,
            "sana_dtype": args.sana_dtype,
            "hook_kind": normalized_hook_kind,
            "hook_kind_requested": args.hook_kind,
            "disc_family": disc.family,
            "disc_arch": args.disc_arch,
            "disc_pretrained": args.disc_pretrained if args.disc_family == "openclip" else None,
            "disc_checkpoint": args.disc_checkpoint,
            "pixio_repo_dir": args.pixio_repo_dir,
            "disc_notes": disc.notes,
            "disc_model_dtype": args.disc_model_dtype,
            "disc_native_image_size_hw": list(disc.native_image_size_hw) if disc.native_image_size_hw is not None else None,
            "canonical_grid_source": args.canonical_grid_source,
            "canonical_grid_hw": list(canonical_grid_hw),
            "sana_native_grid_hw": list(sana_native_grid_hw) if sana_native_grid_hw is not None else None,
            "disc_native_grid_hw": list(disc_native_grid_hw) if disc_native_grid_hw is not None else None,
            "requested_height": args.height,
            "requested_width": args.width,
            "use_resolution_binning": args.use_resolution_binning,
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
            "max_timesteps": args.max_timesteps,
            "intermediate_timestep": args.intermediate_timestep,
            "seed": args.seed,
            "topk": args.topk,
            "total_patch_samples": total_samples,
            "sana_num_layers": len(sana.layer_names),
            "disc_num_layers": len(disc.layer_names),
            "sana_image_patch_size": sana.image_patch_size,
            "disc_patch_size": disc.patch_size,
            "prompt_template": args.prompt_template,
            "label_alias_mode": args.label_alias_mode,
            "clean_caption": args.clean_caption,
            "prompt_enhancement": args.prompt_enhancement,
            "generation_seed_mode": "batch_seed(global_batch_start_index)",
            "save_generated_images": args.save_generated_images,
            "generated_images_subdir": args.generated_images_subdir if args.save_generated_images else None,
            "generated_image_format": args.generated_image_format if args.save_generated_images else None,
            "capture_step_when_multistep": "last",
        }

        with open(save_dir / "run_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        with open(save_dir / "best_buddies.json", "w", encoding="utf-8") as f:
            json.dump(best_buddies, f, indent=2)

        neighbors_dir = save_dir / "neighbors"
        neighbors_dir.mkdir(parents=True, exist_ok=True)
        for i, layer_name in enumerate(sana.layer_names):
            torch.save(
                {
                    "scores": topk_a_scores[i],
                    "disc_layer_idx": topk_a_layers[i],
                    "disc_neuron": topk_a_neurons[i],
                },
                neighbors_dir / f"sana_{layer_name}_top{args.topk}.pt",
            )

        print(f"Saved run metadata to {save_dir / 'run_metadata.json'}")
        print(f"Saved prompt manifest to {save_dir / 'prompt_manifest.json'}")
        print(f"Saved {len(best_buddies):,} mutual top-k pairs to {save_dir / 'best_buddies.json'}")
        if save_images_dir is not None:
            print(f"Saved generated images to {save_images_dir}")
        if best_buddies:
            print("Top 10 pairs:")
            for row in best_buddies[:10]:
                print(
                    f"  {row['sana_layer']}[{row['sana_neuron']}] <-> "
                    f"{row['disc_layer']}[{row['disc_neuron']}]: {row['correlation']:.4f}"
                )
    finally:
        cleanup_distributed(dist_env)


if __name__ == "__main__":
    main()
