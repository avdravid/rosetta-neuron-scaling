
#!/usr/bin/env python3
"""
Visualize matched pMF neurons against one or more target models.

Two input modes:

1. Multi-model Rosetta anchors (--anchors-dir + --run label=dir ...):
   - loads rosetta_anchors.json + per-target-model run_metadata.json
   - treats pMF as the anchor; shows maps for ALL target models on the same images

2. Single pairwise run (--results-dir):
   - loads best_buddies.json + run_metadata.json from one matching-run directory
   - treats each best-buddy pair as a single-match anchor; shows pMF + the one disc model

In both modes:
- chooses images from ONE model's point of view (--select-model and --select-stat)
- renders a dark-theme, searchable single-page viewer (sidebar anchor list +
  main panel) showing the top-activating images per matched neuron. Each example
  row holds the source image plus per-model heatmap/overlay tiles in one
  responsive grid that reflows to fit the window/zoom width.

Notes:
- For exact regeneration on distributed match outputs, this script follows the
  original per-rank batch seeding scheme:
      generation_seed_mode == "batch_seed(global_batch_start)"
- For older single-GPU outputs that used one sequential RNG stream, only prefix
  sampling is exact. Random subsampling is refused for those runs.
- To keep pMF generation exactly aligned with the original run, this script
  regenerates using the ORIGINAL batch schedule from run_metadata.json, not a new
  visualization batch size.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x=None, **kwargs):
        return x if x is not None else []
    tqdm.write = print


# -----------------------------------------------------------------------------
# RNG / helpers
# -----------------------------------------------------------------------------


class SimpleRNG:
    def __init__(self, device: torch.device | str, seed: int = 0):
        self.device = str(device)
        gen_device = "cuda" if str(device).startswith("cuda") else "cpu"
        self.generator = torch.Generator(device=gen_device)
        self.generator.manual_seed(seed)

    def randn(self, shape: Sequence[int]) -> torch.Tensor:
        return torch.randn(shape, generator=self.generator, device=self.device)

    def randint(
        self,
        low: int,
        high: int,
        size: Sequence[int],
        dtype: torch.dtype = torch.int32,
    ) -> torch.Tensor:
        return torch.randint(low, high, size, generator=self.generator, device=self.device, dtype=dtype)


def batch_seed(base_seed: int, global_batch_start: int) -> int:
    return int(
        (int(base_seed) * 6364136223846793005 + int(global_batch_start) * 1442695040888963407 + 1)
        % (2**63 - 1)
    )


def rank_shard_bounds(num_items: int, world_size: int, rank: int) -> Tuple[int, int]:
    base = num_items // world_size
    rem = num_items % world_size
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return start, end


def build_label_schedule(num_images: int, mode: str, fixed_label: int, seed: int) -> torch.Tensor:
    if mode == "fixed":
        return torch.full((num_images,), fixed_label, dtype=torch.int32)
    if mode == "balanced":
        reps = math.ceil(num_images / 1000)
        return torch.arange(1000, dtype=torch.int32).repeat(reps)[:num_images]
    if mode == "random":
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        return torch.randint(0, 1000, (num_images,), generator=g, dtype=torch.int32)
    raise ValueError(f"Unknown label mode: {mode}")


def build_search_indices(
    total_images: int,
    max_search_images: Optional[int],
    sample_mode: str,
    sample_seed: int,
) -> List[int]:
    n = total_images if max_search_images is None else min(total_images, max_search_images)
    if sample_mode == "prefix":
        return list(range(n))
    rng = random.Random(sample_seed)
    idxs = list(range(total_images))
    rng.shuffle(idxs)
    return idxs[:n]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _import_from_repo(repo: str | Path, module_name: str):
    repo = str(Path(repo).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    return __import__(module_name, fromlist=[module_name])


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


def parse_kv_list(values: List[str], arg_name: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected {arg_name} entries like label=value, got: {item}")
        k, v = item.split("=", 1)
        out[k] = v
    return out


def extract_anchor_matches(anchor: dict) -> Dict[str, dict]:
    if "matches" in anchor and isinstance(anchor["matches"], dict):
        return anchor["matches"]
    out = {}
    for k, v in anchor.items():
        if k.startswith("pmf_") or k in {"avg_correlation", "min_correlation", "num_models", "anchor_id"}:
            continue
        if isinstance(v, dict) and ("disc_layer" in v or "disc_layer_idx" in v or "layer" in v):
            out[k] = v
    return out


def sort_anchors(anchors: List[dict]) -> List[dict]:
    def key_fn(x: dict):
        return float(x.get("avg_correlation", x.get("min_correlation", 0.0)))
    return sorted(anchors, key=key_fn, reverse=True)


def derive_disc_label_from_metadata(metadata: dict) -> str:
    """Pick a stable label for the disc tower in a single pairwise run.

    Used by --results-dir mode to synthesize the single-entry run_dirs mapping.
    """
    family = metadata.get("disc_family")
    if family:
        return str(family)
    return "disc"


def best_buddies_to_anchors(best_buddies: List[dict], label: str) -> List[dict]:
    """Wrap each best-buddy pair as a degenerate Rosetta anchor with one match.

    Reuses the anchors code path for pairwise runs: each entry becomes an
    anchor on the pMF side with a single match keyed by `label` (the disc family).
    """
    anchors: List[dict] = []
    for i, pair in enumerate(best_buddies):
        pmf_layer = pair.get("pmf_layer", pair.get("model1_layer"))
        pmf_neuron = pair.get("pmf_neuron", pair.get("model1_neuron"))
        pmf_layer_idx = pair.get("pmf_layer_idx", pair.get("model1_layer_idx"))
        disc_layer = pair.get("disc_layer", pair.get("model2_layer"))
        disc_layer_idx = pair.get("disc_layer_idx", pair.get("model2_layer_idx"))
        disc_neuron = pair.get("disc_neuron", pair.get("model2_neuron"))
        correlation = float(pair["correlation"])
        anchors.append({
            "anchor_id": i,
            "pmf_layer": pmf_layer,
            "pmf_neuron": int(pmf_neuron),
            "pmf_layer_idx": int(pmf_layer_idx) if pmf_layer_idx is not None else None,
            "avg_correlation": correlation,
            "min_correlation": correlation,
            "matches": {
                label: {
                    "disc_layer": disc_layer,
                    "disc_layer_idx": int(disc_layer_idx) if disc_layer_idx is not None else None,
                    "disc_neuron": int(disc_neuron),
                    "correlation": correlation,
                },
            },
        })
    return anchors


# -----------------------------------------------------------------------------
# Hook capture
# -----------------------------------------------------------------------------


class MultiActivationCapture:
    def __init__(self, modules: List[torch.nn.Module], capture_output: Optional[List[bool]] = None):
        self.modules = modules
        self.capture_output = capture_output or [False] * len(modules)
        self.handles = []
        self.activations: List[Optional[torch.Tensor]] = [None] * len(modules)

    def _make_hook(self, idx: int, use_output: bool):
        def hook(_module, inputs, output):
            act = output if use_output else inputs[0]
            if isinstance(act, (tuple, list)):
                act = act[0]
            self.activations[idx] = act.detach()
        return hook

    def register(self):
        for i, module in enumerate(self.modules):
            self.handles.append(module.register_forward_hook(self._make_hook(i, self.capture_output[i])))
        return self

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        self.activations = [None] * len(self.modules)

    def get_and_clear(self) -> List[torch.Tensor]:
        acts = self.activations
        self.activations = [None] * len(self.modules)
        out: List[torch.Tensor] = []
        for i, act in enumerate(acts):
            if act is None:
                raise RuntimeError(f"Missing activation for hook index {i}.")
            out.append(act)
        return out


# -----------------------------------------------------------------------------
# Model specs
# -----------------------------------------------------------------------------


@dataclass
class PMFSpec:
    model: torch.nn.Module
    modules: List[torch.nn.Module]
    layer_names: List[str]
    patch_token_offset: int
    patch_size: int
    num_patches: int
    img_size: int


@dataclass
class TowerSpec:
    family: str
    model: torch.nn.Module
    modules: List[torch.nn.Module]
    layer_names: List[str]
    patch_token_offset: int
    patch_size: int
    preprocess: Callable[[torch.Tensor], torch.Tensor]
    forward: Callable[[torch.Tensor], object]
    native_image_size_hw: Optional[Tuple[int, int]]
    notes: Dict[str, object]


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------


def load_pmf(
    repo: str | Path,
    model_name: str,
    ckpt_path: Optional[str],
    hf_repo: Optional[str],
    ckpt_file: Optional[str],
    device: torch.device,
) -> PMFSpec:
    pmf_mod = _import_from_repo(repo, "pmf")
    pixel_mean_flow = getattr(pmf_mod, "pixelMeanFlow")

    img_size = 16 * int(model_name.split("_")[-1])
    model = pixel_mean_flow(model_name, img_size=img_size).to(device).eval()

    if ckpt_path is None:
        if hf_repo is None or ckpt_file is None:
            raise ValueError("Provide either --pmf-ckpt-path or both --pmf-hf-repo and --pmf-ckpt-file.")
        ckpt_path = hf_hub_download(repo_id=hf_repo, filename=ckpt_file)

    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)

    net = model.net
    modules: List[torch.nn.Module] = []
    names: List[str] = []
    for i, blk in enumerate(net.shared_blocks):
        modules.append(blk.mlp.w2)
        names.append(f"pmf_shared_{i:02d}")
    for i, blk in enumerate(net.u_heads):
        modules.append(blk.mlp.w2)
        names.append(f"pmf_u_{i:02d}")

    return PMFSpec(
        model=model,
        modules=modules,
        layer_names=names,
        patch_token_offset=int(net.prefix_tokens),
        patch_size=int(net.patch_size),
        num_patches=int(net.x_embedder.num_patches),
        img_size=int(model.img_size),
    )


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


def _extract_mlp_proj_module(mlp: torch.nn.Module, layer_idx: int) -> torch.nn.Module:
    for name in ["down_proj", "fc2", "c_proj", "w2", "w3"]:
        if hasattr(mlp, name):
            return getattr(mlp, name)
    raise ValueError(f"Unsupported MLP/projection structure in layer {layer_idx}: {type(mlp)}")


def load_dinov3_local_tower(
    repo: str | Path,
    arch: str,
    weights: str,
    device: torch.device,
    mean: Sequence[float],
    std: Sequence[float],
    input_size_hw: Optional[Tuple[int, int]],
) -> TowerSpec:
    model = torch.hub.load(str(Path(repo).resolve()), arch, source="local", weights=weights)
    model = model.to(device).eval()

    if not hasattr(model, "blocks"):
        raise ValueError("Expected DINOv3 local model with .blocks")

    modules: List[torch.nn.Module] = []
    names: List[str] = []
    for i, blk in enumerate(model.blocks):
        modules.append(_extract_mlp_proj_module(blk.mlp, i))
        names.append(f"disc_block_{i:02d}")

    patch_size_obj = getattr(model, "patch_size", None)
    if isinstance(patch_size_obj, tuple):
        patch_size = int(patch_size_obj[0])
    elif isinstance(patch_size_obj, int):
        patch_size = int(patch_size_obj)
    else:
        raise ValueError("Could not determine local DINOv3 patch size.")

    n_storage_tokens = int(getattr(model, "n_storage_tokens", 0))
    preprocess = _make_image_preprocess(mean, std, resize_hw=input_size_hw)

    def forward(images: torch.Tensor):
        if hasattr(model, "forward_features"):
            return model.forward_features(images)
        return model(images)

    return TowerSpec(
        family="dinov3",
        model=model,
        modules=modules,
        layer_names=names,
        patch_token_offset=1 + n_storage_tokens,
        patch_size=patch_size,
        preprocess=preprocess,
        forward=forward,
        native_image_size_hw=input_size_hw,
        notes={"arch": arch, "loader": "local"},
    )


def load_dinov3_hf_tower(
    model_id: str,
    device: torch.device,
    mean: Sequence[float],
    std: Sequence[float],
    input_size_hw: Optional[Tuple[int, int]],
) -> TowerSpec:
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_id).to(device).eval()

    if hasattr(model, "model") and hasattr(model.model, "layer"):
        blocks = list(model.model.layer)
        n_storage_tokens = int(
            getattr(model, "n_storage_tokens", getattr(getattr(model, "config", object()), "n_storage_tokens", 0))
        )
    elif hasattr(model, "blocks"):
        blocks = list(model.blocks)
        n_storage_tokens = int(getattr(model, "n_storage_tokens", 0))
    else:
        raise ValueError("Unsupported HF DINOv3 structure.")

    modules: List[torch.nn.Module] = []
    names: List[str] = []
    for i, blk in enumerate(blocks):
        modules.append(_extract_mlp_proj_module(blk.mlp, i))
        names.append(f"disc_block_{i:02d}")

    patch_size_obj = getattr(model, "patch_size", None)
    if patch_size_obj is None:
        patch_size_obj = getattr(getattr(model, "config", object()), "patch_size", None)
    if isinstance(patch_size_obj, tuple):
        patch_size = int(patch_size_obj[0])
    elif isinstance(patch_size_obj, int):
        patch_size = int(patch_size_obj)
    else:
        raise ValueError("Could not determine HF DINOv3 patch size.")

    preprocess = _make_image_preprocess(mean, std, resize_hw=input_size_hw)

    def forward(images: torch.Tensor):
        try:
            return model(pixel_values=images)
        except TypeError:
            return model(images)

    return TowerSpec(
        family="dinov3",
        model=model,
        modules=modules,
        layer_names=names,
        patch_token_offset=1 + n_storage_tokens,
        patch_size=patch_size,
        preprocess=preprocess,
        forward=forward,
        native_image_size_hw=input_size_hw,
        notes={"arch": model_id, "loader": "hf"},
    )


def load_openclip_tower(
    model_name: str,
    pretrained: str,
    device: torch.device,
    mean: Optional[Sequence[float]],
    std: Optional[Sequence[float]],
    input_size_hw: Optional[Tuple[int, int]],
) -> TowerSpec:
    import open_clip

    model, _, preprocess_tf = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    model = model.eval()
    visual = model.visual

    modules: List[torch.nn.Module] = []
    names: List[str] = []
    for i, blk in enumerate(visual.transformer.resblocks):
        modules.append(_extract_mlp_proj_module(blk.mlp, i))
        names.append(f"disc_block_{i:02d}")

    kernel = getattr(visual.conv1, "kernel_size", None)
    if isinstance(kernel, tuple):
        patch_size = int(kernel[0])
    elif isinstance(kernel, int):
        patch_size = int(kernel)
    else:
        raise ValueError("Could not determine OpenCLIP patch size.")

    if mean is None or std is None:
        norm_mean = norm_std = None
        for tf in getattr(preprocess_tf, "transforms", []):
            if tf.__class__.__name__ == "Normalize":
                norm_mean = list(tf.mean)
                norm_std = list(tf.std)
                break
        mean = norm_mean or [0.48145466, 0.4578275, 0.40821073]
        std = norm_std or [0.26862954, 0.26130258, 0.27577711]

    native_hw = input_size_hw
    if native_hw is None:
        image_size = getattr(visual, "image_size", None) or getattr(visual, "input_resolution", None)
        native_hw = _as_hw(image_size)

    preprocess = _make_image_preprocess(mean, std, resize_hw=native_hw)

    def forward(images: torch.Tensor):
        return visual(images)

    return TowerSpec(
        family="openclip",
        model=model,
        modules=modules,
        layer_names=names,
        patch_token_offset=1,
        patch_size=patch_size,
        preprocess=preprocess,
        forward=forward,
        native_image_size_hw=native_hw,
        notes={"model_name": model_name, "pretrained": pretrained},
    )


# -----------------------------------------------------------------------------
# Map conversion / rendering helpers
# -----------------------------------------------------------------------------


def _infer_square_grid(num_tokens: int) -> Tuple[int, int]:
    side = int(round(math.sqrt(num_tokens)))
    if side * side != num_tokens:
        raise RuntimeError(f"Expected square token grid, got {num_tokens} tokens.")
    return side, side


def _act_to_map(
    act: torch.Tensor,
    patch_token_offset: int,
    target_grid_hw: Tuple[int, int],
    resample_mode: str,
) -> torch.Tensor:
    x = act[:, patch_token_offset:, :]
    gh, gw = _infer_square_grid(x.shape[1])
    x = x.reshape(x.shape[0], gh, gw, x.shape[-1]).permute(0, 3, 1, 2)
    if (gh, gw) != target_grid_hw:
        if resample_mode in {"bilinear", "bicubic"}:
            x = F.interpolate(x, size=target_grid_hw, mode=resample_mode, align_corners=False)
        else:
            x = F.interpolate(x, size=target_grid_hw, mode=resample_mode)
    return x


def tensor_to_uint8_image(x_m11: torch.Tensor) -> np.ndarray:
    x = x_m11.detach().clamp(-1.0, 1.0)
    x = ((x + 1.0) / 2.0 * 255.0).round().to(torch.uint8)
    return x.permute(1, 2, 0).cpu().numpy()


def colorize_heatmap(
    map2d: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    if vmin is None:
        vmin = float(np.min(map2d))
    if vmax is None:
        vmax = float(np.max(map2d))
    if vmax <= vmin + 1e-12:
        t = np.zeros_like(map2d, dtype=np.float32)
    else:
        t = np.clip((map2d - vmin) / (vmax - vmin), 0.0, 1.0)

    # viridis colormap (dark blue/purple -> teal -> yellow)
    import matplotlib.cm as _mcm
    rgba = _mcm.viridis(t)
    rgb = rgba[..., :3]
    return (rgb * 255).astype(np.uint8)


def overlay_heatmap(image: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    if heat.shape[:2] != image.shape[:2]:
        heat = np.array(
            Image.fromarray(heat).resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)
        )
    return np.clip(
        (1 - alpha) * image.astype(np.float32) + alpha * heat.astype(np.float32),
        0,
        255,
    ).astype(np.uint8)


def pil_to_data_uri(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"


def np_to_uri(arr: np.ndarray, resize_hw: Optional[Tuple[int, int]] = None, resample=Image.NEAREST) -> str:
    img = Image.fromarray(arr)
    if resize_hw is not None:
        img = img.resize((resize_hw[1], resize_hw[0]), resample=resample)
    return pil_to_data_uri(img, fmt="PNG")


# -----------------------------------------------------------------------------
# Exact regeneration iterators
# -----------------------------------------------------------------------------


def determine_generation_mode(meta: dict) -> str:
    if meta.get("generation_seed_mode") == "batch_seed(global_batch_start)":
        return "distributed_batch_seed"
    return "single_sequential"


def verify_generation_consistency(run_meta_by_label: Dict[str, dict]) -> dict:
    first_label = next(iter(run_meta_by_label))
    ref = run_meta_by_label[first_label]
    keys = [
        "pmf_model",
        "num_images",
        "label_mode",
        "fixed_label",
        "seed",
        "num_steps",
        "omega",
        "t_min",
        "t_max",
        "generation_seed_mode",
        "distributed_world_size",
        "batch_size_per_rank",
        "batch_size",
    ]
    for label, meta in run_meta_by_label.items():
        for k in keys:
            if ref.get(k) != meta.get(k):
                # Allow disc-specific differences, but generation should match.
                raise ValueError(
                    f"Generation metadata mismatch between runs '{first_label}' and '{label}' at key '{k}': "
                    f"{ref.get(k)!r} vs {meta.get(k)!r}"
                )
    return ref


def iter_selected_batches_distributed_exact(
    pmf: PMFSpec,
    meta: dict,
    search_indices: List[int],
    label_schedule: torch.Tensor,
    device: torch.device,
) -> Iterable[Tuple[torch.Tensor, List[int], List[int]]]:
    """
    Yield exact original-generation batches for a distributed run.
    Returns:
      images_m11 : [B,3,H,W] exact original batch
      batch_global_indices : list of global indices for this batch
      selected_positions : positions within the batch that belong to search_indices
    """
    total_images = int(meta["num_images"])
    world_size = int(meta.get("distributed_world_size", 1))
    batch_size_per_rank = int(meta.get("batch_size_per_rank", meta.get("batch_size")))
    seed = int(meta["seed"])
    num_steps = int(meta["num_steps"])
    omega = float(meta["omega"])
    t_min = float(meta["t_min"])
    t_max = float(meta["t_max"])

    rank_bounds = [rank_shard_bounds(total_images, world_size, r) for r in range(world_size)]

    grouped: Dict[Tuple[int, int, int], List[int]] = {}
    for idx in search_indices:
        idx = int(idx)
        found = False
        for rank, (shard_start, shard_end) in enumerate(rank_bounds):
            if shard_start <= idx < shard_end:
                batch_start = shard_start + ((idx - shard_start) // batch_size_per_rank) * batch_size_per_rank
                batch_end = min(batch_start + batch_size_per_rank, shard_end)
                key = (rank, batch_start, batch_end)
                grouped.setdefault(key, []).append(idx)
                found = True
                break
        if not found:
            raise RuntimeError(f"Index {idx} out of range for distributed shards.")

    # Process batches in global order
    ordered_keys = sorted(grouped.keys(), key=lambda x: (x[1], x[0]))
    for _rank, batch_start, batch_end in ordered_keys:
        labels = label_schedule[batch_start:batch_end].to(device)
        rng = SimpleRNG(device, seed=batch_seed(seed, batch_start))
        images = pmf.model.generate(
            n_sample=int(batch_end - batch_start),
            rng=rng,
            num_steps=num_steps,
            omega=omega,
            t_min=t_min,
            t_max=t_max,
            labels=labels,
        )
        batch_indices = list(range(batch_start, batch_end))
        selected_positions = [idx - batch_start for idx in grouped[(_rank, batch_start, batch_end)]]
        yield images, batch_indices, selected_positions


def iter_selected_batches_single_prefix_exact(
    pmf: PMFSpec,
    meta: dict,
    search_indices: List[int],
    label_schedule: torch.Tensor,
    device: torch.device,
) -> Iterable[Tuple[torch.Tensor, List[int], List[int]]]:
    """
    Exact regeneration for old single-GPU sequential RNG runs.
    Only supports prefix sampling, because arbitrary random access is not exact.
    """
    n = len(search_indices)
    if search_indices != list(range(n)):
        raise ValueError(
            "Exact regeneration for old single-GPU sequential runs only supports prefix sampling. "
            "Use --sample-mode prefix."
        )

    total_images = int(meta["num_images"])
    if n > total_images:
        raise ValueError("Requested more search indices than total images.")

    seed = int(meta["seed"])
    num_steps = int(meta["num_steps"])
    omega = float(meta["omega"])
    t_min = float(meta["t_min"])
    t_max = float(meta["t_max"])
    orig_batch_size = int(meta.get("batch_size_per_rank", meta.get("batch_size", 1)))

    rng = SimpleRNG(device, seed=seed)
    for batch_start in range(0, n, orig_batch_size):
        batch_end = min(batch_start + orig_batch_size, n)
        labels = label_schedule[batch_start:batch_end].to(device)
        images = pmf.model.generate(
            n_sample=int(batch_end - batch_start),
            rng=rng,
            num_steps=num_steps,
            omega=omega,
            t_min=t_min,
            t_max=t_max,
            labels=labels,
        )
        batch_indices = list(range(batch_start, batch_end))
        selected_positions = list(range(batch_end - batch_start))
        yield images, batch_indices, selected_positions


def iter_selected_generation_batches(
    pmf: PMFSpec,
    meta: dict,
    search_indices: List[int],
    label_schedule: torch.Tensor,
    device: torch.device,
) -> Iterable[Tuple[torch.Tensor, List[int], List[int]]]:
    mode = determine_generation_mode(meta)
    if mode == "distributed_batch_seed":
        yield from iter_selected_batches_distributed_exact(pmf, meta, search_indices, label_schedule, device)
    else:
        yield from iter_selected_batches_single_prefix_exact(pmf, meta, search_indices, label_schedule, device)


def compute_tower_maps_in_minibatches(
    tower: TowerSpec,
    cap: MultiActivationCapture,
    images_m11: torch.Tensor,
    target_grid_hw: Tuple[int, int],
    resample_mode: str,
    batch_size: int,
) -> List[torch.Tensor]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be >= 1")

    chunks_by_layer: Optional[List[List[torch.Tensor]]] = None
    B = images_m11.shape[0]
    for start in range(0, B, batch_size):
        end = min(start + batch_size, B)
        disc_images = tower.preprocess(images_m11[start:end])
        _ = tower.forward(disc_images)
        acts = cap.get_and_clear()
        sub_maps = [
            _act_to_map(act, tower.patch_token_offset, target_grid_hw, resample_mode)
            for act in acts
        ]
        if chunks_by_layer is None:
            chunks_by_layer = [[] for _ in sub_maps]
        for i, m in enumerate(sub_maps):
            chunks_by_layer[i].append(m)

    if chunks_by_layer is None:
        raise RuntimeError("No tower activations were collected.")
    return [torch.cat(chunks, dim=0) for chunks in chunks_by_layer]


# -----------------------------------------------------------------------------
# Top / bottom selection
# -----------------------------------------------------------------------------


def keep_top_k(records: List[dict], rec: dict, k: int) -> None:
    if k <= 0:
        return
    records.append(rec)
    records.sort(key=lambda x: x["score"], reverse=True)
    if len(records) > k:
        del records[k:]


# -----------------------------------------------------------------------------
# HTML rendering — dark-theme single-page viewer (sidebar + main).
# Embedded CSS + JS, self-contained HTML. Mirrors the language viewer.
# -----------------------------------------------------------------------------


CSS = """
:root {
  --bg: #0e1117; --panel: #12161d; --panel2: #161b22; --border: #22272e;
  --text: #fafafa; --muted: #8b949e; --accent1: #7fd3ff; --badge: #1f252d;
}
* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; background: var(--bg); color: var(--text);
  font: 13px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
.app { display: flex; height: 100vh; }
.sidebar { width: 340px; min-width: 340px; background: var(--panel);
  border-right: 1px solid var(--border); display: flex; flex-direction: column; }
.sidebar .hdr { padding: 12px 14px; border-bottom: 1px solid var(--border); }
.sidebar h1 { font-size: 14px; margin: 0 0 4px 0; }
.sidebar .models { font-size: 11px; color: var(--muted); }
.sidebar .filters { padding: 10px 14px; border-bottom: 1px solid var(--border);
  display: flex; flex-direction: column; gap: 8px; }
.sidebar .filters input {
  background: var(--panel2); color: var(--text); border: 1px solid var(--border);
  padding: 4px 6px; font: inherit; border-radius: 4px; width: 100%; }
.sidebar .filters label { font-size: 11px; color: var(--muted); display: flex;
  flex-direction: column; gap: 2px; }
.sidebar .filters button { background: var(--panel2); color: var(--text);
  border: 1px solid var(--border); padding: 4px 8px; border-radius: 4px;
  cursor: pointer; font: inherit; }
.sidebar .filters .row2 { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
.count { padding: 6px 14px; font-size: 11px; color: var(--muted);
  border-bottom: 1px solid var(--border); }
.list { overflow-y: auto; flex: 1; }
.pager { padding: 6px 14px; display: flex; justify-content: space-between;
  align-items: center; border-top: 1px solid var(--border); font-size: 11px; color: var(--muted); }
.pager button { background: var(--panel2); color: var(--text);
  border: 1px solid var(--border); padding: 2px 8px; border-radius: 4px;
  cursor: pointer; font: inherit; }
.anchor-item { padding: 8px 14px; border-bottom: 1px solid var(--border); cursor: pointer; }
.anchor-item:hover { background: var(--panel2); }
.anchor-item.selected { background: #1c2230; border-left: 3px solid var(--accent1); padding-left: 11px; }
.anchor-item .title { font-size: 12px; font-weight: 600; }
.anchor-item .corr { float: right; color: var(--accent1); font-weight: 600; }
.anchor-item .meta { font-size: 11px; color: var(--muted); margin-top: 2px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.main { flex: 1; overflow-y: auto; padding: 16px 20px; }
.main .empty { color: var(--muted); padding: 40px; text-align: center; }
.anchor-header { padding-bottom: 10px; border-bottom: 1px solid var(--border);
  margin-bottom: 14px; }
.anchor-header h2 { font-size: 18px; margin: 0 0 4px 0; }
.anchor-header .sub { font-size: 12px; color: var(--muted); }
.badges { margin-top: 6px; display: flex; gap: 8px; flex-wrap: wrap; }
.badge { display: inline-block; background: var(--badge); color: var(--text);
  border: 1px solid var(--border); border-radius: 4px; padding: 2px 8px; font-size: 11px; }
.badge.corr { color: var(--accent1); border-color: #22577a; }
.example { margin: 14px 0; padding: 10px; background: var(--panel2);
  border: 1px solid var(--border); border-radius: 6px; }
.example .rank { font-size: 11px; color: var(--muted); margin-bottom: 8px; }
/* All tiles for one example live in a single grid row; the column count is set
   inline per example, and each tile fills 1/N of the row, so the grid reflows
   on window resize / zoom while always keeping one example on one line. */
.panel-grid { display: grid; gap: 8px; align-items: start; }
.panel { background: var(--panel); border: 1px solid var(--border);
  border-radius: 6px; padding: 6px; min-width: 0; }
.panel img { display: block; width: 100%; height: auto; border-radius: 4px; }
.panel .cap { font-size: 10px; color: var(--muted); margin-top: 4px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
"""


JS = r"""
(function() {
  const INDEX = window.__INDEX__;            // array of summary entries
  const MASTER = window.__MASTER__;
  const INLINE = window.__ANCHORS__;          // array indexed by anchor_id
  const PAGE_SIZE = 50;

  const state = { q: "", cmin: 0, lmin: 0, lmax: 9999, page: 0, selected: null };

  function loadFromHash() {
    const h = (location.hash || "").replace(/^#/, "");
    if (!h) return;
    for (const p of h.split("&")) {
      const [k, v] = p.split("=");
      if (!k) continue;
      const dv = decodeURIComponent(v || "");
      if (k === "q") state.q = dv;
      else if (k === "cmin") state.cmin = parseFloat(dv);
      else if (k === "l") { const [a,b] = dv.split("-"); state.lmin=+a||0; state.lmax=+b||9999; }
      else if (k === "page") state.page = parseInt(dv, 10) || 0;
      else if (k === "anchor") state.selected = parseInt(dv, 10);
    }
  }
  function saveToHash() {
    const p = [];
    if (state.q) p.push("q=" + encodeURIComponent(state.q));
    if (state.cmin > 0) p.push("cmin=" + state.cmin);
    if (state.lmin > 0 || state.lmax < 9999) p.push("l=" + state.lmin + "-" + state.lmax);
    if (state.page > 0) p.push("page=" + state.page);
    if (state.selected != null) p.push("anchor=" + state.selected);
    history.replaceState(null, "", "#" + p.join("&"));
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }
  function shortModel(m) { return String(m || "").split("/").pop(); }

  function filtered() {
    const q = state.q.trim().toLowerCase();
    const qL = q.match(/^l(\d+)$/);
    const qN = q.match(/^n(\d+)$/);
    const out = [];
    for (const idx of INDEX) {
      if (idx.avg_correlation < state.cmin) continue;
      if (idx.pmf_layer_idx < state.lmin || idx.pmf_layer_idx > state.lmax) continue;
      if (q) {
        let match;
        if (qL) match = (idx.pmf_layer_idx == +qL[1]);
        else if (qN) match = (idx.pmf_neuron == +qN[1]);
        else match = (idx.search_text || "").indexOf(q) !== -1;
        if (!match) continue;
      }
      out.push(idx);
    }
    out.sort((a, b) => b.avg_correlation - a.avg_correlation);
    return out;
  }

  function renderAnchorItem(idx, isSel) {
    const title = `${escapeHtml(idx.pmf_layer)}[${idx.pmf_neuron}]`;
    const matchLine = (idx.matches || []).map(m =>
      `${escapeHtml(shortModel(m.label))} ${escapeHtml(m.layer)}[${m.neuron}] (${(+m.correlation).toFixed(2)})`
    ).join(" · ");
    return `<div class="anchor-item ${isSel ? 'selected' : ''}" data-anchor="${idx.anchor_id}">
      <div><span class="corr">${(+idx.avg_correlation).toFixed(3)}</span>
        <span class="title">${title}</span></div>
      <div class="meta" title="${escapeHtml(matchLine)}">${matchLine}</div>
    </div>`;
  }

  function renderExample(ex, i) {
    const n = (ex.panels || []).length || 1;
    const cols = `grid-template-columns: repeat(${n}, minmax(0, 1fr));`;
    const tiles = (ex.panels || []).map(p =>
      `<div class="panel">
        <img src="${p.uri}" alt="${escapeHtml(p.label)}">
        <div class="cap" title="${escapeHtml(p.label + (p.caption ? ' · ' + p.caption : ''))}">${escapeHtml(p.label)}${p.caption ? ' · ' + escapeHtml(p.caption) : ''}</div>
      </div>`
    ).join("");
    return `<div class="example">
      <div class="rank">#${i+1} · image ${ex.global_index} · selected by ${escapeHtml(ex.selected_by)} ${escapeHtml(ex.selected_stat)} · score ${(+ex.score).toFixed(4)}</div>
      <div class="panel-grid" style="${cols}">${tiles}</div>
    </div>`;
  }

  function renderMain() {
    const mainEl = document.getElementById("main");
    if (state.selected == null) {
      mainEl.innerHTML = '<div class="empty">Select an anchor from the sidebar.</div>';
      return;
    }
    const a = INLINE ? INLINE[state.selected] : null;
    if (!a) { mainEl.innerHTML = '<div class="empty">Anchor not found.</div>'; return; }
    const matchBadges = (a.matches || []).map(m =>
      `<span class="badge">${escapeHtml(shortModel(m.label))} ${escapeHtml(m.layer)}[${m.neuron}] corr ${(+m.correlation).toFixed(3)}</span>`
    ).join("");
    const header = `<div class="anchor-header">
      <h2>${escapeHtml(a.pmf_layer)}[${a.pmf_neuron}]</h2>
      <div class="sub">${escapeHtml(MASTER.anchor_model || 'anchor')} · layer idx ${a.pmf_layer_idx}</div>
      <div class="badges">
        <span class="badge corr">avg corr ${(+a.avg_correlation).toFixed(3)}</span>
        <span class="badge">min corr ${(+a.min_correlation).toFixed(3)}</span>
        <span class="badge">anchor_id ${a.anchor_id}</span>
        ${matchBadges}
      </div></div>`;
    const body = (a.examples || []).length
      ? (a.examples || []).map((e, i) => renderExample(e, i)).join("")
      : '<div class="empty">No activating examples were found for this anchor.</div>';
    mainEl.innerHTML = header + body;
    mainEl.scrollTop = 0;
  }

  function renderSidebar() {
    const all = filtered();
    document.getElementById("count").textContent = `${all.length} anchors`;
    const start = state.page * PAGE_SIZE;
    const page = all.slice(start, start + PAGE_SIZE);
    const listEl = document.getElementById("list");
    listEl.innerHTML = page.map(idx => renderAnchorItem(idx, idx.anchor_id === state.selected)).join("");
    for (const el of listEl.querySelectorAll(".anchor-item")) {
      el.onclick = () => {
        state.selected = parseInt(el.dataset.anchor, 10);
        saveToHash(); renderSidebar(); renderMain();
      };
    }
    document.getElementById("pgInfo").textContent =
      `page ${state.page+1}/${Math.max(1, Math.ceil(all.length/PAGE_SIZE))}`;
  }

  function wire() {
    const q = document.getElementById("q");
    q.value = state.q;
    q.oninput = () => { state.q = q.value; state.page = 0; saveToHash(); renderSidebar(); };
    const cmin = document.getElementById("cmin");
    cmin.value = state.cmin || "";
    cmin.oninput = () => { state.cmin = +cmin.value || 0; state.page = 0; saveToHash(); renderSidebar(); };
    const l = document.getElementById("l");
    l.value = (state.lmin === 0 && state.lmax === 9999) ? "" : `${state.lmin}-${state.lmax}`;
    l.oninput = () => {
      const m = l.value.match(/^(\d+)-(\d+)$/);
      if (m) { state.lmin = +m[1]; state.lmax = +m[2]; }
      else { state.lmin = 0; state.lmax = 9999; }
      state.page = 0; saveToHash(); renderSidebar();
    };
    document.getElementById("reset").onclick = () => {
      Object.assign(state, { q:"", cmin:0, lmin:0, lmax:9999, page:0, selected:null });
      location.hash = ""; init();
    };
    document.getElementById("prev").onclick = () => {
      if (state.page > 0) { state.page--; saveToHash(); renderSidebar(); }
    };
    document.getElementById("next").onclick = () => {
      const all = filtered();
      if ((state.page + 1) * PAGE_SIZE < all.length) { state.page++; saveToHash(); renderSidebar(); }
    };
  }

  function init() {
    loadFromHash();
    wire();
    renderSidebar();
    renderMain();
  }
  document.addEventListener("DOMContentLoaded", init);
})();
"""


def _script_safe(s: str) -> str:
    """Neutralize sequences that would prematurely terminate <script> or trigger the HTML parser."""
    return (s.replace("</", "<\\/")
             .replace("<!--", "<\\!--")
             .replace("]]>", "]]\\u003e"))


def render_html(master: dict, summaries: List[dict], inline: List[dict], title: str) -> str:
    anchor_model = html.escape(str(master.get("anchor_model", "")))
    partner_models = master.get("partner_models", [])
    partners_line = " &middot; ".join(html.escape(str(p)) for p in partner_models)
    num_anchors = master.get("num_anchors", len(summaries))
    top_images = master.get("top_images", 0)
    grid = html.escape(str(master.get("canonical_grid", "")))
    sub = (f"{num_anchors} anchors &middot; top-{top_images} imgs &middot; grid {grid} &middot; "
           f"searched {master.get('num_searched', 0)}/{master.get('total_images', 0)}")

    master_js = _script_safe(json.dumps(master))
    index_js = _script_safe(json.dumps(summaries))
    inline_js = _script_safe(json.dumps(inline))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{html.escape(title)}</title>
<style>{CSS}</style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <div class="hdr">
      <h1>{html.escape(title)}</h1>
      <div class="models">{anchor_model} &larr; {partners_line}</div>
      <div class="models">{sub}</div>
    </div>
    <div class="filters">
      <label>search (L5, N1024, or model name)<input id="q" placeholder="L5 / N1024 / openclip"></label>
      <div class="row2">
        <label>min avg corr <input id="cmin" type="number" step="0.01" min="-1" max="1"></label>
        <label>layer range (a-b) <input id="l" placeholder="e.g. 0-20"></label>
      </div>
      <button id="reset">reset</button>
    </div>
    <div class="count" id="count">0 anchors</div>
    <div class="list" id="list"></div>
    <div class="pager">
      <button id="prev">&lsaquo; prev</button>
      <span id="pgInfo"></span>
      <button id="next">next &rsaquo;</button>
    </div>
  </aside>
  <main class="main" id="main"></main>
</div>
<script>window.__MASTER__ = {master_js};</script>
<script>window.__INDEX__ = {index_js};</script>
<script>window.__ANCHORS__ = {inline_js};</script>
<script>{JS}</script>
</body>
</html>
"""


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    # Input mode A: multi-model Rosetta anchors
    p.add_argument("--anchors-dir", type=str, default=None, help="Directory containing rosetta_anchors.json (multi-model anchors mode)")
    p.add_argument("--anchors-json", type=str, default=None, help="Optional explicit path to rosetta_anchors.json")
    p.add_argument("--run", action="append", default=[], help="Pairwise run mapping label=results_dir. Repeat per target model.")
    # Input mode B: single pairwise run
    p.add_argument("--results-dir", type=str, default=None, help="A single matching-run directory (containing best_buddies.json + run_metadata.json). Mutually exclusive with --anchors-dir / --run.")
    p.add_argument("--pmf-repo", type=str, required=True)
    p.add_argument("--pmf-ckpt-path", type=str, default=None)
    p.add_argument("--pmf-hf-repo", type=str, default="Lyy0725/pMF")
    p.add_argument("--pmf-ckpt-file", type=str, default="pMF-B-16.pt")

    # per-label overrides
    p.add_argument("--disc-repo", action="append", default=[], help="Override local DINO repo per label: label=/path/to/repo")
    p.add_argument("--disc-weights", action="append", default=[], help="Override local DINO checkpoint per label: label=/path/to/ckpt")
    p.add_argument("--disc-pretrained", action="append", default=[], help="Override OpenCLIP pretrained tag per label: label=openai")
    p.add_argument("--disc-input-size", action="append", default=[], help="Override input size per label: label=224")

    p.add_argument("--num-anchors", type=int, default=24)
    p.add_argument("--top-images", type=int, default=4, help="Number of top-activating images to show per anchor")
    p.add_argument("--bottom-images", type=int, default=None,
                   help="Deprecated and ignored. Min-activating images are no longer rendered.")
    p.add_argument("--max-search-images", type=int, default=None)
    p.add_argument("--sample-mode", choices=["prefix", "random"], default="prefix")
    p.add_argument("--sample-seed", type=int, default=0)

    p.add_argument("--select-model", type=str, default="pmf", help="Which model chooses the images: pmf or one of the run labels like dino, clip")
    p.add_argument("--select-stat", choices=["max", "mean"], default="max")

    p.add_argument("--act-resample-mode", choices=["nearest", "bilinear", "bicubic", "area"], default="bilinear")
    p.add_argument("--batch-size", type=int, default=8, help="Minibatch size for target-model forward passes during visualization search.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-html", type=str, default="rosetta_anchor_maps.html")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Validate input mode.
    mode_anchors = bool(args.anchors_dir or args.anchors_json)
    mode_results = bool(args.results_dir)
    if mode_anchors and mode_results:
        raise ValueError("Use either --anchors-dir/--anchors-json (multi-model anchors) OR --results-dir (single pairwise run), not both.")
    if not (mode_anchors or mode_results):
        raise ValueError("Provide --anchors-dir (with --run label=dir ...) for multi-model anchors, OR --results-dir for a single pairwise run.")

    if mode_results:
        results_dir = Path(args.results_dir)
        bb_path = results_dir / "best_buddies.json"
        meta_path = results_dir / "run_metadata.json"
        best_buddies = load_json(bb_path)
        if not isinstance(best_buddies, list) or len(best_buddies) == 0:
            raise ValueError(f"best_buddies.json is empty or invalid at {bb_path}")
        synthetic_label = derive_disc_label_from_metadata(load_json(meta_path))
        anchors = best_buddies_to_anchors(best_buddies, synthetic_label)
        summary = {}
        run_dirs = {synthetic_label: str(results_dir)}
        # Reuse results_dir as the "anchors dir" for title + output-file resolution.
        anchors_dir = results_dir
    else:
        anchors_dir = Path(args.anchors_dir)
        anchors_json = Path(args.anchors_json) if args.anchors_json else anchors_dir / "rosetta_anchors.json"
        anchors_payload = load_json(anchors_json)
        if isinstance(anchors_payload, dict):
            summary = anchors_payload.get("summary", {})
            anchors = anchors_payload.get("anchors", [])
        else:
            summary = {}
            anchors = anchors_payload
        if not isinstance(anchors, list) or len(anchors) == 0:
            raise ValueError(
                f"rosetta_anchors.json is empty or invalid. "
                f"Loaded root type={type(anchors_payload).__name__} from {anchors_json}"
            )
        run_dirs = parse_kv_list(args.run, "--run")
        if not run_dirs:
            raise ValueError("Provide at least one --run label=results_dir.")

    anchors = sort_anchors(anchors)[: args.num_anchors]

    disc_repo_over = parse_kv_list(args.disc_repo, "--disc-repo")
    disc_weights_over = parse_kv_list(args.disc_weights, "--disc-weights")
    disc_pretrained_over = parse_kv_list(args.disc_pretrained, "--disc-pretrained")
    disc_input_size_over = parse_kv_list(args.disc_input_size, "--disc-input-size")

    run_meta_by_label = {label: load_json(Path(run_dir) / "run_metadata.json") for label, run_dir in run_dirs.items()}
    gen_meta = verify_generation_consistency(run_meta_by_label)

    total_images = int(gen_meta["num_images"])
    if args.sample_mode == "random" and determine_generation_mode(gen_meta) != "distributed_batch_seed":
        raise ValueError(
            "Exact random subsampling is only supported for distributed outputs with generation_seed_mode=batch_seed(global_batch_start). "
            "Use --sample-mode prefix for older runs."
        )

    label_schedule = build_label_schedule(
        num_images=total_images,
        mode=gen_meta["label_mode"],
        fixed_label=int(gen_meta.get("fixed_label", 0)),
        seed=int(gen_meta["seed"]),
    )

    pmf_model_name = gen_meta["pmf_model"]
    pmf = load_pmf(args.pmf_repo, pmf_model_name, args.pmf_ckpt_path, args.pmf_hf_repo, args.pmf_ckpt_file, device)
    canonical_side = int(round(math.sqrt(pmf.num_patches)))
    if canonical_side * canonical_side != pmf.num_patches:
        raise ValueError(f"Expected pMF patches to form a square grid, got {pmf.num_patches}")
    canonical_grid_hw = (canonical_side, canonical_side)

    towers: Dict[str, TowerSpec] = {}
    for label, meta in run_meta_by_label.items():
        family = meta["disc_family"]
        arch = meta["disc_arch"]
        native_hw = tuple(meta["disc_native_image_size_hw"]) if meta.get("disc_native_image_size_hw") is not None else None
        if label in disc_input_size_over:
            size = int(disc_input_size_over[label])
            native_hw = (size, size)

        if family == "dinov3":
            if label in disc_repo_over:
                repo = disc_repo_over[label]
                weights = disc_weights_over.get(label)
                if weights is None:
                    raise ValueError(f"Need --disc-weights {label}=... for local DINO run")
                tower = load_dinov3_local_tower(
                    repo=repo,
                    arch=arch,
                    weights=weights,
                    device=device,
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    input_size_hw=native_hw,
                )
            else:
                tower = load_dinov3_hf_tower(
                    model_id=arch,
                    device=device,
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    input_size_hw=native_hw,
                )
        elif family == "openclip":
            pretrained = disc_pretrained_over.get(label) or meta.get("disc_notes", {}).get("pretrained")
            if pretrained is None:
                raise ValueError(f"Need --disc-pretrained {label}=... for OpenCLIP run '{label}'")
            tower = load_openclip_tower(
                model_name=arch,
                pretrained=pretrained,
                device=device,
                mean=None,
                std=None,
                input_size_hw=native_hw,
            )
        else:
            raise ValueError(f"Unsupported disc family in run '{label}': {family}")

        if tower.patch_size != pmf.patch_size:
            raise ValueError(f"Patch size mismatch for {label}: pMF={pmf.patch_size}, target={tower.patch_size}")
        towers[label] = tower

    if args.select_model != "pmf" and args.select_model not in towers:
        raise ValueError(f"--select-model must be 'pmf' or one of: {list(towers.keys())}")

    search_indices = build_search_indices(
        total_images=total_images,
        max_search_images=args.max_search_images,
        sample_mode=args.sample_mode,
        sample_seed=args.sample_seed,
    )

    pmf_cap = MultiActivationCapture(pmf.modules).register()
    tower_caps = {label: MultiActivationCapture(spec.modules).register() for label, spec in towers.items()}

    try:
        top_records: List[List[dict]] = [[] for _ in anchors]

        batch_iter = iter_selected_generation_batches(
            pmf=pmf,
            meta=gen_meta,
            search_indices=search_indices,
            label_schedule=label_schedule,
            device=device,
        )

        # Number of exact-generation batches is not easily known for random distributed mode.
        for images_m11, batch_global_indices, selected_positions in tqdm(batch_iter, desc="anchor search", dynamic_ncols=True):
            pmf_acts = pmf_cap.get_and_clear()
            pmf_maps_by_layer = [
                _act_to_map(act, pmf.patch_token_offset, canonical_grid_hw, args.act_resample_mode) for act in pmf_acts
            ]

            tower_maps_by_label: Dict[str, List[torch.Tensor]] = {}
            for label, tower in towers.items():
                tower_maps_by_label[label] = compute_tower_maps_in_minibatches(
                    tower=tower,
                    cap=tower_caps[label],
                    images_m11=images_m11,
                    target_grid_hw=canonical_grid_hw,
                    resample_mode=args.act_resample_mode,
                    batch_size=args.batch_size,
                )

            image_uint8_batch = [tensor_to_uint8_image(images_m11[b]) for b in range(images_m11.shape[0])]

            for anchor_idx, anchor in enumerate(anchors):
                pmf_layer = anchor["pmf_layer"]
                pmf_neuron = int(anchor["pmf_neuron"])
                pmf_layer_idx = pmf.layer_names.index(pmf_layer)
                pmf_batch_map = pmf_maps_by_layer[pmf_layer_idx][:, pmf_neuron]  # [B,H,W]

                matches = extract_anchor_matches(anchor)
                target_map_tensors: Dict[str, torch.Tensor] = {}
                target_stats_tensors: Dict[str, Dict[str, torch.Tensor]] = {}

                for label, match in matches.items():
                    if label not in towers:
                        continue
                    layer_name = match.get("disc_layer") or match.get("layer")
                    neuron_idx = int(match.get("disc_neuron", match.get("neuron")))
                    layer_idx = towers[label].layer_names.index(layer_name)
                    tmap = tower_maps_by_label[label][layer_idx][:, neuron_idx]
                    target_map_tensors[label] = tmap
                    target_stats_tensors[label] = {
                        "max": tmap.amax(dim=(-1, -2)),
                        "mean": tmap.mean(dim=(-1, -2)),
                    }

                if args.select_model != "pmf" and args.select_model not in target_map_tensors:
                    continue

                pmf_max = pmf_batch_map.amax(dim=(-1, -2))
                pmf_mean = pmf_batch_map.mean(dim=(-1, -2))

                if args.select_model == "pmf":
                    ranking = pmf_max if args.select_stat == "max" else pmf_mean
                else:
                    ranking = target_stats_tensors[args.select_model][args.select_stat]

                for pos in selected_positions:
                    global_idx = batch_global_indices[pos]
                    rec = {
                        "score": float(ranking[pos].item()),
                        "selected_by": args.select_model,
                        "selected_stat": args.select_stat,
                        "global_index": int(global_idx),
                        "image": image_uint8_batch[pos],
                        "pmf_map": pmf_batch_map[pos].detach().cpu().numpy(),
                        "pmf_max": float(pmf_max[pos].item()),
                        "pmf_mean": float(pmf_mean[pos].item()),
                        "targets": {},
                    }
                    for label, tmap in target_map_tensors.items():
                        arr = tmap[pos].detach().cpu().numpy()
                        rec["targets"][label] = {
                            "map": arr,
                            "max": float(np.max(arr)),
                            "mean": float(np.mean(arr)),
                        }

                    keep_top_k(top_records[anchor_idx], rec, args.top_images)

    finally:
        pmf_cap.remove()
        for cap in tower_caps.values():
            cap.remove()

    # -----------------------------------------------------------------------------
    # Build viewer data (one record per anchor) and render the single-page viewer.
    # -----------------------------------------------------------------------------

    run_labels = summary.get("run_labels", [])
    if run_labels:
        title = f"Rosetta anchor maps: {anchors_dir.name} ({', '.join(run_labels)})"
    else:
        title = f"Rosetta anchor maps: {anchors_dir.name}"

    summaries: List[dict] = []
    inline: List[dict] = []
    for anchor_idx, anchor in enumerate(anchors):
        matches = extract_anchor_matches(anchor)
        pmf_layer = anchor["pmf_layer"]
        pmf_neuron = int(anchor["pmf_neuron"])
        pmf_layer_idx = pmf.layer_names.index(pmf_layer)
        avg_corr = float(anchor.get("avg_correlation", anchor.get("min_correlation", 0.0)))
        min_corr = float(anchor.get("min_correlation", avg_corr))

        match_list: List[dict] = []
        for label, match in matches.items():
            match_list.append({
                "label": label,
                "layer": match.get("disc_layer") or match.get("layer"),
                "neuron": int(match.get("disc_neuron", match.get("neuron"))),
                "correlation": float(match.get("correlation", float("nan"))),
            })

        # Convert each kept record into an example with image/heatmap/overlay tiles.
        examples: List[dict] = []
        for rec in sorted(top_records[anchor_idx], key=lambda x: x["score"], reverse=True):
            h, w = int(rec["image"].shape[0]), int(rec["image"].shape[1])
            panels: List[dict] = [
                {"label": "image", "caption": f"#{rec['global_index']}", "uri": np_to_uri(rec["image"])}
            ]
            pmf_heat = colorize_heatmap(rec["pmf_map"])
            panels.append({"label": "pMF heatmap", "caption": "",
                           "uri": np_to_uri(pmf_heat, resize_hw=(h, w))})
            panels.append({"label": "pMF overlay", "caption": "",
                           "uri": np_to_uri(overlay_heatmap(rec["image"], pmf_heat))})
            for label, target in rec["targets"].items():
                theat = colorize_heatmap(target["map"])
                panels.append({"label": f"{label} heatmap", "caption": "",
                               "uri": np_to_uri(theat, resize_hw=(h, w))})
                panels.append({"label": f"{label} overlay", "caption": "",
                               "uri": np_to_uri(overlay_heatmap(rec["image"], theat))})
            examples.append({
                "global_index": int(rec["global_index"]),
                "score": float(rec["score"]),
                "selected_by": rec["selected_by"],
                "selected_stat": rec["selected_stat"],
                "panels": panels,
            })

        search_text = " ".join(
            [pmf_layer.lower(), f"n{pmf_neuron}", f"l{pmf_layer_idx}"]
            + [f"{m['label']} {m['layer']} n{m['neuron']}".lower() for m in match_list]
        )

        common = {
            "anchor_id": anchor_idx,
            "pmf_layer": pmf_layer,
            "pmf_neuron": pmf_neuron,
            "pmf_layer_idx": pmf_layer_idx,
            "avg_correlation": avg_corr,
            "min_correlation": min_corr,
            "matches": match_list,
        }
        summaries.append({**common, "search_text": search_text})
        inline.append({**common, "examples": examples})

    master = {
        "anchor_model": "pMF",
        "partner_models": list(towers.keys()),
        "num_anchors": len(anchors),
        "top_images": args.top_images,
        "canonical_grid": f"{canonical_grid_hw[0]}x{canonical_grid_hw[1]}",
        "num_searched": len(search_indices),
        "total_images": total_images,
    }

    html_out = render_html(master, summaries, inline, title=title)
    out_path = anchors_dir / args.output_html
    out_path.write_text(html_out, encoding="utf-8")
    print(f"Saved HTML to {out_path}")


if __name__ == "__main__":
    main()
