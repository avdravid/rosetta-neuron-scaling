
#!/usr/bin/env python3
"""
Multi-GPU match of pMF one-step generator MLP neurons to a discriminative ViT.

This is a distributed data-parallel adaptation of the single-GPU matching pipeline:
- samples are aligned spatial patch cells rather than tokenizer-aligned spans
- pMF activations come from the single denoising/generation forward pass
- discriminative activations come from the same generated images
- supported discriminative towers include DINOv3, DINOv2, MAE, PixIO, OpenCLIP, and InternViT
- the canonical patch grid can be chosen from either pMF or the discriminative tower
- the non-canonical side is projected onto that grid before computing stats/correlations

Launch with torchrun for multi-GPU, for example:
  torchrun --standalone --nproc_per_node=4 match_pmf_vit_multi_gpu.py ...

Behavior notes:
- Each rank processes a contiguous shard of the global image index range.
- Generation is deterministic for a fixed world size and shard layout because each
  global batch is seeded from (base_seed, global_batch_start_index).
- This may not be bit-identical to single-GPU generation because the batching/sharding
  order changes the RNG schedule. Stats and correlations remain internally consistent
  because both passes reuse the same shard-local batch seeds.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image

try:
    from tqdm.auto import tqdm
except Exception:
    class _TqdmNoop:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable
        def __iter__(self):
            return iter(self.iterable) if self.iterable is not None else iter(())
        def update(self, n=1): pass
        def set_postfix(self, *args, **kwargs): pass
        def close(self): pass
        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): return False
    def tqdm(iterable=None, **kwargs):
        return _TqdmNoop(iterable=iterable, **kwargs)
    tqdm.write = print


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
        dist.barrier()
        dist.destroy_process_group()


def print0(dist_env: DistEnv, *args, **kwargs) -> None:
    if dist_env.is_main:
        print(*args, **kwargs)


def barrier(dist_env: DistEnv) -> None:
    if dist_env.enabled:
        dist.barrier()


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
# RNG helper copied from the user's pMF example pattern.
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
    # Deterministic 63-bit mix; stable across passes for a fixed shard layout.
    return int((int(base_seed) * 6364136223846793005 + int(global_batch_start) * 1442695040888963407 + 1) % (2**63 - 1))


# -----------------------------------------------------------------------------
# Hook utilities
# -----------------------------------------------------------------------------


class MultiActivationCapture:
    """Capture module inputs or outputs for a list of modules."""

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
                raise RuntimeError(f"Missing activation for hook index {i}.")
            if act.ndim != 3:
                raise RuntimeError(f"Expected 3D activation [B, tokens, hidden], got {tuple(act.shape)}")
            out.append(act)
        return out


@dataclass
class TowerSpec:
    family: str
    model: torch.nn.Module
    modules: List[torch.nn.Module]
    layer_names: List[str]
    patch_token_offset: int
    patch_size: int
    expected_num_patches: Optional[int]
    preprocess: Callable[[torch.Tensor], torch.Tensor]
    forward: Callable[[torch.Tensor], object]
    notes: Dict[str, object]
    native_image_size_hw: Optional[Tuple[int, int]] = None


# -----------------------------------------------------------------------------
# pMF loader + hooks
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


_DEF_NOISE_SCALE_HINTS = {
    "pmfDiT_B_16": (7.5, 0.1, 0.8),
    "pmfDiT_B_32": (6.5, 0.1, 0.7),
    "pmfDiT_L_16": (7.0, 0.2, 0.7),
    "pmfDiT_L_32": (7.5, 0.2, 0.6),
    "pmfDiT_H_16": (7.0, 0.2, 0.6),
    "pmfDiT_H_32": (5.5, 0.1, 0.6),
}




def _import_from_repo(repo: str | Path, module_name: str):
    repo = str(Path(repo).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    return __import__(module_name, fromlist=[module_name])


def load_pmf(
    repo: str | Path,
    model_name: str,
    ckpt_path: Optional[str],
    hf_repo: Optional[str],
    ckpt_file: Optional[str],
    device: torch.device,
    verbose: bool = True,
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
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing and verbose:
        print(f"[pMF] missing keys: {len(missing)}")
    if unexpected and verbose:
        print(f"[pMF] unexpected keys: {len(unexpected)}")

    net = model.net
    modules: List[torch.nn.Module] = []
    layer_names: List[str] = []

    for i, block in enumerate(net.shared_blocks):
        if not hasattr(block.mlp, "w2"):
            raise ValueError("Expected pMF SwiGLUMlp with w2 module.")
        modules.append(block.mlp.w2)
        layer_names.append(f"pmf_shared_{i:02d}")

    for i, block in enumerate(net.u_heads):
        if not hasattr(block.mlp, "w2"):
            raise ValueError("Expected pMF SwiGLUMlp with w2 module.")
        modules.append(block.mlp.w2)
        layer_names.append(f"pmf_u_{i:02d}")

    return PMFSpec(
        model=model,
        modules=modules,
        layer_names=layer_names,
        patch_token_offset=int(net.prefix_tokens),
        patch_size=int(net.patch_size),
        num_patches=int(net.x_embedder.num_patches),
        img_size=int(model.img_size),
    )


# -----------------------------------------------------------------------------
# Discriminative towers
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


def _extract_mlp_proj_module(mlp: torch.nn.Module, layer_idx: int) -> torch.nn.Module:
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
    raise ValueError(f"Unsupported MLP/projection structure in layer {layer_idx}: {type(mlp)}")




def load_dinov3_local_tower(
    repo: str | Path,
    arch: str,
    weights: Optional[str],
    device: torch.device,
    mean: Sequence[float],
    std: Sequence[float],
    input_size_hw: Optional[Tuple[int, int]],
) -> TowerSpec:
    if weights is None:
        raise ValueError("DINOv3 local loading requires --disc-weights.")
    model = torch.hub.load(str(Path(repo).resolve()), arch, source="local", weights=weights)
    model = model.to(device).eval()

    if not hasattr(model, "blocks"):
        raise ValueError("Expected DINOv3 local model with .blocks")

    modules: List[torch.nn.Module] = []
    layer_names: List[str] = []
    for i, blk in enumerate(model.blocks):
        if not hasattr(blk, "mlp"):
            raise ValueError(f"Expected .mlp in DINOv3 block {i}")
        modules.append(_extract_mlp_proj_module(blk.mlp, i))
        layer_names.append(f"disc_block_{i:02d}")

    patch_size_obj = getattr(model, "patch_size", None)
    if isinstance(patch_size_obj, tuple):
        patch_size = int(patch_size_obj[0])
    elif isinstance(patch_size_obj, int):
        patch_size = int(patch_size_obj)
    else:
        raise ValueError("Could not determine DINOv3 patch size from local model.")

    n_storage_tokens = int(getattr(model, "n_storage_tokens", 0))
    patch_offset = 1 + n_storage_tokens
    preprocess = _make_image_preprocess(mean, std, resize_hw=input_size_hw)

    def forward(images: torch.Tensor):
        if hasattr(model, "forward_features"):
            return model.forward_features(images)
        return model(images)

    return TowerSpec(
        family="dinov3",
        model=model,
        modules=modules,
        layer_names=layer_names,
        patch_token_offset=patch_offset,
        patch_size=patch_size,
        expected_num_patches=None,
        preprocess=preprocess,
        forward=forward,
        notes={
            "arch": arch,
            "loader": "local",
            "n_storage_tokens": n_storage_tokens,
            "init_source": "pretrained",
            "weights": weights,
        },
        native_image_size_hw=input_size_hw,
    )




def load_dinov2_local_tower(
    repo: str | Path,
    arch: str,
    weights: Optional[str],
    device: torch.device,
    mean: Sequence[float],
    std: Sequence[float],
    input_size_hw: Optional[Tuple[int, int]],
) -> TowerSpec:
    hub_repo = str(Path(repo).resolve())

    # Official DINOv2 hub entrypoints are names like:
    # dinov2_vitb14, dinov2_vitl14, dinov2_vitg14, and _reg variants.
    # If weights is provided, pass it through to the hub entrypoint as the official
    # hub code accepts custom local/URL weights strings. Otherwise use the default
    # pretrained weights for that entrypoint.
    if weights:
        model = torch.hub.load(hub_repo, arch, source="local", weights=weights)
    else:
        model = torch.hub.load(hub_repo, arch, source="local")
    model = model.to(device).eval()

    if not hasattr(model, "blocks"):
        raise ValueError("Expected DINOv2 model with .blocks")

    modules: List[torch.nn.Module] = []
    layer_names: List[str] = []
    for i, blk in enumerate(model.blocks):
        if not hasattr(blk, "mlp"):
            raise ValueError(f"Expected .mlp in DINOv2 block {i}")
        modules.append(_extract_mlp_proj_module(blk.mlp, i))
        layer_names.append(f"disc_block_{i:02d}")

    patch_size_obj = getattr(model, "patch_size", None)
    if isinstance(patch_size_obj, tuple):
        patch_size = int(patch_size_obj[0])
    elif isinstance(patch_size_obj, int):
        patch_size = int(patch_size_obj)
    else:
        raise ValueError("Could not determine DINOv2 patch size from local model.")

    num_register_tokens = int(getattr(model, "num_register_tokens", 0))
    patch_offset = 1 + num_register_tokens
    expected_num_patches = None
    patch_embed = getattr(model, "patch_embed", None)
    if patch_embed is not None and hasattr(patch_embed, "num_patches"):
        try:
            expected_num_patches = int(getattr(patch_embed, "num_patches"))
        except Exception:
            expected_num_patches = None

    preprocess = _make_image_preprocess(mean, std, resize_hw=input_size_hw)

    def forward(images: torch.Tensor):
        if hasattr(model, "forward_features"):
            return model.forward_features(images)
        return model(images)

    return TowerSpec(
        family="dinov2",
        model=model,
        modules=modules,
        layer_names=layer_names,
        patch_token_offset=patch_offset,
        patch_size=patch_size,
        expected_num_patches=expected_num_patches,
        preprocess=preprocess,
        forward=forward,
        notes={
            "arch": arch,
            "loader": "local",
            "num_register_tokens": num_register_tokens,
            "register_variant": arch.endswith("_reg"),
            "init_source": "pretrained",
            "weights": weights,
        },
        native_image_size_hw=input_size_hw,
    )


def _load_torch_checkpoint_maybe_url(path_or_url: str, map_location: str = "cpu"):
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return torch.hub.load_state_dict_from_url(path_or_url, map_location=map_location, check_hash=False)
    return torch.load(path_or_url, map_location=map_location)


def _normalize_state_dict_for_loading(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("model."):
            nk = nk[len("model."):]
        out[nk] = v
    return out


def interpolate_mae_pos_embed(model: torch.nn.Module, checkpoint_model: dict) -> None:
    if "pos_embed" not in checkpoint_model:
        return

    pos_embed_checkpoint = checkpoint_model["pos_embed"]
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches

    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    new_size = int(num_patches ** 0.5)

    if orig_size == new_size:
        return

    print(f"[MAE] Position interpolate from {orig_size}x{orig_size} to {new_size}x{new_size}")

    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]

    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens,
        size=(new_size, new_size),
        mode="bicubic",
        align_corners=False,
    )
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)

    checkpoint_model["pos_embed"] = torch.cat((extra_tokens, pos_tokens), dim=1)




def load_mae_local_tower(
    repo: str | Path,
    arch: str,
    weights: Optional[str],
    device: torch.device,
    mean: Sequence[float],
    std: Sequence[float],
    input_size_hw: Optional[Tuple[int, int]],
) -> TowerSpec:
    repo_path = str(Path(repo).resolve())
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    models_vit = __import__("models_vit", fromlist=["models_vit"])
    if not hasattr(models_vit, arch):
        raise ValueError(
            f"Unsupported MAE arch '{arch}'. Expected one of: "
            f"vit_base_patch16, vit_large_patch16, vit_huge_patch14"
        )

    ctor = getattr(models_vit, arch)

    # IMPORTANT:
    # Build the MAE model at the requested image size so that
    # model.patch_embed.num_patches matches the target grid.
    if input_size_hw is None:
        native_image_size_hw = (224, 224)
    else:
        native_image_size_hw = input_size_hw

    if native_image_size_hw[0] != native_image_size_hw[1]:
        raise ValueError(
            f"MAE loader currently expects a square input size, got {native_image_size_hw}."
        )

    img_size = int(native_image_size_hw[0])

    model = ctor(
        img_size=img_size,
        num_classes=0,
        global_pool=False,
    )
    model = model.to(device).eval()

    if weights is None:
        raise ValueError("MAE local loading requires --disc-weights.")
    checkpoint = _load_torch_checkpoint_maybe_url(weights, map_location="cpu")
    checkpoint_model = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    if not isinstance(checkpoint_model, dict):
        raise ValueError("Unsupported MAE checkpoint format: expected a state dict or dict with key 'model'.")
    checkpoint_model = _normalize_state_dict_for_loading(checkpoint_model)

    state_dict = model.state_dict()
    for k in ["head.weight", "head.bias", "fc_norm.weight", "fc_norm.bias"]:
        if k in checkpoint_model and k in state_dict and checkpoint_model[k].shape != state_dict[k].shape:
            del checkpoint_model[k]

    # MAE official positional interpolation helper.
    try:
        pos_embed_mod = __import__("util.pos_embed", fromlist=["interpolate_pos_embed"])
        interpolate_pos_embed = getattr(pos_embed_mod, "interpolate_pos_embed", None)
        if interpolate_pos_embed is not None:
            interpolate_pos_embed(model, checkpoint_model)
    except Exception as exc:
        print(f"[MAE] Warning: positional interpolation helper failed: {exc}")

    missing, unexpected = model.load_state_dict(checkpoint_model, strict=False)

    if not hasattr(model, "blocks"):
        raise ValueError("Expected MAE VisionTransformer with .blocks")

    modules: List[torch.nn.Module] = []
    layer_names: List[str] = []
    for i, blk in enumerate(model.blocks):
        if not hasattr(blk, "mlp"):
            raise ValueError(f"Expected .mlp in MAE block {i}")
        modules.append(_extract_mlp_proj_module(blk.mlp, i))
        layer_names.append(f"disc_block_{i:02d}")

    patch_size_obj = getattr(model.patch_embed, "patch_size", None)
    if isinstance(patch_size_obj, tuple):
        patch_size = int(patch_size_obj[0])
    elif isinstance(patch_size_obj, int):
        patch_size = int(patch_size_obj)
    else:
        raise ValueError("Could not determine MAE patch size from model.patch_embed.patch_size")

    expected_num_patches = None
    if hasattr(model.patch_embed, "num_patches"):
        try:
            expected_num_patches = int(model.patch_embed.num_patches)
        except Exception:
            expected_num_patches = None

    preprocess = _make_image_preprocess(mean, std, resize_hw=native_image_size_hw)

    def forward(images: torch.Tensor):
        if hasattr(model, "forward_features"):
            return model.forward_features(images)
        return model(images)

    return TowerSpec(
        family="mae",
        model=model,
        modules=modules,
        layer_names=layer_names,
        patch_token_offset=1,  # CLS only
        patch_size=patch_size,
        expected_num_patches=expected_num_patches,
        preprocess=preprocess,
        forward=forward,
        notes={
            "arch": arch,
            "loader": "local",
            "weights": weights,
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
            "img_size": img_size,
            "init_source": "pretrained",
        },
        native_image_size_hw=native_image_size_hw,
    )


def _import_pixio_local_module(repo: str | Path):
    repo_path = Path(repo).resolve()

    # PixIO's source layout is unusual for importers: pixio.py imports sibling
    # modules like "layers.attention" as top-level modules. That means the
    # sys.path entry must usually be the repo's inner "pixio/" source directory,
    # not just the repo root.
    candidate_dirs: List[Path] = []
    if (repo_path / "pixio.py").exists():
        candidate_dirs.append(repo_path)
    if (repo_path / "pixio" / "pixio.py").exists():
        candidate_dirs.append(repo_path / "pixio")

    if not candidate_dirs:
        raise ImportError(
            f"Could not find PixIO source file under {repo_path}. Expected either "
            f"{repo_path / 'pixio.py'} or {repo_path / 'pixio' / 'pixio.py'}."
        )

    last_exc: Optional[Exception] = None
    for candidate in candidate_dirs:
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
        try:
            return __import__("pixio", fromlist=["pixio"])
        except Exception as exc:
            last_exc = exc

    raise ImportError(f"Failed to import PixIO module from {repo_path}") from last_exc




def load_pixio_local_tower(
    repo: str | Path,
    arch: str,
    weights: Optional[str],
    device: torch.device,
    mean: Sequence[float],
    std: Sequence[float],
    input_size_hw: Optional[Tuple[int, int]],
) -> TowerSpec:
    pixio_mod = _import_pixio_local_module(repo)

    if not hasattr(pixio_mod, arch):
        available = [name for name in dir(pixio_mod) if name.startswith("pixio_vit")]
        raise ValueError(
            f"Unsupported PixIO arch '{arch}'. Expected one of: {', '.join(sorted(available))}"
        )

    ctor = getattr(pixio_mod, arch)
    model = ctor(pretrained=None)
    model = model.to(device).eval()

    if weights is None:
        raise ValueError("PixIO local loading requires --disc-weights.")
    checkpoint = _load_torch_checkpoint_maybe_url(weights, map_location="cpu")
    checkpoint_model = checkpoint
    if isinstance(checkpoint, dict):
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            checkpoint_model = checkpoint["model"]
        elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            checkpoint_model = checkpoint["state_dict"]
    if not isinstance(checkpoint_model, dict):
        raise ValueError("Unsupported PixIO checkpoint format: expected a state dict or dict with key 'model'/'state_dict'.")

    checkpoint_model = _normalize_state_dict_for_loading(checkpoint_model)
    missing, unexpected = model.load_state_dict(checkpoint_model, strict=False)

    if not hasattr(model, "blocks"):
        raise ValueError("Expected PixIO model with .blocks")

    modules: List[torch.nn.Module] = []
    layer_names: List[str] = []
    for i, blk in enumerate(model.blocks):
        if not hasattr(blk, "mlp"):
            raise ValueError(f"Expected .mlp in PixIO block {i}")
        modules.append(_extract_mlp_proj_module(blk.mlp, i))
        layer_names.append(f"disc_block_{i:02d}")

    patch_embed = getattr(model, "patch_embed", None)
    if patch_embed is None:
        raise ValueError("Expected PixIO model with .patch_embed")

    patch_size_obj = getattr(patch_embed, "patch_size", None)
    if isinstance(patch_size_obj, tuple):
        patch_size = int(patch_size_obj[0])
    elif isinstance(patch_size_obj, int):
        patch_size = int(patch_size_obj)
    else:
        raise ValueError("Could not determine PixIO patch size from model.patch_embed.patch_size")

    n_cls_tokens = int(getattr(model, "n_cls_tokens", 0))
    patch_offset = n_cls_tokens

    native_image_size_hw = input_size_hw or _as_hw(getattr(patch_embed, "img_size", None)) or (256, 256)

    expected_num_patches: Optional[int]
    if native_image_size_hw is not None:
        expected_num_patches = int((native_image_size_hw[0] // patch_size) * (native_image_size_hw[1] // patch_size))
    else:
        expected_num_patches = None

    preprocess = _make_image_preprocess(mean, std, resize_hw=native_image_size_hw)

    def forward(images: torch.Tensor):
        # Passing an empty block list suppresses the large per-block feature list
        # in the PixIO return value while still running every transformer block
        # and triggering the hooks attached to their MLP projections.
        return model(images, block_ids=[])

    return TowerSpec(
        family="pixio",
        model=model,
        modules=modules,
        layer_names=layer_names,
        patch_token_offset=patch_offset,
        patch_size=patch_size,
        expected_num_patches=expected_num_patches,
        preprocess=preprocess,
        forward=forward,
        notes={
            "arch": arch,
            "loader": "local",
            "weights": weights,
            "n_cls_tokens": n_cls_tokens,
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
            "init_source": "pretrained",
        },
        native_image_size_hw=native_image_size_hw,
    )




def load_dinov3_hf_tower(
    model_id: str,
    device: torch.device,
    mean: Sequence[float],
    std: Sequence[float],
    input_size_hw: Optional[Tuple[int, int]],
) -> TowerSpec:
    try:
        from transformers import AutoModel
    except Exception as exc:
        raise ImportError("DINOv3 HF loading requires transformers with DINOv3 support.") from exc

    model = AutoModel.from_pretrained(model_id)
    model = model.to(device).eval()

    if hasattr(model, "model") and hasattr(model.model, "layer"):
        blocks = list(model.model.layer)
        n_storage_tokens = int(getattr(model, "n_storage_tokens", getattr(getattr(model, "config", object()), "n_storage_tokens", 0)))
    elif hasattr(model, "blocks"):
        blocks = list(model.blocks)
        n_storage_tokens = int(getattr(model, "n_storage_tokens", getattr(getattr(model, "config", object()), "n_storage_tokens", 0)))
    else:
        raise ValueError("Unsupported HF DINOv3 structure; expected .model.layer or .blocks")

    modules: List[torch.nn.Module] = []
    layer_names: List[str] = []
    for i, blk in enumerate(blocks):
        if not hasattr(blk, "mlp"):
            raise ValueError(f"Expected .mlp in HF DINOv3 block {i}")
        modules.append(_extract_mlp_proj_module(blk.mlp, i))
        layer_names.append(f"disc_block_{i:02d}")

    patch_size_obj = getattr(model, "patch_size", None)
    if patch_size_obj is None:
        patch_size_obj = getattr(getattr(model, "config", object()), "patch_size", None)
    if isinstance(patch_size_obj, tuple):
        patch_size = int(patch_size_obj[0])
    elif isinstance(patch_size_obj, int):
        patch_size = int(patch_size_obj)
    else:
        raise ValueError("Could not determine HF DINOv3 patch size.")

    patch_offset = 1 + n_storage_tokens
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
        layer_names=layer_names,
        patch_token_offset=patch_offset,
        patch_size=patch_size,
        expected_num_patches=None,
        preprocess=preprocess,
        forward=forward,
        notes={
            "arch": model_id,
            "loader": "hf",
            "n_storage_tokens": n_storage_tokens,
            "init_source": "pretrained",
        },
        native_image_size_hw=input_size_hw,
    )

def _extract_processor_resize_hw(processor) -> Optional[Tuple[int, int]]:
    if processor is None:
        return None

    for attr in ("crop_size", "size"):
        value = getattr(processor, attr, None)
        if value is None:
            continue
        if isinstance(value, dict):
            if "height" in value and "width" in value:
                return int(value["height"]), int(value["width"])
            if "shortest_edge" in value:
                s = int(value["shortest_edge"])
                return s, s
        return _as_hw(value)

    return None


def _unwrap_internvit_vision_model(model: torch.nn.Module) -> torch.nn.Module:
    candidates = [model]
    for attr in ("vision_model", "model"):
        child = getattr(model, attr, None)
        if isinstance(child, torch.nn.Module):
            candidates.append(child)
            grandchild = getattr(child, "vision_model", None)
            if isinstance(grandchild, torch.nn.Module):
                candidates.append(grandchild)

    for candidate in candidates:
        if hasattr(candidate, "encoder") and hasattr(candidate.encoder, "layers") and hasattr(candidate, "embeddings"):
            return candidate

    raise ValueError(
        "Unsupported InternViT/InternVL structure. Expected a vision model with .encoder.layers and .embeddings. "
        f"Top-level type: {type(model)}"
    )




def load_internvit_hf_tower(
    model_id: str,
    device: torch.device,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
    input_size_hw: Optional[Tuple[int, int]] = None,
) -> TowerSpec:
    try:
        from transformers import AutoImageProcessor, AutoModel
    except Exception as exc:
        raise ImportError("InternViT loading requires transformers with AutoModel/AutoImageProcessor support.") from exc

    bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()) if device.type == "cuda" else False
    internvit_forward_dtype = torch.bfloat16 if bf16_supported else torch.float16 if device.type == "cuda" else torch.float32

    from_pretrained_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if device.type == "cuda":
        from_pretrained_kwargs["torch_dtype"] = internvit_forward_dtype

    model = AutoModel.from_pretrained(
        model_id,
        **from_pretrained_kwargs,
    )
    if device.type == "cuda":
        model = model.to(device=device, dtype=internvit_forward_dtype).eval()
    else:
        model = model.to(device).eval()

    vision_model = _unwrap_internvit_vision_model(model)

    if not hasattr(vision_model, "encoder") or not hasattr(vision_model.encoder, "layers"):
        raise ValueError("Expected InternViT vision model with .encoder.layers")

    blocks = list(vision_model.encoder.layers)
    modules: List[torch.nn.Module] = []
    layer_names: List[str] = []
    for i, blk in enumerate(blocks):
        if not hasattr(blk, "mlp"):
            raise ValueError(f"Expected .mlp in InternViT block {i}")
        modules.append(_extract_mlp_proj_module(blk.mlp, i))
        layer_names.append(f"disc_block_{i:02d}")

    config = getattr(vision_model, "config", getattr(model, "config", None))
    if config is None:
        raise ValueError("Could not determine InternViT config.")

    patch_size_obj = getattr(config, "patch_size", None)
    if patch_size_obj is None:
        patch_embed = getattr(getattr(vision_model, "embeddings", None), "patch_embedding", None)
        if patch_embed is not None:
            kernel = getattr(patch_embed, "kernel_size", None)
            if isinstance(kernel, tuple):
                patch_size_obj = kernel[0]
            else:
                patch_size_obj = kernel

    if isinstance(patch_size_obj, tuple):
        patch_size = int(patch_size_obj[0])
    elif isinstance(patch_size_obj, int):
        patch_size = int(patch_size_obj)
    else:
        raise ValueError("Could not determine InternViT patch size.")

    try:
        processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
    except Exception:
        processor = None

    processor_resize_hw = _extract_processor_resize_hw(processor)
    processor_mean = list(getattr(processor, "image_mean", [])) if processor is not None else []
    processor_std = list(getattr(processor, "image_std", [])) if processor is not None else []

    if mean is None:
        mean = processor_mean or [0.485, 0.456, 0.406]
    if std is None:
        std = processor_std or [0.229, 0.224, 0.225]

    config_image_size_hw = _as_hw(getattr(config, "image_size", None))
    native_image_size_hw = input_size_hw or processor_resize_hw or config_image_size_hw

    if native_image_size_hw is None:
        raise ValueError("Could not determine InternViT input image size.")
    if native_image_size_hw[0] != native_image_size_hw[1]:
        raise ValueError(f"InternViT loader currently expects a square input size, got {native_image_size_hw}.")

    if config_image_size_hw is not None and tuple(native_image_size_hw) != tuple(config_image_size_hw):
        resize_owner = vision_model if hasattr(vision_model, "resize_pos_embeddings") else model if hasattr(model, "resize_pos_embeddings") else None
        if resize_owner is None:
            raise ValueError(
                f"InternViT input-size override requires resize_pos_embeddings support; requested {native_image_size_hw}, "
                f"config image size is {config_image_size_hw}."
            )
        resize_owner.resize_pos_embeddings(
            old_size=int(config_image_size_hw[0]),
            new_size=int(native_image_size_hw[0]),
            patch_size=patch_size,
        )

    expected_num_patches = int((native_image_size_hw[0] // patch_size) * (native_image_size_hw[1] // patch_size))
    preprocess = _make_image_preprocess(mean, std, resize_hw=native_image_size_hw)

    def forward(images: torch.Tensor):
        if images.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=internvit_forward_dtype):
                return vision_model(pixel_values=images, output_hidden_states=False, return_dict=True)
        return vision_model(pixel_values=images, output_hidden_states=False, return_dict=True)

    return TowerSpec(
        family="internvit",
        model=vision_model,
        modules=modules,
        layer_names=layer_names,
        patch_token_offset=1,  # CLS only
        patch_size=patch_size,
        expected_num_patches=expected_num_patches,
        preprocess=preprocess,
        forward=forward,
        notes={
            "arch": model_id,
            "loader": "hf_remote_code",
            "root_model_type": type(vision_model).__name__,
            "forward_autocast_dtype": str(internvit_forward_dtype).replace("torch.", ""),
            "init_source": "pretrained",
        },
        native_image_size_hw=native_image_size_hw,
    )




def load_openclip_tower(
    model_name: str,
    pretrained: Optional[str],
    device: torch.device,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
    input_size_hw: Optional[Tuple[int, int]] = None,
) -> TowerSpec:
    import open_clip

    create_kwargs = {
        "device": device,
    }
    if pretrained is not None:
        create_kwargs["pretrained"] = pretrained

    model, _, preprocess_tf = open_clip.create_model_and_transforms(
        model_name,
        **create_kwargs,
    )
    model = model.eval()
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

    # ------------------------------------------------------------------
    # Case 1: classic OpenCLIP ViT
    # visual.transformer.resblocks
    # ------------------------------------------------------------------
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
                getattr(visual, "image_size", None) or
                getattr(visual, "input_resolution", None)
            )

    # ------------------------------------------------------------------
    # Case 2: timm-backed OpenCLIP towers (EVA / EVA02 / some SigLIP etc.)
    # visual is a TimmModel wrapper, backbone lives in visual.trunk
    # ------------------------------------------------------------------
    elif hasattr(visual, "trunk") and hasattr(visual.trunk, "blocks"):
        visual_kind = "openclip_timm"
        trunk = visual.trunk

        # Respect explicit input-size override when the wrapper/backbone supports it.
        if input_size_hw is not None:
            try:
                if hasattr(visual, "set_input_size"):
                    visual.set_input_size(input_size_hw)
                elif hasattr(trunk, "set_input_size"):
                    trunk.set_input_size(input_size_hw)
            except Exception:
                # Safe fallback: just keep using preprocessing resize below.
                pass

        blocks = list(trunk.blocks)

        patch_offset = int(
            getattr(trunk, "num_prefix_tokens", getattr(visual, "num_prefix_tokens", 1))
        )

        patch_embed = getattr(trunk, "patch_embed", None)
        if patch_embed is None:
            raise ValueError(
                f"OpenCLIP timm visual tower {type(trunk)} has no patch_embed; "
                "cannot infer patch size / patch count."
            )

        patch_size = _infer_patch_size_from_patch_embed(patch_embed)
        expected_num_patches = _infer_num_patches_from_patch_embed(patch_embed)

        if native_image_size_hw is None:
            native_image_size_hw = _as_hw(
                getattr(visual, "image_size", None) or
                getattr(trunk, "img_size", None) or
                getattr(trunk, "image_size", None) or
                getattr(patch_embed, "img_size", None)
            )

    # ------------------------------------------------------------------
    # Case 3: generic timm-style ViT directly exposed as visual.blocks
    # ------------------------------------------------------------------
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
                getattr(visual, "img_size", None) or
                getattr(visual, "image_size", None)
            )

    else:
        raise ValueError(
            "Unsupported OpenCLIP visual tower structure. "
            f"visual type: {type(visual)}. "
            "Expected one of: "
            "visual.transformer.resblocks, visual.trunk.blocks, or visual.blocks."
        )

    if patch_size is None:
        raise ValueError(
            f"Could not determine patch size for OpenCLIP visual tower type {type(visual)} "
            f"(kind={visual_kind})."
        )

    modules: List[torch.nn.Module] = []
    layer_names: List[str] = []
    for i, blk in enumerate(blocks):
        if not hasattr(blk, "mlp"):
            raise ValueError(
                f"Expected .mlp in OpenCLIP block {i}, got {type(blk)} "
                f"(kind={visual_kind})."
            )
        modules.append(_extract_mlp_proj_module(blk.mlp, i))
        layer_names.append(f"disc_block_{i:02d}")

    # Pull normalization stats from the OpenCLIP transform if caller did not override.
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
        # Calling the visual tower directly is enough to execute all transformer blocks
        # and trigger the hooks attached above.
        return visual(images)

    return TowerSpec(
        family="openclip",
        model=model,
        modules=modules,
        layer_names=layer_names,
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
            "init_source": "pretrained",
        },
        native_image_size_hw=native_image_size_hw,
    )

# -----------------------------------------------------------------------------
# Activation slicing / resampling / stats / top-k utilities
# -----------------------------------------------------------------------------


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

    `sorted_samples[d]` holds an ascending sample of layer-d activations (collected during
    stage 1 either via reservoir sampling or by storing every sample). `normalize` returns
    the midrank-transformed input, standardized using the empirical mean/invstd of the
    rank-transformed reservoir. Midrank (average of searchsorted_left and searchsorted_right)
    correctly handles ties; empirical standardization keeps the resulting correlation in
    [-1, 1] even when the activation distribution has lots of ties (e.g. ReLU sparsity).
    """
    sorted_samples: torch.Tensor  # [D, K], float32, ascending along last dim
    rank_mean: torch.Tensor       # [D], mean of midrank-transformed reservoir
    rank_invstd: torch.Tensor     # [D], 1/std of midrank-transformed reservoir
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
    """Given sorted reservoir samples [D, K], return (rank_mean, rank_invstd) of the
    midrank-transformed reservoir. For (nearly) constant neurons we set invstd=0, which
    zeros out their normalized output so dead/saturating neurons can't dominate the
    correlation matmul via numerical noise."""
    K = sorted_samples.shape[1]
    idx_l = torch.searchsorted(sorted_samples, sorted_samples, right=False).to(torch.float32)
    idx_r = torch.searchsorted(sorted_samples, sorted_samples, right=True).to(torch.float32)
    r_self = 0.5 * (idx_l + idx_r) / float(K)
    mean = r_self.mean(dim=-1)
    var = ((r_self - mean.unsqueeze(-1)) ** 2).mean(dim=-1)
    alive = var > var_thresh
    invstd = torch.where(alive, var.clamp(min=var_thresh).rsqrt(), torch.zeros_like(var))
    return mean, invstd


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


Normalizer = Union[LayerStats, QuantileLayerStats]


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


def _infer_square_grid(num_tokens: int) -> Tuple[int, int]:
    side = int(round(math.sqrt(num_tokens)))
    if side * side != num_tokens:
        raise RuntimeError(f"Expected a square token grid, got {num_tokens} patch tokens.")
    return side, side


def _tokens_to_spatial_map(act: torch.Tensor, patch_token_offset: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    if act.ndim != 3:
        raise RuntimeError(f"Expected [B, tokens, hidden], got {tuple(act.shape)}")
    if patch_token_offset >= act.shape[1]:
        raise RuntimeError(f"Patch token offset {patch_token_offset} >= token count {act.shape[1]}; wrong hook point or prefix size.")
    x = act[:, patch_token_offset:, :]
    grid_hw = _infer_square_grid(x.shape[1])
    x = x.reshape(x.shape[0], grid_hw[0], grid_hw[1], x.shape[-1])
    return x, grid_hw


def _flatten_patch_tokens_on_grid(
    act: torch.Tensor,
    patch_token_offset: int,
    target_grid_hw: Optional[Tuple[int, int]],
    resample_mode: str,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    x, native_grid_hw = _tokens_to_spatial_map(act, patch_token_offset)
    effective_grid_hw = native_grid_hw if target_grid_hw is None else target_grid_hw
    if native_grid_hw != effective_grid_hw:
        x = x.permute(0, 3, 1, 2)
        if resample_mode in {"bilinear", "bicubic"}:
            x = F.interpolate(x, size=effective_grid_hw, mode=resample_mode, align_corners=False)
        else:
            x = F.interpolate(x, size=effective_grid_hw, mode=resample_mode)
        x = x.permute(0, 2, 3, 1)
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


def _sync_grid_hw(grid_hw: Tuple[int, int], device: torch.device, dist_env: DistEnv, name: str) -> Tuple[int, int]:
    if not dist_env.enabled:
        return int(grid_hw[0]), int(grid_hw[1])

    t = torch.tensor([int(grid_hw[0]), int(grid_hw[1])], dtype=torch.int64, device=device)
    t_min = t.clone()
    t_max = t.clone()
    dist.all_reduce(t_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(t_max, op=dist.ReduceOp.MAX)
    if not torch.equal(t_min, t_max):
        raise RuntimeError(
            f"Observed inconsistent {name} across ranks: min={tuple(int(x) for x in t_min.tolist())}, "
            f"max={tuple(int(x) for x in t_max.tolist())}"
        )
    return int(t_min[0].item()), int(t_min[1].item())


# -----------------------------------------------------------------------------
# Label schedule and image iteration
# -----------------------------------------------------------------------------


def build_label_schedule(num_images: int, mode: str, fixed_label: int, seed: int) -> torch.Tensor:
    if mode == "fixed":
        labels = torch.full((num_images,), fixed_label, dtype=torch.int32)
    elif mode == "balanced":
        reps = math.ceil(num_images / 1000)
        labels = torch.arange(1000, dtype=torch.int32).repeat(reps)[:num_images]
    elif mode == "random":
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        labels = torch.randint(0, 1000, (num_images,), generator=g, dtype=torch.int32)
    else:
        raise ValueError(f"Unknown label mode: {mode}")
    return labels


@torch.inference_mode()
def _save_image_batch(images_m11: torch.Tensor, labels: torch.Tensor, out_dir: Path, start_index: int, image_format: str = "png") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = images_m11.detach().clamp(-1.0, 1.0)
    imgs = ((imgs + 1.0) / 2.0 * 255.0).round().to(torch.uint8)
    imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
    labels_cpu = labels.detach().cpu().tolist()
    fmt = image_format.lower().lstrip(".")
    for offset, (img, label) in enumerate(zip(imgs, labels_cpu)):
        idx = start_index + offset
        filename = out_dir / f"img_{idx:06d}_class_{int(label):04d}.{fmt}"
        Image.fromarray(img).save(filename)


@torch.inference_mode()
def iter_generated_batches(
    pmf: PMFSpec,
    num_images: int,
    batch_size: int,
    label_schedule: torch.Tensor,
    seed: int,
    num_steps: int,
    omega: float,
    t_min: float,
    t_max: float,
    device: torch.device,
    shard_start: int,
    shard_end: int,
    save_images_dir: Optional[Path] = None,
    save_image_format: str = "png",
) -> Iterable[Tuple[torch.Tensor, torch.Tensor, int]]:
    for start in range(shard_start, shard_end, batch_size):
        end = min(start + batch_size, shard_end)
        labels = label_schedule[start:end].to(device)
        rng = SimpleRNG(device, seed=batch_seed(seed, start))
        images = pmf.model.generate(
            n_sample=int(end - start),
            rng=rng,
            num_steps=num_steps,
            omega=omega,
            t_min=t_min,
            t_max=t_max,
            labels=labels,
        )
        if save_images_dir is not None:
            _save_image_batch(images_m11=images, labels=labels, out_dir=save_images_dir, start_index=start, image_format=save_image_format)
        yield images, labels, start


# -----------------------------------------------------------------------------
# Passes
# -----------------------------------------------------------------------------


@torch.inference_mode()
def compute_layer_stats(
    pmf: PMFSpec,
    disc: TowerSpec,
    num_images: int,
    batch_size: int,
    label_schedule: torch.Tensor,
    seed: int,
    num_steps: int,
    omega: float,
    t_min: float,
    t_max: float,
    device: torch.device,
    canonical_grid_source: str,
    act_resample_mode: str,
    dist_env: DistEnv,
    similarity: str = "pearson",
    spearman_mode: str = "approx",
    spearman_reservoir_size: int = 4096,
    save_images_dir: Optional[Path] = None,
    save_image_format: str = "png",
) -> Tuple[List[Normalizer], List[Normalizer], int, Tuple[int, int], Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    shard_start, shard_end = rank_shard_bounds(num_images, dist_env.world_size, dist_env.rank)
    local_num_images = shard_end - shard_start
    if local_num_images <= 0:
        raise ValueError(f"Rank {dist_env.rank} received no images. Ensure num_images >= world_size.")

    if canonical_grid_source not in {"pmf", "disc"}:
        raise ValueError(f"Unsupported canonical_grid_source: {canonical_grid_source}")

    cap_pmf = MultiActivationCapture(pmf.modules).register()
    cap_disc = MultiActivationCapture(disc.modules).register()
    num_batches = _num_batches(local_num_images, batch_size)

    pmf_collector = _make_side_collector(
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
        pmf_native_grid_hw: Optional[Tuple[int, int]] = None
        disc_native_grid_hw: Optional[Tuple[int, int]] = None

        with tqdm(total=num_batches, desc=f"stats batches [rank{dist_env.rank}]", **tqdm_kwargs(dist_env)) as pbar:
            for batch_idx, (images_m11, _labels, _global_start) in enumerate(
                iter_generated_batches(
                    pmf=pmf,
                    num_images=num_images,
                    batch_size=batch_size,
                    label_schedule=label_schedule,
                    seed=seed,
                    num_steps=num_steps,
                    omega=omega,
                    t_min=t_min,
                    t_max=t_max,
                    device=device,
                    shard_start=shard_start,
                    shard_end=shard_end,
                    save_images_dir=save_images_dir,
                    save_image_format=save_image_format,
                ),
                start=1,
            ):
                pmf_acts = cap_pmf.get_and_clear()
                disc_images = disc.preprocess(images_m11)
                _ = disc.forward(disc_images)
                disc_acts = cap_disc.get_and_clear()

                batch_pmf_native_grid_hw = _tokens_to_spatial_map(pmf_acts[0], pmf.patch_token_offset)[1]
                batch_disc_native_grid_hw = _tokens_to_spatial_map(disc_acts[0], disc.patch_token_offset)[1]

                if pmf_native_grid_hw is None:
                    pmf_native_grid_hw = batch_pmf_native_grid_hw
                elif pmf_native_grid_hw != batch_pmf_native_grid_hw:
                    raise RuntimeError(
                        f"Observed inconsistent pMF native activation grids across batches: "
                        f"{pmf_native_grid_hw} vs {batch_pmf_native_grid_hw}"
                    )

                if disc_native_grid_hw is None:
                    disc_native_grid_hw = batch_disc_native_grid_hw
                elif disc_native_grid_hw != batch_disc_native_grid_hw:
                    raise RuntimeError(
                        f"Observed inconsistent discriminative native activation grids across batches: "
                        f"{disc_native_grid_hw} vs {batch_disc_native_grid_hw}"
                    )

                if canonical_grid_hw is None:
                    if canonical_grid_source == "pmf":
                        canonical_grid_hw = pmf_native_grid_hw
                    else:
                        canonical_grid_hw = disc_native_grid_hw
                    assert canonical_grid_hw is not None
                    canonical_grid_hw = _sync_grid_hw(
                        canonical_grid_hw,
                        device=device,
                        dist_env=dist_env,
                        name=f"canonical grid ({canonical_grid_source})",
                    )

                flat_pmf: List[torch.Tensor] = []
                for act in pmf_acts:
                    x, native_grid_hw = _flatten_patch_tokens_on_grid(
                        act,
                        patch_token_offset=pmf.patch_token_offset,
                        target_grid_hw=canonical_grid_hw,
                        resample_mode=act_resample_mode,
                    )
                    if native_grid_hw != pmf_native_grid_hw:
                        raise RuntimeError(
                            f"Observed inconsistent pMF native activation grids across layers: "
                            f"expected {pmf_native_grid_hw}, got {native_grid_hw}"
                        )
                    flat_pmf.append(x.to(torch.float32))

                flat_disc: List[torch.Tensor] = []
                for act in disc_acts:
                    y, native_grid_hw = _flatten_patch_tokens_on_grid(
                        act,
                        patch_token_offset=disc.patch_token_offset,
                        target_grid_hw=canonical_grid_hw,
                        resample_mode=act_resample_mode,
                    )
                    if native_grid_hw != disc_native_grid_hw:
                        raise RuntimeError(
                            f"Observed inconsistent discriminative native activation grids across layers: "
                            f"expected {disc_native_grid_hw}, got {native_grid_hw}"
                        )
                    flat_disc.append(y.to(torch.float32))

                sample_count = flat_pmf[0].shape[0]
                if any(x.shape[0] != sample_count for x in flat_pmf + flat_disc):
                    raise RuntimeError("Inconsistent canonical patch sample count across layers.")
                local_samples += int(sample_count)

                pmf_collector.update_layers(flat_pmf)
                disc_collector.update_layers(flat_disc)

                processed_images = min(batch_idx * batch_size, local_num_images)
                pbar.update(1)
                pbar.set_postfix(local_images=f"{processed_images}/{local_num_images}", local_samples=f"{local_samples:,}")

        assert canonical_grid_hw is not None

        pmf_normalizers, total_samples = pmf_collector.finalize(dist_env)
        disc_normalizers, disc_total_samples = disc_collector.finalize(dist_env)
        if total_samples != disc_total_samples:
            raise RuntimeError(
                f"Inconsistent sample counts between pMF ({total_samples}) and disc ({disc_total_samples}) collectors."
            )
        return pmf_normalizers, disc_normalizers, total_samples, canonical_grid_hw, pmf_native_grid_hw, disc_native_grid_hw
    finally:
        cap_pmf.remove()
        cap_disc.remove()


@torch.inference_mode()
def accumulate_corr_for_disc_chunk(
    pmf: PMFSpec,
    disc: TowerSpec,
    disc_modules_chunk: List[torch.nn.Module],
    disc_chunk_indices: List[int],
    pmf_normalizers: List[Normalizer],
    disc_normalizers: List[Normalizer],
    num_images: int,
    batch_size: int,
    label_schedule: torch.Tensor,
    seed: int,
    num_steps: int,
    omega: float,
    t_min: float,
    t_max: float,
    device: torch.device,
    compute_dtype: torch.dtype,
    canonical_grid_hw: Tuple[int, int],
    act_resample_mode: str,
    dist_env: DistEnv,
) -> Optional[List[List[torch.Tensor]]]:
    shard_start, shard_end = rank_shard_bounds(num_images, dist_env.world_size, dist_env.rank)
    local_num_images = shard_end - shard_start
    if local_num_images <= 0:
        raise ValueError(f"Rank {dist_env.rank} received no images. Ensure num_images >= world_size.")

    cap_pmf = MultiActivationCapture(pmf.modules).register()
    cap_disc = MultiActivationCapture(disc_modules_chunk).register()
    num_batches = _num_batches(local_num_images, batch_size)

    chunk_start = disc_chunk_indices[0]
    chunk_end = disc_chunk_indices[-1] + 1
    chunk_desc = f"corr batches disc[{chunk_start}:{chunk_end}]"

    try:
        accumulators: Optional[List[List[torch.Tensor]]] = None

        with tqdm(total=num_batches, desc=f"{chunk_desc} [rank{dist_env.rank}]", leave=False, **tqdm_kwargs(dist_env)) as pbar:
            for batch_idx, (images_m11, _labels, _global_start) in enumerate(
                iter_generated_batches(
                    pmf=pmf,
                    num_images=num_images,
                    batch_size=batch_size,
                    label_schedule=label_schedule,
                    seed=seed,
                    num_steps=num_steps,
                    omega=omega,
                    t_min=t_min,
                    t_max=t_max,
                    device=device,
                    shard_start=shard_start,
                    shard_end=shard_end,
                ),
                start=1,
            ):
                pmf_acts = cap_pmf.get_and_clear()
                disc_images = disc.preprocess(images_m11)
                _ = disc.forward(disc_images)
                disc_acts = cap_disc.get_and_clear()

                flat_pmf: List[torch.Tensor] = []
                for i, act in enumerate(pmf_acts):
                    x, _ = _flatten_patch_tokens_on_grid(act, patch_token_offset=pmf.patch_token_offset, target_grid_hw=canonical_grid_hw, resample_mode=act_resample_mode)
                    x = pmf_normalizers[i].normalize(x, device=device, dtype=compute_dtype)
                    flat_pmf.append(x)

                flat_disc: List[torch.Tensor] = []
                for local_j, act in enumerate(disc_acts):
                    j = disc_chunk_indices[local_j]
                    y, _ = _flatten_patch_tokens_on_grid(act, patch_token_offset=disc.patch_token_offset, target_grid_hw=canonical_grid_hw, resample_mode=act_resample_mode)
                    y = disc_normalizers[j].normalize(y, device=device, dtype=compute_dtype)
                    flat_disc.append(y)

                if accumulators is None:
                    accumulators = []
                    for x in flat_pmf:
                        row: List[torch.Tensor] = []
                        for y in flat_disc:
                            row.append(torch.zeros((x.shape[1], y.shape[1]), dtype=torch.float32, device=device))
                        accumulators.append(row)

                assert accumulators is not None
                for i, x in enumerate(flat_pmf):
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
        cap_pmf.remove()
        cap_disc.remove()


# -----------------------------------------------------------------------------
# Best-buddy extraction
# -----------------------------------------------------------------------------


def build_mutual_topk_pairs(
    pmf_layer_names: List[str],
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
                        "pmf_layer_idx": la,
                        "pmf_layer": pmf_layer_names[la],
                        "pmf_neuron": na,
                        "disc_layer_idx": lb,
                        "disc_layer": disc_layer_names[lb],
                        "disc_neuron": nb,
                        "correlation": corr,
                        "rank_in_pmf": rank_a + 1,
                        "rank_in_disc": rank_b + 1,
                    }
                )

    pairs.sort(key=lambda x: float(x["correlation"]), reverse=True)
    return pairs


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # pMF
    parser.add_argument("--pmf-repo", type=str, required=True, help="Path to local pMF repo.")
    parser.add_argument("--pmf-model", type=str, default="pmfDiT_B_16")
    parser.add_argument("--pmf-ckpt-path", type=str, default=None)
    parser.add_argument("--pmf-hf-repo", type=str, default="Lyy0725/pMF")
    parser.add_argument("--pmf-ckpt-file", type=str, default="pMF-B-16.pt")
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--omega", type=float, default=None)
    parser.add_argument("--t-min", type=float, default=None)
    parser.add_argument("--t-max", type=float, default=None)

    # discriminative tower
    parser.add_argument("--disc-family", type=str, choices=["dinov3", "dinov2", "mae", "pixio", "openclip", "internvit"], default="openclip")
    parser.add_argument("--disc-arch", type=str, default="ViT-B-16",
                        help=("For OpenCLIP: model name like ViT-B-16. For DINOv3: either the HF model id "
                              "or the local torch.hub architecture name when --disc-repo is provided. "
                              "For PixIO: constructor name like pixio_vitb16, pixio_vitl16, or pixio_vith16. "
                              "For InternViT: Hugging Face model id or local HF model directory, e.g. "
                              "OpenGVLab/InternViT-6B-448px-V2_5."))
    parser.add_argument("--disc-repo", type=str, default=None,
                        help="Optional local repo path for DINOv3, DINOv2, MAE, or PixIO. For dinov2, mae, and pixio this is required. "
                             "InternViT does not use --disc-repo; pass the HF model id or local HF export via --disc-arch instead.")
    parser.add_argument("--disc-weights", type=str, default=None,
                        help="For local DINOv3/DINOv2: checkpoint / URL / local path passed to the torch.hub entrypoint as weights=.... "
                             "For MAE and PixIO: checkpoint path or URL.")
    parser.add_argument("--disc-pretrained", type=str, default=None,
                        help="For OpenCLIP: pretrained tag like openai or laion2b_s34b_b88k.")
    parser.add_argument("--disc-input-size", type=int, default=None,
                        help="Optional square input size override for the discriminative model. Useful for forcing OpenCLIP to 224, etc.")
    parser.add_argument("--disc-mean", type=float, nargs=3, default=None)
    parser.add_argument("--disc-std", type=float, nargs=3, default=None)

    # matching
    parser.add_argument("--num-images", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8, help="Per-rank batch size in distributed mode.")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument(
        "--report-topk",
        type=str,
        default=None,
        help="Comma-separated list of k values (e.g. '1,2,5,10,20') for which to report mutual best-buddy "
             "counts and write filtered best_buddies_top{k}.json files. Each value must be <= --topk. "
             "The canonical best_buddies.json always corresponds to --topk.",
    )
    parser.add_argument("--disc-chunk-size", type=int, default=1)
    parser.add_argument("--label-mode", choices=["fixed", "balanced", "random"], default="balanced")
    parser.add_argument("--fixed-label", type=int, default=0)
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compute-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--canonical-grid-source", choices=["pmf", "disc"], default="pmf",
                        help="Choose whether the canonical patch grid is taken from pMF or the discriminative tower.")
    parser.add_argument("--act-resample-mode", choices=["nearest", "bilinear", "bicubic", "area"], default="bilinear",
                        help="How to map non-canonical patch maps onto the canonical grid.")
    parser.add_argument(
        "--similarity",
        choices=["pearson", "spearman"],
        default="pearson",
        help=(
            "Similarity metric for matching neurons. "
            "'pearson' (default) is the existing z-scored Pearson correlation. "
            "'spearman' applies per-neuron rank transformation via an empirical CDF before the matmul."
        ),
    )
    parser.add_argument(
        "--spearman-mode",
        choices=["approx", "exact"],
        default="approx",
        help=(
            "Only used when --similarity spearman. "
            "'approx' (default) builds a per-neuron empirical CDF via reservoir sampling of "
            "--spearman-reservoir-size activations per neuron. "
            "'exact' stores every activation on CPU and ranks the full population at finalize time; "
            "memory grows linearly with num_images * patches_per_image * hidden_dim, so this is only "
            "practical at small scale."
        ),
    )
    parser.add_argument(
        "--spearman-reservoir-size",
        type=int,
        default=4096,
        help=(
            "Per-neuron reservoir size used in --spearman-mode approx. Larger = closer to exact Spearman "
            "at higher memory cost. After all-gather across ranks the empirical CDF has "
            "reservoir_size * world_size samples per neuron."
        ),
    )

    # outputs
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--save-full-corr", action="store_true", help="Also save per-layer-pair full correlation matrices.")
    parser.add_argument("--save-generated-images", action="store_true")
    parser.add_argument("--generated-images-subdir", type=str, default="generated_images")
    parser.add_argument("--generated-image-format", choices=["png", "jpg", "jpeg", "webp"], default="png")

    return parser.parse_args()


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

        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        compute_dtype = dtype_map[args.compute_dtype]

        if args.num_images < dist_env.world_size:
            raise ValueError(f"num_images ({args.num_images}) must be >= world_size ({dist_env.world_size}).")

        if args.omega is None or args.t_min is None or args.t_max is None:
            d_omega, d_tmin, d_tmax = _DEF_NOISE_SCALE_HINTS.get(args.pmf_model, (7.5, 0.1, 0.8))
            omega = d_omega if args.omega is None else args.omega
            t_min = d_tmin if args.t_min is None else args.t_min
            t_max = d_tmax if args.t_max is None else args.t_max
        else:
            omega = args.omega
            t_min = args.t_min
            t_max = args.t_max

        if dist_env.is_main:
            os.makedirs(args.save_dir, exist_ok=True)
        barrier(dist_env)
        save_images_dir = Path(args.save_dir) / args.generated_images_subdir if args.save_generated_images else None

        pmf = load_pmf(
            repo=args.pmf_repo,
            model_name=args.pmf_model,
            ckpt_path=args.pmf_ckpt_path,
            hf_repo=args.pmf_hf_repo,
            ckpt_file=args.pmf_ckpt_file,
            device=device,
            verbose=dist_env.is_main,
        )

        disc_input_hw = _as_hw(args.disc_input_size)
        if args.disc_family == "dinov3":
            if args.disc_repo:
                if args.disc_weights is None:
                    raise ValueError("For local DINOv3 loading, provide --disc-weights together with --disc-repo.")
                disc = load_dinov3_local_tower(
                    repo=args.disc_repo,
                    arch=args.disc_arch,
                    weights=args.disc_weights,
                    device=device,
                    mean=args.disc_mean or [0.485, 0.456, 0.406],
                    std=args.disc_std or [0.229, 0.224, 0.225],
                    input_size_hw=disc_input_hw,
                )
            else:
                disc = load_dinov3_hf_tower(
                    model_id=args.disc_arch,
                    device=device,
                    mean=args.disc_mean or [0.485, 0.456, 0.406],
                    std=args.disc_std or [0.229, 0.224, 0.225],
                    input_size_hw=disc_input_hw,
                )
        elif args.disc_family == "dinov2":
            if args.disc_repo is None:
                raise ValueError("For --disc-family dinov2, provide --disc-repo pointing to a local cloned DINOv2 repo.")
            disc = load_dinov2_local_tower(
                repo=args.disc_repo,
                arch=args.disc_arch,
                weights=args.disc_weights,
                device=device,
                mean=args.disc_mean or [0.485, 0.456, 0.406],
                std=args.disc_std or [0.229, 0.224, 0.225],
                input_size_hw=disc_input_hw,
            )
        elif args.disc_family == "mae":
            if args.disc_repo is None:
                raise ValueError("For --disc-family mae, provide --disc-repo pointing to a local cloned facebookresearch/mae repo.")
            if args.disc_weights is None:
                raise ValueError("For --disc-family mae, provide --disc-weights pointing to an MAE checkpoint (local path or URL).")
            disc = load_mae_local_tower(
                repo=args.disc_repo,
                arch=args.disc_arch,
                weights=args.disc_weights,
                device=device,
                mean=args.disc_mean or [0.485, 0.456, 0.406],
                std=args.disc_std or [0.229, 0.224, 0.225],
                input_size_hw=disc_input_hw,
            )
        elif args.disc_family == "pixio":
            if args.disc_repo is None:
                raise ValueError("For --disc-family pixio, provide --disc-repo pointing to a local cloned facebookresearch/pixio repo.")
            if args.disc_weights is None:
                raise ValueError("For --disc-family pixio, provide --disc-weights pointing to a PixIO checkpoint (local path or URL).")
            disc = load_pixio_local_tower(
                repo=args.disc_repo,
                arch=args.disc_arch,
                weights=args.disc_weights,
                device=device,
                mean=args.disc_mean or [0.485, 0.456, 0.406],
                std=args.disc_std or [0.229, 0.224, 0.225],
                input_size_hw=disc_input_hw,
            )
        elif args.disc_family == "internvit":
            disc = load_internvit_hf_tower(
                model_id=args.disc_arch,
                device=device,
                mean=args.disc_mean,
                std=args.disc_std,
                input_size_hw=disc_input_hw,
            )
        else:
            if args.disc_pretrained is None:
                raise ValueError("For --disc-family openclip, provide --disc-pretrained.")
            disc = load_openclip_tower(
                model_name=args.disc_arch,
                pretrained=args.disc_pretrained,
                device=device,
                mean=args.disc_mean,
                std=args.disc_std,
                input_size_hw=disc_input_hw,
            )

        pmf_grid_side = int(round(math.sqrt(pmf.num_patches)))
        if pmf_grid_side * pmf_grid_side != pmf.num_patches:
            raise ValueError(f"Expected pMF patches to form a square grid, got {pmf.num_patches}")
        pmf_expected_grid_hw = (pmf_grid_side, pmf_grid_side)

        disc_expected_grid_hw = None
        if disc.expected_num_patches is not None:
            disc_grid_side = int(round(math.sqrt(disc.expected_num_patches)))
            if disc_grid_side * disc_grid_side == disc.expected_num_patches:
                disc_expected_grid_hw = (disc_grid_side, disc_grid_side)

        label_schedule = build_label_schedule(num_images=args.num_images, mode=args.label_mode, fixed_label=args.fixed_label, seed=args.seed)

        print0(dist_env, f"[setup] distributed world size: {dist_env.world_size}")
        print0(dist_env, f"[setup] per-rank batch size: {args.batch_size}")
        print0(dist_env, "[setup] pMF init source: checkpoint")
        print0(dist_env, "[setup] disc init source: pretrained")
        print0(dist_env, f"[setup] canonical grid source: {args.canonical_grid_source}")
        print0(dist_env, f"[setup] pMF expected native grid: {pmf_expected_grid_hw[0]}x{pmf_expected_grid_hw[1]}")
        if disc.native_image_size_hw is not None:
            print0(dist_env, f"[setup] disc input size: {disc.native_image_size_hw[0]}x{disc.native_image_size_hw[1]}")
        if disc_expected_grid_hw is not None:
            print0(dist_env, f"[setup] disc expected native grid: {disc_expected_grid_hw[0]}x{disc_expected_grid_hw[1]}")

        print0(dist_env, f"[setup] similarity: {args.similarity}")
        if args.similarity == "spearman":
            print0(dist_env, f"[setup] spearman mode: {args.spearman_mode} (reservoir size {args.spearman_reservoir_size})")
        print0(dist_env, "[1/3] Computing per-neuron stats on canonical grid...")
        pmf_normalizers, disc_normalizers, total_samples, canonical_grid_hw, pmf_native_grid_hw, disc_native_grid_hw = compute_layer_stats(
            pmf=pmf,
            disc=disc,
            num_images=args.num_images,
            batch_size=args.batch_size,
            label_schedule=label_schedule,
            seed=args.seed,
            num_steps=args.num_steps,
            omega=omega,
            t_min=t_min,
            t_max=t_max,
            device=device,
            canonical_grid_source=args.canonical_grid_source,
            act_resample_mode=args.act_resample_mode,
            dist_env=dist_env,
            similarity=args.similarity,
            spearman_mode=args.spearman_mode,
            spearman_reservoir_size=args.spearman_reservoir_size,
            save_images_dir=save_images_dir,
            save_image_format=args.generated_image_format,
        )
        print0(dist_env, f"Collected {total_samples:,} aligned canonical patch samples.")
        print0(
            dist_env,
            f"[setup] canonical grid observed as {canonical_grid_hw[0]}x{canonical_grid_hw[1]} "
            f"({args.canonical_grid_source})",
        )
        if pmf_native_grid_hw is not None:
            if pmf_native_grid_hw == canonical_grid_hw:
                print0(dist_env, f"[setup] pMF native activation grid observed as {pmf_native_grid_hw[0]}x{pmf_native_grid_hw[1]} (already canonical)")
            else:
                print0(
                    dist_env,
                    f"[setup] pMF native activation grid observed as {pmf_native_grid_hw[0]}x{pmf_native_grid_hw[1]} "
                    f"-> resampled to {canonical_grid_hw[0]}x{canonical_grid_hw[1]}",
                )
        if disc_native_grid_hw is not None:
            if disc_native_grid_hw == canonical_grid_hw:
                print0(dist_env, f"[setup] disc native activation grid observed as {disc_native_grid_hw[0]}x{disc_native_grid_hw[1]} (already canonical)")
            else:
                print0(
                    dist_env,
                    f"[setup] disc native activation grid observed as {disc_native_grid_hw[0]}x{disc_native_grid_hw[1]} "
                    f"-> resampled to {canonical_grid_hw[0]}x{canonical_grid_hw[1]}",
                )

        print0(dist_env, "[2/3] Accumulating correlations and global top-k neighbors...")
        pmf_dims = {i: st.dim for i, st in enumerate(pmf_normalizers)}
        disc_dims = {j: st.dim for j, st in enumerate(disc_normalizers)}

        topk_a_scores, topk_a_layers, topk_a_neurons = _init_global_topk(pmf_dims, args.topk)
        topk_b_scores, topk_b_layers, topk_b_neurons = _init_global_topk(disc_dims, args.topk)

        corr_dir = Path(args.save_dir) / "corr"
        if args.save_full_corr and dist_env.is_main:
            corr_dir.mkdir(parents=True, exist_ok=True)
        barrier(dist_env)

        chunk_starts = list(range(0, len(disc.modules), args.disc_chunk_size))
        for chunk_start in tqdm(chunk_starts, desc="disc layer chunks", **tqdm_kwargs(dist_env)):
            chunk_end = min(chunk_start + args.disc_chunk_size, len(disc.modules))
            chunk_indices = list(range(chunk_start, chunk_end))
            chunk_modules = [disc.modules[idx] for idx in chunk_indices]
            if dist_env.is_main:
                tqdm.write(f"  - disc layers {chunk_start}:{chunk_end}")

            accumulators = accumulate_corr_for_disc_chunk(
                pmf=pmf,
                disc=disc,
                disc_modules_chunk=chunk_modules,
                disc_chunk_indices=chunk_indices,
                pmf_normalizers=pmf_normalizers,
                disc_normalizers=disc_normalizers,
                num_images=args.num_images,
                batch_size=args.batch_size,
                label_schedule=label_schedule,
                seed=args.seed,
                num_steps=args.num_steps,
                omega=omega,
                t_min=t_min,
                t_max=t_max,
                device=device,
                compute_dtype=compute_dtype,
                canonical_grid_hw=canonical_grid_hw,
                act_resample_mode=args.act_resample_mode,
                dist_env=dist_env,
            )

            if dist_env.is_main:
                assert accumulators is not None
                for i, pmf_layer_name in enumerate(pmf.layer_names):
                    for local_j, disc_idx in enumerate(chunk_indices):
                        corr = (accumulators[i][local_j] / float(total_samples)).cpu()

                        topk_a_scores[i], topk_a_layers[i], topk_a_neurons[i] = _merge_rowwise_topk(
                            topk_a_scores[i], topk_a_layers[i], topk_a_neurons[i], corr, disc_idx
                        )
                        topk_b_scores[disc_idx], topk_b_layers[disc_idx], topk_b_neurons[disc_idx] = _merge_colwise_topk(
                            topk_b_scores[disc_idx], topk_b_layers[disc_idx], topk_b_neurons[disc_idx], corr, i
                        )

                        if args.save_full_corr:
                            out_path = corr_dir / f"corr_{pmf_layer_name}_vs_{disc.layer_names[disc_idx]}.pt"
                            torch.save(corr, out_path)

            barrier(dist_env)

        if not dist_env.is_main:
            return

        print("[3/3] Extracting mutual top-k matches...")
        best_buddies = build_mutual_topk_pairs(
            pmf_layer_names=pmf.layer_names,
            disc_layer_names=disc.layer_names,
            topk_a_scores=topk_a_scores,
            topk_a_layers=topk_a_layers,
            topk_a_neurons=topk_a_neurons,
            topk_b_scores=topk_b_scores,
            topk_b_layers=topk_b_layers,
            topk_b_neurons=topk_b_neurons,
            topk=args.topk,
        )

        report_ks: List[int] = []
        if args.report_topk:
            for tok in args.report_topk.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                k = int(tok)
                if k <= 0:
                    raise ValueError(f"--report-topk values must be positive, got {k}")
                if k > args.topk:
                    raise ValueError(
                        f"--report-topk value {k} exceeds --topk={args.topk}; "
                        "all report thresholds must be <= --topk."
                    )
                report_ks.append(k)
            report_ks = sorted(set(report_ks))

        report_topk_counts: Dict[int, int] = {}
        if report_ks:
            for k in report_ks:
                report_topk_counts[k] = sum(
                    1 for p in best_buddies
                    if int(p["rank_in_pmf"]) <= k and int(p["rank_in_disc"]) <= k
                )

        metadata = {
            "pmf_model": args.pmf_model,
            "disc_family": args.disc_family,
            "disc_arch": args.disc_arch,
            "disc_notes": disc.notes,
            "disc_native_image_size_hw": list(disc.native_image_size_hw) if disc.native_image_size_hw is not None else None,
            "pmf_init_source": "checkpoint",
            "disc_init_source": "pretrained",
            "canonical_grid_source": args.canonical_grid_source,
            "canonical_grid_hw": list(canonical_grid_hw),
            "pmf_expected_grid_hw": list(pmf_expected_grid_hw),
            "pmf_native_grid_hw": list(pmf_native_grid_hw) if pmf_native_grid_hw is not None else None,
            "disc_native_grid_hw": list(disc_native_grid_hw) if disc_native_grid_hw is not None else None,
            "disc_expected_grid_hw": list(disc_expected_grid_hw) if disc_expected_grid_hw is not None else None,
            "act_resample_mode": args.act_resample_mode,
            "similarity": args.similarity,
            "spearman_mode": args.spearman_mode if args.similarity == "spearman" else None,
            "spearman_reservoir_size": args.spearman_reservoir_size if args.similarity == "spearman" and args.spearman_mode == "approx" else None,
            "num_images": args.num_images,
            "batch_size_per_rank": args.batch_size,
            "distributed_world_size": dist_env.world_size,
            "label_mode": args.label_mode,
            "fixed_label": args.fixed_label,
            "num_steps": args.num_steps,
            "omega": omega,
            "t_min": t_min,
            "t_max": t_max,
            "seed": args.seed,
            "topk": args.topk,
            "report_topk": report_ks if report_ks else None,
            "report_topk_counts": {str(k): v for k, v in report_topk_counts.items()} if report_topk_counts else None,
            "total_patch_samples": total_samples,
            "pmf_num_layers": len(pmf.layer_names),
            "disc_num_layers": len(disc.layer_names),
            "pmf_num_patches": pmf.num_patches,
            "pmf_patch_size": pmf.patch_size,
            "disc_patch_size": disc.patch_size,
            "generation_seed_mode": "batch_seed(global_batch_start)",
            "save_generated_images": args.save_generated_images,
            "generated_images_subdir": args.generated_images_subdir if args.save_generated_images else None,
            "generated_image_format": args.generated_image_format if args.save_generated_images else None,
        }

        with open(Path(args.save_dir) / "run_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        with open(Path(args.save_dir) / "best_buddies.json", "w", encoding="utf-8") as f:
            json.dump(best_buddies, f, indent=2)

        for k in report_ks:
            if k == args.topk:
                continue
            filtered = [
                p for p in best_buddies
                if int(p["rank_in_pmf"]) <= k and int(p["rank_in_disc"]) <= k
            ]
            with open(Path(args.save_dir) / f"best_buddies_top{k}.json", "w", encoding="utf-8") as f:
                json.dump(filtered, f, indent=2)

        neighbors_dir = Path(args.save_dir) / "neighbors"
        neighbors_dir.mkdir(parents=True, exist_ok=True)
        for i, layer_name in enumerate(pmf.layer_names):
            torch.save(
                {
                    "scores": topk_a_scores[i],
                    "disc_layer_idx": topk_a_layers[i],
                    "disc_neuron": topk_a_neurons[i],
                },
                neighbors_dir / f"pmf_{layer_name}_top{args.topk}.pt",
            )

        print(f"Saved run metadata to {Path(args.save_dir) / 'run_metadata.json'}")
        print(f"Saved {len(best_buddies):,} mutual top-k pairs (k={args.topk}) to {Path(args.save_dir) / 'best_buddies.json'}")
        if report_topk_counts:
            print("Mutual best-buddy counts by k threshold:")
            for k in report_ks:
                print(f"  k={k:>3}: {report_topk_counts[k]:,}")
        if save_images_dir is not None:
            print(f"Saved generated images to {save_images_dir}")
        if best_buddies:
            print("Top 10 pairs:")
            for row in best_buddies[:10]:
                print(
                    f"  {row['pmf_layer']}[{row['pmf_neuron']}] <-> "
                    f"{row['disc_layer']}[{row['disc_neuron']}]: {row['correlation']:.4f}"
                )
    finally:
        cleanup_distributed(dist_env)

if __name__ == "__main__":
    main()
