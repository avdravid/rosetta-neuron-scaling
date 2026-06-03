"""Typed runtime structures for match_lm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass(frozen=True)
class MatchConfig:
    model1: str
    model2: str
    tokenizer1: Optional[str]
    tokenizer2: Optional[str]
    span_pool: str
    dataset: str
    split: str
    pile_subsets_arg: str
    pile_ratio_by: str
    use_padding: bool
    shard_pile_subsets: bool
    num_samples: int
    max_tokens: int
    max_bytes: int
    batch_size: int
    device: str
    dtype_name: str
    seed: int
    save_dir: str
    only_diagonal: bool
    allow_tf32: bool
    tokenize_batch: int
    b_block: int
    save_neighbors: bool
    top_k: int
    depth_neighbors: Optional[int]
    token_cache: str
    use_fsdp: bool = False
    revision1: Optional[str] = None
    revision2: Optional[str] = None


@dataclass(frozen=True)
class RuntimeContext:
    distributed: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    compute_dtype: torch.dtype


@dataclass(frozen=True)
class LayerSpec:
    names: List[str]
    modules: List[torch.nn.Module]
    capture_output: List[bool]


@dataclass(frozen=True)
class StatsBundle:
    means_a: List[torch.Tensor]
    stds_a: List[torch.Tensor]
    means_b: List[torch.Tensor]
    stds_b: List[torch.Tensor]


@dataclass(frozen=True)
class CorrBlockResult:
    prod_concat: List[torch.Tensor]
    total_n: float
    db_sizes: List[int]


@dataclass(frozen=True)
class CacheBundle:
    token_cache_path: str
    stats_path: Optional[str]
    pile_subsets: List[str]
