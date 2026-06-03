"""Orchestration for match_lm."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import traceback
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer

from data.pile_sampler import (
    is_pile_dataset_name,
    parse_pile_subsets,
    resolve_pile_stats_path,
)

from .match_bytealign import _dtype_nbytes
from .match_cache import (
    _ensure_pile_val_stats_distributed,
    _split_token_budget,
    make_dataloader_from_cache,
    make_sharded_dataloader_from_cache,
    pretokenize_pile_two_models_bytecache_once,
)
from .match_cli import parse_config, resolve_compute_dtype
from .match_corr import compute_corr_token_level_averaged
from .match_dist import dist_active, dist_barrier, init_distributed_from_env
from .match_model import (
    _load_model,
    get_mlp_post_activation_modules,
    make_backbone_forward,
)
from .match_save import save_corr_block_from_prod_concat, save_neighbors_block_from_prod_concat
from .match_stats import compute_stats_token_level_averaged
from .match_types import CacheBundle, MatchConfig, RuntimeContext, StatsBundle


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")


def _prepare_runtime_context(cfg: MatchConfig) -> RuntimeContext:
    distributed, rank, world_size, local_rank, ddp_device = init_distributed_from_env()
    device = ddp_device if distributed else torch.device(cfg.device)

    os.makedirs(cfg.save_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if cfg.allow_tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    compute_dtype = resolve_compute_dtype(cfg.dtype_name)

    return RuntimeContext(
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        compute_dtype=compute_dtype,
    )


def _build_cache_bundle(cfg: MatchConfig) -> CacheBundle:
    pile_subsets = parse_pile_subsets(cfg.pile_subsets_arg)
    if cfg.token_cache.strip():
        token_cache_path = cfg.token_cache
    else:
        tok1_id = cfg.tokenizer1 or cfg.model1
        tok2_id = cfg.tokenizer2 or cfg.model2
        subset_tag = ",".join(pile_subsets) if pile_subsets else "default"
        h = hashlib.md5(
            (
                tok1_id
                + "||"
                + tok2_id
                + f"||dataset={cfg.dataset}||split={cfg.split}||subsets={subset_tag}"
                + f"||pile_ratio_by={cfg.pile_ratio_by}"
                + f"||use_padding={int(cfg.use_padding)}"
                + f"||N={cfg.num_samples}||T={cfg.max_tokens}||B={cfg.max_bytes}||seed={cfg.seed}"
            ).encode("utf-8")
        ).hexdigest()[:10]
        token_cache_path = os.path.join(cfg.save_dir, f"bytecache_{h}.pt")

    return CacheBundle(
        token_cache_path=token_cache_path,
        stats_path=None,
        pile_subsets=pile_subsets,
    )


def _build_allowed_pairs(
    l1: int, l2: int, depth_neighbors: Optional[int]
) -> Optional[Set[Tuple[int, int]]]:
    """Return set of (i, j) layer pairs within J-nearest by normalized depth.

    Returns None when depth_neighbors is None (all pairs allowed).
    """
    if depth_neighbors is None:
        return None
    pairs: Set[Tuple[int, int]] = set()
    for i in range(l1):
        d_i = i / max(1, l1 - 1)
        dists = []
        for j in range(l2):
            d_j = j / max(1, l2 - 1)
            dists.append((abs(d_i - d_j), j))
        dists.sort()
        for _, j in dists[: depth_neighbors]:
            pairs.add((i, j))
    return pairs


def _active_a_for_block(
    allowed_pairs: Optional[Set[Tuple[int, int]]], js: List[int], l1: int
) -> Optional[Set[int]]:
    """Return set of A-layer indices relevant for a B-block, or None if all."""
    if allowed_pairs is None:
        return None
    a_set: Set[int] = set()
    for j in js:
        for i in range(l1):
            if (i, j) in allowed_pairs:
                a_set.add(i)
    return a_set if len(a_set) < l1 else None


def _estimate_block_accumulator_gb(
    means_a_cpu: List[torch.Tensor],
    means_b_cpu: List[torch.Tensor],
    js: List[int],
    compute_dtype: torch.dtype,
    active_a: Optional[Set[int]] = None,
) -> float:
    total_db = sum(int(means_b_cpu[j].numel()) for j in js)
    total_elems = 0
    for i, m_a in enumerate(means_a_cpu):
        if active_a is not None and i not in active_a:
            continue
        total_elems += int(m_a.numel()) * total_db
    return (total_elems * _dtype_nbytes(compute_dtype)) / 1e9


def _save_run_metadata(cfg: MatchConfig, rt: RuntimeContext, cache: CacheBundle) -> None:
    if rt.rank != 0:
        return
    tok1_id = cfg.tokenizer1 or cfg.model1
    tok2_id = cfg.tokenizer2 or cfg.model2
    metadata = {
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "model1": cfg.model1,
        "model2": cfg.model2,
        "tokenizer1": tok1_id,
        "tokenizer2": tok2_id,
        "span_pool": cfg.span_pool,
        "dataset": cfg.dataset,
        "split": cfg.split,
        "pile_subsets": cache.pile_subsets,
        "pile_ratio_by": cfg.pile_ratio_by,
        "use_padding": cfg.use_padding,
        "shard_pile_subsets": cfg.shard_pile_subsets,
        "num_samples": cfg.num_samples,
        "max_tokens": cfg.max_tokens,
        "max_bytes": cfg.max_bytes,
        "batch_size": cfg.batch_size,
        "dtype": cfg.dtype_name,
        "distributed": rt.distributed,
        "world_size": rt.world_size,
        "seed": cfg.seed,
        "token_cache_path": cache.token_cache_path,
        "b_block": cfg.b_block,
        "depth_neighbors": cfg.depth_neighbors,
        "save_neighbors": cfg.save_neighbors,
        "top_k": cfg.top_k,
        "use_fsdp": cfg.use_fsdp,
    }
    with open(os.path.join(cfg.save_dir, "match_run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def _build_or_load_cache(
    cfg: MatchConfig,
    rt: RuntimeContext,
    tok1,
    tok2,
    cache_bundle: CacheBundle,
) -> Dict[str, torch.Tensor]:
    token_cache_path = cache_bundle.token_cache_path

    pile_subsets = cache_bundle.pile_subsets
    if cfg.shard_pile_subsets and rt.distributed and rt.world_size > 1 and pile_subsets:
        pile_subsets = [s for i, s in enumerate(pile_subsets) if (i % rt.world_size) == rt.rank]
        if rt.rank == 0:
            print(
                f"[rank0] Sharding Pile subsets across ranks: "
                f"{len(cache_bundle.pile_subsets)} -> {max(1, len(pile_subsets))} per rank"
            )
        if not pile_subsets:
            if rt.rank == 0:
                print("[rank0] No subsets assigned to rank 0 after sharding; will write empty shard.")

    if not os.path.exists(token_cache_path):
        if rt.distributed and rt.world_size > 1 and is_pile_dataset_name(cfg.dataset):
            if cfg.shard_pile_subsets and pile_subsets:
                local_budget = None
                stats_path = cache_bundle.stats_path
                if stats_path and os.path.exists(stats_path):
                    try:
                        with open(stats_path, "r", encoding="utf-8") as f:
                            stats = json.load(f)
                        if cfg.pile_ratio_by == "equal":
                            total = len(cache_bundle.pile_subsets)
                            shard_total = len(pile_subsets)
                        else:
                            all_weights = stats.get("docs_by_subset", {}) if cfg.pile_ratio_by == "docs" else stats.get("tokens_by_subset", {})
                            total = sum(int(all_weights.get(s, 0)) for s in cache_bundle.pile_subsets)
                            shard_total = sum(int(all_weights.get(s, 0)) for s in pile_subsets)
                        if total > 0 and shard_total > 0:
                            local_budget = max(1, int(cfg.num_samples * (shard_total / total)))
                    except Exception:
                        local_budget = None
                if local_budget is None:
                    local_budget = _split_token_budget(cfg.num_samples, rt.rank, rt.world_size)
                if rt.rank == 0:
                    print(
                        "[rank0] shard_pile_subsets=true: per-rank token budget="
                        f"{local_budget} (subset shard size={len(pile_subsets)})"
                    )
            else:
                local_budget = _split_token_budget(cfg.num_samples, rt.rank, rt.world_size)
            shard_path = f"{token_cache_path}.rank{rt.rank}.pt"
            merge_error_path = f"{token_cache_path}.merge_error.txt"
            if rt.rank == 0 and os.path.exists(merge_error_path):
                try:
                    os.remove(merge_error_path)
                except OSError:
                    pass
            if not os.path.exists(shard_path):
                if rt.rank == 0:
                    print(f"[rank0] Building byte-aligned cache shards: {token_cache_path}.rank*.pt")
                if local_budget <= 0:
                    torch.save({}, shard_path)
                else:
                    try:
                        cache_shard = pretokenize_pile_two_models_bytecache_once(
                            tok1,
                            tok2,
                            total_token_budget=local_budget,
                            dataset_name=cfg.dataset,
                            max_tokens=cfg.max_tokens,
                            max_bytes=cfg.max_bytes,
                            text_batch=cfg.tokenize_batch,
                            seed=cfg.seed + rt.rank,
                            split=cfg.split,
                            pile_subsets=pile_subsets,
                            stats_path=cache_bundle.stats_path,
                            ratio_by=cfg.pile_ratio_by,
                            use_padding=cfg.use_padding,
                            allow_empty=True,
                        )
                    except Exception as exc:
                        print(f"[rank{rt.rank}] cache shard build failed; writing empty shard. error={exc}")
                        print(traceback.format_exc())
                        cache_shard = {}
                    torch.save(cache_shard, shard_path)
            dist_barrier()
            if rt.rank == 0:
                shard_paths = [f"{token_cache_path}.rank{i}.pt" for i in range(rt.world_size)]
                shards_all: List[Dict[str, torch.Tensor]] = []
                for i, p in enumerate(shard_paths):
                    if not os.path.exists(p):
                        print(f"[rank0] Missing shard file for rank {i}: {p}; treating as empty.")
                        shards_all.append({})
                        continue
                    try:
                        shards_all.append(torch.load(p, map_location="cpu"))
                    except Exception as exc:
                        print(f"[rank0] Failed to load shard {p}; treating as empty. error={exc}")
                        shards_all.append({})
                shards = [s for s in shards_all if s]
                merge_error: Optional[str] = None
                if not shards:
                    merge_error = (
                        "All cache shards are empty; cannot build cache. "
                        "Try increasing --num_samples, lowering --seq_length, "
                        "or setting PILE_DEBUG=1 to inspect sampling."
                    )
                else:
                    try:
                        merged: Dict[str, torch.Tensor] = {}
                        for k in shards[0].keys():
                            merged[k] = torch.cat([s[k] for s in shards], dim=0)
                        torch.save(merged, token_cache_path)
                    except Exception as exc:
                        merge_error = f"Failed to merge cache shards: {exc}"
                if merge_error is not None:
                    with open(merge_error_path, "w", encoding="utf-8") as f:
                        f.write(merge_error + "\n")
            dist_barrier()
            if os.path.exists(merge_error_path):
                with open(merge_error_path, "r", encoding="utf-8") as f:
                    msg = f.read().strip()
                raise RuntimeError(msg)
        else:
            if not rt.distributed or rt.rank == 0:
                if rt.rank == 0:
                    print(f"[rank0] Building byte-aligned cache: {token_cache_path}")
                if not is_pile_dataset_name(cfg.dataset):
                    raise ValueError(
                        f"Dataset {cfg.dataset!r} is not recognized as a Pile-shaped corpus. "
                        f"This release matches only on the Pile validation set; point --dataset "
                        f"at a directory containing val.jsonl.zst (see language/README.md)."
                    )
                if rt.rank == 0:
                    print(f"[rank0] Using The Pile with token budget={cfg.num_samples}")
                cache = pretokenize_pile_two_models_bytecache_once(
                    tok1,
                    tok2,
                    total_token_budget=cfg.num_samples,
                    dataset_name=cfg.dataset,
                    max_tokens=cfg.max_tokens,
                    max_bytes=cfg.max_bytes,
                    text_batch=cfg.tokenize_batch,
                    seed=cfg.seed,
                    split=cfg.split,
                    pile_subsets=pile_subsets,
                    stats_path=cache_bundle.stats_path,
                    ratio_by=cfg.pile_ratio_by,
                    use_padding=cfg.use_padding,
                )
                torch.save(cache, token_cache_path)

    dist_barrier()
    return torch.load(token_cache_path, map_location="cpu")


def _run_stats_pass(
    cfg: MatchConfig,
    rt: RuntimeContext,
    model1,
    model2,
    modules_a: List[torch.nn.Module],
    modules_b: List[torch.nn.Module],
    capture_output_a: List[bool],
    capture_output_b: List[bool],
    dataloader_stats,
    forward1,
    forward2,
) -> StatsBundle:
    if rt.rank == 0:
        print("\n=== PASS 1: Stats with canonical span pooling (parallel across ranks) ===")

    disable_inner = bool(rt.distributed and rt.rank != 0)
    means_a, stds_a, means_b, stds_b = compute_stats_token_level_averaged(
        model1,
        model2,
        modules_a,
        modules_b,
        dataloader_stats,
        rt.device,
        forward1,
        forward2,
        rt.compute_dtype,
        span_pool=cfg.span_pool,
        capture_outputA=capture_output_a,
        capture_outputB=capture_output_b,
        disable_tqdm=disable_inner,
    )
    return StatsBundle(means_a=means_a, stds_a=stds_a, means_b=means_b, stds_b=stds_b)


def run(cfg: MatchConfig) -> None:
    rt = _prepare_runtime_context(cfg)
    os.makedirs(cfg.save_dir, exist_ok=True)

    if rt.distributed and rt.rank == 0:
        print(f"[rank0] distributed world_size={rt.world_size}")

    if rt.rank == 0:
        print(f"Loading model A: {cfg.model1}")
    model1 = _load_model(
        cfg.model1,
        rt.compute_dtype,
        rt.device,
        use_fsdp=cfg.use_fsdp,
        revision=cfg.revision1,
    )

    if rt.rank == 0:
        print(f"Loading model B: {cfg.model2}")
    model2 = _load_model(
        cfg.model2,
        rt.compute_dtype,
        rt.device,
        use_fsdp=cfg.use_fsdp,
        revision=cfg.revision2,
    )

    forward1 = make_backbone_forward(model1)
    forward2 = make_backbone_forward(model2)

    tok1_id = cfg.tokenizer1 or cfg.model1
    tok2_id = cfg.tokenizer2 or cfg.model2
    tok1 = AutoTokenizer.from_pretrained(tok1_id, use_fast=True)
    tok2 = AutoTokenizer.from_pretrained(tok2_id, use_fast=True)
    for tok in (tok1, tok2):
        tok.pad_token = tok.pad_token or tok.eos_token
        tok.padding_side = "right"
        tok.truncation_side = "right"

    cache_bundle = _build_cache_bundle(cfg)
    if is_pile_dataset_name(cfg.dataset):
        cache_bundle = CacheBundle(
            token_cache_path=cache_bundle.token_cache_path,
            stats_path=resolve_pile_stats_path(cfg.dataset, tok1),
            pile_subsets=cache_bundle.pile_subsets,
        )
        if rt.distributed and rt.world_size > 1 and cache_bundle.stats_path is not None:
            _ensure_pile_val_stats_distributed(
                cfg.dataset,
                tok1,
                stats_path=cache_bundle.stats_path,
                rank=rt.rank,
                world_size=rt.world_size,
            )

    cache = _build_or_load_cache(cfg, rt, tok1, tok2, cache_bundle)
    dataloader_full = make_dataloader_from_cache(cache, cfg.batch_size)
    if rt.distributed and rt.world_size > 1:
        dataloader_stats = make_sharded_dataloader_from_cache(cache, cfg.batch_size, rt.rank, rt.world_size)
    else:
        dataloader_stats = dataloader_full

    layers_a = get_mlp_post_activation_modules(model1, cfg.model1)
    layers_b = get_mlp_post_activation_modules(model2, cfg.model2)
    modules_a = [m for _, m, _ in layers_a]
    modules_b = [m for _, m, _ in layers_b]
    capture_output_a = [cap for _, _, cap in layers_a]
    capture_output_b = [cap for _, _, cap in layers_b]
    
    l1 = len(modules_a)
    l2 = len(modules_b)

    allowed_pairs = _build_allowed_pairs(l1, l2, cfg.depth_neighbors)
    if allowed_pairs is not None and rt.rank == 0:
        total_pairs = l1 * l2
        print(
            f"[depth_neighbors={cfg.depth_neighbors}] "
            f"Restricted to {len(allowed_pairs)}/{total_pairs} layer pairs "
            f"({100 * len(allowed_pairs) / max(1, total_pairs):.1f}%)"
        )

    stats = _run_stats_pass(
        cfg,
        rt,
        model1,
        model2,
        modules_a,
        modules_b,
        capture_output_a,
        capture_output_b,
        dataloader_stats,
        forward1,
        forward2,
    )

    if rt.rank == 0:
        for i in range(l1):
            torch.save({"mean": stats.means_a[i], "std": stats.stds_a[i]}, os.path.join(cfg.save_dir, f"stats_A_layer{i}.pt"))
        for j in range(l2):
            torch.save({"mean": stats.means_b[j], "std": stats.stds_b[j]}, os.path.join(cfg.save_dir, f"stats_B_layer{j}.pt"))
    dist_barrier()

    means_a_gpu = [m.to(device=rt.device, dtype=rt.compute_dtype) for m in stats.means_a]
    stds_a_gpu = [s.to(device=rt.device, dtype=rt.compute_dtype) for s in stats.stds_a]
    means_b_gpu = [m.to(device=rt.device, dtype=rt.compute_dtype) for m in stats.means_b]
    stds_b_gpu = [s.to(device=rt.device, dtype=rt.compute_dtype) for s in stats.stds_b]
    invstd_a_gpu = [1.0 / s for s in stds_a_gpu]
    invstd_b_gpu = [1.0 / s for s in stds_b_gpu]

    if rt.rank == 0:
        print("\n=== PASS 2: Correlation with canonical span pooling (B-blocks sharded across ranks) ===")
        print("    Each canonical span contributes equally; A/B activations are pooled within spans")

    b_block = max(1, cfg.b_block)
    all_block_starts = list(range(0, l2, b_block))
    my_block_starts = [bs for bi, bs in enumerate(all_block_starts) if (bi % rt.world_size) == rt.rank]

    if rt.distributed:
        print(f"[rank{rt.rank}] cuda:{torch.cuda.current_device()} blocks={len(my_block_starts)}/{len(all_block_starts)}")

    outer_disable = bool(rt.distributed and rt.rank != 0)
    for block_start in my_block_starts:
        js = list(range(block_start, min(block_start + b_block, l2)))
        active_a = _active_a_for_block(allowed_pairs, js, l1)

        # Skip entire block if no A-layers are relevant
        if active_a is not None and len(active_a) == 0:
            if rt.rank == 0:
                print(f"[rank0] Block js={js[0]}..{js[-1]}: skipped (no active A-layers)")
            continue

        modules_b_block = [modules_b[j] for j in js]
        capture_output_b_block = [capture_output_b[j] for j in js]
        means_b_block_gpu = [means_b_gpu[j] for j in js]
        invstd_b_block_gpu = [invstd_b_gpu[j] for j in js]

        if rt.rank == 0:
            est_gb = _estimate_block_accumulator_gb(stats.means_a, stats.means_b, js, rt.compute_dtype, active_a=active_a)
            active_str = f", active_a={len(active_a)}/{l1}" if active_a is not None else ""
            print(f"[rank0] Block js={js[0]}..{js[-1]} (K={len(js)}{active_str}): accumulators ≈ {est_gb:.1f} GB @ {cfg.dtype_name}")

        prod_concat, total_n, db_sizes = compute_corr_token_level_averaged(
            model1,
            model2,
            modules_a,
            modules_b_block,
            dataloader_full,
            rt.device,
            means_a_gpu,
            invstd_a_gpu,
            means_b_block_gpu,
            invstd_b_block_gpu,
            forward1,
            forward2,
            rt.compute_dtype,
            span_pool=cfg.span_pool,
            capture_outputA=capture_output_a,
            capture_outputB_block=capture_output_b_block,
            disable_tqdm=outer_disable,
            active_a_indices=active_a,
        )

        if cfg.save_neighbors:
            save_neighbors_block_from_prod_concat(
                prod_concat,
                total_n,
                db_sizes,
                js,
                cfg.save_dir,
                only_diagonal=cfg.only_diagonal,
                top_k=max(1, cfg.top_k),
                allowed_pairs=allowed_pairs,
            )
        else:
            save_corr_block_from_prod_concat(
                prod_concat,
                total_n,
                db_sizes,
                js,
                cfg.save_dir,
                only_diagonal=cfg.only_diagonal,
                allowed_pairs=allowed_pairs,
            )
        del prod_concat

    dist_barrier()
    _save_run_metadata(cfg, rt, cache_bundle)
    if rt.rank == 0:
        print("\nDone!")

    if dist_active():
        torch.distributed.destroy_process_group()


def main(argv=None) -> None:
    cfg = parse_config(argv)
    run(cfg)


if __name__ == "__main__":
    main()
