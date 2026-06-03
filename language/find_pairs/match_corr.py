"""Correlation computation for match_lm."""

from __future__ import annotations

from typing import List, Set, Tuple, Optional

import torch
from tqdm import tqdm

from .match_stats import (
    _aggregate_spans_vectorized,
    _build_canonical_spans_for_batch,
    _precompute_span_flat_indices,
)
from .match_model import MultiActivationCapture


@torch.inference_mode()
def compute_corr_block_prod_concat_byte_aligned(
    model1,
    model2,
    modulesA: List[torch.nn.Module],
    modulesB_block: List[torch.nn.Module],
    dataloader_full,
    device: torch.device,
    meansA_gpu: List[torch.Tensor],
    invstdA_gpu: List[torch.Tensor],
    meansB_block_gpu: List[torch.Tensor],
    invstdB_block_gpu: List[torch.Tensor],
    forward_fn1,
    forward_fn2,
    compute_dtype: torch.dtype,
    *,
    disable_tqdm: bool = False,
) -> Tuple[List[torch.Tensor], float, List[int]]:
    """
    DEPRECATED: Use compute_corr_token_level_averaged instead.
    This function uses byte-weighted correlation which can produce values > 1.
    """
    raise RuntimeError(
        "compute_corr_block_prod_concat_byte_aligned is deprecated. "
        "Use compute_corr_token_level_averaged instead for correct token-level correlation."
    )


@torch.inference_mode()
def compute_corr_token_level_averaged(
    model1,
    model2,
    modulesA: List[torch.nn.Module],
    modulesB_block: List[torch.nn.Module],
    dataloader_full,
    device: torch.device,
    meansA_gpu: List[torch.Tensor],
    invstdA_gpu: List[torch.Tensor],
    meansB_block_gpu: List[torch.Tensor],
    invstdB_block_gpu: List[torch.Tensor],
    forward_fn1,
    forward_fn2,
    compute_dtype: torch.dtype,
    *,
    span_pool: str,
    capture_outputA: Optional[List[bool]] = None,
    capture_outputB_block: Optional[List[bool]] = None,
    disable_tqdm: bool = False,
    active_a_indices: Optional[Set[int]] = None,
) -> Tuple[List[torch.Tensor], float, List[int]]:
    """
    Unweighted canonical-span correlation.

    For each canonical span (shared A/B boundaries):
    1. Pool A tokens within the span (mean/max/median)
    2. Pool B tokens within the span (mean/max/median)
    3. Standardize using precomputed mean/std
    4. Accumulate unweighted outer product of standardized activations

    Returns:
        prod_concat: list of (DA, total_DB) tensors (unweighted sums)
        sum_weights: total sample count (use to divide prod_concat)
        DB_sizes: sizes for each B layer in the block
    """
    L1 = len(modulesA)
    K = len(modulesB_block)

    capture_outputA = capture_outputA or [False] * len(modulesA)
    capture_outputB_block = capture_outputB_block or [False] * len(modulesB_block)
    capA = MultiActivationCapture(modulesA, capture_output=capture_outputA).register()
    capB = MultiActivationCapture(modulesB_block, capture_output=capture_outputB_block).register()

    DB_sizes = [int(m.numel()) for m in meansB_block_gpu]
    offsets = [0]
    for db in DB_sizes:
        offsets.append(offsets[-1] + db)
    total_DB = offsets[-1]

    prod_concat: List[Optional[torch.Tensor]] = [None] * L1
    sum_weights = 0.0

    try:
        for batch in tqdm(dataloader_full, desc=f"Corr block (K={K}, unweighted)", leave=False, disable=disable_tqdm):
            # Build canonical spans (CPU)
            span_b, span_a_tokens, span_b_tokens = _build_canonical_spans_for_batch(
                batch["byte_offsets1"], batch["attention_mask1"],
                batch["byte_offsets2"], batch["attention_mask2"],
                batch["aligned_len"],
            )
            U = len(span_b)
            if U == 0:
                continue

            # Forward model A
            input_ids1 = batch["input_ids1"].to(device, non_blocking=True).long()
            attention_mask1 = batch["attention_mask1"].to(device, non_blocking=True)
            _ = forward_fn1(input_ids=input_ids1, attention_mask=attention_mask1)
            actsA = capA.get_and_clear(expected_batch=int(input_ids1.shape[0]), expected_seq=int(input_ids1.shape[1]))

            # Forward model B
            input_ids2 = batch["input_ids2"].to(device, non_blocking=True).long()
            attention_mask2 = batch["attention_mask2"].to(device, non_blocking=True)
            _ = forward_fn2(input_ids=input_ids2, attention_mask=attention_mask2)
            actsB = capB.get_and_clear(expected_batch=int(input_ids2.shape[0]), expected_seq=int(input_ids2.shape[1]))

            # Precompute flat indices once per batch (reused across all layers)
            flat_bA, flat_tA, span_idsA, countsA = _precompute_span_flat_indices(
                span_b, span_a_tokens, U, device
            )
            flat_bB, flat_tB, span_idsB, countsB = _precompute_span_flat_indices(
                span_b, span_b_tokens, U, device
            )

            # --- Extract RAW A activations for active layers (vectorized) ---
            raw_x_layers = [
                _aggregate_spans_vectorized(
                    actsA[i], flat_bA, flat_tA, span_idsA, countsA, U, span_pool, compute_dtype=compute_dtype
                )
                if (active_a_indices is None or i in active_a_indices)
                else None
                for i in range(L1)
            ]

            # --- Build RAW pooled B activations for this B-block (vectorized) ---
            raw_avg_B_list = [
                _aggregate_spans_vectorized(
                    actsB[k], flat_bB, flat_tB, span_idsB, countsB, U, span_pool, compute_dtype=compute_dtype
                )
                for k in range(K)
            ]

            # --- Standardize activations ---
            x_layers = []
            for i in range(L1):
                x = raw_x_layers[i]
                if x is None:
                    x_layers.append(None)
                    continue
                x.sub_(meansA_gpu[i]).mul_(invstdA_gpu[i])
                x_layers.append(x)

            avg_B_concat = torch.empty((U, total_DB), device=device, dtype=compute_dtype)
            off = 0
            for k in range(K):
                DB = DB_sizes[k]
                avg_B_k = raw_avg_B_list[k]
                avg_B_k.sub_(meansB_block_gpu[k]).mul_(invstdB_block_gpu[k])
                avg_B_concat[:, off:off + DB] = avg_B_k
                off += DB

            del raw_x_layers, raw_avg_B_list

            # --- Unweighted accumulation for each A layer ---
            weighted_avg_B = avg_B_concat  # (U, total_DB)

            for i in range(L1):
                x = x_layers[i]
                if x is None:
                    continue
                DA = x.shape[1]
                if prod_concat[i] is None:
                    prod_concat[i] = torch.zeros((DA, total_DB), device=device, dtype=compute_dtype)

                # Weighted outer product: sum over samples of w_s * x_s * avg_B_s
                prod_concat[i].addmm_(x.transpose(0, 1), weighted_avg_B)

            sum_weights += float(U)

            del x_layers, avg_B_concat, weighted_avg_B
            del flat_bA, flat_tA, span_idsA, countsA
            del flat_bB, flat_tB, span_idsB, countsB
            del actsA, actsB, input_ids1, input_ids2, attention_mask1, attention_mask2

        if sum_weights < 1e-8:
            raise RuntimeError("No valid canonical spans found (sum_weights ~ 0).")

        for i, p in enumerate(prod_concat):
            if p is None and (active_a_indices is None or i in active_a_indices):
                raise RuntimeError(f"Missing accumulator for A layer {i}.")

        return prod_concat, sum_weights, DB_sizes  # type: ignore

    finally:
        capA.remove()
        capB.remove()
