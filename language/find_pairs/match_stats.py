"""Statistics computation for match_lm."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import bisect

import torch
import torch.distributed as dist
from tqdm import tqdm

from .match_bytealign import _build_intervals_from_offsets, _module_activation_dim
from .match_dist import dist_active
from .match_model import MultiActivationCapture

def _token_byte_weights_cpu(
    byte_offsets: torch.Tensor,          # (B,S,2) int16/int32 CPU
    attention_mask: torch.Tensor,        # (B,S) uint8 CPU
    aligned_len: torch.Tensor,           # (B,) int32 CPU
) -> torch.Tensor:
    """
    For each token, weight = number of aligned bytes assigned to that token
    (clipped to [0, aligned_len) and ignoring zero-span tokens).
    Returns int32 weights (B,S) on CPU.
    """
    start = byte_offsets[..., 0].to(torch.int32)
    end = byte_offsets[..., 1].to(torch.int32)
    al = aligned_len.to(torch.int32).unsqueeze(1)

    clipped_end = torch.minimum(end, al)
    w = clipped_end - start
    w = torch.clamp(w, min=0)
    # ignore tokens beyond aligned_len and masked tokens
    w = w * (start < al).to(torch.int32)
    w = w * attention_mask.to(torch.int32)
    return w


@torch.inference_mode()
def compute_stats_all_layers_distributed_byte_aligned(
    model,
    modules: List[torch.nn.Module],
    dataloader_shard,
    device: torch.device,
    forward_fn,
    compute_dtype: torch.dtype,
    *,
    which: int,  # 1 or 2 (select batch keys)
    eps_val: float = 1e-4,
    disable_tqdm: bool = False,
):
    """
    DEPRECATED: This function computes stats using individual model byte weights,
    which can lead to correlation values outside [-1, 1] when tokenizations differ.
    Use compute_stats_both_models_overlap_aligned instead.
    """
    raise RuntimeError(
        "compute_stats_all_layers_distributed_byte_aligned is deprecated. "
        "Use compute_stats_both_models_overlap_aligned instead for correct correlation computation."
    )


@torch.inference_mode()
def compute_stats_both_models_overlap_aligned(
    model1,
    model2,
    modulesA: List[torch.nn.Module],
    modulesB: List[torch.nn.Module],
    dataloader_shard,
    device: torch.device,
    forward_fn1,
    forward_fn2,
    compute_dtype: torch.dtype,
    *,
    eps_val: float = 1e-4,
    disable_tqdm: bool = False,
):
    """
    DEPRECATED: Use compute_stats_token_level_averaged instead.
    This function computes overlap-byte-weighted stats which can lead to
    semantic overweighting of long tokens.
    """
    raise RuntimeError(
        "compute_stats_both_models_overlap_aligned is deprecated. "
        "Use compute_stats_token_level_averaged instead for correct token-level correlation."
    )


def _build_canonical_spans_for_batch(
    boA: torch.Tensor, amA: torch.Tensor,
    boB: torch.Tensor, amB: torch.Tensor,
    aligned_len: torch.Tensor,
) -> Tuple[List[int], List[List[int]], List[List[int]]]:
    """
    Build canonical spans by intersecting A/B token boundaries.

    For each batch item:
      - Collect token boundaries (start/end) from A and B.
      - Intersect boundary sets to get shared boundaries.
      - Canonical spans are ranges between consecutive shared boundaries.
      - For each span, collect overlapping A tokens and B tokens.

    Returns:
      span_b:          list of batch indices, length U
      span_a_tokens:   list of token-index lists for A, length U
      span_b_tokens:   list of token-index lists for B, length U
    """
    B = boA.shape[0]
    span_b: List[int] = []
    span_a_tokens: List[List[int]] = []
    span_b_tokens: List[List[int]] = []

    for b in range(B):
        al = int(aligned_len[b].item())
        if al <= 0:
            continue

        intsA = _build_intervals_from_offsets(boA[b], amA[b], al)
        intsB = _build_intervals_from_offsets(boB[b], amB[b], al)
        if not intsA or not intsB:
            continue

        boundariesA = {0, al}
        boundariesB = {0, al}
        for s, e, _ in intsA:
            boundariesA.add(s)
            boundariesA.add(e)
        for s, e, _ in intsB:
            boundariesB.add(s)
            boundariesB.add(e)

        shared = sorted(boundariesA & boundariesB)
        if len(shared) < 2:
            continue

        span_count = len(shared) - 1
        span_A: List[List[int]] = [[] for _ in range(span_count)]
        span_B: List[List[int]] = [[] for _ in range(span_count)]

        for s, e, tA in intsA:
            start_idx = max(0, bisect.bisect_right(shared, s) - 1)
            end_idx = max(0, bisect.bisect_left(shared, e))
            for si in range(start_idx, end_idx):
                if shared[si] < e and shared[si + 1] > s:
                    span_A[si].append(tA)

        for s, e, tB in intsB:
            start_idx = max(0, bisect.bisect_right(shared, s) - 1)
            end_idx = max(0, bisect.bisect_left(shared, e))
            for si in range(start_idx, end_idx):
                if shared[si] < e and shared[si + 1] > s:
                    span_B[si].append(tB)

        for si in range(span_count):
            if not span_A[si] or not span_B[si]:
                continue
            span_b.append(b)
            span_a_tokens.append(span_A[si])
            span_b_tokens.append(span_B[si])

    return span_b, span_a_tokens, span_b_tokens


def _precompute_span_flat_indices(
    span_b: List[int],
    span_tokens: List[List[int]],
    U: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Precompute flat indices for vectorized span aggregation.

    Returns:
        flat_b:       (P,) batch indices for each token-pair
        flat_t:       (P,) token indices for each token-pair
        span_ids:     (P,) which span each token-pair belongs to
        span_counts:  (U,) number of tokens per span (float32)
    """
    all_b: List[int] = []
    all_t: List[int] = []
    all_span: List[int] = []
    counts: List[int] = []
    for i in range(U):
        toks = span_tokens[i]
        n = len(toks)
        b = span_b[i]
        all_b.extend([b] * n)
        all_t.extend(toks)
        all_span.extend([i] * n)
        counts.append(n)

    return (
        torch.tensor(all_b, dtype=torch.long, device=device),
        torch.tensor(all_t, dtype=torch.long, device=device),
        torch.tensor(all_span, dtype=torch.long, device=device),
        torch.tensor(counts, dtype=torch.float32, device=device),
    )


def _aggregate_spans_vectorized(
    acts_layer: torch.Tensor,
    flat_b: torch.Tensor,
    flat_t: torch.Tensor,
    span_ids: torch.Tensor,
    span_counts: torch.Tensor,
    U: int,
    pool: str,
    *,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Vectorized span aggregation using scatter operations.

    acts_layer: (B, S, H) tensor
    flat_b, flat_t, span_ids: precomputed from _precompute_span_flat_indices
    span_counts: (U,) number of tokens per span
    pool: mean | max | median
    Returns: (U, H)
    """
    H = int(acts_layer.shape[2])
    device = acts_layer.device

    # Gather all activations at once: (P, H)
    all_acts = acts_layer[flat_b, flat_t].to(compute_dtype)
    expanded_ids = span_ids.unsqueeze(1).expand(-1, H)

    if pool == "mean":
        result = torch.zeros(U, H, device=device, dtype=compute_dtype)
        result.scatter_add_(0, expanded_ids, all_acts)
        result /= span_counts.to(compute_dtype).unsqueeze(1).clamp(min=1)
        return result
    elif pool == "max":
        result = torch.full((U, H), float('-inf'), device=device, dtype=compute_dtype)
        result.scatter_reduce_(0, expanded_ids, all_acts, reduce="amax", include_self=False)
        return result
    elif pool == "median":
        # Median cannot be easily vectorized; fall back to per-span loop
        out = torch.empty((U, H), device=device, dtype=compute_dtype)
        offset = 0
        for i in range(U):
            n = int(span_counts[i].item())
            token_acts = all_acts[offset:offset + n]
            out[i] = token_acts.float().median(dim=0).values.to(dtype=compute_dtype)
            offset += n
        return out
    else:
        raise ValueError(f"Unsupported span pool: {pool}")


def _aggregate_spans_for_layer(
    acts_layer: torch.Tensor,
    span_b: List[int],
    span_tokens: List[Any],
    pool: str,
    *,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Aggregate token activations for each span (loop-based, legacy).

    acts_layer: (B, S, H) tensor
    span_b: list of batch indices (len U)
    span_tokens: list of 1D token-index tensors (len U)
    pool: mean | max | median
    Returns: (U, H)
    """
    U = len(span_b)
    H = int(acts_layer.shape[2])
    device = acts_layer.device
    out = torch.empty((U, H), device=device, dtype=compute_dtype)
    for i in range(U):
        b = span_b[i]
        toks = span_tokens[i]
        token_acts = acts_layer[b, toks].to(dtype=compute_dtype)
        if pool == "mean":
            out[i] = token_acts.mean(dim=0)
        elif pool == "max":
            out[i] = token_acts.max(dim=0).values
        elif pool == "median":
            out[i] = token_acts.float().median(dim=0).values.to(dtype=compute_dtype)
        else:
            raise ValueError(f"Unsupported span pool: {pool}")
    return out


def _welford_combine(
    countA: torch.Tensor, meanA: torch.Tensor, m2A: torch.Tensor,
    countB: torch.Tensor, meanB: torch.Tensor, m2B: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combine two sets of Welford statistics using parallel algorithm.
    Avoids catastrophic cancellation by never computing E[x²] - E[x]² directly.

    All tensors should be float32 for numerical stability.
    """
    count = countA + countB
    # Avoid division by zero
    count_safe = count.clamp(min=1.0)
    delta = meanB - meanA
    mean = meanA + delta * countB / count_safe
    # M2 = M2_a + M2_b + delta^2 * count_a * count_b / count
    m2 = m2A + m2B + delta * delta * countA * countB / count_safe
    return count, mean, m2


def _welford_combine_weighted(
    sum_wA: torch.Tensor, meanA: torch.Tensor, m2A: torch.Tensor,
    sum_wB: torch.Tensor, meanB: torch.Tensor, m2B: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combine two sets of weighted Welford statistics.
    Uses sum of weights instead of count.

    All tensors should be float32 for numerical stability.
    """
    sum_w = sum_wA + sum_wB
    sum_w_safe = sum_w.clamp(min=1e-8)
    delta = meanB - meanA
    mean = meanA + delta * sum_wB / sum_w_safe
    m2 = m2A + m2B + delta * delta * sum_wA * sum_wB / sum_w_safe
    return sum_w, mean, m2


@torch.inference_mode()
def compute_stats_token_level_averaged(
    model1,
    model2,
    modulesA: List[torch.nn.Module],
    modulesB: List[torch.nn.Module],
    dataloader_shard,
    device: torch.device,
    forward_fn1,
    forward_fn2,
    compute_dtype: torch.dtype,
    *,
    span_pool: str,
    capture_outputA: Optional[List[bool]] = None,
    capture_outputB: Optional[List[bool]] = None,
    eps_val: float = 1e-4,
    disable_tqdm: bool = False,
):
    """
    Compute per-neuron mean/std using canonical spans with UNWEIGHTED averaging.

    For each canonical span (defined by shared A/B token boundaries):
      - Aggregate A tokens within the span (mean/max/median)
      - Aggregate B tokens within the span (mean/max/median)

    Uses Welford's algorithm with float32 accumulation to avoid catastrophic
    cancellation when computing variance (which occurs with E[x²] - E[x]²).
    """
    L1 = len(modulesA)
    L2 = len(modulesB)

    capture_outputA = capture_outputA or [False] * len(modulesA)
    capture_outputB = capture_outputB or [False] * len(modulesB)
    HsA = [_module_activation_dim(m, cap) for m, cap in zip(modulesA, capture_outputA)]
    HsB = [_module_activation_dim(m, cap) for m, cap in zip(modulesB, capture_outputB)]
    if len(set(HsA)) != 1:
        raise RuntimeError("Non-uniform layer dims in model A not supported.")
    if len(set(HsB)) != 1:
        raise RuntimeError("Non-uniform layer dims in model B not supported.")
    HA = HsA[0]
    HB = HsB[0]

    capA = MultiActivationCapture(modulesA, capture_output=capture_outputA).register()
    capB = MultiActivationCapture(modulesB, capture_output=capture_outputB).register()

    # Welford accumulators for model A (float32 for numerical stability).
    # We treat each canonical span equally (unweighted), but keep sum_w for aggregation.
    sum_wA = torch.zeros((L1, 1), device=device, dtype=torch.float32)
    meanA = torch.zeros((L1, HA), device=device, dtype=torch.float32)
    m2A = torch.zeros((L1, HA), device=device, dtype=torch.float32)

    # Welford accumulators for model B averaged (unweighted across canonical spans).
    sum_wB = torch.zeros((L2, 1), device=device, dtype=torch.float32)
    meanB = torch.zeros((L2, HB), device=device, dtype=torch.float32)
    m2B = torch.zeros((L2, HB), device=device, dtype=torch.float32)

    try:
        for batch in tqdm(dataloader_shard, desc="Stats (canonical spans, unweighted)", leave=False, disable=disable_tqdm):
            # Build canonical spans (CPU)
            span_b, span_a_tokens, span_b_tokens = _build_canonical_spans_for_batch(
                batch["byte_offsets1"], batch["attention_mask1"],
                batch["byte_offsets2"], batch["attention_mask2"],
                batch["aligned_len"],
            )
            U = len(span_b)
            if U == 0:
                continue

            # --- Forward pass model A ---
            input_ids1 = batch["input_ids1"].to(device, non_blocking=True).long()
            attention_mask1 = batch["attention_mask1"].to(device, non_blocking=True)
            _ = forward_fn1(input_ids=input_ids1, attention_mask=attention_mask1)
            actsA = capA.get_and_clear(expected_batch=int(input_ids1.shape[0]), expected_seq=int(input_ids1.shape[1]))

            # --- Forward pass model B ---
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

            # --- Aggregate activations for all layers (vectorized) ---
            # Model A: x_layers[li] = (U, HA)
            x_layers = [
                _aggregate_spans_vectorized(
                    actsA[li], flat_bA, flat_tA, span_idsA, countsA, U, span_pool, compute_dtype=compute_dtype
                ).float()
                for li in range(L1)
            ]

            # Model B: avg_B_layers[li] = (U, HB)
            avg_B_layers = [
                _aggregate_spans_vectorized(
                    actsB[li], flat_bB, flat_tB, span_idsB, countsB, U, span_pool, compute_dtype=compute_dtype
                ).float()
                for li in range(L2)
            ]

            # --- Unweighted stats (each canonical span counts once) ---
            sample_weights = torch.ones((U,), device=device, dtype=torch.float32)
            batch_sum_w = sample_weights.sum()  # scalar = U

            # --- Accumulate unweighted stats for model A ---
            for li, x in enumerate(x_layers):
                # Unweighted batch statistics
                w_norm = sample_weights / batch_sum_w.clamp(min=1e-8)  # (U,) normalized weights
                batch_weighted_mean = (w_norm.unsqueeze(1) * x).sum(dim=0)  # (HA,)
                batch_weighted_m2 = (sample_weights.unsqueeze(1) * (x - batch_weighted_mean) ** 2).sum(dim=0)  # (HA,)
                # Combine with running statistics
                sum_wA[li], meanA[li], m2A[li] = _welford_combine_weighted(
                    sum_wA[li], meanA[li], m2A[li],
                    batch_sum_w.unsqueeze(0), batch_weighted_mean, batch_weighted_m2
                )

            # --- Accumulate unweighted stats for model B ---
            for li, avg_B in enumerate(avg_B_layers):
                # Unweighted batch statistics
                w_norm = sample_weights / batch_sum_w.clamp(min=1e-8)  # (U,) normalized weights
                batch_weighted_mean = (w_norm.unsqueeze(1) * avg_B).sum(dim=0)  # (HB,)
                batch_weighted_m2 = (sample_weights.unsqueeze(1) * (avg_B - batch_weighted_mean) ** 2).sum(dim=0)  # (HB,)
                # Combine with running statistics
                sum_wB[li], meanB[li], m2B[li] = _welford_combine_weighted(
                    sum_wB[li], meanB[li], m2B[li],
                    batch_sum_w.unsqueeze(0), batch_weighted_mean, batch_weighted_m2
                )

            del x_layers, avg_B_layers, sample_weights
            del actsA, actsB, input_ids1, input_ids2, attention_mask1, attention_mask2
            del flat_bA, flat_tA, span_idsA, countsA
            del flat_bB, flat_tB, span_idsB, countsB

        # All-reduce across ranks for weighted Welford stats
        if dist_active():
            # For aggregation across ranks, use sum_w (counts) and local means.

            local_sum_wA = sum_wA.clone()
            local_sum_wB = sum_wB.clone()
            local_meanA = meanA.clone()
            local_meanB = meanB.clone()

            # Step 1: Compute weighted sums using LOCAL sum_w (before all-reduce)
            weighted_meanA = meanA * sum_wA  # local_sum_w * local_mean
            weighted_meanB = meanB * sum_wB

            # Step 2: All-reduce sum_w and weighted means
            dist.all_reduce(sum_wA, op=dist.ReduceOp.SUM)
            dist.all_reduce(sum_wB, op=dist.ReduceOp.SUM)
            dist.all_reduce(weighted_meanA, op=dist.ReduceOp.SUM)
            dist.all_reduce(weighted_meanB, op=dist.ReduceOp.SUM)

            # Step 3: Compute global means = sum(sum_w_i * mean_i) / sum(sum_w_i)
            global_meanA = weighted_meanA / sum_wA.clamp(min=1e-8)
            global_meanB = weighted_meanB / sum_wB.clamp(min=1e-8)

            # Step 4: All-reduce M2
            # Exact parallel Welford with correction term:
            # M2_total = sum(M2_i) + sum(sum_w_i * (mean_i - global_mean)^2)
            dist.all_reduce(m2A, op=dist.ReduceOp.SUM)
            dist.all_reduce(m2B, op=dist.ReduceOp.SUM)
            deltaA = local_meanA - global_meanA
            deltaB = local_meanB - global_meanB
            m2A_correction = local_sum_wA * (deltaA * deltaA)
            m2B_correction = local_sum_wB * (deltaB * deltaB)
            dist.all_reduce(m2A_correction, op=dist.ReduceOp.SUM)
            dist.all_reduce(m2B_correction, op=dist.ReduceOp.SUM)
            m2A = m2A + m2A_correction
            m2B = m2B + m2B_correction

            meanA = global_meanA
            meanB = global_meanB

        total_sum_wA = sum_wA[0, 0].item()
        total_sum_wB = sum_wB[0, 0].item()

        if total_sum_wA < 1e-8:
            raise RuntimeError("No valid canonical spans found (global sum_weights ~ 0).")

        eps = torch.tensor(eps_val, device=device, dtype=torch.float32)

        # Compute variance from M2: var = M2 / sum_weights
        varA = m2A / sum_wA.clamp(min=1e-8)
        varA = torch.clamp(varA, min=0.0)
        stdA = torch.sqrt(varA + eps)
        stdA = torch.clamp(stdA, min=eps)

        varB = m2B / sum_wB.clamp(min=1e-8)
        varB = torch.clamp(varB, min=0.0)
        stdB = torch.sqrt(varB + eps)
        stdB = torch.clamp(stdB, min=eps)

        # Convert to compute_dtype for output (maintains compatibility)
        meansA_cpu = [meanA[li].to(compute_dtype).detach().cpu() for li in range(L1)]
        stdsA_cpu = [stdA[li].to(compute_dtype).detach().cpu() for li in range(L1)]
        meansB_cpu = [meanB[li].to(compute_dtype).detach().cpu() for li in range(L2)]
        stdsB_cpu = [stdB[li].to(compute_dtype).detach().cpu() for li in range(L2)]

        return meansA_cpu, stdsA_cpu, meansB_cpu, stdsB_cpu

    finally:
        capA.remove()
        capB.remove()
