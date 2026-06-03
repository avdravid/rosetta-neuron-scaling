#!/usr/bin/env python3
"""
Find Best Buddy Neuron Pairs (GLOBAL across layers) from saved neighbor files.

Expected inputs (produced by match_lm.py with --save_neighbors):
  - nn_A_layer{i}_vs_B_layer{j}_top{T}.pt   : top-T neighbors in B(layer j) for each A neuron in layer i
  - nn_B_layer{j}_vs_A_layer{i}_top{T}.pt   : top-T neighbors in A(layer i) for each B neuron in layer j

Each .pt contains either:
  - (neighbors, scores)   where neighbors is int tensor and scores is float tensor
  - {"neighbors": ..., "scores": ...}

Workflow:
1) Load neighbor-file inventory (choose the largest T per layer-pair if multiple exist).
2) Build global top-K neighbor lists:
   - For every A neuron across all layers: global top-K over ALL B layers (merge per-layer candidates).
   - For every B neuron across all layers: global top-K over ALL A layers.
   This is done with vectorized torch.topk (no Python inner loops over neurons).
3) Best buddy pairs are mutual top-K neighbors (ranks recorded).
4) Save JSON + stats; print top pairs; optional sanity check.

Note:
- Global top-K is exact if every per-layer-pair file has T >= K.
"""

import argparse
import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


# ----------------------------- data structures -----------------------------

NeuronKey = Tuple[int, int]  # (layer, neuron_index)


@dataclass
class BestBuddyPair:
    model1_layer: int
    model1_neuron: int
    model2_layer: int
    model2_neuron: int
    correlation: float
    rank_in_model1: int  # 1-based
    rank_in_model2: int  # 1-based


# ----------------------------- file patterns -----------------------------

_RE_A = re.compile(r"^nn_A_layer(\d+)_vs_B_layer(\d+)_top(\d+)\.pt$")
_RE_B = re.compile(r"^nn_B_layer(\d+)_vs_A_layer(\d+)_top(\d+)\.pt$")


def _load_neighbors_file(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (neighbors, scores) as CPU tensors.
    Supports tuple/list (neighbors, scores) OR dict {"neighbors":..., "scores":...}.
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        neighbors = obj["neighbors"]
        scores = obj["scores"]
    elif isinstance(obj, (tuple, list)) and len(obj) == 2:
        neighbors, scores = obj
    else:
        raise ValueError(f"{os.path.basename(path)}: expected (neighbors,scores) or dict, got {type(obj)}")
    if not torch.is_tensor(neighbors) or not torch.is_tensor(scores):
        raise ValueError(f"{os.path.basename(path)}: neighbors/scores must be tensors.")
    if neighbors.ndim != 2 or scores.ndim != 2:
        raise ValueError(f"{os.path.basename(path)}: expected 2D tensors, got {neighbors.shape}, {scores.shape}")
    if neighbors.shape != scores.shape:
        raise ValueError(f"{os.path.basename(path)}: neighbors and scores shape mismatch.")
    return neighbors, scores


def list_neighbor_files(nn_dir: str):
    """
    Returns:
      filesA: dict[(layerA, layerB)] = (topT, filename) (largest T wins per pair)
      filesB: dict[(layerB, layerA)] = (topT, filename)
    """
    filesA: Dict[Tuple[int, int], Tuple[int, str]] = {}
    filesB: Dict[Tuple[int, int], Tuple[int, str]] = {}

    for f in os.listdir(nn_dir):
        m = _RE_A.match(f)
        if m:
            la, lb, top = map(int, m.groups())
            key = (la, lb)
            prev = filesA.get(key)
            if prev is None or top > prev[0]:
                filesA[key] = (top, f)
            continue

        m = _RE_B.match(f)
        if m:
            lb, la, top = map(int, m.groups())
            key = (lb, la)
            prev = filesB.get(key)
            if prev is None or top > prev[0]:
                filesB[key] = (top, f)

    return filesA, filesB


def build_pairs_by_layer(
    files: Dict[Tuple[int, int], Tuple[int, str]],
    *,
    left_is_A: bool,
) -> Dict[int, List[Tuple[int, int, str]]]:
    """
    For A->B files: key=(layerA, layerB), group by layerA.
    For B->A files: key=(layerB, layerA), group by layerB.
    Returns mapping:
      left_layer -> list[(right_layer, topT, filename)]
    """
    out: Dict[int, List[Tuple[int, int, str]]] = {}
    for (l_left, l_right), (top, fname) in files.items():
        out.setdefault(l_left, []).append((l_right, top, fname))
    for l in out:
        out[l].sort(key=lambda x: x[0])
    return out


def infer_layer_sizes(
    nn_dir: str,
    filesA: Dict[Tuple[int, int], Tuple[int, str]],
    filesB: Dict[Tuple[int, int], Tuple[int, str]],
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Infer #neurons per layer for model A (from A->B files) and model B (from B->A files).
    Returns:
      sizeA[layerA] = DA
      sizeB[layerB] = DB
    """
    sizeA: Dict[int, int] = {}
    sizeB: Dict[int, int] = {}

    # infer A sizes
    for (la, lb), (_, fname) in filesA.items():
        if la in sizeA:
            continue
        path = os.path.join(nn_dir, fname)
        neigh, _ = _load_neighbors_file(path)
        sizeA[la] = int(neigh.shape[0])

    # infer B sizes
    for (lb, la), (_, fname) in filesB.items():
        if lb in sizeB:
            continue
        path = os.path.join(nn_dir, fname)
        neigh, _ = _load_neighbors_file(path)
        sizeB[lb] = int(neigh.shape[0])

    if not sizeA:
        raise FileNotFoundError(f"No nn_A_layer*_vs_B_layer*_top*.pt files found in {nn_dir}")
    if not sizeB:
        raise FileNotFoundError(f"No nn_B_layer*_vs_A_layer*_top*.pt files found in {nn_dir}")

    return sizeA, sizeB


def make_offsets(size_by_layer: Dict[int, int]) -> Tuple[Dict[int, int], np.ndarray, np.ndarray]:
    """
    Build:
      offset_by_layer[layer] = global offset
      layers_sorted: array of layer ids in sorted order
      offsets_prefix: array of prefix sums aligned with layers_sorted, length = len(layers_sorted)+1
    Allows decoding global id -> (layer, neuron) even if layers are not contiguous.
    """
    layers_sorted = np.array(sorted(size_by_layer.keys()), dtype=np.int32)
    offsets_prefix = np.zeros((len(layers_sorted) + 1,), dtype=np.int64)
    offset_by_layer: Dict[int, int] = {}

    cur = 0
    for idx, layer in enumerate(layers_sorted):
        offset_by_layer[int(layer)] = int(cur)
        cur += int(size_by_layer[int(layer)])
        offsets_prefix[idx + 1] = cur

    return offset_by_layer, layers_sorted, offsets_prefix


def decode_global_ids(
    gids: np.ndarray,
    layers_sorted: np.ndarray,
    offsets_prefix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    gids: (N,) global neuron ids
    returns (layer_ids, neuron_ids) both int32/int64 arrays
    """
    # block index in [0..len(layers_sorted)-1]
    block = np.searchsorted(offsets_prefix, gids, side="right") - 1
    block = block.astype(np.int32)
    layer = layers_sorted[block].astype(np.int32)
    neuron = (gids - offsets_prefix[block]).astype(np.int64)
    return layer, neuron


# ----------------------------- global top-K merging -----------------------------

def compute_global_topk_for_A(
    nn_dir: str,
    pairsA_by_layerA: Dict[int, List[Tuple[int, int, str]]],
    offsetB_by_layer: Dict[int, int],
    sizeA: Dict[int, int],
    *,
    K: int,
    min_correlation: float,
    same_layer_only: bool,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
    """
    Build global top-K neighbors for all A neurons across all layers:
      returns:
        A_scores: (N_A, K) float32
        A_partnerB: (N_A, K) int64 global ids of B neurons
        top_used_per_pair: record of smallest topT used (for warnings)
    """
    device_t = torch.device(device)
    layerA_list = sorted(sizeA.keys())
    # global indexing for A neurons
    offsetA_by_layer, layersA_sorted, offsetsA_prefix = make_offsets(sizeA)
    N_A = int(offsetsA_prefix[-1])

    A_scores = np.full((N_A, K), -np.inf, dtype=np.float32)
    A_partnerB = np.full((N_A, K), -1, dtype=np.int64)

    # track minimum topT encountered (helps warn if <K)
    min_top_seen = {}

    for la in tqdm(layerA_list, desc="Merging global top-K for model A"):
        DA = int(sizeA[la])
        if DA <= 0:
            continue

        best_scores = torch.full((DA, K), float("-inf"), dtype=torch.float32, device=device_t)
        best_ids = torch.full((DA, K), -1, dtype=torch.int64, device=device_t)

        candidates = pairsA_by_layerA.get(la, [])
        if same_layer_only:
            candidates = [c for c in candidates if c[0] == la]

        for lb, topT, fname in candidates:
            min_top_seen[(la, lb)] = min(min_top_seen.get((la, lb), topT), topT)

            path = os.path.join(nn_dir, fname)
            neigh, score = _load_neighbors_file(path)

            # Use only first K per layer (sufficient for exact global top-K)
            k_use = min(K, int(score.shape[1]))
            neigh = neigh[:, :k_use].to(torch.int64)
            score = score[:, :k_use].to(torch.float32)

            if float(min_correlation) > float("-inf"):
                score = score.masked_fill(score < float(min_correlation), float("-inf"))

            base = int(offsetB_by_layer[lb])
            cand_ids = (neigh + base).to(torch.int64)

            # move to device
            cand_scores = score.to(device_t, non_blocking=False)
            cand_ids = cand_ids.to(device_t, non_blocking=False)

            # merge: topK over concat
            cat_scores = torch.cat([best_scores, cand_scores], dim=1)
            cat_ids = torch.cat([best_ids, cand_ids], dim=1)

            new_scores, idx = torch.topk(cat_scores, K, dim=1, largest=True, sorted=True)
            new_ids = cat_ids.gather(1, idx)

            best_scores = new_scores
            best_ids = new_ids

        start = offsetA_by_layer[la]
        A_scores[start:start + DA, :] = best_scores.cpu().numpy()
        A_partnerB[start:start + DA, :] = best_ids.cpu().numpy()

    return A_scores, A_partnerB, min_top_seen


def compute_global_topk_for_B(
    nn_dir: str,
    pairsB_by_layerB: Dict[int, List[Tuple[int, int, str]]],
    offsetA_by_layer: Dict[int, int],
    sizeB: Dict[int, int],
    *,
    K: int,
    min_correlation: float,
    same_layer_only: bool,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
    """
    Build global top-K neighbors for all B neurons across all layers:
      returns:
        B_scores: (N_B, K) float32
        B_partnerA: (N_B, K) int64 global ids of A neurons
        min_top_seen per pair
    """
    device_t = torch.device(device)
    layerB_list = sorted(sizeB.keys())
    offsetB_by_layer, layersB_sorted, offsetsB_prefix = make_offsets(sizeB)
    N_B = int(offsetsB_prefix[-1])

    B_scores = np.full((N_B, K), -np.inf, dtype=np.float32)
    B_partnerA = np.full((N_B, K), -1, dtype=np.int64)

    min_top_seen = {}

    for lb in tqdm(layerB_list, desc="Merging global top-K for model B"):
        DB = int(sizeB[lb])
        if DB <= 0:
            continue

        best_scores = torch.full((DB, K), float("-inf"), dtype=torch.float32, device=device_t)
        best_ids = torch.full((DB, K), -1, dtype=torch.int64, device=device_t)

        candidates = pairsB_by_layerB.get(lb, [])
        if same_layer_only:
            candidates = [c for c in candidates if c[0] == lb]  # here "right layer" is A layer index

        for la, topT, fname in candidates:
            min_top_seen[(lb, la)] = min(min_top_seen.get((lb, la), topT), topT)

            path = os.path.join(nn_dir, fname)
            neigh, score = _load_neighbors_file(path)

            k_use = min(K, int(score.shape[1]))
            neigh = neigh[:, :k_use].to(torch.int64)
            score = score[:, :k_use].to(torch.float32)

            if float(min_correlation) > float("-inf"):
                score = score.masked_fill(score < float(min_correlation), float("-inf"))

            base = int(offsetA_by_layer[la])
            cand_ids = (neigh + base).to(torch.int64)

            cand_scores = score.to(device_t, non_blocking=False)
            cand_ids = cand_ids.to(device_t, non_blocking=False)

            cat_scores = torch.cat([best_scores, cand_scores], dim=1)
            cat_ids = torch.cat([best_ids, cand_ids], dim=1)

            new_scores, idx = torch.topk(cat_scores, K, dim=1, largest=True, sorted=True)
            new_ids = cat_ids.gather(1, idx)

            best_scores = new_scores
            best_ids = new_ids

        start = offsetB_by_layer[lb]
        B_scores[start:start + DB, :] = best_scores.cpu().numpy()
        B_partnerA[start:start + DB, :] = best_ids.cpu().numpy()

    return B_scores, B_partnerA, min_top_seen


# ----------------------------- best-buddy computation -----------------------------

def find_global_best_buddies_from_neighbors(
    nn_dir: str,
    k: int,
    min_correlation: float = 0.0,
    same_layer_only: bool = False,
    device: str = "cpu",
    strict_topk: bool = False,
) -> List[BestBuddyPair]:
    """
    Main entry: compute global top-K for both directions from neighbor files, then find mutual pairs.
    """
    filesA, filesB = list_neighbor_files(nn_dir)
    if not filesA or not filesB:
        raise FileNotFoundError(
            f"Expected neighbor files in {nn_dir}:\n"
            f"  nn_A_layer{{i}}_vs_B_layer{{j}}_top{{T}}.pt and nn_B_layer{{j}}_vs_A_layer{{i}}_top{{T}}.pt\n"
            f"Found A-files={len(filesA)} B-files={len(filesB)}"
        )

    sizeA, sizeB = infer_layer_sizes(nn_dir, filesA, filesB)

    offsetA_by_layer, layersA_sorted, offsetsA_prefix = make_offsets(sizeA)
    offsetB_by_layer, layersB_sorted, offsetsB_prefix = make_offsets(sizeB)
    N_A = int(offsetsA_prefix[-1])
    N_B = int(offsetsB_prefix[-1])

    pairsA_by_la = build_pairs_by_layer(filesA, left_is_A=True)
    pairsB_by_lb = build_pairs_by_layer(filesB, left_is_A=False)

    # compute global KNN lists
    A_scores, A_partnerB, minTopA = compute_global_topk_for_A(
        nn_dir, pairsA_by_la, offsetB_by_layer, sizeA,
        K=k, min_correlation=min_correlation, same_layer_only=same_layer_only, device=device
    )
    B_scores, B_partnerA, minTopB = compute_global_topk_for_B(
        nn_dir, pairsB_by_lb, offsetA_by_layer, sizeB,
        K=k, min_correlation=min_correlation, same_layer_only=same_layer_only, device=device
    )

    # warn about per-pair topT < k
    badA = [(la, lb, t) for (la, lb), t in minTopA.items() if t < k]
    badB = [(lb, la, t) for (lb, la), t in minTopB.items() if t < k]
    if badA or badB:
        msg = (
            f"[warn] Some neighbor files have topT < requested K={k}. "
            f"Global top-K may be incomplete.\n"
            f"  A->B bad pairs: {len(badA)}  B->A bad pairs: {len(badB)}"
        )
        print(msg)
        if strict_topk:
            raise RuntimeError("strict_topk enabled and some files have topT < K.")

    # Build edge lists and intersect
    # edges from A: (a,b) with corr and rank1
    K = k
    a_idx = np.arange(N_A, dtype=np.int64)
    a_rep = np.repeat(a_idx, K)  # (N_A*K,)
    b_flat = A_partnerB.reshape(-1)
    corr_flat = A_scores.reshape(-1)
    rank1 = np.tile(np.arange(1, K + 1, dtype=np.int16), N_A)

    mask1 = (b_flat >= 0) & np.isfinite(corr_flat) & (corr_flat >= float(min_correlation))
    a1 = a_rep[mask1]
    b1 = b_flat[mask1].astype(np.int64)
    corr1 = corr_flat[mask1].astype(np.float32)
    r1 = rank1[mask1].astype(np.int16)

    code1 = a1 * np.int64(N_B) + b1  # unique encoding

    # edges from B: (a,b) with rank2
    b_idx = np.arange(N_B, dtype=np.int64)
    b_rep = np.repeat(b_idx, K)
    a2_flat = B_partnerA.reshape(-1)
    rank2 = np.tile(np.arange(1, K + 1, dtype=np.int16), N_B)

    mask2 = (a2_flat >= 0)
    a2 = a2_flat[mask2].astype(np.int64)
    b2 = b_rep[mask2]
    r2 = rank2[mask2].astype(np.int16)

    code2 = a2 * np.int64(N_B) + b2

    # sort and intersect
    order1 = np.argsort(code1)
    code1s = code1[order1]
    corr1s = corr1[order1]
    r1s = r1[order1]

    order2 = np.argsort(code2)
    code2s = code2[order2]
    r2s = r2[order2]

    inter, idx1, idx2 = np.intersect1d(code1s, code2s, return_indices=True)

    if inter.size == 0:
        return []

    corr_m = corr1s[idx1]
    r1_m = r1s[idx1].astype(np.int32)
    r2_m = r2s[idx2].astype(np.int32)

    a_m = (inter // np.int64(N_B)).astype(np.int64)
    b_m = (inter % np.int64(N_B)).astype(np.int64)

    # decode globals -> (layer, neuron)
    a_layer, a_neu = decode_global_ids(a_m, layersA_sorted, offsetsA_prefix)
    b_layer, b_neu = decode_global_ids(b_m, layersB_sorted, offsetsB_prefix)

    buddies: List[BestBuddyPair] = []
    for t in range(inter.size):
        buddies.append(
            BestBuddyPair(
                model1_layer=int(a_layer[t]),
                model1_neuron=int(a_neu[t]),
                model2_layer=int(b_layer[t]),
                model2_neuron=int(b_neu[t]),
                correlation=float(corr_m[t]),
                rank_in_model1=int(r1_m[t]),
                rank_in_model2=int(r2_m[t]),
            )
        )

    buddies.sort(key=lambda x: x.correlation, reverse=True)
    return buddies


# ----------------------------- analysis / printing -----------------------------

def analyze_best_buddies(buddies: List[BestBuddyPair]) -> dict:
    stats = {
        "total_pairs": len(buddies),
        "correlation_stats": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
        "rank_stats": {"mean_rank_model1": 0.0, "mean_rank_model2": 0.0},
        "pairs_by_layerpair": {},
    }
    if not buddies:
        return stats

    corrs = np.array([b.correlation for b in buddies], dtype=np.float64)
    r1 = np.array([b.rank_in_model1 for b in buddies], dtype=np.float64)
    r2 = np.array([b.rank_in_model2 for b in buddies], dtype=np.float64)

    stats["correlation_stats"]["mean"] = float(corrs.mean())
    stats["correlation_stats"]["std"] = float(corrs.std())
    stats["correlation_stats"]["min"] = float(corrs.min())
    stats["correlation_stats"]["max"] = float(corrs.max())
    stats["rank_stats"]["mean_rank_model1"] = float(r1.mean())
    stats["rank_stats"]["mean_rank_model2"] = float(r2.mean())

    by_lp: Dict[str, int] = {}
    for b in buddies:
        key = f"{b.model1_layer}_vs_{b.model2_layer}"
        by_lp[key] = by_lp.get(key, 0) + 1
    stats["pairs_by_layerpair"] = dict(sorted(by_lp.items(), key=lambda kv: (-kv[1], kv[0])))
    return stats


def save_best_buddies(buddies: List[BestBuddyPair], output_path: str) -> None:
    with open(output_path, "w") as f:
        json.dump([asdict(b) for b in buddies], f, indent=2)
    print(f"Saved best buddies to {output_path}")


def print_top_buddies(buddies: List[BestBuddyPair], top_n: int = 20) -> None:
    if not buddies:
        print("\n=== Top Best Buddy Pairs ===\n(no pairs)")
        return

    top_n = min(top_n, len(buddies))
    print(f"\n=== Top {top_n} Best Buddy Pairs (GLOBAL across layers) ===")
    print(f"{'L1':>4} {'N1':>7} {'L2':>4} {'N2':>7} {'Corr':>9} {'R1':>4} {'R2':>4}")
    print("-" * 46)
    for b in buddies[:top_n]:
        print(
            f"{b.model1_layer:>4} {b.model1_neuron:>7} "
            f"{b.model2_layer:>4} {b.model2_neuron:>7} "
            f"{b.correlation:>9.5f} {b.rank_in_model1:>4} {b.rank_in_model2:>4}"
        )


def verify_sanity_check_identity(buddies: List[BestBuddyPair]) -> None:
    if not buddies:
        print("\n=== Sanity Check ===\n(no pairs)")
        return
    total = len(buddies)
    diag = sum(
        1 for b in buddies
        if (b.model1_layer == b.model2_layer) and (b.model1_neuron == b.model2_neuron)
    )
    pct = 100.0 * diag / total
    print("\n=== Sanity Check (identity-style) ===")
    print(f"Diagonal matches (same layer & same neuron idx): {diag}/{total} ({pct:.1f}%)")


def save_correlation_histogram(
    buddies: List[BestBuddyPair],
    output_path: str,
    bin_width: float = 0.10,
    cross_activations_path: str = None,
) -> None:
    """
    Save histograms of best buddy correlations and activations.

    Args:
        buddies: List of best buddy pairs
        output_path: Path to save the histogram image
        bin_width: Width of each bin (default 0.10)
        cross_activations_path: Optional path to cross_activations.json for activation histogram
    """
    if not buddies:
        print("[warn] No buddies to plot histogram.")
        return

    corrs = np.array([b.correlation for b in buddies], dtype=np.float64)

    # Load activation data if available
    activations_m1 = []
    activations_m2 = []
    if cross_activations_path and os.path.exists(cross_activations_path):
        try:
            with open(cross_activations_path, "r") as f:
                cross_data = json.load(f)
            for pair in cross_data.get("pairs", []):
                for ex in pair.get("examples", []):
                    if "activation" in ex:
                        activations_m1.append(ex["activation"])
                    if "cross_activation" in ex:
                        activations_m2.append(ex["cross_activation"])
            print(f"Loaded {len(activations_m1)} model1 activations, {len(activations_m2)} model2 activations")
        except Exception as e:
            print(f"[warn] Could not load cross_activations: {e}")

    has_activations = len(activations_m1) > 0 or len(activations_m2) > 0

    # Create figure with 1 or 2 subplots
    if has_activations:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        ax_corr = axes[0]
        ax_act = axes[1]
    else:
        fig, ax_corr = plt.subplots(figsize=(10, 6))

    # --- Correlation histogram with 0.10 bin width ---
    corr_min = np.floor(corrs.min() / bin_width) * bin_width
    corr_max = np.ceil(corrs.max() / bin_width) * bin_width
    corr_bins = np.arange(corr_min, corr_max + bin_width, bin_width)

    n, bin_edges, patches = ax_corr.hist(
        corrs, bins=corr_bins, edgecolor="black", alpha=0.7, color="steelblue"
    )

    # Add statistics as text
    stats_text = (
        f"N = {len(corrs)}\n"
        f"Mean = {corrs.mean():.4f}\n"
        f"Std = {corrs.std():.4f}\n"
        f"Min = {corrs.min():.4f}\n"
        f"Max = {corrs.max():.4f}\n"
        f"Median = {np.median(corrs):.4f}"
    )
    ax_corr.text(
        0.02, 0.98, stats_text,
        transform=ax_corr.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax_corr.set_xlabel("Correlation", fontsize=12)
    ax_corr.set_ylabel("Count", fontsize=12)
    ax_corr.set_title("Best Buddy Pair Correlation Distribution", fontsize=14)
    ax_corr.grid(True, alpha=0.3)

    # Add vertical lines for mean and median
    ax_corr.axvline(corrs.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean ({corrs.mean():.3f})")
    ax_corr.axvline(np.median(corrs), color="green", linestyle=":", linewidth=1.5, label=f"Median ({np.median(corrs):.3f})")
    ax_corr.legend(loc="upper right")

    # --- Activation histogram with 0.10 bin width ---
    if has_activations:
        all_acts = []
        labels = []
        colors = []

        if activations_m1:
            all_acts.append(np.array(activations_m1, dtype=np.float64))
            labels.append("Model 1")
            colors.append("steelblue")
        if activations_m2:
            all_acts.append(np.array(activations_m2, dtype=np.float64))
            labels.append("Model 2 (cross)")
            colors.append("coral")

        # Compute bins for activations with 0.10 width
        all_vals = np.concatenate(all_acts)
        act_min = np.floor(all_vals.min() / bin_width) * bin_width
        act_max = np.ceil(all_vals.max() / bin_width) * bin_width
        act_bins = np.arange(act_min, act_max + bin_width, bin_width)

        ax_act.hist(
            all_acts, bins=act_bins, edgecolor="black", alpha=0.7,
            color=colors, label=labels
        )

        # Add statistics
        act_stats_lines = []
        for i, (acts, label) in enumerate(zip(all_acts, labels)):
            act_stats_lines.append(
                f"{label}:\n"
                f"  N = {len(acts)}\n"
                f"  Mean = {acts.mean():.4f}\n"
                f"  Std = {acts.std():.4f}\n"
                f"  Min = {acts.min():.4f}\n"
                f"  Max = {acts.max():.4f}"
            )
        act_stats_text = "\n".join(act_stats_lines)
        ax_act.text(
            0.98, 0.98, act_stats_text,
            transform=ax_act.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        ax_act.set_xlabel("Activation", fontsize=12)
        ax_act.set_ylabel("Count", fontsize=12)
        ax_act.set_title("Best Buddy Pair Activation Distribution", fontsize=14)
        ax_act.grid(True, alpha=0.3)
        ax_act.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved histogram to {output_path}")


# ----------------------------- main -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Find GLOBAL best-buddy neuron pairs from saved neighbor files")
    parser.add_argument("--corr_dir", type=str, default="./outputs",
                        help="Directory containing nn_A_* and nn_B_* neighbor files (from --save_neighbors)")
    parser.add_argument("--k", type=int, default=5,
                        help="Global top-K neighbors per neuron (must be <= per-file topT for exactness)")
    parser.add_argument("--min_correlation", type=float, default=0.0,
                        help="Minimum correlation threshold to consider")
    parser.add_argument("--same_layer_only", action="store_true",
                        help="Only consider layer i vs layer i (by layer index)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for merging/topk (cpu or cuda)")
    parser.add_argument("--strict_topk", action="store_true",
                        help="Error if any per-pair neighbor file has topT < K")
    parser.add_argument("--output", type=str, default="best_buddies.json",
                        help="Output JSON filename (written inside corr_dir)")
    parser.add_argument("--top_n", type=int, default=20,
                        help="How many top pairs to print")
    parser.add_argument("--sanity_check", action="store_true",
                        help="Run identity-style sanity check (useful if model1==model2)")
    args = parser.parse_args()

    # normalize device
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available; falling back to CPU.")
        args.device = "cpu"

    print(f"Finding GLOBAL best buddies from neighbor files with K={args.k}")
    print(f"dir: {args.corr_dir}")
    print(f"min_correlation: {args.min_correlation}")
    print(f"same_layer_only: {args.same_layer_only}")
    print(f"device: {args.device}")

    buddies = find_global_best_buddies_from_neighbors(
        nn_dir=args.corr_dir,
        k=args.k,
        min_correlation=args.min_correlation,
        same_layer_only=args.same_layer_only,
        device=args.device,
        strict_topk=args.strict_topk,
    )

    filesA, filesB = list_neighbor_files(args.corr_dir)
    sizeA, sizeB = infer_layer_sizes(args.corr_dir, filesA, filesB)
    total_A = sum(sizeA.values())
    total_B = sum(sizeB.values())

    stats = analyze_best_buddies(buddies)
    print("\n=== Statistics ===")
    print(f"Total neurons: model1={total_A}  model2={total_B}")
    print(f"Total best buddy pairs: {stats['total_pairs']}")
    print(
        f"Correlation: mean={stats['correlation_stats']['mean']:.5f}  "
        f"std={stats['correlation_stats']['std']:.5f}  "
        f"range=[{stats['correlation_stats']['min']:.5f}, {stats['correlation_stats']['max']:.5f}]"
    )
    print(
        f"Average ranks: model1={stats['rank_stats']['mean_rank_model1']:.2f}  "
        f"model2={stats['rank_stats']['mean_rank_model2']:.2f}"
    )

    lp_items = list(stats["pairs_by_layerpair"].items())
    if lp_items:
        print("\nPairs by layer-pair (top 10):")
        for k_, v_ in lp_items[:10]:
            print(f"  {k_}: {v_}")

    print_top_buddies(buddies, top_n=args.top_n)

    if args.sanity_check:
        verify_sanity_check_identity(buddies)

    out_path = os.path.join(args.corr_dir, args.output)
    save_best_buddies(buddies, out_path)

    stats_path = os.path.join(args.corr_dir, "best_buddies_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {stats_path}")

    # Save correlation histogram for sanity check
    hist_path = os.path.join(args.corr_dir, "best_buddies_correlation_histogram.png")
    save_correlation_histogram(buddies, hist_path)


if __name__ == "__main__":
    main()
