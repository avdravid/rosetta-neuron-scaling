#!/usr/bin/env python3
"""
compute_rosetta_anchors.py

Given best_buddies.json files for anchor-vs-other models, compute the
intersection of anchor neurons present in all pairs (rosetta anchors).

Outputs:
- rosetta_anchors.json: list of anchor neurons with per-model buddies + avg correlation
- rosetta_anchor_buddies.json: best_buddies-like list for anchor activations
- filtered_best_buddies_<model>.json: per-model best_buddies filtered to rosetta anchors
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib
import numpy as np


def load_best_buddies(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        out = []
        for v in obj.values():
            if isinstance(v, list):
                out.extend(v)
        return out
    raise ValueError(f"Unknown best buddies format: {type(obj)}")


def anchor_key(b: dict) -> Tuple[int, int]:
    return int(b.get("model1_layer", 0)), int(b.get("model1_neuron", 0))


def pick_best_per_anchor(buddies: List[dict]) -> Dict[Tuple[int, int], dict]:
    best: Dict[Tuple[int, int], dict] = {}
    for b in buddies:
        try:
            key = anchor_key(b)
            corr = float(b.get("correlation", 0.0))
        except Exception:
            continue
        prev = best.get(key)
        if prev is None or corr > float(prev.get("correlation", 0.0)):
            best[key] = b
    return best


def main():
    p = argparse.ArgumentParser(description="Compute rosetta anchors from anchor-vs-other best_buddies files")
    p.add_argument("--anchor_model", type=str, required=True)
    p.add_argument("--models", type=str, nargs="+", required=True, help="Other model names (same order as buddies paths)")
    p.add_argument("--buddies", type=str, nargs="+", required=True, help="Paths to best_buddies.json (anchor as model1)")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--top_k", type=int, default=None, help="Keep only top K rosetta anchors by avg correlation")
    p.add_argument("--save_all", action="store_true", help="Save full rosetta list (ignore top_k)")
    args = p.parse_args()

    if len(args.models) != len(args.buddies):
        raise ValueError("models and buddies must have the same length")

    os.makedirs(args.output_dir, exist_ok=True)

    per_model_best: Dict[str, Dict[Tuple[int, int], dict]] = {}
    for model_name, buddy_path in zip(args.models, args.buddies):
        buddies = load_best_buddies(buddy_path)
        per_model_best[model_name] = pick_best_per_anchor(buddies)

    # Intersection of anchor neurons across all models
    model_keys = [set(d.keys()) for d in per_model_best.values()]
    if not model_keys:
        rosetta_keys = set()
    else:
        rosetta_keys = set.intersection(*model_keys)

    rosetta_list = []
    for (l1, n1) in rosetta_keys:
        per_model = []
        corrs = []
        for model_name in args.models:
            b = per_model_best[model_name].get((l1, n1))
            if not b:
                continue
            corr = float(b.get("correlation", 0.0))
            corrs.append(corr)
            per_model.append({
                "model": model_name,
                "model2_layer": int(b.get("model2_layer", 0)),
                "model2_neuron": int(b.get("model2_neuron", 0)),
                "correlation": corr,
            })
        if not per_model:
            continue
        avg_corr = sum(corrs) / max(1, len(corrs))
        rosetta_list.append({
            "model1_layer": int(l1),
            "model1_neuron": int(n1),
            "avg_correlation": float(avg_corr),
            "per_model": per_model,
        })

    rosetta_list.sort(key=lambda x: float(x.get("avg_correlation", 0.0)), reverse=True)
    if (not args.save_all) and args.top_k is not None:
        rosetta_list = rosetta_list[: int(args.top_k)]

    rosetta_path = os.path.join(args.output_dir, "rosetta_anchors.json")
    with open(rosetta_path, "w", encoding="utf-8") as f:
        json.dump(rosetta_list, f, indent=2)

    # Anchor buddies file (for collect_activations)
    anchor_buddies = []
    for r in rosetta_list:
        per_model = r.get("per_model", [])
        if not per_model:
            continue
        first = per_model[0]
        anchor_buddies.append({
            "model1_layer": r["model1_layer"],
            "model1_neuron": r["model1_neuron"],
            "model2_layer": first.get("model2_layer", 0),
            "model2_neuron": first.get("model2_neuron", 0),
            "correlation": r.get("avg_correlation", 0.0),
        })
    anchor_buddies_path = os.path.join(args.output_dir, "rosetta_anchor_buddies.json")
    with open(anchor_buddies_path, "w", encoding="utf-8") as f:
        json.dump(anchor_buddies, f, indent=2)

    # Per-model filtered buddies
    for model_name in args.models:
        filtered = []
        best_map = per_model_best[model_name]
        for r in rosetta_list:
            key = (int(r["model1_layer"]), int(r["model1_neuron"]))
            b = best_map.get(key)
            if b:
                filtered.append(b)
        out_path = os.path.join(args.output_dir, f"filtered_best_buddies_{model_name.replace('/', '_')}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(filtered, f, indent=2)

    # Save avg-correlation histogram + stats for sanity checking
    stats = {
        "anchor": args.anchor_model,
        "models": args.models,
        "count": len(rosetta_list),
        "mean": 0.0,
        "std": 0.0,
        "min": 0.0,
        "max": 0.0,
        "median": 0.0,
    }
    avg_corrs = [float(r.get("avg_correlation", 0.0)) for r in rosetta_list]
    if avg_corrs:
        stats.update({
            "mean": float(np.mean(avg_corrs)),
            "std": float(np.std(avg_corrs)),
            "min": float(np.min(avg_corrs)),
            "max": float(np.max(avg_corrs)),
            "median": float(np.median(avg_corrs)),
        })

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.hist(avg_corrs, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
        plt.title("Rosetta avg correlations")
        plt.xlabel("Average correlation")
        plt.ylabel("Count")
        plt.tight_layout()
        hist_path = os.path.join(args.output_dir, "rosetta_avg_corr_hist.png")
        plt.savefig(hist_path, dpi=150)
        plt.close()

    stats_path = os.path.join(args.output_dir, "rosetta_avg_corr_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved {len(rosetta_list)} rosetta anchors to {rosetta_path}")


if __name__ == "__main__":
    main()
