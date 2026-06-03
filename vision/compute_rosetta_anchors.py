#!/usr/bin/env python3
"""
Compute Rosetta anchors across multiple pairwise anchor-vs-discriminative matching runs.

Expected input directories are outputs from pairwise matcher scripts such as
match_pmf_vit_multigpu.py or similar pipelines, each containing:
  - run_metadata.json
  - best_buddies.json

The anchor side is configurable via --anchor-prefix. For example:
  --anchor-prefix pmf   -> reads pmf_layer / pmf_neuron / rank_in_pmf
  --anchor-prefix sana  -> reads sana_layer / sana_neuron / rank_in_sana

An anchor neuron becomes a "Rosetta anchor" if it has at least one mutual-match
entry in *every* supplied results directory.

For each results directory, the script keeps only the strongest buddy (highest
correlation) for each anchor neuron. Then it intersects anchor neurons across all
runs and ranks them by:
  1) average correlation across runs
  2) minimum correlation across runs
  3) sum correlation across runs

Outputs:
  - rosetta_anchors.json
  - rosetta_anchors.csv
  - rosetta_anchor_summary.json

Example with pMF anchors:
  python compute_rosetta_anchors_updated.py \
      --results-dirs ./pmf_dinovitb16_50000 ./pmf_openclip_50000 \
      --anchor-prefix pmf \
      --output-dir ./pmf_rosetta_anchors

Example with Sana anchors:
  python compute_rosetta_anchors_updated.py \
      --results-dirs ./sana_dino ./sana_clip \
      --anchor-prefix sana \
      --output-dir ./sana_rosetta_anchors

Optional per-run labels:
  python compute_rosetta_anchors_updated.py \
      --results-dirs ./sana_dino ./sana_clip \
      --labels dino clip \
      --anchor-prefix sana \
      --output-dir ./sana_rosetta_anchors
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


AnchorKey = Tuple[str, int]  # (anchor_layer, anchor_neuron)


@dataclass(frozen=True)
class AnchorSpec:
    prefix: str

    @property
    def layer_idx_field(self) -> str:
        return f"{self.prefix}_layer_idx"

    @property
    def layer_field(self) -> str:
        return f"{self.prefix}_layer"

    @property
    def neuron_field(self) -> str:
        return f"{self.prefix}_neuron"

    @property
    def rank_field(self) -> str:
        return f"rank_in_{self.prefix}"


@dataclass
class RunInfo:
    label: str
    results_dir: Path
    metadata: Dict[str, Any]
    best_buddies: List[Dict[str, Any]]
    anchor_model_id: Optional[str]
    disc_family: str
    disc_arch: str

    @property
    def disc_display_name(self) -> str:
        fam = self.disc_family or "unknown"
        arch = self.disc_arch or "unknown"
        return f"{fam}:{arch}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dirs",
        type=str,
        nargs="+",
        required=True,
        help="One or more pairwise matcher output directories.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="*",
        default=None,
        help="Optional labels for the supplied result dirs. Must match the number of dirs if provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where rosetta_anchors.json/csv/summary will be written.",
    )
    parser.add_argument(
        "--anchor-prefix",
        type=str,
        default="pmf",
        help=(
            "String prefix used for the anchor-side fields in best_buddies.json. "
            "Examples: 'pmf' -> pmf_layer/pmf_neuron/rank_in_pmf, "
            "'sana' -> sana_layer/sana_neuron/rank_in_sana."
        ),
    )
    parser.add_argument(
        "--anchor-model-metadata-key",
        type=str,
        default=None,
        help=(
            "Optional metadata key used to check that all runs share the same anchor model. "
            "If omitted, the script tries to infer one from metadata and skips the check if it cannot."
        ),
    )
    parser.add_argument(
        "--min-correlation",
        type=float,
        default=None,
        help="If set, discard best-buddy pairs below this correlation before intersecting.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="If set, keep only the top N anchors in the JSON/CSV outputs.",
    )
    parser.add_argument(
        "--sort-by",
        choices=["avg", "min", "sum"],
        default="avg",
        help="Primary ranking metric for anchors.",
    )
    parser.add_argument(
        "--allow-mixed-anchor-models",
        action="store_true",
        help="By default, all result dirs must use the same anchor model when that metadata can be inferred.",
    )
    parser.add_argument(
        "--allow-mixed-pmf-models",
        action="store_true",
        help=(
            "Deprecated compatibility alias for --allow-mixed-anchor-models. "
            "Useful when reusing old pMF-oriented command lines."
        ),
    )
    return parser.parse_args()


def normalize_anchor_prefix(raw_prefix: str) -> str:
    prefix = raw_prefix.strip()
    if not prefix:
        raise ValueError("--anchor-prefix must be a non-empty string.")
    return prefix


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_label(results_dir: Path, metadata: Dict[str, Any]) -> str:
    disc_family = str(metadata.get("disc_family", "unknown"))
    disc_arch = str(metadata.get("disc_arch", "unknown"))
    base = f"{disc_family}:{disc_arch}"
    # Keep filesystem-friendly but readable.
    safe = base.replace("/", "_")
    return safe


def infer_anchor_model_id(
    metadata: Dict[str, Any],
    anchor_prefix: str,
    anchor_model_metadata_key: Optional[str],
) -> Optional[str]:
    if anchor_model_metadata_key:
        value = metadata.get(anchor_model_metadata_key)
        return None if value is None else stringify_model_id(value)

    preferred_keys = [
        f"{anchor_prefix}_model",
        f"{anchor_prefix}_model_name",
        f"{anchor_prefix}_pretrained",
        f"{anchor_prefix}_source",
    ]
    for key in preferred_keys:
        if key in metadata and metadata[key] is not None:
            return stringify_model_id(metadata[key])

    notes_key = f"{anchor_prefix}_notes"
    notes = metadata.get(notes_key)
    if isinstance(notes, dict):
        for key in ("model_name", "pretrained", "model", "name"):
            if key in notes and notes[key] is not None:
                return stringify_model_id(notes[key])

    # canonical_grid_source is often a coarse family identifier such as "pmf" or "sana".
    # It is better than nothing for a soft sanity check, but not necessarily a full model id.
    canonical_grid_source = metadata.get("canonical_grid_source")
    if canonical_grid_source is not None:
        canonical_grid_source = str(canonical_grid_source)
        if canonical_grid_source == anchor_prefix:
            return f"canonical_grid_source:{canonical_grid_source}"

    return None


def stringify_model_id(value: Any) -> str:
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def load_runs(
    results_dirs: Sequence[str],
    labels: Optional[Sequence[str]],
    anchor_prefix: str,
    anchor_model_metadata_key: Optional[str],
) -> List[RunInfo]:
    if labels is not None and len(labels) not in (0, len(results_dirs)):
        raise ValueError("--labels must either be omitted or have the same length as --results-dirs.")

    runs: List[RunInfo] = []
    for idx, raw_dir in enumerate(results_dirs):
        results_dir = Path(raw_dir).resolve()
        metadata_path = results_dir / "run_metadata.json"
        buddies_path = results_dir / "best_buddies.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing {metadata_path}")
        if not buddies_path.exists():
            raise FileNotFoundError(f"Missing {buddies_path}")

        metadata = read_json(metadata_path)
        best_buddies = read_json(buddies_path)
        if not isinstance(best_buddies, list):
            raise ValueError(f"Expected {buddies_path} to contain a list, got {type(best_buddies).__name__}")

        disc_family = str(metadata.get("disc_family", "unknown"))
        disc_arch = str(metadata.get("disc_arch", "unknown"))
        anchor_model_id = infer_anchor_model_id(metadata, anchor_prefix, anchor_model_metadata_key)

        label = labels[idx] if labels else infer_label(results_dir, metadata)

        runs.append(
            RunInfo(
                label=label,
                results_dir=results_dir,
                metadata=metadata,
                best_buddies=best_buddies,
                anchor_model_id=anchor_model_id,
                disc_family=disc_family,
                disc_arch=disc_arch,
            )
        )
    return runs


def validate_runs(
    runs: Sequence[RunInfo],
    allow_mixed_anchor_models: bool,
    anchor_prefix: str,
    anchor_model_metadata_key: Optional[str],
) -> None:
    if len(runs) < 2:
        raise ValueError("Need at least two result dirs to compute cross-run anchors.")

    if not allow_mixed_anchor_models:
        known_ids = [run.anchor_model_id for run in runs if run.anchor_model_id is not None]
        if len(known_ids) == len(runs):
            unique_ids = sorted(set(known_ids))
            if len(unique_ids) != 1:
                source_desc = anchor_model_metadata_key or f"auto-inferred metadata for anchor prefix '{anchor_prefix}'"
                raise ValueError(
                    "All runs must share the same anchor model unless "
                    "--allow-mixed-anchor-models is set. "
                    f"Using {source_desc}, got: {unique_ids}"
                )
        # If some ids are unknown, skip this check rather than fail spuriously.

    labels = [run.label for run in runs]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Run labels must be unique. Got: {labels}")



def require_row_field(row: Dict[str, Any], field: str, run: RunInfo) -> Any:
    if field not in row:
        available = ", ".join(sorted(row.keys()))
        raise KeyError(
            f"Missing required field '{field}' in best_buddies entry for run '{run.label}' "
            f"({run.results_dir}). Available fields: {available}"
        )
    return row[field]



def extract_best_per_anchor(
    run: RunInfo,
    anchor_spec: AnchorSpec,
    min_correlation: Optional[float],
) -> Dict[AnchorKey, Dict[str, Any]]:
    """
    Keep the strongest match per anchor neuron for this run.
    """
    best: Dict[AnchorKey, Dict[str, Any]] = {}
    for row in run.best_buddies:
        if not isinstance(row, dict):
            raise ValueError(
                f"Each entry in best_buddies.json must be an object/dict. Got {type(row).__name__} in run '{run.label}'."
            )

        corr = float(require_row_field(row, "correlation", run))
        if min_correlation is not None and corr < min_correlation:
            continue

        anchor_layer = str(require_row_field(row, anchor_spec.layer_field, run))
        anchor_neuron = int(require_row_field(row, anchor_spec.neuron_field, run))
        key: AnchorKey = (anchor_layer, anchor_neuron)

        prev = best.get(key)
        if prev is None or corr > float(prev["correlation"]):
            anchor_layer_idx = int(row.get(anchor_spec.layer_idx_field, -1))
            rank_in_anchor = int(row.get(anchor_spec.rank_field, -1))
            disc_layer_idx = int(row.get("disc_layer_idx", -1))
            disc_layer = str(require_row_field(row, "disc_layer", run))
            disc_neuron = int(require_row_field(row, "disc_neuron", run))
            rank_in_disc = int(row.get("rank_in_disc", -1))

            best[key] = {
                "anchor_prefix": anchor_spec.prefix,
                "anchor_layer_idx": anchor_layer_idx,
                "anchor_layer": anchor_layer,
                "anchor_neuron": anchor_neuron,
                anchor_spec.layer_idx_field: anchor_layer_idx,
                anchor_spec.layer_field: anchor_layer,
                anchor_spec.neuron_field: anchor_neuron,
                "disc_layer_idx": disc_layer_idx,
                "disc_layer": disc_layer,
                "disc_neuron": disc_neuron,
                "correlation": corr,
                "rank_in_anchor": rank_in_anchor,
                anchor_spec.rank_field: rank_in_anchor,
                "rank_in_disc": rank_in_disc,
                "disc_family": run.disc_family,
                "disc_arch": run.disc_arch,
                "disc_label": run.label,
                "results_dir": str(run.results_dir),
            }
    return best



def intersect_anchor_keys(per_run_best: Dict[str, Dict[AnchorKey, Dict[str, Any]]]) -> List[AnchorKey]:
    key_sets = [set(mapping.keys()) for mapping in per_run_best.values()]
    if not key_sets:
        return []
    common = set.intersection(*key_sets)
    return sorted(common, key=lambda x: (x[0], x[1]))



def build_anchor_records(
    common_keys: Sequence[AnchorKey],
    per_run_best: Dict[str, Dict[AnchorKey, Dict[str, Any]]],
    anchor_spec: AnchorSpec,
    sort_by: str,
    top_n: Optional[int],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for key in common_keys:
        run_matches: Dict[str, Dict[str, Any]] = {}
        corrs: List[float] = []
        anchor_layer_idx = None

        for label, mapping in per_run_best.items():
            match = dict(mapping[key])  # shallow copy
            run_matches[label] = match
            corrs.append(float(match["correlation"]))
            if anchor_layer_idx is None:
                anchor_layer_idx = int(match["anchor_layer_idx"])

        avg_corr = sum(corrs) / len(corrs)
        min_corr = min(corrs)
        max_corr = max(corrs)
        sum_corr = sum(corrs)

        record = {
            "anchor_prefix": anchor_spec.prefix,
            "anchor_layer_idx": int(anchor_layer_idx if anchor_layer_idx is not None else -1),
            "anchor_layer": key[0],
            "anchor_neuron": key[1],
            anchor_spec.layer_idx_field: int(anchor_layer_idx if anchor_layer_idx is not None else -1),
            anchor_spec.layer_field: key[0],
            anchor_spec.neuron_field: key[1],
            "num_models": len(run_matches),
            "avg_correlation": avg_corr,
            "min_correlation": min_corr,
            "max_correlation": max_corr,
            "sum_correlation": sum_corr,
            "matches": run_matches,
        }
        records.append(record)

    sort_field = {
        "avg": "avg_correlation",
        "min": "min_correlation",
        "sum": "sum_correlation",
    }[sort_by]

    records.sort(
        key=lambda r: (
            float(r[sort_field]),
            float(r["min_correlation"]),
            float(r["avg_correlation"]),
            float(r["sum_correlation"]),
            str(r["anchor_layer"]),
            int(r["anchor_neuron"]),
        ),
        reverse=True,
    )

    if top_n is not None:
        records = records[:top_n]
    return records



def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)



def write_csv(
    path: Path,
    records: Sequence[Dict[str, Any]],
    run_labels: Sequence[str],
    anchor_spec: AnchorSpec,
) -> None:
    base_fields = [
        "anchor_prefix",
        "anchor_layer_idx",
        "anchor_layer",
        "anchor_neuron",
        anchor_spec.layer_idx_field,
        anchor_spec.layer_field,
        anchor_spec.neuron_field,
        "num_models",
        "avg_correlation",
        "min_correlation",
        "max_correlation",
        "sum_correlation",
    ]
    per_run_fields: List[str] = []
    for label in run_labels:
        per_run_fields.extend(
            [
                f"{label}__disc_family",
                f"{label}__disc_arch",
                f"{label}__disc_layer_idx",
                f"{label}__disc_layer",
                f"{label}__disc_neuron",
                f"{label}__correlation",
                f"{label}__rank_in_anchor",
                f"{label}__{anchor_spec.rank_field}",
                f"{label}__rank_in_disc",
                f"{label}__results_dir",
            ]
        )
    fieldnames = base_fields + per_run_fields

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            row = {k: rec[k] for k in base_fields}
            matches = rec["matches"]
            for label in run_labels:
                m = matches[label]
                row[f"{label}__disc_family"] = m["disc_family"]
                row[f"{label}__disc_arch"] = m["disc_arch"]
                row[f"{label}__disc_layer_idx"] = m["disc_layer_idx"]
                row[f"{label}__disc_layer"] = m["disc_layer"]
                row[f"{label}__disc_neuron"] = m["disc_neuron"]
                row[f"{label}__correlation"] = m["correlation"]
                row[f"{label}__rank_in_anchor"] = m["rank_in_anchor"]
                row[f"{label}__{anchor_spec.rank_field}"] = m[anchor_spec.rank_field]
                row[f"{label}__rank_in_disc"] = m["rank_in_disc"]
                row[f"{label}__results_dir"] = m["results_dir"]
            writer.writerow(row)



def main() -> None:
    args = parse_args()
    anchor_prefix = normalize_anchor_prefix(args.anchor_prefix)
    anchor_spec = AnchorSpec(prefix=anchor_prefix)
    allow_mixed_anchor_models = bool(args.allow_mixed_anchor_models or args.allow_mixed_pmf_models)

    runs = load_runs(
        args.results_dirs,
        args.labels,
        anchor_prefix=anchor_prefix,
        anchor_model_metadata_key=args.anchor_model_metadata_key,
    )
    validate_runs(
        runs,
        allow_mixed_anchor_models=allow_mixed_anchor_models,
        anchor_prefix=anchor_prefix,
        anchor_model_metadata_key=args.anchor_model_metadata_key,
    )

    per_run_best: Dict[str, Dict[AnchorKey, Dict[str, Any]]] = {}
    for run in runs:
        mapping = extract_best_per_anchor(
            run,
            anchor_spec=anchor_spec,
            min_correlation=args.min_correlation,
        )
        per_run_best[run.label] = mapping

    common_keys = intersect_anchor_keys(per_run_best)
    anchor_records = build_anchor_records(
        common_keys=common_keys,
        per_run_best=per_run_best,
        anchor_spec=anchor_spec,
        sort_by=args.sort_by,
        top_n=args.top_n,
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    known_anchor_model_ids = sorted({run.anchor_model_id for run in runs if run.anchor_model_id is not None})
    summary = {
        "anchor_prefix": anchor_prefix,
        "anchor_field_names": {
            "layer_idx": anchor_spec.layer_idx_field,
            "layer": anchor_spec.layer_field,
            "neuron": anchor_spec.neuron_field,
            "rank": anchor_spec.rank_field,
        },
        "anchor_model_metadata_key": args.anchor_model_metadata_key,
        "num_runs": len(runs),
        "run_labels": [run.label for run in runs],
        "run_infos": [
            {
                "label": run.label,
                "results_dir": str(run.results_dir),
                "anchor_model_id": run.anchor_model_id,
                "disc_family": run.disc_family,
                "disc_arch": run.disc_arch,
                "num_best_buddies": len(run.best_buddies),
                "num_unique_anchor_neurons": len(per_run_best[run.label]),
            }
            for run in runs
        ],
        "anchor_model_set": known_anchor_model_ids,
        "anchor_model_consistency_check": {
            "enabled": not allow_mixed_anchor_models,
            "skipped_due_to_missing_metadata": len(known_anchor_model_ids) == 0 or any(
                run.anchor_model_id is None for run in runs
            ),
        },
        "min_correlation_filter": args.min_correlation,
        "sort_by": args.sort_by,
        "top_n": args.top_n,
        "num_common_anchor_neurons": len(common_keys),
        "num_output_anchors": len(anchor_records),
    }

    anchors_json = {
        "summary": summary,
        "anchors": anchor_records,
    }

    write_json(output_dir / "rosetta_anchor_summary.json", summary)
    write_json(output_dir / "rosetta_anchors.json", anchors_json)
    write_csv(output_dir / "rosetta_anchors.csv", anchor_records, [run.label for run in runs], anchor_spec)

    print(f"Anchor prefix: {anchor_prefix}")
    print(f"Wrote summary to {output_dir / 'rosetta_anchor_summary.json'}")
    print(f"Wrote anchors JSON to {output_dir / 'rosetta_anchors.json'}")
    print(f"Wrote anchors CSV to {output_dir / 'rosetta_anchors.csv'}")
    print(f"Found {len(anchor_records):,} anchors across {len(runs)} runs.")

    if summary["anchor_model_consistency_check"]["skipped_due_to_missing_metadata"] and not allow_mixed_anchor_models:
        print(
            "Note: anchor-model consistency check was skipped for at least one run because the metadata "
            "did not expose a reliable anchor model id. Pass --anchor-model-metadata-key to enforce it explicitly."
        )

    if anchor_records:
        print("Top 10 anchors:")
        for rec in anchor_records[:10]:
            parts = []
            for label in [run.label for run in runs]:
                m = rec["matches"][label]
                parts.append(
                    f"{label} -> {m['disc_layer']}[{m['disc_neuron']}] ({m['correlation']:.4f})"
                )
            joined = "; ".join(parts)
            print(
                f"  {rec['anchor_layer']}[{rec['anchor_neuron']}] "
                f"avg={rec['avg_correlation']:.4f}, min={rec['min_correlation']:.4f} :: {joined}"
            )


if __name__ == "__main__":
    main()
