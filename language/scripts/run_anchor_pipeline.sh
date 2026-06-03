#!/bin/bash
# =============================================================================
# Rosetta Anchor Pipeline (multi-model via anchor)
# =============================================================================
# Notes:
# - The first model is the anchor; all remaining models are compared to it.
# - DATASET/SPLIT are threaded into both matching and cache-building.
# - TOKENIZER1 applies to the anchor model; TOKENIZER2 applies to every non-anchor
#   model if you explicitly set it. If left empty, each model uses its own default tokenizer.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/match_lm.py" ] && [ -d "$SCRIPT_DIR/find_pairs" ]; then
  REPO_ROOT="$SCRIPT_DIR"
else
  REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
fi
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 ANCHOR_MODEL OTHER_MODEL [OTHER_MODEL ...]"
  echo "Optional env vars: NUM_SAMPLES, START_STEP, TOP_K_ROSETTA, DATASET, SPLIT"
  exit 1
fi

ANCHOR_MODEL="$1"
shift
MODELS=("$@")

NUM_SAMPLES="${NUM_SAMPLES:-10000000}"
if [ -z "${START_STEP+x}" ]; then
  if [ "${#MODELS[@]}" -eq 1 ]; then
    START_STEP=3
  else
    START_STEP=1
  fi
fi
TOP_K_ROSETTA="${TOP_K_ROSETTA:-1000}"
STOP_AFTER_ROSETTA="${STOP_AFTER_ROSETTA:-false}"

if ! [[ "$START_STEP" =~ ^[1-6]$ ]]; then
  echo "Error: START_STEP must be between 1 and 6 (got: $START_STEP)"
  echo "Usage: $0 ANCHOR_MODEL OTHER_MODEL [OTHER_MODEL ...]"
  exit 1
fi

# If only one other model, delegate to the pairwise pipeline.
if [ "${#MODELS[@]}" -eq 1 ]; then
  "$REPO_ROOT/run_pipeline.sh" "$ANCHOR_MODEL" "${MODELS[0]}" "$NUM_SAMPLES" "$START_STEP"
  exit 0
fi

OUTPUT_BASE="${OUTPUT_BASE:-./outputs_anchor}"
ROSETTA_DIR="${OUTPUT_BASE}/rosetta"
CACHE_DIR="${CACHE_DIR:-${OUTPUT_BASE}/cache}"
ACT_DIR_ANCHOR="${OUTPUT_BASE}/act_anchor"
MANIFEST="${ROSETTA_DIR}/manifest.json"

mkdir -p "$OUTPUT_BASE" "$ROSETTA_DIR" "$CACHE_DIR" "$ACT_DIR_ANCHOR"

# Shared knobs (aligned with run_pipeline.sh where possible)
SEQ_LENGTH="${SEQ_LENGTH:-1024}"
BATCH_SIZE="${BATCH_SIZE:-8}"
K_NEIGHBORS="${K_NEIGHBORS:-1}"
SAVE_NEIGHBORS_TOPK="${SAVE_NEIGHBORS_TOPK:-100}"
B_BLOCK="${B_BLOCK:-2}"
TOKENIZE_BATCH="${TOKENIZE_BATCH:-256}"
MIN_CHARS="${MIN_CHARS:-100}"
TOP_K_ACTIVATIONS="${TOP_K_ACTIVATIONS:-20}"
PER_SAMPLE_TOP="${PER_SAMPLE_TOP:-3}"
CONTEXT_SIZE="${CONTEXT_SIZE:-10}"
BUDDY_TOP_PAIRS="${BUDDY_TOP_PAIRS:-$TOP_K_ROSETTA}"
SEQ_LENGTH_MODEL2="${SEQ_LENGTH_MODEL2:-1024}"
ALLOW_POSITION_FALLBACK="${ALLOW_POSITION_FALLBACK:-false}"
DTYPE="${DTYPE:-bfloat16}"
SPAN_POOL="${SPAN_POOL:-mean}"
CROSS_POOL="${CROSS_POOL:-$SPAN_POOL}"
SEED="${SEED:-42}"
DEPTH_NEIGHBORS="${DEPTH_NEIGHBORS:-}"

# Dataset selection. Default: the Pile validation set placed at <repo>/pile/
# (download instructions in language/README.md). Override with another local
# Pile mirror by pointing DATASET at a directory containing val.jsonl.zst.
DATASET="${DATASET:-$REPO_ROOT/../pile}"
SPLIT="${SPLIT:-val}"
PILE_SUBSETS="${PILE_SUBSETS:-}"
PILE_RATIO_BY="${PILE_RATIO_BY:-tokens}"
USE_PADDING="${USE_PADDING:-false}"
SHARD_PILE_SUBSETS="${SHARD_PILE_SUBSETS:-false}"

# Special-token handling
ADD_SPECIAL_TOKENS="${ADD_SPECIAL_TOKENS:-false}"
MAP_SPECIAL_TOKENS="${MAP_SPECIAL_TOKENS:-true}"
REQUIRE_CHAR_SPAN="${REQUIRE_CHAR_SPAN:-true}"
ALLOW_SPECIAL_TOKENS_WITHOUT_CHAR_SPAN="${ALLOW_SPECIAL_TOKENS_WITHOUT_CHAR_SPAN:-false}"

# torchrun
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NPROC_MATCH="${NPROC_MATCH:-$NPROC_PER_NODE}"
NPROC_ACT="${NPROC_ACT:-$NPROC_PER_NODE}"
NPROC_CROSS="${NPROC_CROSS:-$NPROC_PER_NODE}"
MASTER_PORT_MATCH="${MASTER_PORT_MATCH:-10101}"
MASTER_PORT_ACT="${MASTER_PORT_ACT:-10102}"
MASTER_PORT_CROSS="${MASTER_PORT_CROSS:-10103}"

# Optional tokenizer overrides
TOKENIZER1="${TOKENIZER1:-}"
TOKENIZER2="${TOKENIZER2:-}"

safe_name() {
  echo "$1" | tr '/' '_'
}

write_manifest() {
  local labels_str=""
  local paths_str=""
  local model
  local safe
  local out_path

  for model in "${MODELS[@]}"; do
    safe="$(safe_name "$model")"
    out_path="${OUTPUT_BASE}/cross_${safe}.json"
    labels_str+="$model"$'\n'
    paths_str+="$out_path"$'\n'
  done

  MODEL_LABELS_STR="$labels_str" \
  CROSS_PATHS_STR="$paths_str" \
  ANCHOR_MODEL_STR="$ANCHOR_MODEL" \
  ROSETTA_PATH_STR="${ROSETTA_DIR}/rosetta_anchors.json" \
  ACT_ANCHOR_DIR_STR="$ACT_DIR_ANCHOR" \
  MANIFEST_PATH_STR="$MANIFEST" \
  python - <<'PY'
import json, os
labels = [x for x in os.environ.get("MODEL_LABELS_STR", "").splitlines() if x]
paths = [x for x in os.environ.get("CROSS_PATHS_STR", "").splitlines() if x]
manifest = {
    "anchor_model": os.environ.get("ANCHOR_MODEL_STR", "Anchor"),
    "model_labels": labels,
    "rosetta_path": os.environ.get("ROSETTA_PATH_STR", ""),
    "cross_paths": paths,
    "act_anchor_dir": os.environ.get("ACT_ANCHOR_DIR_STR", ""),
}
out_path = os.environ["MANIFEST_PATH_STR"]
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
print("Saved manifest to", out_path)
PY
}

echo "Anchor model: $ANCHOR_MODEL"
echo "Other models: ${MODELS[*]}"
echo "NUM_SAMPLES=$NUM_SAMPLES START_STEP=$START_STEP TOP_K_ROSETTA=$TOP_K_ROSETTA"
echo "DATASET=$DATASET SPLIT=$SPLIT"

# -----------------------------------------------------------------------------
# Step 1: match_lm.py for each anchor-vs-model
# -----------------------------------------------------------------------------
if [ "$START_STEP" -le 1 ]; then
  echo "=============================================="
  echo "Step 1: match_lm.py (per anchor-vs-model pair)"
  echo "=============================================="

  for MODEL in "${MODELS[@]}"; do
    SAFE="$(safe_name "$MODEL")"
    CORR_OUTPUT_DIR="${OUTPUT_BASE}/${SAFE}/correlations"
    TOKEN_CACHE_PATH="${CACHE_DIR}/bytecache_${SAFE}.pt"
    mkdir -p "$CORR_OUTPUT_DIR"

    MATCH_TOKENIZER_ARGS=()
    if [ -n "$TOKENIZER1" ]; then
      MATCH_TOKENIZER_ARGS+=(--tokenizer1 "$TOKENIZER1")
    fi
    if [ -n "$TOKENIZER2" ]; then
      MATCH_TOKENIZER_ARGS+=(--tokenizer2 "$TOKENIZER2")
    fi

    MATCH_DATASET_ARGS=(--dataset "$DATASET" --split "$SPLIT" --pile_ratio_by "$PILE_RATIO_BY")
    if [ -n "$PILE_SUBSETS" ]; then
      MATCH_DATASET_ARGS+=(--pile_subsets "$PILE_SUBSETS")
    fi
    if [ "$USE_PADDING" = "true" ]; then
      MATCH_DATASET_ARGS+=(--use_padding)
    fi
    if [ "$SHARD_PILE_SUBSETS" = "true" ]; then
      MATCH_DATASET_ARGS+=(--shard_pile_subsets)
    fi

    MATCH_ARGS=(
      --model1 "$ANCHOR_MODEL"
      --model2 "$MODEL"
      --num_samples "$NUM_SAMPLES"
      --seq_length "$SEQ_LENGTH"
      --batch_size "$BATCH_SIZE"
      --span_pool "$SPAN_POOL"
      --save_dir "$CORR_OUTPUT_DIR"
      --dtype "$DTYPE"
      --seed "$SEED"
      --b_block "$B_BLOCK"
      --tokenize_batch "$TOKENIZE_BATCH"
      --save_neighbors
      --top_k "$SAVE_NEIGHBORS_TOPK"
      --token_cache "$TOKEN_CACHE_PATH"
      "${MATCH_TOKENIZER_ARGS[@]}"
      "${MATCH_DATASET_ARGS[@]}"
    )
    if [ -n "$DEPTH_NEIGHBORS" ]; then
      MATCH_ARGS+=(--depth_neighbors "$DEPTH_NEIGHBORS")
    fi
    if [ "${USE_FSDP:-false}" = "true" ]; then
      MATCH_ARGS+=(--use_fsdp)
    fi

    if [ "$NPROC_MATCH" -gt 1 ]; then
      torchrun --nproc_per_node="$NPROC_MATCH" --master_port="$MASTER_PORT_MATCH" \
        "$REPO_ROOT/match_lm.py" "${MATCH_ARGS[@]}"
    else
      python "$REPO_ROOT/match_lm.py" "${MATCH_ARGS[@]}"
    fi
  done
else
  echo "[Skipping Step 1: match_lm.py]"
fi

# -----------------------------------------------------------------------------
# Step 2: find_best_buddies.py for each pair
# -----------------------------------------------------------------------------
count_pairs() {
  BUDDIES_JSON_PATH="$1" python -c '
import json, os
with open(os.environ["BUDDIES_JSON_PATH"]) as f:
    data = json.load(f)
pairs = data.get("buddies", data) if isinstance(data, dict) else data
print(len(pairs))
'
}

if [ "$START_STEP" -le 2 ]; then
  echo "=============================================="
  echo "Step 2: find_best_buddies.py (per pair)"
  echo "=============================================="

  PAIRWISE_SUMMARY=()
  for MODEL in "${MODELS[@]}"; do
    SAFE="$(safe_name "$MODEL")"
    CORR_OUTPUT_DIR="${OUTPUT_BASE}/${SAFE}/correlations"
    python "$REPO_ROOT/find_best_buddies.py" \
      --corr_dir "$CORR_OUTPUT_DIR" \
      --k "$K_NEIGHBORS" \
      --output "best_buddies.json"
    BUDDIES_JSON="${CORR_OUTPUT_DIR}/best_buddies.json"
    if [ -f "$BUDDIES_JSON" ]; then
      N="$(count_pairs "$BUDDIES_JSON")"
      PAIRWISE_SUMMARY+=("  ${MODEL}: ${N} best-buddy pairs (K=${K_NEIGHBORS})")
    fi
  done

  if [ "${#PAIRWISE_SUMMARY[@]}" -gt 0 ]; then
    echo ""
    echo "=============================================="
    echo "PAIRWISE MATCHING COMPLETE (K=${K_NEIGHBORS}):"
    for line in "${PAIRWISE_SUMMARY[@]}"; do
      echo "$line"
    done
    echo "=============================================="
  fi

  if [ -n "${K_NEIGHBORS_LIST:-}" ]; then
    echo ""
    echo "=============================================="
    echo "K-sweep: re-running find_best_buddies for K in [${K_NEIGHBORS_LIST}] across all pairs"
    echo "=============================================="
    SWEEP_LINES=()
    for K in $K_NEIGHBORS_LIST; do
      for MODEL in "${MODELS[@]}"; do
        SAFE="$(safe_name "$MODEL")"
        CORR_OUTPUT_DIR="${OUTPUT_BASE}/${SAFE}/correlations"
        SWEEP_OUT="best_buddies_k${K}.json"
        SWEEP_PATH="${CORR_OUTPUT_DIR}/${SWEEP_OUT}"
        python "$REPO_ROOT/find_best_buddies.py" \
          --corr_dir "$CORR_OUTPUT_DIR" \
          --k "$K" \
          --output "$SWEEP_OUT" \
          > "${CORR_OUTPUT_DIR}/best_buddies_k${K}.log" 2>&1
        if [ -f "$SWEEP_PATH" ]; then
          N="$(count_pairs "$SWEEP_PATH")"
          SWEEP_LINES+=("  K=${K}  ${MODEL}: ${N} best-buddy pairs")
        else
          SWEEP_LINES+=("  K=${K}  ${MODEL}: <failed — see best_buddies_k${K}.log>")
        fi
      done
    done
    echo ""
    echo "=============================================="
    echo "PAIRWISE K-SWEEP RESULTS:"
    for line in "${SWEEP_LINES[@]}"; do
      echo "$line"
    done
    echo "=============================================="
  fi
else
  echo "[Skipping Step 2: find_best_buddies.py]"
fi

# -----------------------------------------------------------------------------
# Step 3: build cache (shared)
# -----------------------------------------------------------------------------
if [ "$START_STEP" -le 3 ] && [ "$STOP_AFTER_ROSETTA" != "true" ]; then
  echo "=============================================="
  echo "Step 3: build activation cache"
  echo "=============================================="

  CACHE_TOKENIZER_ARGS=()
  if [ -n "$TOKENIZER1" ]; then
    CACHE_TOKENIZER_ARGS+=(--tokenizer "$TOKENIZER1")
  else
    CACHE_TOKENIZER_ARGS+=(--tokenizer "$ANCHOR_MODEL")
  fi

  CACHE_ARGS=(
    --build_cache
    --cache_dir "$CACHE_DIR"
    --dataset "$DATASET"
    --split "$SPLIT"
    --cache_size "$NUM_SAMPLES"
    --min_chars "$MIN_CHARS"
    --seq_length "$SEQ_LENGTH"
    --seed "$SEED"
    "${CACHE_TOKENIZER_ARGS[@]}"
  )
  if [ -n "$PILE_SUBSETS" ]; then
    CACHE_ARGS+=(--pile_subsets "$PILE_SUBSETS")
  fi

  python "$REPO_ROOT/collect_activations.py" "${CACHE_ARGS[@]}"
else
  echo "[Skipping Step 3: build activation cache]"
fi

# -----------------------------------------------------------------------------
# Step 4: compute rosetta anchors + collect activations for anchor model
# -----------------------------------------------------------------------------
if [ "$START_STEP" -le 4 ]; then
  echo "=============================================="
  echo "Step 4: compute_rosetta_anchors.py + collect_activations.py"
  echo "=============================================="

  BUDDY_PATHS=()
  MODEL_NAMES=()
  for MODEL in "${MODELS[@]}"; do
    SAFE="$(safe_name "$MODEL")"
    CORR_OUTPUT_DIR="${OUTPUT_BASE}/${SAFE}/correlations"
    BUDDY_PATHS+=("${CORR_OUTPUT_DIR}/best_buddies.json")
    MODEL_NAMES+=("$MODEL")
  done

  # Always save the FULL intersection set; TOP_K_ROSETTA is only the viewer cap.
  python "$REPO_ROOT/compute_rosetta_anchors.py" \
    --anchor_model "$ANCHOR_MODEL" \
    --models "${MODEL_NAMES[@]}" \
    --buddies "${BUDDY_PATHS[@]}" \
    --output_dir "$ROSETTA_DIR" \
    --save_all

  PRIMARY_ROSETTA="${ROSETTA_DIR}/rosetta_anchors.json"
  if [ -f "$PRIMARY_ROSETTA" ]; then
    NUM_ANCHORS="$(count_pairs "$PRIMARY_ROSETTA")"
    echo ""
    echo "=============================================="
    echo "ROSETTA ANCHORS DISCOVERED (K=${K_NEIGHBORS}): ${NUM_ANCHORS} total saved to ${PRIMARY_ROSETTA}"
    echo "Viewer will render top ${TOP_K_ROSETTA} by avg correlation (raise TOP_K_ROSETTA to show more)."
    echo "=============================================="
  fi

  if [ -n "${K_NEIGHBORS_LIST:-}" ]; then
    echo ""
    echo "=============================================="
    echo "K-sweep: computing intersection anchors for K in [${K_NEIGHBORS_LIST}]"
    echo "=============================================="
    INTERSECTION_LINES=()
    for K in $K_NEIGHBORS_LIST; do
      K_ROSETTA_DIR="${ROSETTA_DIR}/k${K}"
      mkdir -p "$K_ROSETTA_DIR"
      K_BUDDY_PATHS=()
      for MODEL in "${MODELS[@]}"; do
        SAFE="$(safe_name "$MODEL")"
        K_BUDDY_PATHS+=("${OUTPUT_BASE}/${SAFE}/correlations/best_buddies_k${K}.json")
      done
      python "$REPO_ROOT/compute_rosetta_anchors.py" \
        --anchor_model "$ANCHOR_MODEL" \
        --models "${MODEL_NAMES[@]}" \
        --buddies "${K_BUDDY_PATHS[@]}" \
        --output_dir "$K_ROSETTA_DIR" \
        --save_all \
        > "${K_ROSETTA_DIR}/compute.log" 2>&1
      K_ROSETTA_PATH="${K_ROSETTA_DIR}/rosetta_anchors.json"
      if [ -f "$K_ROSETTA_PATH" ]; then
        N="$(count_pairs "$K_ROSETTA_PATH")"
        INTERSECTION_LINES+=("  K=${K}: ${N} intersection anchors  (${K_ROSETTA_PATH})")
      else
        INTERSECTION_LINES+=("  K=${K}: <failed — see ${K_ROSETTA_DIR}/compute.log>")
      fi
    done
    echo ""
    echo "=============================================="
    echo "INTERSECTION ANCHORS K-SWEEP:"
    for line in "${INTERSECTION_LINES[@]}"; do
      echo "$line"
    done
    echo "=============================================="
  fi

  if [ "$STOP_AFTER_ROSETTA" = "true" ]; then
    echo "STOP_AFTER_ROSETTA=true: stopping after Rosetta anchor computation."
    echo "Rosetta anchors: ${ROSETTA_DIR}/rosetta_anchors.json"
    exit 0
  fi

  ANCHOR_TOKENIZER_ARGS=()
  if [ -n "$TOKENIZER1" ]; then
    ANCHOR_TOKENIZER_ARGS+=(--tokenizer "$TOKENIZER1")
  fi

  SPECIAL_TOKEN_ARGS=()
  if [ "$ADD_SPECIAL_TOKENS" = "true" ]; then
    SPECIAL_TOKEN_ARGS+=(--add_special_tokens)
  fi
  REQUIRE_CHAR_SPAN_ARGS=()
  if [ "$REQUIRE_CHAR_SPAN" = "true" ]; then
    REQUIRE_CHAR_SPAN_ARGS+=(--require_char_span)
  fi
  SPECIAL_SPAN_ARGS=()
  if [ "$ALLOW_SPECIAL_TOKENS_WITHOUT_CHAR_SPAN" = "true" ]; then
    SPECIAL_SPAN_ARGS+=(--allow_special_tokens_without_char_span)
  fi

  ACT_ARGS=(
    --cache_dir "$CACHE_DIR"
    --model "$ANCHOR_MODEL"
    "${ANCHOR_TOKENIZER_ARGS[@]}"
    "${SPECIAL_TOKEN_ARGS[@]}"
    "${SPECIAL_SPAN_ARGS[@]}"
    --output_dir "$ACT_DIR_ANCHOR"
    --best_buddies_path "${ROSETTA_DIR}/rosetta_anchor_buddies.json"
    --which_model 1
    --layers all
    --seq_length "$SEQ_LENGTH"
    --batch_size "$BATCH_SIZE"
    --dtype "$DTYPE"
    --top_k "$TOP_K_ACTIVATIONS"
    --per_sample_top "$PER_SAMPLE_TOP"
    --context_size "$CONTEXT_SIZE"
    --buddy_top_pairs "$BUDDY_TOP_PAIRS"
    "${REQUIRE_CHAR_SPAN_ARGS[@]}"
  )

  if [ "$NPROC_ACT" -gt 1 ]; then
    torchrun --nproc_per_node="$NPROC_ACT" --master_port="$MASTER_PORT_ACT" \
      "$REPO_ROOT/collect_activations.py" "${ACT_ARGS[@]}"
  else
    python "$REPO_ROOT/collect_activations.py" "${ACT_ARGS[@]}"
  fi
else
  echo "[Skipping Step 4: compute_rosetta_anchors.py + collect_activations.py]"
fi

# -----------------------------------------------------------------------------
# Step 5: compute cross activations for each model (anchor -> model)
# -----------------------------------------------------------------------------
if [ "$START_STEP" -le 5 ]; then
  echo "=============================================="
  echo "Step 5: compute_cross_activations.py (per model)"
  echo "=============================================="

  for MODEL in "${MODELS[@]}"; do
    SAFE="$(safe_name "$MODEL")"
    FILTERED="${ROSETTA_DIR}/filtered_best_buddies_${SAFE}.json"
    OUT_PATH="${OUTPUT_BASE}/cross_${SAFE}.json"

    CROSS_TOKENIZER2_ARGS=()
    if [ -n "$TOKENIZER2" ]; then
      CROSS_TOKENIZER2_ARGS+=(--tokenizer2 "$TOKENIZER2")
    fi

    SPECIAL_TOKEN_ARGS=()
    if [ "$ADD_SPECIAL_TOKENS" = "true" ]; then
      SPECIAL_TOKEN_ARGS+=(--add_special_tokens)
    fi
    MAP_SPECIAL_TOKEN_ARGS=()
    if [ "$ADD_SPECIAL_TOKENS" = "true" ] && [ "$MAP_SPECIAL_TOKENS" = "true" ]; then
      MAP_SPECIAL_TOKEN_ARGS+=(--map_special_tokens)
    fi
    FALLBACK_ARGS=()
    if [ "$ALLOW_POSITION_FALLBACK" = "true" ]; then
      FALLBACK_ARGS+=(--allow_position_fallback)
    fi

    CROSS_ARGS=(
      --buddies_path "$FILTERED"
      --act1_dir "$ACT_DIR_ANCHOR"
      --model2 "$MODEL"
      "${CROSS_TOKENIZER2_ARGS[@]}"
      "${SPECIAL_TOKEN_ARGS[@]}"
      "${MAP_SPECIAL_TOKEN_ARGS[@]}"
      --cache_dir "$CACHE_DIR"
      --output_path "$OUT_PATH"
      --dtype "$DTYPE"
      --pool "$CROSS_POOL"
      --context_size "$CONTEXT_SIZE"
      --seq_length "$SEQ_LENGTH_MODEL2"
      --max_pairs "$TOP_K_ROSETTA"
      --max_examples "$TOP_K_ACTIVATIONS"
      "${FALLBACK_ARGS[@]}"
    )

    if [ "$NPROC_CROSS" -gt 1 ]; then
      torchrun --nproc_per_node="$NPROC_CROSS" --master_port="$MASTER_PORT_CROSS" \
        "$REPO_ROOT/compute_cross_activations.py" "${CROSS_ARGS[@]}"
    else
      python "$REPO_ROOT/compute_cross_activations.py" "${CROSS_ARGS[@]}"
    fi
  done
else
  echo "[Skipping Step 5: compute_cross_activations.py]"
fi

# -----------------------------------------------------------------------------
# Step 6: HTML viewer (top-activating sequences per rosetta neuron across N models)
# -----------------------------------------------------------------------------
if [ "$START_STEP" -le 6 ]; then
  echo "=============================================="
  echo "Step 6: visualize.py"
  echo "=============================================="

  write_manifest
  python "$REPO_ROOT/visualize.py" \
    --manifest "$MANIFEST" \
    --num-anchors "$TOP_K_ROSETTA" \
    --output "${OUTPUT_BASE}/index.html"
else
  echo "[Skipping Step 6: visualize.py]"
fi

echo ""
echo "=============================================="
echo "Anchor Pipeline Complete!"
echo "=============================================="
echo "  - Rosetta anchors: ${ROSETTA_DIR}/rosetta_anchors.json"
echo "  - Anchor activations: ${ACT_DIR_ANCHOR}/"
echo "  - HTML viewer: ${OUTPUT_BASE}/index.html"
