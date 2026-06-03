#!/bin/bash
# =============================================================================
# Rosetta Neurons Pipeline (NEIGHBORS + cross-tokenizer + MULTI-GPU activations)
# =============================================================================

set -euo pipefail

# Locate the language/ root so this script works whether invoked from scripts/ or
# from the repo root. All entry-point .py files live one level above scripts/.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/match_lm.py" ] && [ -d "$SCRIPT_DIR/find_pairs" ]; then
  REPO_ROOT="$SCRIPT_DIR"
else
  REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
fi
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"

MODEL1="${1:-EleutherAI/pythia-1b}"
MODEL2="${2:-unsloth/Llama-3.2-1B}"
NUM_SAMPLES="${3:-10000000}"
START_STEP="${4:-4}"

export TOKENIZERS_PARALLELISM=false
# Limit native thread pools to avoid PyGILState_Release crashes on Python shutdown.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"

# Validate START_STEP
if ! [[ "$START_STEP" =~ ^[1-6]$ ]]; then
  echo "Error: START_STEP must be between 1 and 6 (got: $START_STEP)"
  echo "Usage: $0 [MODEL1] [MODEL2] [NUM_SAMPLES] [START_STEP]"
  echo ""
  echo "Steps:"
  echo "  1 - match_lm.py (compute correlations)"
  echo "  2 - find_best_buddies.py (find mutual neighbors)"
  echo "  3 - build activation cache"
  echo "  4 - collect_activations.py (model1 activations)"
  echo "  5 - compute_cross_activations.py (model2 cross-activations)"
  echo "  6 - visualize.py (generate HTML viewer)"
  echo ""
  echo "Example: $0 MODEL1 MODEL2 10000 1"
  exit 1
fi

echo "Starting pipeline from step $START_STEP"

SEQ_LENGTH="${SEQ_LENGTH:-1024}"
BATCH_SIZE="${BATCH_SIZE:-8}"
K_NEIGHBORS="${K_NEIGHBORS:-1}"

SAVE_NEIGHBORS_TOPK="${SAVE_NEIGHBORS_TOPK:-100}"




B_BLOCK="${B_BLOCK:-2}"


TOKENIZE_BATCH="${TOKENIZE_BATCH:-256}"
MIN_CHARS="${MIN_CHARS:-100}"

DEPTH_NEIGHBORS="${DEPTH_NEIGHBORS:-}"
USE_FSDP="${USE_FSDP:-false}"

# Viewer cap on how many top-K matched pairs (sorted by correlation) to render.
# The underlying cross_activations.json still contains every best-buddy pair.
TOP_K_ROSETTA="${TOP_K_ROSETTA:-1000}"

TOP_K_ACTIVATIONS="${TOP_K_ACTIVATIONS:-20}"
PER_SAMPLE_TOP="${PER_SAMPLE_TOP:-3}"


CONTEXT_SIZE="${CONTEXT_SIZE:-10}"
BUDDY_TOP_PAIRS="${BUDDY_TOP_PAIRS:-100}"


SEQ_LENGTH_MODEL2="${SEQ_LENGTH_MODEL2:-1024}"
ALLOW_POSITION_FALLBACK="${ALLOW_POSITION_FALLBACK:-false}"
DTYPE="${DTYPE:-bfloat16}"

# Span pooling used inside match_lm canonical spans: mean|max|median
SPAN_POOL="${SPAN_POOL:-mean}"

# Cross pooling in Step 5. If not explicitly set, match correlation pooling.
CROSS_POOL="${CROSS_POOL:-$SPAN_POOL}"



# Dataset selection. Default: the Pile validation set placed at <repo>/pile/
# (download instructions in language/README.md). Override with another local
# Pile mirror by pointing DATASET at a directory containing val.jsonl.zst.
DATASET="${DATASET:-$REPO_ROOT/../pile}"
SPLIT="${SPLIT:-val}"
PILE_SUBSETS="${PILE_SUBSETS:-}"

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

# Optional HF revision tags (e.g., "step143000" for Pythia training checkpoints).
# Currently only wired into step 1 (match_lm); step 4/5 still load from the default branch.
REVISION1="${REVISION1:-}"
REVISION2="${REVISION2:-}"

OUTPUT_BASE="${OUTPUT_BASE:-./outputs_cross}"
CORR_OUTPUT_DIR="${OUTPUT_BASE}/correlations"
CACHE_DIR="${CORR_OUTPUT_DIR}/cache"
ACT_DIR1="${OUTPUT_BASE}/act_m1"
# Allow sweep drivers to share a byte cache across many OUTPUT_BASE dirs by
# pointing TOKEN_CACHE at a common file (tokenizer-keyed, model-independent).
TOKEN_CACHE="${TOKEN_CACHE:-${CACHE_DIR}/bytecache.pt}"

mkdir -p "$CORR_OUTPUT_DIR" "$CACHE_DIR" "$ACT_DIR1" "$(dirname "$TOKEN_CACHE")"




# -----------------------------------------------------------------------------
# Step 1: compute correlations and save neighbors (MULTI-GPU)
# -----------------------------------------------------------------------------
if [ "$START_STEP" -le 1 ]; then
  echo "=============================================="
  echo "Step 1: match_lm.py (MULTI-GPU)"
  echo "=============================================="

  MATCH_TOKENIZER_ARGS=()
  if [ -n "$TOKENIZER1" ]; then
    MATCH_TOKENIZER_ARGS+=(--tokenizer1 "$TOKENIZER1")
  fi
  if [ -n "$TOKENIZER2" ]; then
    MATCH_TOKENIZER_ARGS+=(--tokenizer2 "$TOKENIZER2")
  fi

  MATCH_REVISION_ARGS=()
  if [ -n "$REVISION1" ]; then
    MATCH_REVISION_ARGS+=(--revision1 "$REVISION1")
  fi
  if [ -n "$REVISION2" ]; then
    MATCH_REVISION_ARGS+=(--revision2 "$REVISION2")
  fi

  MATCH_EXTRA_ARGS=()
  if [ -n "$DEPTH_NEIGHBORS" ]; then
    MATCH_EXTRA_ARGS+=(--depth_neighbors "$DEPTH_NEIGHBORS")
  fi
  if [ "$USE_FSDP" = "true" ]; then
    MATCH_EXTRA_ARGS+=(--use_fsdp)
  fi

  MATCH_ARGS=(
    --model1 "$MODEL1"
    --model2 "$MODEL2"
    --dataset "$DATASET"
    --split "$SPLIT"
    --num_samples "$NUM_SAMPLES"
    --seq_length "$SEQ_LENGTH"
    --batch_size "$BATCH_SIZE"
    --span_pool "$SPAN_POOL"
    --save_dir "$CORR_OUTPUT_DIR"
    --dtype "$DTYPE"
    --b_block "$B_BLOCK"
    --tokenize_batch "$TOKENIZE_BATCH"
    --save_neighbors
    --top_k "$SAVE_NEIGHBORS_TOPK"
    --token_cache "$TOKEN_CACHE"
    "${MATCH_TOKENIZER_ARGS[@]}"
    "${MATCH_REVISION_ARGS[@]}"
    "${MATCH_EXTRA_ARGS[@]}"
  )
  if [ -n "$PILE_SUBSETS" ]; then
    MATCH_ARGS+=(--pile_subsets "$PILE_SUBSETS")
  fi

  if [ "$NPROC_MATCH" -gt 1 ]; then
    torchrun --nproc_per_node="$NPROC_MATCH" --master_port="$MASTER_PORT_MATCH" match_lm.py "${MATCH_ARGS[@]}"
  else
    python match_lm.py "${MATCH_ARGS[@]}"
  fi

  echo ""
  echo "Neighbors saved to: $CORR_OUTPUT_DIR"
  echo ""
else
  echo "[Skipping Step 1: match_lm.py]"
fi

# -----------------------------------------------------------------------------
# Step 2: find best buddy pairs from neighbors
# -----------------------------------------------------------------------------
if [ "$START_STEP" -le 2 ]; then
  echo "=============================================="
  echo "Step 2: find_best_buddies.py"
  echo "=============================================="

  python find_best_buddies.py \
    --corr_dir "$CORR_OUTPUT_DIR" \
    --k "$K_NEIGHBORS" \
    --output "best_buddies.json"

  BUDDIES_JSON="${CORR_OUTPUT_DIR}/best_buddies.json"
  echo ""
  echo "Best buddies saved to: ${BUDDIES_JSON}"

  count_pairs() {
    BUDDIES_JSON_PATH="$1" python -c '
import json, os
with open(os.environ["BUDDIES_JSON_PATH"]) as f:
    data = json.load(f)
pairs = data.get("buddies", data) if isinstance(data, dict) else data
print(len(pairs))
'
  }

  if [ -f "$BUDDIES_JSON" ]; then
    NUM_MATCHES="$(count_pairs "$BUDDIES_JSON")"
    echo "=============================================="
    echo "MATCHING COMPLETE (K=${K_NEIGHBORS}): ${NUM_MATCHES} best-buddy pairs"
    echo "=============================================="
  fi

  if [ -n "${K_NEIGHBORS_LIST:-}" ]; then
    echo ""
    echo "=============================================="
    echo "K-sweep: re-running find_best_buddies for K in [${K_NEIGHBORS_LIST}]"
    echo "=============================================="
    SWEEP_LINES=()
    for K in $K_NEIGHBORS_LIST; do
      SWEEP_OUT="best_buddies_k${K}.json"
      SWEEP_PATH="${CORR_OUTPUT_DIR}/${SWEEP_OUT}"
      python find_best_buddies.py \
        --corr_dir "$CORR_OUTPUT_DIR" \
        --k "$K" \
        --output "$SWEEP_OUT" \
        > "${CORR_OUTPUT_DIR}/best_buddies_k${K}.log" 2>&1
      if [ -f "$SWEEP_PATH" ]; then
        N="$(count_pairs "$SWEEP_PATH")"
        SWEEP_LINES+=("  K=${K}: ${N} best-buddy pairs  (${SWEEP_OUT})")
      else
        SWEEP_LINES+=("  K=${K}: <failed — see best_buddies_k${K}.log>")
      fi
    done
    echo ""
    echo "=============================================="
    echo "K-SWEEP RESULTS:"
    for line in "${SWEEP_LINES[@]}"; do
      echo "$line"
    done
    echo "=============================================="
  fi
  echo ""
else
  echo "[Skipping Step 2: find_best_buddies.py]"
fi

if [ "${STOP_AFTER_ROSETTA:-false}" = "true" ]; then
  echo "STOP_AFTER_ROSETTA=true: stopping after best_buddies (pairwise Rosetta output)."
  echo "Best buddies: ${CORR_OUTPUT_DIR}/best_buddies.json"
  exit 0
fi

# -----------------------------------------------------------------------------
# Step 3: build activation cache
# -----------------------------------------------------------------------------
if [ "$START_STEP" -le 3 ]; then
  echo "=============================================="
  echo "Step 3: build activation cache"
  echo "=============================================="

  CACHE_TOKENIZER_ARGS=()
  if [ -n "$TOKENIZER1" ]; then
    CACHE_TOKENIZER_ARGS+=(--tokenizer "$TOKENIZER1")
  else
    CACHE_TOKENIZER_ARGS+=(--tokenizer "$MODEL1")
  fi

  python collect_activations.py \
    --build_cache \
    --cache_dir "$CACHE_DIR" \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --cache_size "$NUM_SAMPLES" \
    --min_chars "$MIN_CHARS" \
    ${PILE_SUBSETS:+--pile_subsets "$PILE_SUBSETS"} \
    "${CACHE_TOKENIZER_ARGS[@]}" 

  echo ""
  echo "Cache built in: $CACHE_DIR"
  echo ""
else
  echo "[Skipping Step 3: build activation cache]"
fi

# -----------------------------------------------------------------------------
# Step 4: collect activations for model1 (MULTI-GPU)
# -----------------------------------------------------------------------------
if [ "$START_STEP" -le 4 ]; then
  echo "=============================================="
  echo "Step 4: collect_activations.py (MULTI-GPU)"
  echo "=============================================="

  COLLECT_TOKENIZER1_ARGS=()
  if [ -n "$TOKENIZER1" ]; then
    COLLECT_TOKENIZER1_ARGS+=(--tokenizer "$TOKENIZER1")
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

  if [ "$NPROC_ACT" -gt 1 ]; then
    torchrun --nproc_per_node="$NPROC_ACT" --master_port="$MASTER_PORT_ACT" collect_activations.py \
      --cache_dir "$CACHE_DIR" \
      --model "$MODEL1" \
      "${COLLECT_TOKENIZER1_ARGS[@]}" \
      "${SPECIAL_TOKEN_ARGS[@]}" \
      "${SPECIAL_SPAN_ARGS[@]}" \
      --output_dir "$ACT_DIR1" \
      --best_buddies_path "${CORR_OUTPUT_DIR}/best_buddies.json" --which_model 1 \
      --layers all \
      --seq_length "$SEQ_LENGTH" --batch_size "$BATCH_SIZE" --dtype "$DTYPE" \
      --top_k "$TOP_K_ACTIVATIONS" --per_sample_top "$PER_SAMPLE_TOP" --context_size "$CONTEXT_SIZE" \
      --buddy_top_pairs "$BUDDY_TOP_PAIRS" \
      "${REQUIRE_CHAR_SPAN_ARGS[@]}"
  else
    python collect_activations.py \
      --cache_dir "$CACHE_DIR" \
      --model "$MODEL1" \
      "${COLLECT_TOKENIZER1_ARGS[@]}" \
      "${SPECIAL_TOKEN_ARGS[@]}" \
      "${SPECIAL_SPAN_ARGS[@]}" \
      --output_dir "$ACT_DIR1" \
      --best_buddies_path "${CORR_OUTPUT_DIR}/best_buddies.json" --which_model 1 \
      --layers all \
      --seq_length "$SEQ_LENGTH" --batch_size "$BATCH_SIZE" --dtype "$DTYPE" \
      --top_k "$TOP_K_ACTIVATIONS" --per_sample_top "$PER_SAMPLE_TOP" --context_size "$CONTEXT_SIZE" \
      --buddy_top_pairs "$BUDDY_TOP_PAIRS" \
      "${REQUIRE_CHAR_SPAN_ARGS[@]}"
  fi

  echo ""
  echo "Model1 activations written to: $ACT_DIR1"
  echo ""
else
  echo "[Skipping Step 4: collect_activations.py]"
fi

# -----------------------------------------------------------------------------
# Step 5: compute cross activations for model2 (MULTI-GPU)
# -----------------------------------------------------------------------------
if [ "$START_STEP" -le 5 ]; then
  echo "=============================================="
  echo "Step 5: compute_cross_activations.py (MULTI-GPU)"
  echo "=============================================="

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

  FALLBACK_ARG=()
  if [ "$ALLOW_POSITION_FALLBACK" = true ]; then
    FALLBACK_ARG+=(--allow_position_fallback)
  fi

  if [ "$NPROC_CROSS" -gt 1 ]; then
    torchrun --nproc_per_node="$NPROC_CROSS" --master_port="$MASTER_PORT_CROSS" compute_cross_activations.py \
      --buddies_path "${CORR_OUTPUT_DIR}/best_buddies.json" \
      --act1_dir "$ACT_DIR1" \
      --model2 "$MODEL2" \
      "${CROSS_TOKENIZER2_ARGS[@]}" \
      "${SPECIAL_TOKEN_ARGS[@]}" \
      "${MAP_SPECIAL_TOKEN_ARGS[@]}" \
      --cache_dir "$CACHE_DIR" \
      --output_path "${OUTPUT_BASE}/cross_activations.json" \
      --dtype "$DTYPE" \
      --pool "$CROSS_POOL" \
      --context_size "$CONTEXT_SIZE" \
      --seq_length "$SEQ_LENGTH_MODEL2" \
      --max_pairs "$TOP_K_ROSETTA" \
      --max_examples "$TOP_K_ACTIVATIONS" \
      "${FALLBACK_ARG[@]}"
  else
    python compute_cross_activations.py \
      --buddies_path "${CORR_OUTPUT_DIR}/best_buddies.json" \
      --act1_dir "$ACT_DIR1" \
      --model2 "$MODEL2" \
      "${CROSS_TOKENIZER2_ARGS[@]}" \
      "${SPECIAL_TOKEN_ARGS[@]}" \
      "${MAP_SPECIAL_TOKEN_ARGS[@]}" \
      --cache_dir "$CACHE_DIR" \
      --output_path "${OUTPUT_BASE}/cross_activations.json" \
      --dtype "$DTYPE" \
      --pool "$CROSS_POOL" \
      --context_size "$CONTEXT_SIZE" \
      --seq_length "$SEQ_LENGTH_MODEL2" \
      --max_pairs "$TOP_K_ROSETTA" \
      --max_examples "$TOP_K_ACTIVATIONS" \
      "${FALLBACK_ARG[@]}"
  fi

  echo ""
  echo "Cross activations: ${OUTPUT_BASE}/cross_activations.json"
  echo ""
else
  echo "[Skipping Step 5: compute_cross_activations.py]"
fi

# -----------------------------------------------------------------------------
# Step 6: visualize (single process)
# -----------------------------------------------------------------------------
if [ "$START_STEP" -le 6 ]; then
  echo "=============================================="
  echo "Step 6: visualize.py"
  echo "=============================================="

  python visualize.py \
    --cross-activations "${OUTPUT_BASE}/cross_activations.json" \
    --num-anchors "$TOP_K_ROSETTA" \
    --output "${OUTPUT_BASE}/index.html"
else
  echo "[Skipping Step 6: visualize.py]"
fi

echo ""
echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo "  - Neighbors:     $CORR_OUTPUT_DIR/nn_*.pt"
echo "  - Best buddies:  $CORR_OUTPUT_DIR/best_buddies.json"
echo "  - Activations:   $ACT_DIR1/"
echo "  - Cross JSON:    ${OUTPUT_BASE}/cross_activations.json"
echo "  - HTML:          ${OUTPUT_BASE}/index.html"
echo ""
