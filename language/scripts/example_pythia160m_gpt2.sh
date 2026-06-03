#!/bin/bash
# Pairwise Rosetta match: Pythia-160M vs GPT-2 (124M).
# Small models — the full pairwise pipeline (correlations -> best-buddies ->
# activations -> cross-activations -> HTML viewer) runs quickly on a single GPU.
# Dispatched through match.sh (2 models -> pairwise pipeline).
set -euo pipefail

# DATASET defaults to the Pile val set at <repo>/pile/ via match.sh; override by
# pointing DATASET at another local Pile mirror.
export SPLIT="${SPLIT:-val}"
export NUM_SAMPLES="${NUM_SAMPLES:-10000000}"

# Single-GPU by default; bump for more parallelism on the distributed steps.
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

export K_NEIGHBORS="${K_NEIGHBORS:-1}"
# Sweep K for the buddy step (matching runs once; only step 2 is repeated, cheap).
# Keep SAVE_NEIGHBORS_TOPK >= max(K) so all K values are exact.
export K_NEIGHBORS_LIST="${K_NEIGHBORS_LIST:-1 2 5 10}"
export SAVE_NEIGHBORS_TOPK="${SAVE_NEIGHBORS_TOPK:-100}"
export SPAN_POOL="${SPAN_POOL:-mean}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export SEQ_LENGTH="${SEQ_LENGTH:-1024}"

MODEL1="${MODEL1:-EleutherAI/pythia-160m}"
MODEL2="${MODEL2:-gpt2}"

# Tokenizers: each model uses its own by default (leave TOKENIZER1/TOKENIZER2 unset).

export OUTPUT_BASE="${OUTPUT_BASE:-outputs_cross/pythia160m_gpt2}"
mkdir -p "$OUTPUT_BASE"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/../match.sh" "$MODEL1" "$MODEL2" 2>&1 \
  | tee "$OUTPUT_BASE/run.log"
