#!/bin/bash
# Multi-model Rosetta anchors: Pythia-160M (anchor) vs GPT-2 (124M) and OPT-125M.
# Small models — the full pipeline (match -> best-buddies -> Rosetta anchors ->
# anchor activations -> per-model cross-activations -> HTML viewer) runs quickly
# on a single GPU. Dispatched through match.sh (3 models -> anchor pipeline).
set -euo pipefail

# DATASET defaults to the Pile val set at <repo>/pile/ via match.sh; override by
# pointing DATASET at another local Pile mirror.
export SPLIT="${SPLIT:-val}"
export NUM_SAMPLES="${NUM_SAMPLES:-10000000}"

# Single-GPU by default; bump for more parallelism on the distributed steps.
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

# Run the full pipeline through the viewer (these models are small enough).
export STOP_AFTER_ROSETTA="${STOP_AFTER_ROSETTA:-false}"

export K_NEIGHBORS="${K_NEIGHBORS:-1}"
# Sweep K for the buddy step (matching runs once; only step 2 is repeated, cheap).
# Keep SAVE_NEIGHBORS_TOPK >= max(K) so all K values are exact.
export K_NEIGHBORS_LIST="${K_NEIGHBORS_LIST:-1 2 5 10}"
export SAVE_NEIGHBORS_TOPK="${SAVE_NEIGHBORS_TOPK:-100}"
export SPAN_POOL="${SPAN_POOL:-mean}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export SEQ_LENGTH="${SEQ_LENGTH:-1024}"

ANCHOR_MODEL="${ANCHOR_MODEL:-EleutherAI/pythia-160m}"
OTHER1="${OTHER1:-gpt2}"
OTHER2="${OTHER2:-facebook/opt-125m}"

# Tokenizers: each model uses its own by default (leave TOKENIZER1/TOKENIZER2 unset).

export OUTPUT_BASE="${OUTPUT_BASE:-outputs_anchor/pythia160m_gpt2_opt125m}"
mkdir -p "$OUTPUT_BASE"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/../match.sh" "$ANCHOR_MODEL" "$OTHER1" "$OTHER2" 2>&1 \
  | tee "$OUTPUT_BASE/run.log"
