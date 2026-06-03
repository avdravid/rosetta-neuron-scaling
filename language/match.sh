#!/usr/bin/env bash
# match.sh — unified language-matching entry point.
#
# Just list the models. The script dispatches to the right pipeline:
#   2 models     -> scripts/run_pipeline.sh        (pairwise)
#   3+ models    -> scripts/run_anchor_pipeline.sh (multi-model, first is anchor)
#
# Usage:
#   bash match.sh MODEL1 MODEL2 [MODEL3 ...]
#
# Examples:
#   # pairwise
#   bash match.sh EleutherAI/pythia-1.4b facebook/opt-1.3b
#
#   # multi-model with Pythia-6.9B as anchor
#   bash match.sh EleutherAI/pythia-6.9b facebook/opt-6.7b Qwen/Qwen2.5-7B
#
#   # tighten memory for big models
#   NPROC_PER_NODE=8 BATCH_SIZE=2 B_BLOCK=1 \
#     bash match.sh EleutherAI/pythia-12b facebook/opt-13b Qwen/Qwen2.5-14B
#
# Env-var knobs (forwarded to the underlying pipeline):
#   NUM_SAMPLES=10000000       total token budget over the Pile val corpus
#                              (~10M tokens; lower this for a smoke test)
#   NPROC_PER_NODE=1           GPU count for distributed steps (uses torchrun)
#   BATCH_SIZE=8               model forward batch size (reduce if OOM)
#   B_BLOCK=2                  correlation B-layer block size (reduce if OOM)
#   K_NEIGHBORS=1              K for mutual top-K best-buddies. Default 1 =
#                              strict mutual nearest neighbor. Set higher (e.g.
#                              5) to loosen and get more pairs.
#   TOP_K_ROSETTA=1000         viewer cap — how many top anchors (by avg
#                              correlation) to render in index.html. The full
#                              rosetta_anchors.json on disk always contains
#                              every intersection anchor; this only limits
#                              the HTML for browseability. Raise to see more.
#   K_NEIGHBORS_LIST=<unset>   space-separated list of extra K values to sweep
#                              in find_best_buddies (writes best_buddies_kN.json
#                              for each; primary K_NEIGHBORS drives the rest).
#   USE_FSDP=false             FSDP-shard both models in the matching step (set
#                              true for 30B+ where two models won't fit per rank)
#   DEPTH_NEIGHBORS=<unset>    only correlate each A-layer against its K nearest
#                              B-layers (by normalized depth) instead of all of
#                              them. Major step-1 speedup; the result is an
#                              approximation that drops cross-depth buddies.
#                              Typical values: 4-8. Unset = full grid (exact).
#   SEQ_LENGTH=1024            sequence length
#   DTYPE=bfloat16             compute dtype
#   SPAN_POOL=mean             span pooling: mean | max | median
#   DATASET=<repo>/pile        the Pile val set placed at language/../pile/
#                              (default; override with another local Pile mirror)
#   SPLIT=val
#   OUTPUT_BASE=./outputs_cross   (pairwise)   or  ./outputs_anchor  (multi-model)
#   STOP_AFTER_ROSETTA=false   stop after best-buddies / Rosetta anchors (skip viewer)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -lt 2 ]; then
  sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
  exit 1
fi

NUM_SAMPLES="${NUM_SAMPLES:-10000000}"

if [ "$#" -eq 2 ]; then
  echo "[match.sh] 2 models -> pairwise pipeline"
  echo "[match.sh]   MODEL1 = $1"
  echo "[match.sh]   MODEL2 = $2"
  echo "[match.sh]   NUM_SAMPLES = $NUM_SAMPLES   NPROC_PER_NODE = ${NPROC_PER_NODE:-1}   BATCH_SIZE = ${BATCH_SIZE:-8}   B_BLOCK = ${B_BLOCK:-2}"
  exec bash "$SCRIPT_DIR/scripts/run_pipeline.sh" "$1" "$2" "$NUM_SAMPLES" 1
else
  ANCHOR="$1"
  shift
  echo "[match.sh] $# other models -> anchor pipeline"
  echo "[match.sh]   ANCHOR = $ANCHOR"
  echo "[match.sh]   OTHERS = $*"
  echo "[match.sh]   NUM_SAMPLES = $NUM_SAMPLES   NPROC_PER_NODE = ${NPROC_PER_NODE:-1}   BATCH_SIZE = ${BATCH_SIZE:-8}   B_BLOCK = ${B_BLOCK:-2}"
  export NUM_SAMPLES
  exec bash "$SCRIPT_DIR/scripts/run_anchor_pipeline.sh" "$ANCHOR" "$@"
fi
