#!/bin/bash
# Example: aggregate Rosetta anchors from two pairwise matching runs.
# Each --results-dirs entry is the output dir from a prior match_*.py run
# (must contain best_buddies.json + run_metadata.json). Anchor-side neurons
# present in *every* run become "Rosetta anchors", ranked by avg correlation.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

python compute_rosetta_anchors.py \
  --results-dirs ./flux_pixio_run ./flux_openclip_run \
  --labels pixio openclip \
  --anchor-prefix flux \
  --output-dir ./anchors_flux_pixio_openclip

# Alternative: pMF anchors across two discriminative ViTs.
# python compute_rosetta_anchors.py \
#   --results-dirs ./pmf_dinovitb16_50000 ./pmf_clipvitb16_50000 \
#   --labels dino clip \
#   --anchor-prefix pmf \
#   --output-dir ./pmf_rosetta_anchors_vitb16
