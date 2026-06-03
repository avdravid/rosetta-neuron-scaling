#!/bin/bash
# Example: render an HTML viewer for previously-computed Rosetta anchors.
# Loads the anchor list, re-runs the pMF generator + the discriminative ViTs
# on a sample of images, and writes the top-activating images per anchor into a
# searchable single-page viewer.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

python visualize.py \
  --anchors-dir ./pmf_rosetta_anchors_vitb16 \
  --run dino=./pmf_dinovitb16_50000 \
  --run clip=./pmf_clipvitb16_50000 \
  --pmf-repo ./pMF \
  --pmf-hf-repo Lyy0725/pMF \
  --pmf-ckpt-file pMF-B-16.pt \
  --disc-repo dino=./dinov3 \
  --disc-weights dino=./dinov3_checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --disc-pretrained clip=openai \
  --num-anchors 200 \
  --top-images 4 \
  --max-search-images 5000 \
  --sample-mode random \
  --sample-seed 32 \
  --batch-size 8 \
  --select-model pmf \
  --select-stat max
