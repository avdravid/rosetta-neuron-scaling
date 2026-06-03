#!/bin/bash
# Example: pairwise pMF (generative) vs OpenCLIP (discriminative) matching.
# Generates 1 image at the fixed ImageNet label 90, computes per-pixel
# correlations between pMF MLP units and OpenCLIP ViT-B/16 patch features,
# and writes neighbors + best_buddies into ./test_pmf_openclip/.
#
# Requires the pMF repo cloned alongside (see vision/third_party.md).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

python match.py --gen-family pmf \
  --pmf-repo pMF \
  --pmf-model pmfDiT_B_16 \
  --pmf-hf-repo Lyy0725/pMF \
  --pmf-ckpt-file pMF-B-16.pt \
  --disc-family openclip \
  --disc-arch ViT-B-16 \
  --disc-pretrained openai \
  --disc-input-size 224 \
  --num-images 1 \
  --batch-size 1 \
  --label-mode fixed \
  --fixed-label 90 \
  --seed 32 \
  --save-dir ./test_pmf_openclip \
  --topk 1 \
  --disc-chunk-size 4 \
  --save-generated-images
