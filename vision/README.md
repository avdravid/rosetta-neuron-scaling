# Rosetta Neurons — Vision

Find "Rosetta neurons" — neurons that are shared across vision models — by computing correlation matrices on MLP / FFN activations from a generative model and one or more discriminative ViTs. Generative and discriminative activations are aligned on a canonical spatial patch grid, then mutual top-K neighbors and Rosetta anchors are computed analogously to the [language pipeline](../language/).

## Installation

From the repo root:

```bash
pip install -r requirements.txt
```

Or for just this pipeline: `pip install -r vision/requirements.txt`. Needs `torch`, `torchvision`, `open_clip_torch`, `timm`, `diffusers`, `huggingface_hub`, `numpy`, `pillow`, `matplotlib`, `einops`, `omegaconf`.

Several entry points additionally need external model repos (pMF, DINOv2/v3, PixIO, MAE, Sana) cloned alongside this tree — see [`third_party.md`](third_party.md).

## Core concept

For each model pair we:

1. **Generate** a fixed batch of images with the generative model (pMF, FLUX, Sana, …) using a deterministic seed schedule.
2. **Capture** MLP / FFN post-activation feature maps from both the generative and discriminative models on those same images.
3. **Resample** both activation maps to a canonical patch grid (typically the generative model's native grid, e.g. 16 × 16 for pMF-B/16; 14 × 14 for OpenCLIP ViT-B/16).
4. **Correlate** every generative-side channel with every discriminative-side channel across patches × images and save mutual top-K neighbors → `best_buddies.json`.
5. **Aggregate** best-buddies across multiple discriminative models to produce **Rosetta anchors**: generative-side neurons that have at least one mutual match in *every* run.

The output of step 4 is a "matching run directory" (containing `best_buddies.json`, `run_metadata.json`, and per-layer correlation neighbor files). Step 5 consumes one such directory per discriminative model and outputs a `rosetta_anchors.json`.

## Pipeline scripts

### `match.py` — unified entry point

[`match.py`](match.py) is a thin dispatcher that forwards to the right family-specific matching script based on `--gen-family`:

| `--gen-family` | Generator | Discriminative towers |
|---|---|---|
| `pmf` | pMF one-step generator | DINOv2, DINOv3, OpenCLIP, MAE, PixIO, InternViT |
| `flux` | FLUX.2-klein-4B | OpenCLIP, PixIO, InternViT, DINOv3 |
| `sana` | Sana (large DiT) | OpenCLIP, PixIO |

```bash
# pMF + OpenCLIP, single GPU
python match.py --gen-family pmf \
    --pmf-repo pMF --pmf-model pmfDiT_B_16 \
    --pmf-hf-repo Lyy0725/pMF --pmf-ckpt-file pMF-B-16.pt \
    --disc-family openclip --disc-arch ViT-B-16 --disc-pretrained openai \
    --num-images 1000 --save-dir ./pmf_openclip

# FLUX + DINOv3, distributed
torchrun --standalone --nproc_per_node=8 match.py --gen-family flux \
    --disc-family dinov3 --disc-arch facebook/dinov3-vitb16-pretrain-lvd1689m \
    --num-images 1000 --batch-size 2 --save-dir ./flux_dinov3
```

To see the full CLI for a given family, run `python match.py --gen-family <family> --help`. Each family-specific script ([`match_pmf_vit_multigpu.py`](match_pmf_vit_multigpu.py), [`match_flux.py`](match_flux.py), [`match_large_dit.py`](match_large_dit.py)) is also independently invokable if you prefer.

### `scripts/example_match.sh` — worked example

Generates images with pMF-B/16 and matches its MLP units against an OpenCLIP ViT-B/16 vision tower. Writes neighbors + best-buddies to `./test_pmf_openclip/`.

```bash
bash scripts/example_match.sh
```

### `scripts/example_anchors.sh` — aggregate Rosetta anchors

Consumes two or more matching-run directories and intersects their best-buddies on the generative side to produce ranked Rosetta anchors.

```bash
bash scripts/example_anchors.sh
```

The underlying script is [`compute_rosetta_anchors.py`](compute_rosetta_anchors.py). The `--anchor-prefix` flag selects which side of the pair is the anchor (e.g. `pmf`, `flux`, `sana`) — that prefix must match the key naming in each input run's `best_buddies.json`.

### `visualize.py` — unified HTML viewer

[`visualize.py`](visualize.py) re-runs the generative model + the discriminative tower(s) on a sample of images and writes a self-contained, dark-theme single-page viewer of the top-activating image tiles per matched pMF neuron. The page has a searchable, filterable sidebar (search by `L<idx>` pMF layer, `N<neuron>`, or disc-model name; filter by min avg correlation and layer range; paginated) and a main panel where each example's source image plus per-model heatmap/overlay tiles sit in one responsive grid row that reflows to fit the window/zoom width. It accepts two input modes:

```bash
# Mode A — multi-model Rosetta anchors (intersection over N pairwise runs)
python visualize.py \
    --anchors-dir ./pmf_rosetta_anchors_vitb16 \
    --run dino=./pmf_dinovitb16_50000 \
    --run clip=./pmf_clipvitb16_50000 \
    --pmf-repo pMF --pmf-hf-repo Lyy0725/pMF --pmf-ckpt-file pMF-B-16.pt \
    --num-anchors 200 --top-images 4

# Mode B — single pairwise matching run (no anchor aggregation needed)
python visualize.py \
    --results-dir ./pmf_dinovitb16_50000 \
    --pmf-repo pMF --pmf-hf-repo Lyy0725/pMF --pmf-ckpt-file pMF-B-16.pt \
    --num-anchors 24 --top-images 4
```

In Mode B, each best-buddy pair is treated as a degenerate one-match anchor and rendered with the same HTML template. [`scripts/example_visualize_anchors.sh`](scripts/example_visualize_anchors.sh) demonstrates Mode A.

## Analysis utilities

- [`compute_rosetta_anchors.py`](compute_rosetta_anchors.py) — aggregate anchors from N runs
- [`count_best_buddies.py`](count_best_buddies.py) — summary statistics over a matching run

## Pipeline flow

```
match.py / match_*.py (per generative ↔ discriminative pair)
   → best_buddies.json + neighbors
       → visualize.py --results-dir DIR              (single pairwise run)
       → compute_rosetta_anchors.py (intersect over N pairs)
           → rosetta_anchors.json
               → visualize.py --anchors-dir DIR --run ... (multi-model)
```
