# Rosetta Neurons — Language

Find "Rosetta neurons" — neurons that are shared across language models — by computing correlation matrices on MLP activations across a shared dataset. Uses byte-level alignment to handle different tokenizers across models.

## Installation

### Python dependencies

From the repo root:

```bash
pip install -r requirements.txt
```

Or for just this pipeline: `pip install -r language/requirements.txt`. Needs `torch>=2.0.0`, `transformers>=4.30.0`, `sentencepiece`, `datasets`, `matplotlib`, `numpy`, `tqdm`.

### Dataset

The default `DATASET` is the **Pile validation set**, expected at the top of your clone of this repo. Download it from [this Google Drive folder](https://drive.google.com/drive/folders/1klmBdOi5hh5C508vbMFmpizqHgUS3zmR?usp=sharing) and place the file so your tree looks like:

```
rosetta-neuron-scaling/        <- wherever you cloned this repo
├── language/
├── vision/
├── common/
└── pile/
    └── val.jsonl.zst           <- ~450 MB
```

The shell scripts auto-discover this via a path computed from their own runtime location (`$REPO_ROOT/../pile`), so this works no matter where you clone the repo — `~/code/rosetta-neurons-scaling/`, `/opt/rosetta-neuron-scaling/`, anywhere.

To use a different Pile mirror, override via env var:

```bash
DATASET=/data/pile-full SPLIT=train bash match.sh A B
```

This code matches only on Pile-shaped corpora (a directory containing `val.jsonl.zst` or similar). The matcher needs the per-subset prior statistics the Pile sampler computes, so plain HuggingFace text datasets aren't supported as a drop-in replacement.

## Core concept

The system aligns activations across models with different tokenizers by mapping everything to UTF-8 byte positions. **Canonical spans** are defined by the intersection of token boundaries in both models, and correlations are computed by pooling activations within each span (`mean` / `max` / `median`). A **best-buddy** pair is a mutual top-K nearest neighbor between neurons in two models by correlation. A **Rosetta neuron** is an anchor-model neuron that has best-buddy matches in *all* models in a group.

For the deep dive into how `match_lm.py` builds the cache, computes stats, and produces per-layer neighbors, see [`docs/match_lm_pipeline.md`](docs/match_lm_pipeline.md).

## Pipeline scripts

### `match.sh` — unified entry point (recommended)

Just list the models. The dispatcher routes:

- **2 models** → pairwise pipeline ([`scripts/run_pipeline.sh`](scripts/run_pipeline.sh))
- **3+ models** → multi-model anchor pipeline ([`scripts/run_anchor_pipeline.sh`](scripts/run_anchor_pipeline.sh)); the first model is the anchor

```bash
# pairwise
bash match.sh EleutherAI/pythia-1.4b facebook/opt-1.3b

# multi-model with Pythia-6.9B as anchor
bash match.sh EleutherAI/pythia-6.9b facebook/opt-6.7b Qwen/Qwen2.5-7B

# tighten memory + go multi-GPU
NPROC_PER_NODE=8 BATCH_SIZE=2 B_BLOCK=1 \
  bash match.sh EleutherAI/pythia-12b facebook/opt-13b Qwen/Qwen2.5-14B

# quick approximation — only correlate each A-layer against its 6 nearest
# B-layers by normalized depth. ~10x speedup at step 1; drops far away cross-depth
# buddies.
NPROC_PER_NODE=8 DEPTH_NEIGHBORS=6 \
  bash match.sh EleutherAI/pythia-12b Qwen/Qwen2.5-14B
```

Knobs are passed as env vars and forwarded to whichever underlying pipeline gets selected:

| Variable | Default | Purpose |
|---|---|---|
| `NUM_SAMPLES` | `10000000` | Total token budget over the Pile val corpus (~10M tokens — a reasonable default for meaningful correlations).|
| `NPROC_PER_NODE` | `1` | GPU count for distributed steps (uses `torchrun`) |
| `BATCH_SIZE` | `8` | Model forward batch size — reduce if OOM |
| `B_BLOCK` | `2` | Correlation B-layer block size — reduce if OOM |
| `USE_FSDP` | `false` | FSDP-shard both models in the matching step (step 1). Needed for 30B+ where two model copies don't fit per rank. Steps 4–5 always load one un-sharded model per rank, so you still need a per-rank GPU large enough for *one* model. |
| `K_NEIGHBORS` | `1` | K for mutual top-K best-buddies. Default `1` = strict mutual nearest neighbor (each side's single highest-correlated partner). Set higher (e.g. `5`) to loosen and surface more pairs. |
| `K_NEIGHBORS_LIST` | *(unset)* | Space-separated list of additional K values to sweep at the best-buddies step (`"1 2 5 10"`). Writes `best_buddies_kN.json` for each; the primary `K_NEIGHBORS` still drives the downstream activation + viewer steps. Cheap — step 1 isn't repeated. |
| `TOP_K_ROSETTA` | `1000` | **Viewer cap.** How many top anchors (sorted by avg correlation) the HTML renders. The on-disk `rosetta_anchors.json` always contains *every* intersection anchor; this knob only limits the HTML for browseability. Raise to see more in the viewer. |
| `DEPTH_NEIGHBORS` | *(unset, full grid)* | Approximation knob — only correlate each A-side layer against its `K` nearest B-side layers (by normalized depth) instead of all of them. Cuts step-1 work from `O(L_A·L_B)` to `O(L_A·K)`. Typical `K=4–8` gives a ~10× speedup with negligible loss in same-depth buddies; will miss long-range cross-depth pairs. Set when you want a quick approximation; leave unset for the exhaustive run. |
| `SEQ_LENGTH` | `1024` | Sequence length |
| `DTYPE` | `bfloat16` | Compute dtype |
| `SPAN_POOL` | `mean` | Span pooling: `mean` / `max` / `median` |
| `DATASET` | `<repo>/pile/` | Path to the Pile validation set (default — see [Installation](#installation)). Override with another local Pile mirror. |
| `SPLIT` | `val` | Dataset split |
| `OUTPUT_BASE` | `./outputs_cross` (pairwise) or `./outputs_anchor` (multi-model) | Output root |
| `STOP_AFTER_ROSETTA` | `false` | Stop after best-buddies / Rosetta-anchor computation (skip activation collection + viewer) |

After matching, the pipeline writes a single HTML viewer at `${OUTPUT_BASE}/index.html` that shows the top-activating sequences for each Rosetta neuron across all selected target models.

### `visualize.py` — render the HTML viewer (also standalone)

The viewer step (step 6) is also runnable on its own against existing outputs — useful for re-rendering with a different `TOP_K_ROSETTA` cap without rerunning the matching pipeline:

```bash
# Multi-model anchor mode — reads the manifest written by run_anchor_pipeline.sh
python visualize.py \
  --manifest outputs_anchor/rosetta/manifest.json \
  --num-anchors 1000 \
  --output outputs_anchor/index.html

# Pairwise mode — reads cross_activations.json from run_pipeline.sh
python visualize.py \
  --cross-activations outputs_cross/cross_activations.json \
  --num-anchors 1000 \
  --output outputs_cross/index.html
```

The viewer is a single self-contained HTML page with a sidebar (filter by avg correlation, anchor layer range, or search `L8`/`N859`/`8/859`) and a main panel showing per-anchor top-activating example contexts across all selected models side-by-side.

The underlying pipelines are also directly invokable if you need finer control (e.g. resuming from a specific step) — see below.

### `scripts/run_pipeline.sh` — two-model pipeline (direct)

End-to-end matching for a single pair of models: correlation → best buddies → activation cache → top-activating examples for model 1 → cross-activations on model 2 → HTML viewer.

```bash
bash scripts/run_pipeline.sh MODEL1 MODEL2 NUM_SAMPLES [START_STEP]
```

| Arg | Default | Description |
|-----|---------|-------------|
| `MODEL1` | `EleutherAI/pythia-1.4b` | First model (HuggingFace ID or local path) |
| `MODEL2` | `facebook/opt-1.3b` | Second model |
| `NUM_SAMPLES` | `10000000` | Total token budget over the Pile val corpus (~10M tokens) |
| `START_STEP` | `4` | Step to resume from (1–6) |

**Steps:**

1. `match_lm.py` — compute pairwise neuron correlations across layers
2. `find_best_buddies.py` — find mutual top-K neighbor pairs
3. Build activation cache from dataset
4. `collect_activations.py` — collect top-activating examples for model 1 neurons
5. `compute_cross_activations.py` — compute model 2 activations on model 1's top examples
6. `visualize.py` — generate HTML viewer (top-activating sequences per buddy pair)

**Environment variables (selected):**

| Variable | Default | Description |
|----------|---------|-------------|
| `DATASET` | `<repo>/pile/` | Path to the Pile validation set (default — see [Installation](#installation)) or another local Pile mirror |
| `SPLIT` | `val` | Dataset split |
| `PILE_SUBSETS` | *(empty)* | Comma-separated Pile subset names to restrict to |
| `NPROC_PER_NODE` | `1` | Default GPU count for all distributed steps |
| `NPROC_MATCH` / `NPROC_ACT` / `NPROC_CROSS` | `$NPROC_PER_NODE` | Per-step GPU overrides |
| `K_NEIGHBORS` | `1` | K for mutual top-K best-buddies. Default = strict mutual top-1. |
| `K_NEIGHBORS_LIST` | *(unset)* | Sweep extra K values (e.g. `"1 2 5 10"`); writes a `best_buddies_kN.json` per K. |
| `TOP_K_ROSETTA` | `1000` | Viewer cap on how many top anchors the HTML renders (the JSON on disk always contains all of them). |
| `DEPTH_NEIGHBORS` | *(unset, full grid)* | Approximation: correlate each A-layer against its K nearest B-layers only (typical K=4–8 gives ~10× step-1 speedup). |
| `TOKENIZER1` / `TOKENIZER2` | *(empty)* | Override tokenizers |
| `DTYPE` | `bfloat16` | Compute dtype |
| `SPAN_POOL` | `mean` | Span pooling: `mean`, `max`, `median` |

**Examples:**
EleutherAI/pythia-1.4b facebook/opt-1.3b
```bash
# Basic two-model comparison
bash scripts/run_pipeline.sh EleutherAI/pythia-1.4b facebook/opt-1.3b 10000

# Multi-GPU
NPROC_PER_NODE=4 bash scripts/run_pipeline.sh MODEL1 MODEL2 50000

# Resume from step 4
bash scripts/run_pipeline.sh MODEL1 MODEL2 10000 4
```

### `scripts/run_anchor_pipeline.sh` — multi-model anchor pipeline

Extends the two-model pipeline to 3+ models using an **anchor model**: pairwise correlations between the anchor and each other model, then Rosetta anchor neurons (best-buddies in *every* other model).

```bash
bash scripts/run_anchor_pipeline.sh ANCHOR_MODEL OTHER_MODEL1 OTHER_MODEL2 [OTHER_MODEL3 ...]
```

If only one other model is provided, this delegates to `run_pipeline.sh`.

**Steps** (parallel structure to the two-model pipeline):

1. `match_lm.py` per anchor-vs-model pair
2. `find_best_buddies.py` per pair
3. Build shared activation cache
4. `compute_rosetta_anchors.py` — intersect best-buddies across all pairs; collect anchor activations
5. `compute_cross_activations.py` per other model
6. `visualize.py` — multi-model HTML viewer (top-activating sequences per Rosetta anchor across all target models)

**Environment variables (selected):**

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_SAMPLES` | `10000000` | Total token budget over the Pile val corpus |
| `START_STEP` | `1` (or `3` if single other model) | Step to resume from |
| `TOP_K_ROSETTA` | `1000` | Viewer cap on how many top anchors the HTML renders (the JSON on disk always contains all of them) |
| `OUTPUT_BASE` | `./outputs_anchor` | Output directory |
| `STOP_AFTER_ROSETTA` | `false` | Stop after anchor computation (skip activation collection + viewer) |

**Examples:**

```bash
# Three-model comparison with Pythia anchor
bash scripts/run_anchor_pipeline.sh EleutherAI/pythia-1.4b gpt2-xl facebook/opt-1.3b

# Multi-GPU with custom output dir
NPROC_PER_NODE=4 OUTPUT_BASE=./my_run \
  bash scripts/run_anchor_pipeline.sh EleutherAI/pythia-6.9b facebook/opt-6.7b Qwen/Qwen2.5-7B
```

### Worked examples

[`scripts/example_pythia160m_gpt2_opt125m.sh`](scripts/example_pythia160m_gpt2_opt125m.sh) (multi-model anchors: Pythia-160M anchor vs GPT-2 and OPT-125M) and [`scripts/example_pythia160m_gpt2.sh`](scripts/example_pythia160m_gpt2.sh) (pairwise Pythia-160M vs GPT-2) These are example model sets, GPU counts, and knobs as starting points you can copy and modify.

## Architecture

### Pipeline flow

**Two-model:**
```
match_lm.py → find_best_buddies.py → collect_activations.py → compute_cross_activations.py → visualize.py
```

**Multi-model (anchor):**
```
match_lm.py (per pair) → find_best_buddies.py (per pair) → compute_rosetta_anchors.py
  → collect_activations.py → compute_cross_activations.py (per model) → visualize.py
```

### Byte-level alignment

Models with different tokenizers produce different token boundaries for the same text. The system maps every token activation to the UTF-8 byte range it covers, then defines **canonical spans** from the intersection of token boundaries and pools activations within each span before computing correlations.

### Distributed execution

All compute-heavy scripts support `torchrun` multi-GPU execution:

- `match_lm.py` — shards data across ranks in the stats pass; shards B-layer blocks in the correlation pass. With `USE_FSDP=true`, also FSDP-shards both models' parameters across ranks (`FULL_SHARD`, one unit per transformer block) — essential at 30B+ where two un-sharded model copies don't fit.
- `collect_activations.py` — shards documents by `doc_index % world_size`, all-reduces stats, rank 0 merges. Each rank loads one un-sharded model copy.
- `compute_cross_activations.py` — shards buddy pairs across ranks, rank 0 merges. Each rank loads one un-sharded model copy.

### Supported model architectures

MLP post-activation hook detection is automatic for:

- **Pythia / GPT-NeoX**: `gpt_neox.layers.{i}.mlp.dense_4h_to_h`
- **GPT-2**: `transformer.h.{i}.mlp.c_proj`
- **OPT**: `model.decoder.layers.{i}.fc2`
- **Qwen**: `model.layers.{i}.mlp.down_proj`
