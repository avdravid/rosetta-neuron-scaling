# Match LM Pipeline Deep Dive

This document explains exactly what `match_lm.py` (and its core implementation in `find_pairs/match_pipeline.py`) does, end to end. The goal is to produce neuron-neuron correlations across two language models despite different tokenizers, and save per-layer top-K neighbors for best-buddy matching.

## High-Level Flow

1. Build or load a byte-aligned token cache for model1 and model2.
2. Pass 1: compute per-neuron mean and std for both models using canonical spans (unweighted).
3. Pass 2: compute correlation accumulators between every A-layer and B-layer block (unweighted).
4. Save per-layer top-K neighbor lists (or full correlation matrices if `--save_neighbors` is not set).
5. Save metadata about the run.

## Inputs and Outputs

**Inputs**
- Model IDs: `--model1`, `--model2`
- Optional tokenizer overrides: `--tokenizer1`, `--tokenizer2`
- Dataset selection:
  - `--dataset` (Pile-shaped directory containing `val.jsonl.zst`) plus `--split`
  - Pile knobs: `--pile_subsets`, `--pile_ratio_by`, `--use_padding` (`true` keeps short docs and pads; `false` skips short docs)
- Sampling controls: `--num_samples` (total token budget), `--seq_length/--max_tokens`, `--max_bytes`
- Compute controls: `--batch_size`, `--dtype`, `--b_block`, `--seed`, `--span_pool`, `--allow_tf32`
- Distributed sampling: `--shard_pile_subsets`
- Cache controls: `--tokenize_batch`, `--token_cache`
- Output controls: `--save_neighbors`, `--top_k`, `--only_diagonal`
in
**Outputs (in `--save_dir`)**
- `stats_A_layer{i}.pt`, `stats_B_layer{j}.pt`: per-layer mean/std
- `nn_A_layer{i}_vs_B_layer{j}_top{K}.pt`: top-K neighbors for A neurons
- `nn_B_layer{j}_vs_A_layer{i}_top{K}.pt`: top-K neighbors for B neurons
- `corr_layer{i}_vs_layer{j}.pt`: full correlation matrices when `--save_neighbors` is not set
- `match_run_metadata.json`: run configuration and cache info

## Cache Construction and Byte Alignment

Different tokenizers produce different token boundaries. This pipeline aligns activations by mapping each token to a UTF-8 byte span:

1. Tokenize the same raw text with both tokenizers using fast tokenizers and `return_offsets_mapping=True`.
2. Convert character offsets to byte offsets for each token.
3. Compute the overlapping byte region between the two tokenizations (`aligned_len = min(coverage1, coverage2)`).
4. Store `input_ids`, `attention_mask`, `byte_offsets`, and `aligned_len` for both models.

The cache is stored as a single `.pt` file containing CPU tensors:
- `input_ids1`, `attention_mask1`, `byte_offsets1`
- `input_ids2`, `attention_mask2`, `byte_offsets2`
- `aligned_len`

The cache is shared by both the stats and correlation passes.

### Tokenizer Overrides

If `--tokenizer1` or `--tokenizer2` is provided, those tokenizers are used for cache building and alignment, while the models still load from `--model1` and `--model2`.

### Offset Fallbacks

If a fast tokenizer returns zero-length offsets for all non-padding tokens in a sample, the cache builder synthesizes monotonic unit-length spans for those tokens (and prints a warning). This keeps alignment logic working even when offsets are missing, but it is less faithful to true byte alignment.

### Canonical Spans (Boundary Intersection)

Canonical spans are defined by **shared token boundaries** between the two tokenizations:

1. Collect all token boundary byte positions (start and end) from model A.
2. Collect all token boundary byte positions from model B.
3. Intersect these boundary sets.
4. The canonical spans are the byte ranges between consecutive shared boundaries.

Each canonical span becomes a single sample for correlation after pooling within the span.

## Pass 1: Stats (Canonical Spans)

The stats pass computes per-neuron mean and std using **canonical spans** with unweighted averaging.

For each batch:
1. Build canonical spans from the **intersection of token boundaries** in A and B.
2. For each span:
   - Aggregate A activations over A tokens that overlap the span.
   - Aggregate B activations over B tokens that overlap the span.
   - Aggregation uses `--span_pool` (`mean`, `max`, or `median`).
3. Update per-layer Welford statistics for both models using those per-span activations.

Important properties:
- Each canonical span contributes exactly once (no byte length weighting).
- Both models are pooled into the same span unit before statistics.
- Distributed runs aggregate stats across ranks using weighted Welford with exact correction.

Outputs:
- `stats_A_layer{i}.pt` and `stats_B_layer{j}.pt`, each containing `mean` and `std`.

## Pass 2: Correlation (B-Block Sharding)

The correlation pass computes standardized dot products between A-layer activations and B-layer activations.

For each B-layer block (controlled by `--b_block`):
1. For each batch, compute standardized activations for all A layers and the B block.
2. For each canonical span:
   - Pool A tokens within the span, then standardize (using stats from Pass 1).
   - Pool B tokens within the span, then standardize.
3. Accumulate unweighted outer products for each A layer against the concatenated B-block activations.
4. At the end, divide by total sample count to yield correlations.

The output is either:
- Top-K neighbor lists per layer pair (`--save_neighbors`), or
- Full correlation matrices per layer pair.

## Best-Buddy Matching (Next Stage)

`find_best_buddies.py` reads the neighbor files and computes mutual top-K pairs across all layers. These are the "best buddies" used by downstream activation and visualization stages.

## Distributed Execution

- **Stats pass**: data is sharded by rank (each rank processes a subset of cached samples).
- **Correlation pass**: B-layer blocks are sharded across ranks; each rank writes its own neighbor files.
- **Cache build (Pile + distributed)**: each rank builds a cache shard and rank 0 merges them into the final cache file.

The final output is correct if all B-layer blocks are computed and written.

## Important Config Knobs

- `--num_samples`: total token budget over the Pile corpus.
- `--seq_length/--max_tokens`: truncation and padding length per model.
- `--use_padding`: for Pile, keep docs shorter than `seq_length` and pad them; if unset, short docs are skipped.
- `--max_bytes`: truncates text by UTF-8 bytes before tokenization.
- `--b_block`: number of B layers per correlation block (memory vs speed).
- `--top_k`: neighbors saved per neuron (must be >= best-buddy K).
- `--dtype`: compute dtype for activations and correlation accumulation.
- `--span_pool`: pooling within canonical spans (`mean`, `max`, `median`).
- `--only_diagonal`: only compute/save same-layer pairs (`i == j`).
- `--token_cache`: explicit cache path (skip hash-based default).
- `--tokenize_batch`: tokenizer batch size used when building caches.
- `--shard_pile_subsets`: distribute Pile subsets across ranks in distributed cache builds.

## Failure Modes and Checks

- If tokenizers do not provide offsets (non-fast tokenizer), cache build will fail.
- If there are no overlapping canonical spans, the stats/correlation pass raises a clear error.
- If a cache shard is empty in distributed runs, it is treated as empty and merged safely, but a fully empty cache raises an error.
- Use `match_run_metadata.json` to confirm the cache and tokenizer configuration of a run.
