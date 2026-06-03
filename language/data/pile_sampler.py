"""Pile sampling helpers."""
from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import json
import os
import random
import subprocess

from datasets import load_dataset
from tqdm import tqdm


PILE_SUBSETS: List[str] = [
    "Pile-CC",
    "PubMed Central",
    "Books3",
    "OpenWebText2",
    "ArXiv",
    "Github",
    "FreeLaw",
    "StackExchange",
    "USPTO Backgrounds",
    "PubMed Abstracts",
    "Gutenberg (PG-19)",
    "OpenSubtitles",
    "Wikipedia (en)",
    "DM Mathematics",
    "HackerNews",
    "Enron Emails",
    "NIH ExPorter",
    "CC-News",
    "OpenWebText",
    "Books1",
    "EuroParl",
    "Ubuntu IRC",
]


def is_pile_dataset_name(name: str) -> bool:
    name = name.strip()
    return name in {
        "pile",
        "EleutherAI/pile",
        "monology/pile-uncopyrighted",
        "monology/pile-uncopyrighted-parquet",
        "/datasets/pile/current",
    } or _is_local_pile_path(name)

def _pile_dataset_id(name: str) -> str:
    return "EleutherAI/pile" if name.strip() == "pile" else name.strip()

def _is_local_pile_path(name: str) -> bool:
    if not name:
        return False
    if os.path.isdir(name):
        return os.path.exists(os.path.join(name, "val.jsonl.zst"))
    return name.endswith(".jsonl.zst") and os.path.exists(name)

def _iter_jsonl_zst(path: str) -> Iterable[dict]:
    proc = subprocess.Popen(
        ["zstdcat", path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except Exception:
            continue
    proc.stdout.close()
    proc.wait()

def _iter_local_pile(root_or_file: str, split: str) -> Iterable[dict]:
    if os.path.isfile(root_or_file) and root_or_file.endswith(".jsonl.zst"):
        yield from _iter_jsonl_zst(root_or_file)
        return
    root = root_or_file
    if split in {"val", "valid", "validation"}:
        path = os.path.join(root, "val.jsonl.zst")
        yield from _iter_jsonl_zst(path)
        return
    if split in {"test", "testing"}:
        path = os.path.join(root, "test.jsonl.zst")
        yield from _iter_jsonl_zst(path)
        return
    if split == "train":
        train_dir = os.path.join(root, "train")
        files = sorted(
            f for f in os.listdir(train_dir) if f.endswith(".jsonl.zst")
        )
        for fname in files:
            yield from _iter_jsonl_zst(os.path.join(train_dir, fname))
        return
    raise ValueError(f"Unknown split: {split}")

def _shuffle_stream(stream: Iterable[dict], seed: int, buffer_size: int) -> Iterable[dict]:
    if buffer_size <= 1:
        yield from stream
        return
    rng = random.Random(seed)
    buf: List[dict] = []
    for item in stream:
        buf.append(item)
        if len(buf) >= buffer_size:
            idx = rng.randrange(len(buf))
            yield buf.pop(idx)
    while buf:
        idx = rng.randrange(len(buf))
        yield buf.pop(idx)


def _load_pile_stream(dataset_name: str, split: str, seed: int, buffer_size: int):
    dataset_id = _pile_dataset_id(dataset_name)
    if _is_local_pile_path(dataset_id):
        stream = _iter_local_pile(dataset_id, split)
        return _shuffle_stream(stream, seed, buffer_size)
    if dataset_id in {"pile", "EleutherAI/pile"}:
        try:
            ds = load_dataset(
                "EleutherAI/pile",
                split=split,
                streaming=True,
                trust_remote_code=True,
            )
        except RuntimeError as exc:
            msg = str(exc)
            if "Dataset scripts are no longer supported" in msg:
                fallback_id = "monology/pile-uncopyrighted"
                ds = load_dataset(
                    fallback_id,
                    split=split,
                    streaming=True,
                )
            else:
                raise
    else:
        ds = load_dataset(
            dataset_id,
            split=split,
            streaming=True,
        )
    ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
    return ds

def parse_pile_subsets(pile_subsets: Optional[str]) -> Optional[List[str]]:
    if not pile_subsets:
        return None
    parts = [p.strip() for p in pile_subsets.split(",")]
    return [p for p in parts if p]


def get_pile_subset_name(ex: dict) -> Optional[str]:
    meta = ex.get("meta") or {}
    return meta.get("pile_set_name") or ex.get("pile_set_name")


def truncate_utf8_to_max_bytes(text: str, max_bytes: Optional[int]) -> str:
    if max_bytes is None:
        return text
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text
    return raw[:max_bytes].decode("utf-8", errors="ignore")


def _count_tokens(tokenizer, text: str, max_tokens: Optional[int]) -> int:
    enc = tokenizer(
        text,
        add_special_tokens=False,
        truncation=(max_tokens is not None),
        max_length=max_tokens,
    )
    ids = enc.get("input_ids") or []
    return len(ids)


def _build_equal_token_quotas(
    total_token_budget: int,
    subsets: List[str],
) -> Dict[str, int]:
    if total_token_budget <= 0:
        raise ValueError("total_token_budget must be > 0")
    if not subsets:
        raise ValueError("subsets must be non-empty")
    base = total_token_budget // len(subsets)
    remainder = total_token_budget % len(subsets)
    quotas: Dict[str, int] = {}
    for i, subset in enumerate(subsets):
        quotas[subset] = base + (1 if i < remainder else 0)
    return quotas


def iter_pile_texts_equal_token_budget(
    tokenizer,
    total_token_budget: int,
    *,
    dataset_name: str = "EleutherAI/pile",
    split: str = "train",
    min_chars: int = 50,
    seed: int = 42,
    buffer_size: int = 10000,
    pile_subsets: Optional[List[str]] = None,
    text_field: str = "text",
    max_bytes: Optional[int] = None,
    max_tokens: Optional[int] = None,
) -> Iterator[Tuple[str, str, int]]:
    """
    Yield (text, subset, token_count) with equal token quotas per subset.

    If some subsets can't meet quota, a second pass redistributes the remaining
    budget across any subset.
    """
    subsets = pile_subsets or list(PILE_SUBSETS)
    quotas = _build_equal_token_quotas(total_token_budget, subsets)
    tokens_by_subset = {s: 0 for s in subsets}
    total_tokens = 0

    def stream(seed_offset: int) -> Iterable[dict]:
        return _load_pile_stream(dataset_name, split, seed + seed_offset, buffer_size)

    def maybe_yield(
        ex: dict,
        enforce_quota: bool,
    ) -> Optional[Tuple[str, str, int]]:
        nonlocal total_tokens
        subset = get_pile_subset_name(ex)
        if subset not in quotas:
            return None
        if enforce_quota and tokens_by_subset[subset] >= quotas[subset]:
            return None
        text = ex.get(text_field, "")
        if not isinstance(text, str) or len(text) < min_chars:
            return None
        text = truncate_utf8_to_max_bytes(text, max_bytes)
        token_count = _count_tokens(tokenizer, text, max_tokens)
        if token_count <= 0:
            return None
        tokens_by_subset[subset] += token_count
        total_tokens += token_count
        return text, subset, token_count

    for ex in stream(seed_offset=0):
        if total_tokens >= total_token_budget:
            break
        item = maybe_yield(ex, enforce_quota=True)
        if item is None:
            continue
        yield item

    if total_tokens < total_token_budget:
        for ex in stream(seed_offset=1):
            if total_tokens >= total_token_budget:
                break
            item = maybe_yield(ex, enforce_quota=False)
            if item is None:
                continue
            yield item


def _stats_path_for_tokenizer(base_dir: str, tokenizer_id: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in tokenizer_id)
    return os.path.join(base_dir, f"pile_val_stats_{safe}.json")

def resolve_pile_stats_path(
    dataset_name: str,
    tokenizer,
    *,
    stats_path: Optional[str] = None,
) -> str:
    dataset_id = _pile_dataset_id(dataset_name)
    if not _is_local_pile_path(dataset_id):
        raise ValueError("Proportional sampling requires a local Pile path for stats.")
    val_path = resolve_pile_val_path(dataset_name)
    if stats_path is None:
        default_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "rosetta")
        cache_dir = os.environ.get("PILE_STATS_CACHE_DIR", default_cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        stats_path = _stats_path_for_tokenizer(cache_dir, getattr(tokenizer, "name_or_path", "tokenizer"))
    return stats_path

def resolve_pile_val_path(dataset_name: str) -> str:
    dataset_id = _pile_dataset_id(dataset_name)
    if not _is_local_pile_path(dataset_id):
        raise ValueError("Proportional sampling requires a local Pile path for stats.")
    root = dataset_id if os.path.isdir(dataset_id) else os.path.dirname(dataset_id)
    val_path = os.path.join(root, "val.jsonl.zst")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Missing Pile val file: {val_path}")
    return val_path


def _compute_val_stats_local(
    val_path: str,
    tokenizer,
    *,
    batch_size: int = 128,
) -> Dict[str, object]:
    doc_counts: Dict[str, int] = {}
    token_counts: Dict[str, int] = {}
    total_docs = 0
    total_tokens = 0

    batch_texts: List[str] = []
    batch_subsets: List[str] = []

    def flush():
        nonlocal total_tokens
        if not batch_texts:
            return
        enc = tokenizer(
            batch_texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )
        lengths = [len(x) for x in enc.get("input_ids", [])]
        for subset, n_tok in zip(batch_subsets, lengths):
            token_counts[subset] = token_counts.get(subset, 0) + n_tok
            total_tokens += n_tok
        batch_texts.clear()
        batch_subsets.clear()

    pbar = tqdm(desc="Scanning val for subset token stats", unit="docs")
    for ex in _iter_jsonl_zst(val_path):
        meta = ex.get("meta") or {}
        subset = meta.get("pile_set_name") or "unknown"
        text = ex.get("text")
        if not isinstance(text, str):
            continue
        doc_counts[subset] = doc_counts.get(subset, 0) + 1
        total_docs += 1
        batch_texts.append(text)
        batch_subsets.append(subset)
        if len(batch_texts) >= batch_size:
            flush()
        pbar.update(1)
    flush()
    pbar.close()

    return {
        "val_path": val_path,
        "total_docs": total_docs,
        "total_tokens": total_tokens,
        "docs_by_subset": doc_counts,
        "tokens_by_subset": token_counts,
    }

def compute_val_stats_shard(
    val_path: str,
    tokenizer,
    *,
    rank: int,
    world_size: int,
    batch_size: int = 128,
) -> Dict[str, object]:
    if world_size <= 0:
        raise ValueError("world_size must be > 0")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must be in [0, world_size)")

    doc_counts: Dict[str, int] = {}
    token_counts: Dict[str, int] = {}
    total_docs = 0
    total_tokens = 0

    batch_texts: List[str] = []
    batch_subsets: List[str] = []

    def flush():
        nonlocal total_tokens
        if not batch_texts:
            return
        enc = tokenizer(
            batch_texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )
        lengths = [len(x) for x in enc.get("input_ids", [])]
        for subset, n_tok in zip(batch_subsets, lengths):
            token_counts[subset] = token_counts.get(subset, 0) + n_tok
            total_tokens += n_tok
        batch_texts.clear()
        batch_subsets.clear()

    pbar = tqdm(desc=f"Scanning val shard {rank}/{world_size}", unit="docs")
    for idx, ex in enumerate(_iter_jsonl_zst(val_path)):
        if (idx % world_size) != rank:
            continue
        meta = ex.get("meta") or {}
        subset = meta.get("pile_set_name") or "unknown"
        text = ex.get("text")
        if not isinstance(text, str):
            continue
        doc_counts[subset] = doc_counts.get(subset, 0) + 1
        total_docs += 1
        batch_texts.append(text)
        batch_subsets.append(subset)
        if len(batch_texts) >= batch_size:
            flush()
        pbar.update(1)
    flush()
    pbar.close()

    return {
        "val_path": val_path,
        "total_docs": total_docs,
        "total_tokens": total_tokens,
        "docs_by_subset": doc_counts,
        "tokens_by_subset": token_counts,
    }

def merge_val_stats(shard_paths: List[str], out_path: str) -> Dict[str, object]:
    merged_docs: Dict[str, int] = {}
    merged_tokens: Dict[str, int] = {}
    total_docs = 0
    total_tokens = 0
    val_path = None

    for path in shard_paths:
        with open(path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        val_path = val_path or stats.get("val_path")
        total_docs += int(stats.get("total_docs", 0))
        total_tokens += int(stats.get("total_tokens", 0))
        docs_by_subset = stats.get("docs_by_subset", {}) or {}
        tokens_by_subset = stats.get("tokens_by_subset", {}) or {}
        for k, v in docs_by_subset.items():
            merged_docs[k] = merged_docs.get(k, 0) + int(v)
        for k, v in tokens_by_subset.items():
            merged_tokens[k] = merged_tokens.get(k, 0) + int(v)

    merged = {
        "val_path": val_path,
        "total_docs": total_docs,
        "total_tokens": total_tokens,
        "docs_by_subset": merged_docs,
        "tokens_by_subset": merged_tokens,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    return merged


def _load_or_build_val_stats(
    dataset_name: str,
    tokenizer,
    *,
    stats_path: Optional[str] = None,
) -> Dict[str, object]:
    dataset_id = _pile_dataset_id(dataset_name)
    if not _is_local_pile_path(dataset_id):
        raise ValueError("Proportional sampling requires a local Pile path for stats.")
    root = dataset_id if os.path.isdir(dataset_id) else os.path.dirname(dataset_id)
    val_path = os.path.join(root, "val.jsonl.zst")
    if stats_path is None:
        stats_path = resolve_pile_stats_path(dataset_name, tokenizer, stats_path=stats_path)
    if os.path.exists(stats_path):
        print(f"[pile] Loading cached val stats: {stats_path}")
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        tokens_by_subset = stats.get("tokens_by_subset")
        total_tokens = stats.get("total_tokens", 0)
        if isinstance(tokens_by_subset, dict) and sum(tokens_by_subset.values()) > 0 and total_tokens:
            return stats
        print(f"[pile] Cached stats invalid or empty; rebuilding: {stats_path}")
    batch_size = int(os.environ.get("PILE_STATS_BATCH_SIZE", "128"))
    tok_name = getattr(tokenizer, "name_or_path", "tokenizer")
    print(f"[pile] Building val stats (one-time scan): {val_path}")
    print(f"[pile] Tokenizer: {tok_name} | batch_size={batch_size}")
    print(f"[pile] Cache will be saved to: {stats_path}")
    stats = _compute_val_stats_local(val_path, tokenizer, batch_size=batch_size)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats


def _filtered_token_stats_path(stats_path: str, context_length: int) -> str:
    return f"{stats_path}.ge{int(context_length)}tok.json"


def _load_or_build_filtered_token_stats(
    dataset_name: str,
    tokenizer,
    *,
    context_length: int,
    stats_path: Optional[str] = None,
) -> Dict[str, object]:
    """
    Build/load val token stats after filtering out docs with token length < context_length.
    """
    if context_length <= 0:
        raise ValueError("context_length must be > 0")
    base_stats_path = resolve_pile_stats_path(dataset_name, tokenizer, stats_path=stats_path)
    filtered_path = _filtered_token_stats_path(base_stats_path, context_length)

    if os.path.exists(filtered_path):
        with open(filtered_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        tokens_by_subset = stats.get("tokens_by_subset")
        total_tokens = int(stats.get("total_tokens", 0) or 0)
        if isinstance(tokens_by_subset, dict) and total_tokens > 0:
            return stats

    val_path = resolve_pile_val_path(dataset_name)
    batch_size = int(os.environ.get("PILE_STATS_BATCH_SIZE", "128"))
    print(
        "[pile] Building filtered val stats "
        f"(min_tokens={context_length}): {val_path}"
    )

    doc_counts: Dict[str, int] = {}
    token_counts: Dict[str, int] = {}
    total_docs = 0
    total_tokens = 0

    batch_texts: List[str] = []
    batch_subsets: List[str] = []

    def flush():
        nonlocal total_tokens, total_docs
        if not batch_texts:
            return
        enc = tokenizer(
            batch_texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )
        lengths = [len(x) for x in enc.get("input_ids", [])]
        for subset, n_tok in zip(batch_subsets, lengths):
            if n_tok < context_length:
                continue
            doc_counts[subset] = doc_counts.get(subset, 0) + 1
            token_counts[subset] = token_counts.get(subset, 0) + n_tok
            total_docs += 1
            total_tokens += n_tok
        batch_texts.clear()
        batch_subsets.clear()

    pbar = tqdm(desc=f"Scanning val with min_tokens={context_length}", unit="docs")
    for ex in _iter_jsonl_zst(val_path):
        meta = ex.get("meta") or {}
        subset = meta.get("pile_set_name") or "unknown"
        text = ex.get("text")
        if not isinstance(text, str):
            continue
        batch_texts.append(text)
        batch_subsets.append(subset)
        if len(batch_texts) >= batch_size:
            flush()
        pbar.update(1)
    flush()
    pbar.close()

    stats = {
        "val_path": val_path,
        "context_length": int(context_length),
        "total_docs": total_docs,
        "total_tokens": total_tokens,
        "docs_by_subset": doc_counts,
        "tokens_by_subset": token_counts,
    }
    with open(filtered_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats


def _allocate_subset_budgets(
    total_token_budget: int,
    weights_by_subset: Dict[str, int],
    subsets: List[str],
) -> Dict[str, int]:
    weights = {s: int(weights_by_subset.get(s, 0)) for s in subsets}
    total = sum(weights.values())
    if total <= 0:
        return {s: 0 for s in subsets}
    raw = {s: (total_token_budget * weights[s]) / total for s in subsets}
    floors = {s: int(raw[s]) for s in subsets}
    remainder = total_token_budget - sum(floors.values())
    if remainder > 0:
        frac = sorted(subsets, key=lambda s: raw[s] - floors[s], reverse=True)
        for s in frac[:remainder]:
            floors[s] += 1
    return floors


def iter_pile_text_windows_proportional_budget(
    tokenizer,
    total_token_budget: int,
    *,
    context_length: int,
    dataset_name: str,
    split: str,
    min_chars: int = 50,
    seed: int = 42,
    buffer_size: int = 10000,
    pile_subsets: Optional[List[str]] = None,
    stats_path: Optional[str] = None,
    batch_size: Optional[int] = None,
    ratio_by: str = "tokens",
    use_padding: bool = False,
) -> Iterator[Tuple[str, str, int]]:
    """
    Yield (text, subset, token_count) windows where subset budgets are proportional
    to token or doc distribution in the *val* set (or equal quotas).

    For docs shorter than ``context_length``, behavior depends on ``use_padding``:
    if true, short docs are kept (downstream padding will fill); if false, they are skipped.
    """
    subsets = pile_subsets or list(PILE_SUBSETS)
    rng = random.Random(seed)

    stats = _load_or_build_val_stats(dataset_name, tokenizer, stats_path=stats_path)
    tokens_by_subset = stats.get("tokens_by_subset", {})
    docs_by_subset = stats.get("docs_by_subset", {})
    ratio_by = (ratio_by or "tokens").strip().lower()
    budget_source_weights = tokens_by_subset
    if ratio_by == "docs":
        budget_source_weights = docs_by_subset
        budgets = _allocate_subset_budgets(total_token_budget, docs_by_subset, subsets)
    elif ratio_by == "equal":
        budgets = _build_equal_token_quotas(total_token_budget, subsets)
        budget_source_weights = budgets
    else:
        token_weights = tokens_by_subset
        if not use_padding:
            filtered = _load_or_build_filtered_token_stats(
                dataset_name,
                tokenizer,
                context_length=context_length,
                stats_path=stats_path,
            )
            filtered_tokens = filtered.get("tokens_by_subset", {})
            if isinstance(filtered_tokens, dict) and sum(int(v) for v in filtered_tokens.values()) > 0:
                token_weights = filtered_tokens
        budget_source_weights = token_weights
        budgets = _allocate_subset_budgets(total_token_budget, token_weights, subsets)

    remaining_tokens = dict(budgets)
    remaining_sequences = {s: max(0, (budgets[s] + context_length - 1) // context_length) for s in subsets}
    remaining_subset_tokens_total = {s: int(budget_source_weights.get(s, 0)) for s in subsets}

    # For very small budgets, proportional per-subset quotas can stall while waiting
    # for rare subsets. Fall back to any-subset sampling in that case.
    use_any_subset = total_token_budget < (context_length * max(4, len(subsets) // 2))
    if (not use_padding) and ratio_by == "tokens":
        # Keep per-subset quotas active so filtered-token proportional budgets are respected.
        use_any_subset = False
    remaining_any_tokens = total_token_budget

    stream = _load_pile_stream(dataset_name, split, seed, buffer_size)

    fast_small_budget = total_token_budget <= (context_length * 4)

    debug = os.environ.get("PILE_DEBUG", "").strip() not in ("", "0", "false", "False")
    debug_every = int(os.environ.get("PILE_DEBUG_EVERY", "5000"))
    docs_seen = 0
    docs_too_short = 0
    docs_bad_subset = 0
    docs_quota_full = 0
    windows_yielded = 0
    docs_token_short = 0
    debug_tok_len = os.environ.get("PILE_DEBUG_TOKENS", "").strip() not in ("", "0", "false", "False")
    docs_batched = 0
    enc_empty = 0
    zero_token_docs = 0
    offsets_missing = 0
    offsets_all_none = 0
    offsets_len_mismatch = 0
    tok_count_total = 0
    tok_count_min = None
    tok_count_max = None

    if batch_size is None:
        batch_size = int(os.environ.get("PILE_WINDOW_BATCH_SIZE", "32"))
    batch_size = max(1, batch_size)
    if debug_tok_len:
        batch_size = 1
    if debug:
        print(
            "[pile-debug] settings "
            f"total_token_budget={total_token_budget} context_length={context_length} "
            f"use_any_subset={use_any_subset} fast_small_budget={fast_small_budget} "
            f"batch_size={batch_size}"
        )

    batch_texts: List[str] = []
    batch_subsets: List[str] = []

    def remaining_any() -> bool:
        if use_any_subset:
            return remaining_any_tokens > 0
        return any(
            remaining_tokens[s] > 0 and remaining_sequences[s] > 0
            for s in subsets
        )

    def flush_batch() -> Iterator[Tuple[str, str, int]]:
        nonlocal batch_texts, batch_subsets, windows_yielded, docs_token_short, docs_too_short
        nonlocal enc_empty, zero_token_docs, tok_count_total, tok_count_min, tok_count_max
        nonlocal offsets_missing, offsets_all_none, offsets_len_mismatch
        if not batch_texts:
            return iter(())

        if use_any_subset:
            _texts = batch_texts
            _subsets = batch_subsets
            batch_texts = []
            batch_subsets = []
            def _iter_any():
                nonlocal remaining_any_tokens, enc_empty, tok_count_total, tok_count_min, tok_count_max
                for text, subset in zip(_texts, _subsets):
                    enc = tokenizer(
                        text,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=context_length,
                    )
                    doc_tokens = len(enc.get("input_ids") or [])
                    if (not use_padding) and doc_tokens < context_length:
                        if debug:
                            docs_too_short += 1
                        continue
                    if debug and debug_tok_len:
                        print(f"[pile-debug] doc_tokens={doc_tokens}", flush=True)
                    if doc_tokens == 0:
                        if debug:
                            enc_empty += 1
                        continue
                    if debug:
                        tok_count_total += doc_tokens
                        if tok_count_min is None or doc_tokens < tok_count_min:
                            tok_count_min = doc_tokens
                        if tok_count_max is None or doc_tokens > tok_count_max:
                            tok_count_max = doc_tokens
                    if debug:
                        if windows_yielded == 0:
                            print(
                                f"[pile-debug] yielding first window tok_count={doc_tokens}",
                                flush=True,
                            )
                        windows_yielded += 1
                        if doc_tokens < context_length:
                            docs_token_short += 1
                    yield text, subset, doc_tokens
                    remaining_any_tokens -= doc_tokens
                    if remaining_any_tokens <= 0:
                        break

            return _iter_any()

        enc = tokenizer(
            batch_texts,
            add_special_tokens=False,
            return_offsets_mapping=True,
            padding=False,
            truncation=False,
        )
        input_ids_list = enc.get("input_ids") or []
        offsets_list = enc.get("offset_mapping") or []
        if not input_ids_list:
            sample_len = len(batch_texts[0]) if batch_texts else -1
            sample_snippet = batch_texts[0][:200].replace("\n", "\\n") if batch_texts else ""
            raise RuntimeError(
                "[pile] Tokenizer returned empty input_ids for a non-empty batch. "
                f"tokenizer={tokenizer.__class__.__name__} is_fast={getattr(tokenizer, 'is_fast', False)} "
                f"batch_texts={len(batch_texts)} first_len={sample_len} first_snippet={sample_snippet}"
            )
        if (not offsets_list or len(offsets_list) != len(input_ids_list)) and (not use_any_subset):
            raise RuntimeError(
                "[pile] Tokenizer did not return usable offset mappings for window sampling. "
                "This path requires a fast tokenizer with return_offsets_mapping support. "
                f"tokenizer={tokenizer.__class__.__name__} is_fast={getattr(tokenizer, 'is_fast', False)} "
                f"input_ids_list={len(input_ids_list)} offsets_list={len(offsets_list)}"
            )
        if debug and (not offsets_list or len(offsets_list) != len(input_ids_list)):
            print(
                "[pile-debug] offsets missing or mismatched "
                f"input_ids_list={len(input_ids_list)} offsets_list={len(offsets_list)}",
                flush=True,
            )
        if debug and input_ids_list:
            lengths = [len(x) for x in input_ids_list]
            tok_count_total += sum(lengths)
            if lengths:
                min_len = min(lengths)
                max_len = max(lengths)
                if tok_count_min is None or min_len < tok_count_min:
                    tok_count_min = min_len
                if tok_count_max is None or max_len > tok_count_max:
                    tok_count_max = max_len
        if debug and not input_ids_list:
            enc_empty += 1
            first_len = len(batch_texts[0]) if batch_texts else -1
            print(
                "[pile-debug] tokenizer returned empty input_ids; "
                f"batch_texts={len(batch_texts)} first_len={first_len}",
                flush=True,
            )

        _texts = batch_texts
        _subsets = batch_subsets
        batch_texts = []
        batch_subsets = []

        def _iter_items():
            nonlocal remaining_any_tokens
            nonlocal windows_yielded, docs_token_short, docs_too_short
            if use_any_subset and input_ids_list and (not offsets_list or len(offsets_list) != len(input_ids_list)):
                for text, subset, input_ids in zip(_texts, _subsets, input_ids_list):
                    tok_count = min(len(input_ids), context_length)
                    if debug:
                        if windows_yielded == 0:
                            print(
                                f"[pile-debug] yielding first window tok_count={tok_count}",
                                flush=True,
                            )
                        windows_yielded += 1
                        if len(input_ids) < context_length:
                            docs_token_short += 1
                    yield text, subset, tok_count
                    remaining_any_tokens -= tok_count
                    if remaining_any_tokens <= 0:
                        break
                return
            for text, subset, input_ids, offsets in zip(
                _texts, _subsets, input_ids_list, offsets_list
            ):
                doc_tokens = len(input_ids)
                if debug:
                    if offsets is None:
                        offsets_missing += 1
                    else:
                        if len(offsets) != doc_tokens:
                            offsets_len_mismatch += 1
                        if offsets and all(o[0] is None or o[1] is None for o in offsets):
                            offsets_all_none += 1
                if debug and debug_tok_len:
                    print(f"[pile-debug] doc_tokens={doc_tokens}", flush=True)
                if debug and doc_tokens == 0 and zero_token_docs < 5:
                    zero_token_docs += 1
                    snippet = text[:200].replace("\n", "\\n")
                    print(
                        "[pile-debug] zero_token_doc "
                        f"len_chars={len(text)} snippet={snippet}",
                        flush=True,
                    )
                if (not use_padding) and doc_tokens < context_length:
                    if debug:
                        docs_too_short += 1
                    continue

                if use_any_subset and doc_tokens > 0:
                    # For tiny budgets, skip windowing entirely and use full doc.
                    tok_count = min(doc_tokens, context_length)
                    if debug:
                        if windows_yielded == 0:
                            print(
                                f"[pile-debug] yielding first window tok_count={tok_count}",
                                flush=True,
                            )
                        windows_yielded += 1
                        if doc_tokens < context_length:
                            docs_token_short += 1
                    yield text, subset, tok_count
                    remaining_any_tokens -= tok_count
                    if remaining_any_tokens <= 0:
                        break
                    continue

                if doc_tokens < context_length:
                    if use_padding and (not use_any_subset):
                        # Keep short docs as a single padded sequence downstream.
                        # Budget accounting uses real tokens so padding does not
                        # contribute artificial observations.
                        tok_count = min(doc_tokens, remaining_tokens[subset])
                        if tok_count > 0:
                            if debug:
                                windows_yielded += 1
                                docs_token_short += 1
                            yield text, subset, tok_count
                            remaining_tokens[subset] = max(0, remaining_tokens[subset] - tok_count)
                        remaining_subset_tokens_total[subset] = max(
                            0, remaining_subset_tokens_total[subset] - doc_tokens
                        )
                    else:
                        if debug:
                            docs_too_short += 1
                    continue

                if use_any_subset:
                    k = 1 if remaining_any_tokens > 0 else 0
                elif fast_small_budget:
                    k = 1 if remaining_sequences[subset] > 0 else 0
                else:
                    # IID per-doc sampling within subset: take at most one window per doc
                    # until the per-subset sequence budget is exhausted.
                    k = 1 if remaining_sequences[subset] > 0 else 0

                for _ in range(k):
                    # Greedy segment: take the longest prefix window (<= context_length)
                    # that has valid start/end offsets.
                    if not offsets:
                        if debug:
                            print("[pile-debug] window_reject=empty_offsets", flush=True)
                        break
                    start = 0
                    while start < len(offsets) and (offsets[start][0] is None or offsets[start][1] is None):
                        start += 1
                    if start >= len(offsets):
                        if debug:
                            print("[pile-debug] window_reject=no_valid_start", flush=True)
                        break
                    end = min(start + context_length - 1, len(offsets) - 1)
                    while end > start and (offsets[end][0] is None or offsets[end][1] is None):
                        end -= 1
                    if end < start:
                        if debug:
                            print("[pile-debug] window_reject=no_valid_end", flush=True)
                        break
                    start_char = offsets[start][0]
                    end_char = offsets[end][1]
                    if start_char is None or end_char is None or end_char <= start_char:
                        if debug:
                            print(
                                "[pile-debug] window_reject=invalid_span "
                                f"start_char={start_char} end_char={end_char}",
                                flush=True,
                            )
                        break
                    slice_text = text[start_char:end_char]
                    if debug:
                        windows_yielded += 1
                    yield slice_text, subset, (end - start + 1)
                    if use_any_subset:
                        remaining_any_tokens -= (end - start + 1)
                        if remaining_any_tokens <= 0:
                            break
                    else:
                        remaining_sequences[subset] -= 1
                        remaining_tokens[subset] = max(0, remaining_tokens[subset] - (end - start + 1))
                        if remaining_sequences[subset] <= 0 or remaining_tokens[subset] <= 0:
                            break

                if not use_any_subset:
                    remaining_subset_tokens_total[subset] = max(
                        0, remaining_subset_tokens_total[subset] - doc_tokens
                    )

        return _iter_items()

    for ex in stream:
        if not remaining_any():
            break
        docs_seen += 1
        if debug and docs_seen % debug_every == 0:
            avg_tokens = (tok_count_total / docs_batched) if docs_batched else 0.0
            print(
                f"[pile-debug] docs_seen={docs_seen} windows={windows_yielded} "
                f"too_short={docs_too_short} token_short={docs_token_short} "
                f"bad_subset={docs_bad_subset} quota_full={docs_quota_full} "
                f"docs_batched={docs_batched} enc_empty={enc_empty} zero_token_docs={zero_token_docs} "
                f"offsets_missing={offsets_missing} offsets_len_mismatch={offsets_len_mismatch} "
                f"offsets_all_none={offsets_all_none} "
                f"use_any_subset={use_any_subset} batch_size={batch_size} "
                f"tok_min={tok_count_min} tok_avg={avg_tokens:.1f} tok_max={tok_count_max}",
                flush=True,
            )
        subset = get_pile_subset_name(ex)
        if not use_any_subset and subset not in remaining_tokens:
            if debug:
                docs_bad_subset += 1
            continue
        if not use_any_subset and (remaining_tokens[subset] <= 0 or remaining_sequences[subset] <= 0):
            if debug:
                docs_quota_full += 1
            continue
        text = ex.get("text", "")
        if not isinstance(text, str) or len(text) < min_chars:
            if debug:
                docs_too_short += 1
            continue
        batch_texts.append(text)
        batch_subsets.append(subset)
        docs_batched += 1
        if len(batch_texts) >= batch_size:
            for item in flush_batch():
                yield item
            if not remaining_any():
                break

    if batch_texts and remaining_any():
        for item in flush_batch():
            yield item

    if debug:
        print(
            "[pile-debug] done "
            f"docs_seen={docs_seen} docs_batched={docs_batched} "
            f"windows={windows_yielded} enc_empty={enc_empty} "
            f"offsets_missing={offsets_missing} offsets_len_mismatch={offsets_len_mismatch} "
            f"offsets_all_none={offsets_all_none} "
            f"offsets_valid={offsets_valid_tokens} offsets_total={offsets_total_tokens}"
        )
