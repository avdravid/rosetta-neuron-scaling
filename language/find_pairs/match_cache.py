"""Cache building and dataloader helpers for match_lm."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from data.pile_sampler import (
    iter_pile_text_windows_proportional_budget,
    _load_pile_stream,
    resolve_pile_val_path,
    compute_val_stats_shard,
    merge_val_stats,
)

from .match_bytealign import truncate_utf8_to_max_bytes, _tokenize_with_byte_offsets
from .match_dist import dist_barrier


def _split_token_budget(total_token_budget: int, rank: int, world_size: int) -> int:
    if total_token_budget <= 0:
        return 0
    base = total_token_budget // world_size
    remainder = total_token_budget % world_size
    return base + (1 if rank < remainder else 0)


def _ensure_pile_val_stats_distributed(
    dataset_name: str,
    tokenizer,
    *,
    stats_path: str,
    rank: int,
    world_size: int,
) -> None:
    if world_size <= 1:
        return

    def _stats_valid(path: str) -> bool:
        try:
            with open(path, "r", encoding="utf-8") as f:
                stats = json.load(f)
            tokens_by_subset = stats.get("tokens_by_subset")
            total_tokens = stats.get("total_tokens", 0)
            return isinstance(tokens_by_subset, dict) and sum(tokens_by_subset.values()) > 0 and total_tokens
        except Exception:
            return False

    if os.path.exists(stats_path):
        if _stats_valid(stats_path):
            return
        if rank == 0:
            print(f"[pile] Cached stats invalid or empty; rebuilding: {stats_path}")
            try:
                os.remove(stats_path)
            except OSError:
                pass
        dist_barrier()

    shard_path = f"{stats_path}.rank{rank}.json"
    if not os.path.exists(shard_path):
        val_path = resolve_pile_val_path(dataset_name)
        batch_size = int(os.environ.get("PILE_STATS_BATCH_SIZE", "128"))
        stats = compute_val_stats_shard(
            val_path,
            tokenizer,
            rank=rank,
            world_size=world_size,
            batch_size=batch_size,
        )
        with open(shard_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

    dist_barrier()
    if rank == 0 and not os.path.exists(stats_path):
        shard_paths = [f"{stats_path}.rank{i}.json" for i in range(world_size)]
        merge_val_stats(shard_paths, stats_path)
    dist_barrier()


def pretokenize_pile_two_models_bytecache_once(
    tokenizer1,
    tokenizer2,
    total_token_budget: int,
    *,
    dataset_name: str,
    max_tokens: int,
    max_bytes: int,
    min_chars: int = 100,
    text_batch: int = 256,
    seed: int = 42,
    buffer_size: int = 10000,
    split: str = "train",
    pile_subsets: Optional[List[str]] = None,
    stats_path: Optional[str] = None,
    ratio_by: str = "tokens",
    use_padding: bool = False,
    allow_empty: bool = False,
):
    """
    Build a cache from The Pile using configurable subset token budgeting.

    Returns CPU tensors:
      input_ids1, attention_mask1, byte_offsets1,
      input_ids2, attention_mask2, byte_offsets2,
      aligned_len_bytes (N,)
    """
    offsets_dtype = torch.int16 if max_bytes <= 32767 else torch.int32

    chunks: Dict[str, List[torch.Tensor]] = {
        "input_ids1": [], "attention_mask1": [], "byte_offsets1": [],
        "input_ids2": [], "attention_mask2": [], "byte_offsets2": [],
        "aligned_len": [],
    }

    buf: List[str] = []
    got_tokens = 0
    yielded_windows = 0

    # Large local shuffle buffers and large sampler batches can dominate startup
    # time for tiny token budgets (common in distributed runs).
    expected_windows = max(1, (total_token_budget + max(1, max_tokens) - 1) // max(1, max_tokens))
    adaptive_buffer_size = min(buffer_size, max(1, expected_windows * 8))
    adaptive_window_batch = max(1, min(32, expected_windows))
    if adaptive_buffer_size < buffer_size:
        print(
            f"[pile] Reducing stream shuffle buffer {buffer_size} -> {adaptive_buffer_size} "
            f"for token budget={total_token_budget}"
        )
    if adaptive_window_batch < 32:
        print(
            f"[pile] Reducing window tokenizer batch 32 -> {adaptive_window_batch} "
            f"for token budget={total_token_budget}"
        )

    pbar = tqdm(total=total_token_budget, desc="Building byte-aligned cache (Pile)", unit="tokens")
    for text, _subset, token_count in iter_pile_text_windows_proportional_budget(
        tokenizer1,
        total_token_budget,
        context_length=max_tokens,
        dataset_name=dataset_name,
        split=split,
        min_chars=min_chars,
        seed=seed,
        buffer_size=adaptive_buffer_size,
        pile_subsets=pile_subsets,
        stats_path=stats_path,
        batch_size=adaptive_window_batch,
        ratio_by=ratio_by,
        use_padding=use_padding,
    ):
        if got_tokens >= total_token_budget:
            break
        text = truncate_utf8_to_max_bytes(text, max_bytes)
        buf.append(text)
        yielded_windows += 1
        got_tokens += token_count
        pbar.update(token_count)

        if len(buf) >= text_batch:
            ids1, am1, bo1, cov1 = _tokenize_with_byte_offsets(
                tokenizer1, buf, max_tokens, offsets_dtype=offsets_dtype
            )
            ids2, am2, bo2, cov2 = _tokenize_with_byte_offsets(
                tokenizer2, buf, max_tokens, offsets_dtype=offsets_dtype
            )

            aligned = torch.minimum(cov1, cov2).to(torch.int32)
            keep = aligned > 0
            if torch.any(keep):
                k = keep.nonzero(as_tuple=False).squeeze(1)
                chunks["input_ids1"].append(ids1[k])
                chunks["attention_mask1"].append(am1[k])
                chunks["byte_offsets1"].append(bo1[k])
                chunks["input_ids2"].append(ids2[k])
                chunks["attention_mask2"].append(am2[k])
                chunks["byte_offsets2"].append(bo2[k])
                chunks["aligned_len"].append(aligned[k])
            buf = []

    pbar.close()

    if buf:
        ids1, am1, bo1, cov1 = _tokenize_with_byte_offsets(
            tokenizer1, buf, max_tokens, offsets_dtype=offsets_dtype
        )
        ids2, am2, bo2, cov2 = _tokenize_with_byte_offsets(
            tokenizer2, buf, max_tokens, offsets_dtype=offsets_dtype
        )
        aligned = torch.minimum(cov1, cov2).to(torch.int32)
        keep = aligned > 0
        if torch.any(keep):
            k = keep.nonzero(as_tuple=False).squeeze(1)
            chunks["input_ids1"].append(ids1[k])
            chunks["attention_mask1"].append(am1[k])
            chunks["byte_offsets1"].append(bo1[k])
            chunks["input_ids2"].append(ids2[k])
            chunks["attention_mask2"].append(am2[k])
            chunks["byte_offsets2"].append(bo2[k])
            chunks["aligned_len"].append(aligned[k])

    if not chunks["aligned_len"]:
        print(
            "[pile] Proportional window sampler produced no aligned samples; "
            "retrying with direct-doc fallback."
        )
        fallback_buf: List[str] = []
        fallback_tokens = 0
        fallback_docs = 0
        fallback_stream = _load_pile_stream(dataset_name, split, seed + 12345, 1)
        for ex in fallback_stream:
            if fallback_tokens >= total_token_budget:
                break
            text = ex.get("text", "")
            if not isinstance(text, str) or len(text) < min_chars:
                continue
            text = truncate_utf8_to_max_bytes(text, max_bytes)
            enc = tokenizer1(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_tokens,
            )
            doc_tokens = len(enc.get("input_ids") or [])
            if doc_tokens <= 0:
                continue
            if (not use_padding) and doc_tokens < max_tokens:
                continue
            fallback_docs += 1
            fallback_tokens += doc_tokens
            fallback_buf.append(text)
            if len(fallback_buf) >= text_batch:
                ids1, am1, bo1, cov1 = _tokenize_with_byte_offsets(
                    tokenizer1, fallback_buf, max_tokens, offsets_dtype=offsets_dtype
                )
                ids2, am2, bo2, cov2 = _tokenize_with_byte_offsets(
                    tokenizer2, fallback_buf, max_tokens, offsets_dtype=offsets_dtype
                )
                aligned = torch.minimum(cov1, cov2).to(torch.int32)
                keep = aligned > 0
                if torch.any(keep):
                    k = keep.nonzero(as_tuple=False).squeeze(1)
                    chunks["input_ids1"].append(ids1[k])
                    chunks["attention_mask1"].append(am1[k])
                    chunks["byte_offsets1"].append(bo1[k])
                    chunks["input_ids2"].append(ids2[k])
                    chunks["attention_mask2"].append(am2[k])
                    chunks["byte_offsets2"].append(bo2[k])
                    chunks["aligned_len"].append(aligned[k])
                fallback_buf = []
        if fallback_buf:
            ids1, am1, bo1, cov1 = _tokenize_with_byte_offsets(
                tokenizer1, fallback_buf, max_tokens, offsets_dtype=offsets_dtype
            )
            ids2, am2, bo2, cov2 = _tokenize_with_byte_offsets(
                tokenizer2, fallback_buf, max_tokens, offsets_dtype=offsets_dtype
            )
            aligned = torch.minimum(cov1, cov2).to(torch.int32)
            keep = aligned > 0
            if torch.any(keep):
                k = keep.nonzero(as_tuple=False).squeeze(1)
                chunks["input_ids1"].append(ids1[k])
                chunks["attention_mask1"].append(am1[k])
                chunks["byte_offsets1"].append(bo1[k])
                chunks["input_ids2"].append(ids2[k])
                chunks["attention_mask2"].append(am2[k])
                chunks["byte_offsets2"].append(bo2[k])
                chunks["aligned_len"].append(aligned[k])
        print(
            f"[pile] Direct-doc fallback scanned_docs={fallback_docs} "
            f"token_budget_target={total_token_budget}"
        )

    out: Dict[str, torch.Tensor] = {}
    for k, parts in chunks.items():
        if not parts:
            if allow_empty:
                print(
                    "[pile] No aligned samples for this shard; "
                    f"yielded_windows={yielded_windows} token_budget={total_token_budget}. "
                    "Returning empty shard."
                )
                return {}
            raise RuntimeError(
                f"Missing cached field {k}. "
                f"yielded_windows={yielded_windows} token_budget={total_token_budget}. "
                "No aligned samples were kept."
            )
        out[k] = torch.cat(parts, dim=0)

    return out
class ByteAlignedPairDataset(torch.utils.data.Dataset):
    def __init__(self, cache: Dict[str, torch.Tensor]):
        # required fields
        self.input_ids1 = cache["input_ids1"]
        self.attn1 = cache["attention_mask1"]
        self.off1 = cache["byte_offsets1"]
        self.input_ids2 = cache["input_ids2"]
        self.attn2 = cache["attention_mask2"]
        self.off2 = cache["byte_offsets2"]
        self.aligned = cache["aligned_len"]

        N = self.input_ids1.shape[0]
        assert self.input_ids2.shape[0] == N
        assert self.aligned.shape[0] == N

    def __len__(self):
        return int(self.input_ids1.shape[0])

    def __getitem__(self, i):
        return {
            "input_ids1": self.input_ids1[i],
            "attention_mask1": self.attn1[i],
            "byte_offsets1": self.off1[i],
            "input_ids2": self.input_ids2[i],
            "attention_mask2": self.attn2[i],
            "byte_offsets2": self.off2[i],
            "aligned_len": self.aligned[i],
        }


def make_dataloader_from_cache(cache: Dict[str, torch.Tensor], batch_size: int):
    ds = ByteAlignedPairDataset(cache)
    pin = torch.cuda.is_available()
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin,
        drop_last=False,
    )


def make_sharded_dataloader_from_cache(
    cache: Dict[str, torch.Tensor],
    batch_size: int,
    rank: int,
    world_size: int,
):
    ds_full = ByteAlignedPairDataset(cache)
    N = len(ds_full)
    idx = list(range(rank, N, world_size))
    ds = torch.utils.data.Subset(ds_full, idx)

    pin = torch.cuda.is_available()
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin,
        drop_last=False,
    )
