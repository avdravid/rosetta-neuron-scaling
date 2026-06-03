"""Byte-alignment helpers for match_lm."""

from __future__ import annotations

from typing import List, Tuple

import torch


def truncate_utf8_to_max_bytes(text: str, max_bytes: int) -> str:
    """
    Truncate by UTF-8 bytes, safely (won't leave a partial multibyte char at end).
    """
    b = text.encode("utf-8", errors="ignore")
    if len(b) <= max_bytes:
        return text
    b = b[:max_bytes]
    return b.decode("utf-8", errors="ignore")


def _char_to_byte_cumsum(text: str) -> List[int]:
    """
    cum[i] = number of UTF-8 bytes in text[:i], for i in [0..len(text)].
    """
    cum = [0]
    total = 0
    for ch in text:
        total += len(ch.encode("utf-8"))
        cum.append(total)
    return cum


def _tokenize_with_byte_offsets(
    tokenizer,
    texts: List[str],
    max_tokens: int,
    *,
    offsets_dtype: torch.dtype = torch.int32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Tokenize texts and return:
      input_ids:      (B, S) int32
      attention_mask: (B, S) uint8
      byte_offsets:   (B, S, 2) offsets in UTF-8 bytes (start, end), int32/int16
      coverage_bytes: (B,) int32  max byte_end among valid tokens
    """
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            f"Tokenizer {tokenizer.__class__.__name__} is not fast; "
            "byte-level alignment needs return_offsets_mapping from a fast tokenizer."
        )

    enc = tokenizer(
        texts,
        truncation=True,
        max_length=max_tokens,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"].to(torch.int32)
    attention_mask = enc["attention_mask"].to(torch.uint8)
    offsets_char = enc["offset_mapping"]  # (B,S,2) int64 in char indices

    B, S, _ = offsets_char.shape
    byte_offsets = torch.zeros((B, S, 2), dtype=offsets_dtype)

    coverage = torch.zeros((B,), dtype=torch.int32)

    synthetic_span_fallback = 0

    # Convert char offsets -> byte offsets per sample.
    for b in range(B):
        t = texts[b]
        cum = _char_to_byte_cumsum(t)  # len = len(t)+1
        cum_t = torch.tensor(cum, dtype=torch.int32)

        oc = offsets_char[b]  # (S,2)
        # clamp for safety
        starts = oc[:, 0].clamp(0, len(t)).to(torch.long)
        ends = oc[:, 1].clamp(0, len(t)).to(torch.long)

        bs = cum_t[starts]
        be = cum_t[ends]
        if offsets_dtype != torch.int32:
            bs = bs.to(offsets_dtype)
            be = be.to(offsets_dtype)

        byte_offsets[b, :, 0] = bs
        byte_offsets[b, :, 1] = be

        span = (be.to(torch.int32) - bs.to(torch.int32))
        valid = (attention_mask[b].to(torch.bool)) & (span > 0)
        if torch.any(valid):
            coverage[b] = int(be.to(torch.int32)[valid].max().item())
        else:
            valid_attn = attention_mask[b].to(torch.bool)
            if torch.any(valid_attn):
                # Fallback for tokenizers/configurations that return zero-length
                # offsets for all non-padding tokens. Use synthetic monotonic
                # unit spans so overlap logic can still run; padding stays masked.
                idx = valid_attn.nonzero(as_tuple=False).squeeze(1)
                n = int(idx.numel())
                syn_start = torch.arange(n, dtype=torch.int32)
                syn_end = syn_start + 1
                if offsets_dtype != torch.int32:
                    syn_start = syn_start.to(offsets_dtype)
                    syn_end = syn_end.to(offsets_dtype)
                byte_offsets[b, idx, 0] = syn_start
                byte_offsets[b, idx, 1] = syn_end
                coverage[b] = n
                synthetic_span_fallback += 1
            else:
                coverage[b] = 0

    if synthetic_span_fallback > 0:
        print(
            "[cache] Warning: synthesized token spans for "
            f"{synthetic_span_fallback}/{B} samples due to zero offset mappings."
        )

    return input_ids, attention_mask, byte_offsets, coverage


def _module_in_dim(m: torch.nn.Module) -> int:
    if hasattr(m, "in_features"):
        return int(m.in_features)  # type: ignore
    # Handle transformers Conv1D (used in GPT-2): weight shape is (in_features, out_features)
    if hasattr(m, "nf") and hasattr(m, "nx"):
        return int(m.nx)  # nx is input features for Conv1D
    if hasattr(m, "weight"):
        # For Linear: weight shape is (out_features, in_features), so shape[1] is input
        # For Conv1D: weight shape is (in_features, out_features), so shape[0] is input
        # We detect Conv1D by checking if shape[0] < shape[1] (typical MLP expands then contracts)
        # Actually safer: check module class name
        class_name = m.__class__.__name__
        if class_name == "Conv1D":
            return int(m.weight.shape[0])  # Conv1D: (in_features, out_features)
        return int(m.weight.shape[1])  # Linear: (out_features, in_features)
    raise ValueError(f"Cannot infer input dim for module type={type(m)}")


def _module_out_dim(m: torch.nn.Module) -> int:
    if hasattr(m, "out_features"):
        return int(m.out_features)  # type: ignore
    # Handle transformers Conv1D (used in GPT-2)
    if hasattr(m, "nf") and hasattr(m, "nx"):
        return int(m.nf)  # nf is output features for Conv1D
    if hasattr(m, "weight"):
        class_name = m.__class__.__name__
        if class_name == "Conv1D":
            return int(m.weight.shape[1])  # Conv1D: (in_features, out_features)
        return int(m.weight.shape[0])  # Linear: (out_features, in_features)
    raise ValueError(f"Cannot infer output dim for module type={type(m)}")


def _module_activation_dim(m: torch.nn.Module, capture_output: bool) -> int:
    """Get the dimension of activations captured from this module."""
    return _module_out_dim(m) if capture_output else _module_in_dim(m)


def _dtype_nbytes(dtype: torch.dtype) -> int:
    if dtype in (torch.float16, torch.bfloat16):
        return 2
    if dtype == torch.float32:
        return 4
    if dtype == torch.float64:
        return 8
    return 4


def _build_intervals_from_offsets(
    byte_offsets_1d: torch.Tensor,   # (S,2) CPU int
    attention_mask_1d: torch.Tensor, # (S,) CPU uint8
    aligned_len: int,
) -> List[Tuple[int, int, int]]:
    """
    Build a sorted list of intervals (start_byte, end_byte, token_index),
    clipped to [0, aligned_len).
    """
    mask = attention_mask_1d.bool()
    starts = byte_offsets_1d[:, 0].to(torch.int32)
    ends = byte_offsets_1d[:, 1].to(torch.int32)
    al = torch.tensor(aligned_len, dtype=torch.int32)
    ends_clipped = torch.minimum(ends, al)
    valid = mask & (ends > starts) & (starts < al) & (ends_clipped > starts)
    indices = valid.nonzero(as_tuple=False).view(-1)
    if indices.numel() == 0:
        return []
    s_vals = starts[indices].tolist()
    e_vals = ends_clipped[indices].tolist()
    t_vals = indices.tolist()
    intervals = list(zip(s_vals, e_vals, t_vals))
    # tokenizers should already be monotonic; sort defensively
    intervals.sort(key=lambda x: x[0])
    return intervals


def _build_overlap_segments_for_batch(
    boA: torch.Tensor, amA: torch.Tensor,
    boB: torch.Tensor, amB: torch.Tensor,
    aligned_len: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build overlap segments across the batch.

    Returns CPU tensors:
      seg_b:  (M,) int64 batch index
      seg_tA: (M,) int64 token index in A
      seg_tB: (M,) int64 token index in B
      seg_w:  (M,) int32 overlap length in bytes
    """
    B = boA.shape[0]
    seg_b: List[int] = []
    seg_tA: List[int] = []
    seg_tB: List[int] = []
    seg_w: List[int] = []

    for b in range(B):
        al = int(aligned_len[b].item())
        if al <= 0:
            continue

        intsA = _build_intervals_from_offsets(boA[b], amA[b], al)
        intsB = _build_intervals_from_offsets(boB[b], amB[b], al)
        if not intsA or not intsB:
            continue

        i = 0
        j = 0
        while i < len(intsA) and j < len(intsB):
            sA, eA, tA = intsA[i]
            sB, eB, tB = intsB[j]
            s = sA if sA > sB else sB
            e = eA if eA < eB else eB
            if e > s:
                seg_b.append(b)
                seg_tA.append(tA)
                seg_tB.append(tB)
                seg_w.append(e - s)

            # advance the interval that ends first
            if eA <= eB:
                i += 1
            else:
                j += 1

    if not seg_b:
        # empty
        return (
            torch.empty((0,), dtype=torch.int64),
            torch.empty((0,), dtype=torch.int64),
            torch.empty((0,), dtype=torch.int64),
            torch.empty((0,), dtype=torch.int32),
        )

    return (
        torch.tensor(seg_b, dtype=torch.int64),
        torch.tensor(seg_tA, dtype=torch.int64),
        torch.tensor(seg_tB, dtype=torch.int64),
        torch.tensor(seg_w, dtype=torch.int32),
    )
