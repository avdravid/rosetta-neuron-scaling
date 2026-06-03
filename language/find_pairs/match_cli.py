"""CLI parsing for match_lm."""

from __future__ import annotations

import argparse

import torch

from .match_types import MatchConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1", type=str, default="EleutherAI/pythia-1b")
    parser.add_argument("--model2", type=str, default="EleutherAI/pythia-2.8b")
    parser.add_argument(
        "--revision1",
        type=str,
        default="",
        help="HF revision tag for model1 (e.g., 'step1000'). Empty = default (main).",
    )
    parser.add_argument(
        "--revision2",
        type=str,
        default="",
        help="HF revision tag for model2.",
    )
    parser.add_argument(
        "--tokenizer1",
        type=str,
        default="",
        help="Optional tokenizer override for model1.",
    )
    parser.add_argument(
        "--tokenizer2",
        type=str,
        default="",
        help="Optional tokenizer override for model2.",
    )
    parser.add_argument(
        "--span_pool",
        type=str,
        default="mean",
        choices=["mean", "max", "median"],
        help="Pooling over tokens within a canonical span.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="/datasets/pile/current",
        help="Path to a Pile-shaped dataset directory containing val.jsonl.zst (or similar).",
    )
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument(
        "--pile_subsets",
        type=str,
        default="",
        help="Comma-separated list of Pile subsets (optional override).",
    )
    parser.add_argument(
        "--pile_ratio_by",
        type=str,
        default="tokens",
        choices=["tokens", "docs", "equal", "iid"],
        help=(
            "Pile subset sampling ratios: tokens (val token distribution), "
            "docs (val doc distribution), equal (equal token quota per subset), "
            "or iid (ignore subsets and sample windows iid)."
        ),
    )
    parser.add_argument(
        "--use_padding",
        action="store_true",
        help="If set, require Pile docs to be at least seq_length tokens before sampling.",
    )
    parser.add_argument(
        "--shard_pile_subsets",
        action="store_true",
        help="Shard Pile subsets across distributed ranks (each rank samples a distinct subset set).",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=10_000_000,
        help="Total token budget over the Pile corpus.",
    )
    parser.add_argument(
        "--seq_length",
        "--max_tokens",
        dest="max_tokens",
        type=int,
        default=256,
        help="Max tokens per model (padding=max_length). Alias: --seq_length.",
    )
    parser.add_argument(
        "--max_bytes",
        type=int,
        default=8096,
        help="Max UTF-8 bytes per sample before tokenization.",
    )
    parser.add_argument("--batch_size", type=int, default=4)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./outputs")

    parser.add_argument("--only_diagonal", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--tokenize_batch", type=int, default=256)
    parser.add_argument("--b_block", type=int, default=2)

    parser.add_argument(
        "--depth_neighbors",
        type=int,
        default=None,
        help="Only correlate each A-layer with the J nearest B-layers by normalized depth. None = all pairs.",
    )
    parser.add_argument("--save_neighbors", action="store_true")
    parser.add_argument("--top_k", type=int, default=100)

    parser.add_argument("--token_cache", type=str, default="", help="Optional explicit cache path.")
    parser.add_argument(
        "--use_fsdp",
        action="store_true",
        help="Wrap both models with FSDP (FULL_SHARD, transformer-layer auto-wrap) so weights "
             "shard across ranks. Required to fit 30B+ model pairs on 8x80GB.",
    )
    return parser


def parse_config(argv=None) -> MatchConfig:
    args = build_parser().parse_args(argv)
    return MatchConfig(
        model1=args.model1,
        model2=args.model2,
        tokenizer1=args.tokenizer1 or None,
        tokenizer2=args.tokenizer2 or None,
        span_pool=args.span_pool,
        dataset=args.dataset,
        split=args.split,
        pile_subsets_arg=args.pile_subsets,
        pile_ratio_by=args.pile_ratio_by,
        use_padding=bool(args.use_padding),
        shard_pile_subsets=bool(args.shard_pile_subsets),
        num_samples=int(args.num_samples),
        max_tokens=int(args.max_tokens),
        max_bytes=int(args.max_bytes),
        batch_size=int(args.batch_size),
        device=args.device,
        dtype_name=args.dtype,
        seed=int(args.seed),
        save_dir=args.save_dir,
        only_diagonal=bool(args.only_diagonal),
        allow_tf32=bool(args.allow_tf32),
        tokenize_batch=int(args.tokenize_batch),
        b_block=int(args.b_block),
        save_neighbors=bool(args.save_neighbors),
        top_k=int(args.top_k),
        depth_neighbors=args.depth_neighbors,
        token_cache=args.token_cache,
        use_fsdp=bool(args.use_fsdp),
        revision1=args.revision1 or None,
        revision2=args.revision2 or None,
    )


def resolve_compute_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32
