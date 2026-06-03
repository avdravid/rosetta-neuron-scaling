#!/usr/bin/env python3
"""Unified entry point for vision matching.

Pick a generative model family with `--gen-family`; all remaining arguments
are forwarded to the family-specific matching script:

    pmf   ->  match_pmf_vit_multigpu.py   (pMF one-step generator)
    flux  ->  match_flux.py               (FLUX.2-klein-4B)
    sana  ->  match_large_dit.py          (Sana / large DiT)

Examples:

    # pMF + OpenCLIP ViT-B/16, single GPU
    python match.py --gen-family pmf \\
        --pmf-repo pMF --pmf-model pmfDiT_B_16 \\
        --pmf-hf-repo Lyy0725/pMF --pmf-ckpt-file pMF-B-16.pt \\
        --disc-family openclip --disc-arch ViT-B-16 --disc-pretrained openai \\
        --num-images 1000 --batch-size 8 --save-dir ./pmf_openclip

    # FLUX + DINOv3, distributed
    torchrun --standalone --nproc_per_node=8 match.py --gen-family flux \\
        --disc-family dinov3 --disc-arch facebook/dinov3-vitb16-pretrain-lvd1689m \\
        --num-images 1000 --batch-size 2 --save-dir ./flux_dinov3

    # Sana + PixIO
    torchrun --standalone --nproc_per_node=8 match.py --gen-family sana \\
        --disc-family pixio --disc-arch pixio_vith16 \\
        --disc-checkpoint ./pixio_checkpoints/pixio_vith16.pth \\
        --num-images 1000 --batch-size 2 --save-dir ./sana_pixio

To see the full CLI for a given family, run with `--help` *after* `--gen-family`:

    python match.py --gen-family pmf --help
"""

from __future__ import annotations

import argparse
import os
import sys

_FAMILY_TO_MODULE = {
    "pmf":  "match_pmf_vit_multigpu",
    "flux": "match_flux",
    "sana": "match_large_dit",
}


def _split_gen_family(argv: list[str]) -> tuple[str, list[str]]:
    """Pull --gen-family / --gen-family=X out of argv, return (family, rest)."""
    rest: list[str] = []
    family: str | None = None
    it = iter(argv)
    for tok in it:
        if tok == "--gen-family":
            family = next(it, None)
        elif tok.startswith("--gen-family="):
            family = tok.split("=", 1)[1]
        else:
            rest.append(tok)
    if family is None:
        raise SystemExit(
            "match.py: missing required --gen-family {pmf,flux,sana}.\n"
            "Run `python match.py --help` for the dispatcher overview, or\n"
            "`python match.py --gen-family <family> --help` for that family's CLI."
        )
    if family not in _FAMILY_TO_MODULE:
        raise SystemExit(
            f"match.py: unknown --gen-family {family!r}; "
            f"expected one of {sorted(_FAMILY_TO_MODULE)}."
        )
    return family, rest


def main() -> None:
    # If the user asked for top-level --help and did NOT pick a family, show our docstring.
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]) and not any(
        arg == "--gen-family" or arg.startswith("--gen-family=") for arg in sys.argv[1:]
    ):
        print(__doc__)
        return

    family, rest = _split_gen_family(sys.argv[1:])
    module_name = _FAMILY_TO_MODULE[family]

    # Rewrite sys.argv so the inner script's argparse sees clean args.
    inner_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{module_name}.py")
    sys.argv = [inner_script] + rest

    # Make sure the inner module can be imported from this directory.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    inner = __import__(module_name)
    inner.main()


if __name__ == "__main__":
    main()
