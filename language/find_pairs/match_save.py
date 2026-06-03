"""Output saving helpers for match_lm."""

from __future__ import annotations

import os
from typing import List, Optional, Set, Tuple

import torch


def save_neighbors_block_from_prod_concat(
    prod_concat: List[torch.Tensor],
    total_n: float,
    DB_sizes: List[int],
    js_block: List[int],
    save_dir: str,
    *,
    only_diagonal: bool,
    top_k: int,
    allowed_pairs: Optional[Set[Tuple[int, int]]] = None,
):
    L1 = len(prod_concat)
    offsets = [0]
    for db in DB_sizes:
        offsets.append(offsets[-1] + db)

    for i in range(L1):
        prod_i = prod_concat[i]
        if prod_i is None:
            continue
        for k, j in enumerate(js_block):
            if only_diagonal and i != j:
                continue
            if allowed_pairs is not None and (i, j) not in allowed_pairs:
                continue
            sl = prod_i[:, offsets[k]:offsets[k + 1]]

            kA = min(top_k, sl.shape[1])
            kB = min(top_k, sl.shape[0])

            valsA, idxA = torch.topk(sl, k=kA, dim=1, largest=True, sorted=True)
            valsB, idxB = torch.topk(sl, k=kB, dim=0, largest=True, sorted=True)
            valsA = valsA / total_n
            valsB = valsB / total_n
            torch.nan_to_num_(valsA, nan=0.0, posinf=0.0, neginf=0.0)
            torch.nan_to_num_(valsB, nan=0.0, posinf=0.0, neginf=0.0)

            pathA = os.path.join(save_dir, f"nn_A_layer{i}_vs_B_layer{j}_top{top_k}.pt")
            torch.save((idxA.to(torch.int32).cpu(), valsA.cpu()), pathA)

            pathB = os.path.join(save_dir, f"nn_B_layer{j}_vs_A_layer{i}_top{top_k}.pt")
            torch.save((idxB.transpose(0, 1).to(torch.int32).cpu(),
                        valsB.transpose(0, 1).cpu()), pathB)

            del valsA, idxA, valsB, idxB

        prod_concat[i] = None  # type: ignore
        del prod_i


def save_corr_block_from_prod_concat(
    prod_concat: List[torch.Tensor],
    total_n: float,
    DB_sizes: List[int],
    js_block: List[int],
    save_dir: str,
    *,
    only_diagonal: bool,
    allowed_pairs: Optional[Set[Tuple[int, int]]] = None,
):
    L1 = len(prod_concat)
    offsets = [0]
    for db in DB_sizes:
        offsets.append(offsets[-1] + db)

    for i in range(L1):
        prod_i = prod_concat[i]
        if prod_i is None:
            continue
        for k, j in enumerate(js_block):
            if only_diagonal and i != j:
                continue
            if allowed_pairs is not None and (i, j) not in allowed_pairs:
                continue
            sl = prod_i[:, offsets[k]:offsets[k + 1]]
            corr_cpu = sl.cpu()
            corr_cpu.div_(total_n)
            torch.nan_to_num_(corr_cpu, nan=0.0, posinf=0.0, neginf=0.0)
            out_path = os.path.join(save_dir, f"corr_layer{i}_vs_layer{j}.pt")
            torch.save(corr_cpu, out_path)
            del corr_cpu

        prod_concat[i] = None  # type: ignore
        del prod_i
