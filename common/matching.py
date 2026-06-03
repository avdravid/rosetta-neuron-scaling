"""Schema-agnostic primitives shared by the language and vision matching pipelines.

Both pipelines reduce their per-modality features (byte-aligned token spans for
language, spatial patch grids for vision) to a 2D tensor of activations, then
compute a Pearson correlation matrix between neurons of model A and model B and
find mutual top-K neighbors. Those two steps are what lives here.
"""

from __future__ import annotations

import torch


def correlation_matrix(
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Pearson correlation between every column of ``a`` and every column of ``b``.

    Args:
        a: ``(N, Da)`` activations of model A across N aligned samples / spans.
        b: ``(N, Db)`` activations of model B across the same N samples / spans.
        eps: numerical floor for the per-neuron standard deviation.

    Returns:
        ``(Da, Db)`` correlation matrix; ``out[i, j]`` is ``corr(a[:, i], b[:, j])``.
    """
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"correlation_matrix: a and b must share dim 0, got {a.shape} vs {b.shape}"
        )
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    a = a - a.mean(dim=0, keepdim=True)
    b = b - b.mean(dim=0, keepdim=True)
    a_std = a.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    b_std = b.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    a = a / a_std
    b = b / b_std
    n = a.shape[0]
    return (a.T @ b) / n


def mutual_top_k(corr: torch.Tensor, k: int) -> torch.Tensor:
    """Find mutual top-K neighbor pairs ("best buddies") in a correlation matrix.

    A pair ``(i, j)`` is mutual if ``j`` is among the top-K columns for row ``i``
    AND ``i`` is among the top-K rows for column ``j``.

    Args:
        corr: ``(Da, Db)`` correlation matrix from :func:`correlation_matrix`.
        k: neighborhood size on each side.

    Returns:
        ``(M, 2)`` long tensor of mutual ``(row, col)`` indices, M <= min(Da, Db).
    """
    if corr.ndim != 2:
        raise ValueError(f"mutual_top_k: corr must be 2D, got shape {tuple(corr.shape)}")
    if k <= 0:
        raise ValueError(f"mutual_top_k: k must be positive, got {k}")
    da, db = corr.shape
    k_row = min(k, db)
    k_col = min(k, da)
    row_nn = corr.topk(k_row, dim=1).indices  # (Da, k_row): top columns for each row
    col_nn = corr.topk(k_col, dim=0).indices  # (k_col, Db): top rows for each column
    row_mask = torch.zeros((da, db), dtype=torch.bool, device=corr.device)
    col_mask = torch.zeros((da, db), dtype=torch.bool, device=corr.device)
    row_mask.scatter_(1, row_nn, True)
    col_mask.scatter_(0, col_nn, True)
    return (row_mask & col_mask).nonzero(as_tuple=False)
