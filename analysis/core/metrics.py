"""Shared numerical helpers for Pareto-front analysis.

These operate on raw numpy arrays so they are decoupled from the
pipeline on-disk formats; the loaders (`load_smoo`, `load_pdq`) feed
them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def genotype_matrix(trace: pd.DataFrame, image_dim: int) -> np.ndarray:
    """Extract the image-gene slice as an ``int64`` matrix from a trace."""
    mat = np.stack(trace["genotype"].to_list()).astype(np.int64)
    return mat[:, :image_dim]


def n_active_per_row(img_geno: np.ndarray) -> np.ndarray:
    """Count non-zero (i.e. perturbed) image genes per row."""
    return (img_geno != 0).sum(axis=1).astype(np.int64)


def pareto_front_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return indices of 2D minimisation Pareto-optimal points.

    Standard sweep: sort by ``x`` ascending; keep the point if its ``y``
    beats the running-minimum ``y`` by more than a numerical tolerance.
    """
    order = np.argsort(x, kind="mergesort")
    keep, best_y = [], np.inf
    for i in order:
        if y[i] < best_y - 1e-12:
            keep.append(i)
            best_y = y[i]
    return np.asarray(keep, dtype=np.int64)


def hypervolume_2d(
    px: np.ndarray, py: np.ndarray, ref_x: float, ref_y: float,
) -> float:
    """2D hypervolume dominated by ``(px, py)`` below the reference point.

    Points with ``px ≥ ref_x`` or ``py ≥ ref_y`` are filtered out first.
    Returns ``0.0`` on empty input or when the filter leaves no points.
    """
    if len(px) == 0:
        return 0.0
    order = np.argsort(px)
    px, py = px[order], py[order]
    valid = (px < ref_x) & (py < ref_y)
    px, py = px[valid], py[valid]
    if len(px) == 0:
        return 0.0
    hv = 0.0
    for i in range(len(px)):
        next_x = px[i + 1] if i + 1 < len(px) else ref_x
        hv += (next_x - px[i]) * (ref_y - py[i])
    return float(hv)
