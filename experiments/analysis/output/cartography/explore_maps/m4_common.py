"""Shared loading/normalization for the m4 time-map prototypes.

Axes convention for pooled-per-cell views: d_img_sem and d_txt_sem are divided
by each seed's own q99 (per-seed scale differs up to ~5x within a cell), then
clipped to [0, 1.25]. State this in every caption.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE = "/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography/exp100"
OUT = "/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography/explore_maps"

WALL_BOA = "boa constrictor L0-1"
WALL_CELLO = "cello L1-1"
EASY_MARIMBA = "marimba L2-1"
EASY_IGUANA = "green iguana L2-0"

SMOO_COLS = ["source", "generation", "anchor_class", "target_class",
             "level_anchor", "level_target", "seed_dir", "g_pair",
             "d_img_sem", "d_txt_sem", "n_active_img", "n_active_txt",
             "pair_margin"]


def add_cell(df):
    df["cell"] = (df.target_class.astype(str) + " L" +
                  df.level_anchor.astype(str) + "-" + df.level_target.astype(str))
    return df


def load_smoo(cells=None, columns=SMOO_COLS):
    df = pq.read_table(f"{BASE}/points.parquet", columns=columns).to_pandas()
    df = df[(df.source == "smoo") & (df.anchor_class == "junco")].copy()
    add_cell(df)
    if cells is not None:
        df = df[df.cell.isin(cells)].copy()
    # per-seed q99 normalization of semantic axes
    for c in ("d_img_sem", "d_txt_sem"):
        q = df.groupby("seed_dir")[c].transform(lambda s: s.quantile(0.99))
        df[c + "_n"] = (df[c] / q).clip(upper=1.25)
    return df


def load_transects(cells=None):
    t = pq.read_table(f"{BASE}/transects.parquet").to_pandas()
    add_cell(t)
    if cells is not None:
        t = t[t.cell.isin(cells)].copy()
    return t


def binned_median_g(d, nbins=20, min_n=5, xcol="d_img_sem_n", ycol="d_txt_sem_n",
                    extent=(0, 1.25, 0, 1.25)):
    """Median g_pair on a 2D grid; bins with < min_n points are NaN.

    Returns (grid, counts, xedges, yedges)."""
    xe = np.linspace(extent[0], extent[1], nbins + 1)
    ye = np.linspace(extent[2], extent[3], nbins + 1)
    cnt, _, _ = np.histogram2d(d[xcol], d[ycol], bins=[xe, ye])
    ssum, _, _ = np.histogram2d(d[xcol], d[ycol], bins=[xe, ye], weights=d.g_pair)
    # median via sort-free approximation is wrong; do true median with pandas
    xb = np.clip(np.digitize(d[xcol], xe) - 1, 0, nbins - 1)
    yb = np.clip(np.digitize(d[ycol], ye) - 1, 0, nbins - 1)
    med = pd.Series(d.g_pair.values).groupby([xb, yb]).median()
    grid = np.full((nbins, nbins), np.nan)
    for (i, j), v in med.items():
        grid[i, j] = v
    grid[cnt < min_n] = np.nan
    return grid, cnt, xe, ye
