"""Prototype a2 — region maps in COMBINATORIAL planes with surveyed border stakes.

Planes:
  A: x = hamming_to_anchor (normalized by genome length), y = n_active_txt
  B: x = rank_sum_img_norm, y = rank_sum_txt_norm

Columns: pdq_s1 (anchor-centered random) | pdq_s2 (path-constrained) | s2 + stakes.
Stakes = hamming-1 argmax straddle midpoints (exact surveyed border crossings):
black = junco<->boa, gold edge = boa->ostrich.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

sys.path.insert(0, "/Users/kaiser/Projects/Masterarbeit")
from analysis.core.style import apply_style, save_fig

ROOT = Path("/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography")
OUT = ROOT / "explore_maps"

CLASS_COLORS = {
    "junco": "#937860",
    "ostrich": "#E6A817",
    "green iguana": "#55A868",
    "boa constrictor": "#C44E52",
    "cello": "#4C72B0",
    "marimba": "#CCB974",
}

apply_style()

cols = ["source", "seed_dir", "pred_label", "top_gap", "n_active_img", "n_active_txt",
        "rank_sum_img_norm", "rank_sum_txt_norm", "hamming_to_anchor", "image_dim"]
pts = pd.read_parquet(ROOT / "exp100/points.parquet", columns=cols,
                      filters=[("prompt_regime", "==", "cat6")])
pts = pts[pts.source.isin(["pdq_s1", "pdq_s2"])]
pts["ham_norm"] = pts.hamming_to_anchor / (pts.image_dim + 19)

sp = pd.read_parquet(ROOT / "exp100/straddle_pairs.parquet")
am = sp[sp.boundary_kind == "argmax"].copy()
am["ham_norm"] = am.hamming_to_anchor_after / (am.image_dim + 19)
am_jb = am[am.label_after.isin(["junco", "boa constrictor"]) &
           am.label_before.isin(["junco", "boa constrictor"])]
am_os = am[(am.label_after == "ostrich") | (am.label_before == "ostrich")]
print(f"argmax stakes: junco<->boa {len(am_jb)}, ostrich {len(am_os)}")

PLANES = {
    "A": dict(x="ham_norm", y="n_active_txt", xl="hamming to anchor (norm.)",
              yl="n_active_txt", xlim=(0, 1.0), ylim=(-0.5, 19.5),
              xe=np.linspace(0, 1.0, 29), ye=np.arange(-0.5, 20.5, 1.0),
              sx="ham_norm", sy="m_n_active_txt"),
    "B": dict(x="rank_sum_img_norm", y="rank_sum_txt_norm", xl="rank_sum_img (norm.)",
              yl="rank_sum_txt (norm.)", xlim=(0, 1.0), ylim=(0, 1.0),
              xe=np.linspace(0, 1.0, 29), ye=np.linspace(0, 1.0, 29),
              sx="m_rank_sum_img_norm", sy="m_rank_sum_txt_norm"),
}


def region_img(df, xc, yc, xe, ye, min_n=4):
    nx, ny = len(xe) - 1, len(ye) - 1
    img = np.zeros((ny, nx, 4))
    ix = np.clip(np.digitize(df[xc], xe) - 1, 0, nx - 1)
    iy = np.clip(np.digitize(df[yc], ye) - 1, 0, ny - 1)
    g = pd.DataFrame({"ix": ix, "iy": iy, "lbl": df.pred_label.to_numpy()})
    for (bx, by), sub in g.groupby(["ix", "iy"]):
        if len(sub) < min_n:
            continue
        vc = sub.lbl.value_counts()
        share = vc.iloc[0] / len(sub)
        r, gg, b = to_rgb(CLASS_COLORS[vc.index[0]])
        a = 0.25 + 0.75 * (share - 1 / 3) / (2 / 3)
        img[by, bx] = (r, gg, b, np.clip(a, 0.15, 1.0))
    return img


fig, axes = plt.subplots(2, 3, figsize=(14.5, 9))
for row, (pk, P) in enumerate(PLANES.items()):
    for col, (src, title) in enumerate([
            ("pdq_s1", "s1: anchor-centered random"),
            ("pdq_s2", "s2: shrink walks (path-constrained)"),
            ("pdq_s2", "s2 + surveyed border stakes")]):
        ax = axes[row, col]
        sub = pts[pts.source == src]
        img = region_img(sub, P["x"], P["y"], P["xe"], P["ye"])
        ax.imshow(img, origin="lower", aspect="auto", interpolation="nearest",
                  extent=(P["xe"][0], P["xe"][-1], P["ye"][0], P["ye"][-1]), zorder=1)
        if col == 2:
            ax.scatter(am_jb[P["sx"]], am_jb[P["sy"]], s=3, c="black", alpha=0.4,
                       linewidths=0, zorder=3)
            ax.scatter(am_os[P["sx"]], am_os[P["sy"]], s=22, c="#E6A817",
                       edgecolors="black", linewidths=0.6, zorder=4)
        ax.set_xlim(*P["xlim"]); ax.set_ylim(*P["ylim"])
        ax.set_xlabel(P["xl"]); ax.set_ylabel(P["yl"])
        ax.set_title(f"{title}  (n={len(sub)})", fontsize=10)
        ax.grid(False)

handles = [Line2D([], [], marker="s", ls="", color=CLASS_COLORS[l], label=l, markersize=9)
           for l in ["junco", "boa constrictor", "ostrich"]]
handles += [Line2D([], [], marker="o", ls="", color="black", label="junco<->boa stake (hamming-1)", markersize=4),
            Line2D([], [], marker="o", ls="", markerfacecolor="#E6A817", markeredgecolor="black",
                   label="boa->ostrich stake", markersize=7)]
fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.012))
fig.suptitle("Argmax region maps — cat6 prompt, combinatorial planes — bin = majority argmax, alpha = share; blank <4 pts\n"
             "stake coords: midpoint descriptors of hamming-1 argmax flips (straddle_pairs)", fontsize=12)
save_fig(fig, OUT / "m2_02_region_combinatorial.png")

# where do stakes sit relative to the regions? quick numbers
print("\nstake n_active_txt distribution (junco<->boa):")
print(am_jb.m_n_active_txt.describe(percentiles=[.1, .5, .9]).round(2))
print("\nstake gene modality:")
print(am.gene_modality.value_counts())
