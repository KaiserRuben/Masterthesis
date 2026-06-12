"""Prototype b — SUB-BASIN cartography: what argmax hides beneath the boa sea.

Row 1: runner-up class maps (points whose argmax = boa): which class is 2nd?
        s1 semantic plane | s2 rank-sum plane | s2 hamming x n_active_txt plane.
Row 2: submerged pair-target lead fields mean(lp_target - lp_junco) over the
        s2 rank-sum plane, restricted to cells targeting that class:
        green iguana | cello | marimba. Diverging map centered 0;
        |lead| < 0.38 lp (repeat noise) shown white.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb, TwoSlopeNorm
from matplotlib.lines import Line2D

sys.path.insert(0, "/Users/kaiser/Projects/Masterarbeit")
from analysis.core.style import apply_style, save_fig

ROOT = Path("/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography")
OUT = ROOT / "explore_maps"

CAT6 = ["junco", "ostrich", "green iguana", "boa constrictor", "cello", "marimba"]
CLASS_COLORS = {
    "junco": "#937860", "ostrich": "#E6A817", "green iguana": "#55A868",
    "boa constrictor": "#C44E52", "cello": "#4C72B0", "marimba": "#CCB974",
}
NOISE_LP = 0.38

apply_style()

cols = ["source", "seed_dir", "pred_label", "logprobs", "d_img_sem", "d_txt_sem",
        "n_active_txt", "rank_sum_img_norm", "rank_sum_txt_norm",
        "hamming_to_anchor", "image_dim", "target_class"]
pts = pd.read_parquet(ROOT / "exp100/points.parquet", columns=cols,
                      filters=[("prompt_regime", "==", "cat6")])
pts = pts[pts.source.isin(["pdq_s1", "pdq_s2"])].reset_index(drop=True)
lp = np.stack(pts["logprobs"].to_numpy())
order = np.argsort(-lp, axis=1)
pts["runner"] = np.array(CAT6)[order[:, 1]]
for i, c in enumerate(CAT6):
    pts[f"lp_{i}"] = lp[:, i]
pts["ham_norm"] = pts.hamming_to_anchor / (pts.image_dim + 19)

s1 = pts[pts.source == "pdq_s1"].copy()
q99 = s1.groupby("seed_dir").d_img_sem.transform(lambda s: s.quantile(0.99))
s1["x_sem"] = (s1.d_img_sem / q99).clip(0, 1.15)

s2 = pts[pts.source == "pdq_s2"]


def cat_img(df, xc, yc, xe, ye, label_col, min_n=4):
    nx, ny = len(xe) - 1, len(ye) - 1
    img = np.zeros((ny, nx, 4))
    ix = np.clip(np.digitize(df[xc], xe) - 1, 0, nx - 1)
    iy = np.clip(np.digitize(df[yc], ye) - 1, 0, ny - 1)
    g = pd.DataFrame({"ix": ix, "iy": iy, "lbl": df[label_col].to_numpy()})
    for (bx, by), sub in g.groupby(["ix", "iy"]):
        if len(sub) < min_n:
            continue
        vc = sub.lbl.value_counts()
        share = vc.iloc[0] / len(sub)
        r, gg, b = to_rgb(CLASS_COLORS[vc.index[0]])
        a = np.clip(0.25 + 0.75 * (share - 1 / 3) / (2 / 3), 0.15, 1.0)
        img[by, bx] = (r, gg, b, a)
    return img


def field_img(df, xc, yc, xe, ye, vals, min_n=4):
    nx, ny = len(xe) - 1, len(ye) - 1
    out = np.full((ny, nx), np.nan)
    ix = np.clip(np.digitize(df[xc], xe) - 1, 0, nx - 1)
    iy = np.clip(np.digitize(df[yc], ye) - 1, 0, ny - 1)
    g = pd.DataFrame({"ix": ix, "iy": iy, "v": vals})
    agg = g.groupby(["ix", "iy"]).v.agg(["mean", "size"])
    for (bx, by), row in agg.iterrows():
        if row["size"] >= min_n:
            out[by, bx] = row["mean"]
    return out


fig, axes = plt.subplots(2, 3, figsize=(14.5, 9))

# --- row 1: runner-up maps among boa-top points ---------------------------
boa1 = s1[s1.pred_label == "boa constrictor"]
boa2 = s2[s2.pred_label == "boa constrictor"]
panels1 = [
    (boa1, "x_sem", "d_txt_sem", np.linspace(0, 1.15, 16), np.linspace(0, 1, 16),
     "s1 semantic plane", "image dist (per-seed norm.)", "text dist (cosine)"),
    (boa2, "rank_sum_img_norm", "rank_sum_txt_norm", np.linspace(0, 1, 29), np.linspace(0, 1, 29),
     "s2 rank-sum plane", "rank_sum_img (norm.)", "rank_sum_txt (norm.)"),
    (boa2, "ham_norm", "n_active_txt", np.linspace(0, 1, 29), np.arange(-0.5, 20.5, 1),
     "s2 hamming x n_active_txt", "hamming to anchor (norm.)", "n_active_txt"),
]
for ax, (df, xc, yc, xe, ye, title, xl, yl) in zip(axes[0], panels1):
    img = cat_img(df, xc, yc, xe, ye, "runner")
    ax.imshow(img, origin="lower", aspect="auto", interpolation="nearest",
              extent=(xe[0], xe[-1], ye[0], ye[-1]), zorder=1)
    ax.set_title(f"runner-up | argmax=boa — {title} (n={len(df)})", fontsize=9.5)
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(False)

# --- row 2: submerged target lead fields (s2, rank plane) ------------------
xe = np.linspace(0, 1, 29); ye = np.linspace(0, 1, 29)
for ax, tgt in zip(axes[1], ["green iguana", "cello", "marimba"]):
    sub = s2[s2.target_class == tgt]
    lead = sub[f"lp_{CAT6.index(tgt)}"] - sub["lp_0"]
    F = field_img(sub, "rank_sum_img_norm", "rank_sum_txt_norm", xe, ye, lead.to_numpy())
    Fm = np.where(np.abs(F) < NOISE_LP, 0.0, F)
    vmax = np.nanquantile(np.abs(F), 0.99)
    pc = ax.imshow(Fm, origin="lower", aspect="auto", interpolation="nearest",
                   extent=(0, 1, 0, 1), cmap="RdBu_r",
                   norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=min(vmax, 1e-6) and vmax))
    # actually want: vmin=-vmax fixed; positive side likely absent
    fig.colorbar(pc, ax=ax, shrink=0.8, label=f"lp({tgt}) - lp(junco)")
    ax.set_title(f"submerged lead: {tgt}-target cells (s2, n={len(sub)})", fontsize=9.5)
    ax.set_xlabel("rank_sum_img (norm.)"); ax.set_ylabel("rank_sum_txt (norm.)")
    ax.grid(False)
    print(f"{tgt}: lead range [{np.nanmin(F):.2f}, {np.nanmax(F):.2f}], "
          f"frac bins > -2 lp: {np.nanmean(F > -2):.3f}")

handles = [Line2D([], [], marker="s", ls="", color=CLASS_COLORS[c], label=c, markersize=9)
           for c in CAT6 if c != "boa constrictor"]
fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.012))
fig.suptitle("Sub-basin cartography — cat6 prompt — beneath the boa argmax sea\n"
             "row 1: 2nd-place class (bin majority, alpha = share); row 2: pair-target logprob lead vs junco "
             f"(white = |lead| < {NOISE_LP} lp noise)", fontsize=12)
save_fig(fig, OUT / "m2_03_subbasin.png")

# diagnostics: runner-up by region
print("\nrunner-up among boa-top, by source:")
print(pts[pts.pred_label == "boa constrictor"].groupby("source").runner.value_counts(normalize=True).round(3))
