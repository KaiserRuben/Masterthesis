"""L3 basin onion: argmax composition and pair-flip probability vs hamming radius.

Radius = hamming_to_anchor / (image_dim + 19)  (genome lengths differ per seed).
Pooled composition c_k(r); per-source junco share (sampling-bias check);
per-cell flip probability near the anchor vs the evolutionary hardness ranking.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "/Users/kaiser/Projects/Masterarbeit")
from analysis.core.style import apply_style, save_fig

ROOT = Path("/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography")
OUT = ROOT / "explore"

CLASS_COLORS = {
    "junco": "#937860", "ostrich": "#E6A817", "green iguana": "#55A868",
    "boa constrictor": "#C44E52", "cello": "#4C72B0", "marimba": "#CCB974",
}
TARGET_COLORS = {  # cell target classes
    "boa constrictor": "#C44E52", "cello": "#4C72B0", "green iguana": "#55A868",
    "marimba": "#CCB974", "ostrich": "#E6A817", "junco": "#937860",
}

apply_style()

cols = ["source", "pred_label", "pair_margin", "hamming_to_anchor", "image_dim",
        "n_active_txt", "target_class", "level_anchor", "level_target",
        "anchor_word", "target_word", "seed_idx"]
pts = pd.read_parquet(ROOT / "exp100/points.parquet", columns=cols,
                      filters=[("prompt_regime", "==", "cat6")])
pts = pts[pts.source.isin(["pdq_anchor", "pdq_s1", "pdq_s2"])].copy()
# drop the single reversed-direction probe cell (target_class == junco)
pts = pts[pts.target_class != "junco"]
pts["ham_norm"] = pts.hamming_to_anchor / (pts.image_dim + 19)
pts["flipped"] = pts.pair_margin < 0

BINS = np.linspace(0, 1.0, 21)
pts["rbin"] = pd.cut(pts.ham_norm, BINS, labels=0.5 * (BINS[:-1] + BINS[1:]))

# ------------------------------------------------ fig 1: pooled onion + source check + flip prob
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

# (a) stacked composition
comp = pts.groupby("rbin", observed=True)["pred_label"].value_counts(normalize=True).unstack(fill_value=0)
r = comp.index.astype(float)
bottom = np.zeros(len(comp))
for lbl in ["junco", "boa constrictor", "ostrich"]:
    if lbl in comp:
        axes[0].fill_between(r, bottom, bottom + comp[lbl], color=CLASS_COLORS[lbl],
                             alpha=0.85, label=lbl, step="mid")
        bottom += comp[lbl].to_numpy()
n_per = pts.groupby("rbin", observed=True).size()
axes[0].set_xlabel("hamming to anchor (norm.)")
axes[0].set_ylabel("argmax composition")
axes[0].set_title("(a) pooled basin onion")
axes[0].legend(loc="center right")
ax2 = axes[0].twinx()
ax2.plot(n_per.index.astype(float), n_per.values, color="k", lw=0.8, ls=":")
ax2.set_yscale("log"); ax2.set_ylabel("n points", fontsize=8)
ax2.grid(False); ax2.spines["right"].set_visible(True)

# (b) junco share per source (bias check)
for src, ls in [("pdq_s1", "-"), ("pdq_s2", "--")]:
    sub = pts[pts.source == src]
    js = sub.groupby("rbin", observed=True).apply(
        lambda g: (g.pred_label == "junco").mean(), include_groups=False)
    nn = sub.groupby("rbin", observed=True).size()
    js = js[nn >= 30]
    axes[1].plot(js.index.astype(float), js.values, ls, color="#2274A5", label=src)
anch = (pts[pts.source == "pdq_anchor"].pred_label == "junco").mean()
axes[1].scatter([0], [anch], marker="*", s=140, color="#937860", zorder=5,
                label=f"anchors ({anch:.0%} junco)")
axes[1].set_xlabel("hamming to anchor (norm.)")
axes[1].set_ylabel("junco argmax share")
axes[1].set_title("(b) junco share by source")
axes[1].set_ylim(0, 1)
axes[1].legend()

# (c) pair-flip probability per target class
for tgt, g in pts.groupby("target_class"):
    fp = g.groupby("rbin", observed=True)["flipped"].mean()
    nn = g.groupby("rbin", observed=True).size()
    fp = fp[nn >= 30]
    axes[2].plot(fp.index.astype(float), fp.values, color=TARGET_COLORS[tgt], label=tgt)
axes[2].set_xlabel("hamming to anchor (norm.)")
axes[2].set_ylabel("p(pair_margin < 0)")
axes[2].set_title("(c) pair-flip probability by target class")
axes[2].set_ylim(0, 1)
axes[2].legend(fontsize=8)
save_fig(fig, OUT / "v2_basin_onion.png")

# ------------------------------------------------ fig 2: per-cell flip prob curves, grid by target
targets = ["boa constrictor", "green iguana", "ostrich", "cello", "marimba"]
fig, axes = plt.subplots(1, 5, figsize=(17, 3.6), sharey=True)
lvl_style = {0: "-", 1: "--", 2: ":"}
for ax, tgt in zip(axes, targets):
    g = pts[pts.target_class == tgt]
    for (la, lt), gg in g.groupby(["level_anchor", "level_target"]):
        fp = gg.groupby("rbin", observed=True)["flipped"].mean()
        nn = gg.groupby("rbin", observed=True).size()
        fp = fp[nn >= 20]
        ax.plot(fp.index.astype(float), fp.values, ls=lvl_style[lt],
                color=plt.cm.plasma(la / 2.5), lw=1.4, label=f"a{la} t{lt}")
    ax.set_title(tgt, color=TARGET_COLORS[tgt])
    ax.set_xlabel("ham. norm")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=6, ncol=2)
axes[0].set_ylabel("p(pair flip)")
fig.suptitle("Pair-flip probability vs radius, per cell (color = level_anchor, style = level_target)", y=1.04)
save_fig(fig, OUT / "v2_basin_percell_flip.png")

# ------------------------------------------------ near-anchor metrics vs hardness
near = pts[pts.ham_norm < 0.15]
tab = near.groupby(["target_class", "level_anchor", "level_target"]).agg(
    flip_near=("flipped", "mean"),
    junco_share=("pred_label", lambda s: (s == "junco").mean()),
    n=("flipped", "size"),
).round(3)
print("\n=== near-anchor (ham_norm<0.15) flip rate + junco share per cell ===")
print(tab.sort_values("flip_near", ascending=False).to_string())

# basin radius: largest radius below which junco share >= 0.5 (often undefined)
rows = []
for key, g in pts.groupby(["target_class", "level_anchor", "level_target"]):
    js = g.groupby("rbin", observed=True).apply(
        lambda x: (x.pred_label == "junco").mean(), include_groups=False)
    nn = g.groupby("rbin", observed=True).size()
    js = js[nn >= 20]
    below = js[js < 0.5]
    r50 = float(below.index.astype(float).min()) if len(below) else np.nan
    rows.append((*key, r50, float(js.iloc[0]) if len(js) else np.nan))
r50 = pd.DataFrame(rows, columns=["target", "la", "lt", "r50", "junco_share_innermost"])
print("\n=== junco-basin half radius (first bin with junco share < 0.5) ===")
print(r50.to_string(index=False))
