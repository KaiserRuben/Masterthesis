"""L2 transect profiles + folding evidence.

Walk key = (seed_dir, flip_id)  -- flip_id alone is NOT unique.
For each stage-2 walk: order accepted steps by step index, count
  - pair_margin sign changes (pair-boundary re-crossings)
  - pred_label switches (argmax region re-entries)
|crossings| > 1 = re-entrant boundary = folding evidence.

Outputs: crossing-count distributions per cell + gallery of most-folded walks.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, "/Users/kaiser/Projects/Masterarbeit")
from analysis.core.style import apply_style, save_fig

ROOT = Path("/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography")
OUT = ROOT / "explore"

CLASS_COLORS = {
    "junco": "#937860", "ostrich": "#E6A817", "green iguana": "#55A868",
    "boa constrictor": "#C44E52", "cello": "#4C72B0", "marimba": "#CCB974",
}

apply_style()

tr = pd.read_parquet(ROOT / "exp100/transects.parquet")
tr["walk"] = tr.seed_dir.astype(str) + "::" + tr.flip_id.astype(str)
print(f"transect rows: {len(tr)}, walks: {tr.walk.nunique()}")

# ---------------------------------------------------------------- crossing counts
def sign_changes(m: np.ndarray) -> int:
    s = np.sign(m)
    s = s[s != 0]
    return int(np.sum(s[1:] != s[:-1])) if len(s) > 1 else 0

rows = []
for walk, g in tr.groupby("walk"):
    g = g.sort_values("step")
    acc = g[g.accepted]
    if len(acc) < 2:
        continue
    m = acc.pair_margin.to_numpy()
    labels = acc.pred_label.to_numpy()
    lbl_switch = int(np.sum(labels[1:] != labels[:-1]))
    # hamming positions of margin sign changes
    s = np.sign(m)
    pos = acc.hamming_to_anchor.to_numpy()
    nz = s != 0
    s_nz, pos_nz = s[nz], pos[nz]
    cross_pos = pos_nz[1:][s_nz[1:] != s_nz[:-1]]
    rows.append(dict(
        walk=walk, target=g.target_class.iloc[0],
        la=g.level_anchor.iloc[0], lt=g.level_target.iloc[0],
        seed_idx=g.seed_idx.iloc[0], n_acc=len(acc),
        ham_start=acc.hamming_to_anchor.iloc[0], ham_end=acc.hamming_to_anchor.iloc[-1],
        crossings=sign_changes(m), label_switches=lbl_switch,
        cross_pos=cross_pos.tolist(),
        margin_start=m[0], margin_end=m[-1],
    ))
W = pd.DataFrame(rows)
print(f"\nwalks analyzed: {len(W)}")
print("\n=== pair_margin crossing-count distribution ===")
print(W.crossings.describe().round(2))
print(W.crossings.value_counts().sort_index().to_string())
print(f"\nshare with >1 crossing (re-entrant / folded): {(W.crossings > 1).mean():.2%}")
print(f"share with >3 crossings: {(W.crossings > 3).mean():.2%}")
print("\n=== argmax label-switch distribution ===")
print(W.label_switches.describe().round(2))
print("\n=== per target class ===")
print(W.groupby("target")[["crossings", "label_switches", "n_acc"]]
      .agg(["median", "max", "count"]).round(1).to_string())
print("\n=== per cell, sorted by median crossings ===")
tab = W.groupby(["target", "la", "lt"]).agg(
    med_cross=("crossings", "median"), max_cross=("crossings", "max"),
    med_lbl=("label_switches", "median"), n_walks=("walk", "count")).round(1)
print(tab.sort_values("med_cross", ascending=False).to_string())

# normalize crossings by chain length (longer chains, more chances)
W["cross_per_100"] = 100 * W.crossings / W.n_acc
print("\ncrossings per 100 accepted steps:", W.cross_per_100.describe().round(2).to_dict())

# ---------------------------------------------------------------- fig 1: distributions
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
bins = np.arange(-0.5, W.crossings.max() + 1.5)
axes[0].hist(W.crossings, bins=bins, color="#2274A5", edgecolor="white")
axes[0].axvline(1.5, color="k", ls="--", lw=1)
axes[0].text(1.7, axes[0].get_ylim()[1] * 0.55, "folded\n(>1 crossing)", fontsize=9)
axes[0].set_xlabel("pair_margin sign changes (accepted chain)")
axes[0].set_ylabel("walks")
axes[0].set_title("(a) pair-boundary crossings per walk")

for tgt, g in W.groupby("target"):
    axes[1].scatter(g.crossings + np.random.default_rng(1).uniform(-0.2, 0.2, len(g)),
                    g.label_switches + np.random.default_rng(2).uniform(-0.2, 0.2, len(g)),
                    s=18, color=CLASS_COLORS[tgt], label=tgt, alpha=0.7)
axes[1].plot([0, W.crossings.max()], [0, W.crossings.max()], "k:", lw=0.8)
axes[1].set_xlabel("pair_margin crossings")
axes[1].set_ylabel("argmax label switches")
axes[1].set_title("(b) margin crossings vs region re-entries")
axes[1].legend(fontsize=7)

allpos = np.concatenate([np.asarray(p) for p in W.cross_pos if len(p)])
axes[2].hist(allpos, bins=30, color="#D64933", edgecolor="white")
axes[2].set_xlabel("hamming_to_anchor at crossing")
axes[2].set_ylabel("crossing events")
axes[2].set_title("(c) where on the walk crossings happen")
save_fig(fig, OUT / "v2_crossing_counts.png")

# ---------------------------------------------------------------- fig 2: gallery of most folded walks
# rank by noise-robust (debounced) crossings so the gallery shows real folds
def hysteresis_crossings(m: np.ndarray, band: float) -> int:
    state, n = 0, 0
    for v in m:
        if v > band:
            if state == -1:
                n += 1
            state = 1
        elif v < -band:
            if state == 1:
                n += 1
            state = -1
    return n

W["crossings_hyst"] = [
    hysteresis_crossings(
        tr[(tr.walk == w) & tr.accepted].sort_values("step").pair_margin.to_numpy(), 0.38)
    for w in W.walk]
top = W.sort_values(["crossings_hyst", "crossings"], ascending=False).head(6)
fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=False)
for ax, (_, w) in zip(axes.flat, top.iterrows()):
    g = tr[tr.walk == w.walk].sort_values("step")
    acc = g[g.accepted]
    rej = g[~g.accepted]
    x = acc.hamming_to_anchor.to_numpy()
    m = acc.pair_margin.to_numpy()
    lbl = acc.pred_label.to_numpy()
    # region bands under the curve
    ymin, ymax = min(m.min(), -0.1), max(m.max(), 0.1)
    for i in range(len(x) - 1):
        ax.axvspan(x[i], x[i + 1], color=CLASS_COLORS.get(lbl[i], "#ddd"),
                   alpha=0.18, lw=0)
    ax.plot(x, m, color="k", lw=0.9, zorder=3)
    ax.scatter(rej.hamming_to_anchor, rej.pair_margin, s=4, c="#999999",
               marker="x", alpha=0.4, zorder=2, linewidths=0.5)
    ax.axhline(0, color="#C44E52", lw=1, ls="--", zorder=4)
    ax.invert_xaxis()  # walk goes toward anchor
    ax.set_title(f"{w['target']} (a{w['la']},t{w['lt']}) seed {w['seed_idx']} — "
                 f"{w['crossings']} raw / {w['crossings_hyst']} robust crossings, "
                 f"{w['label_switches']} region switches", fontsize=9)
    ax.axhspan(-0.38, 0.38, color="#999999", alpha=0.12, zorder=0)
    ax.set_xlabel("hamming to anchor (walk direction →)")
    ax.set_ylabel("pair_margin")
handles = [Line2D([], [], marker="s", ls="", color=CLASS_COLORS[l], alpha=0.4, label=l)
           for l in ["junco", "boa constrictor", "ostrich"]]
handles.append(Line2D([], [], marker="x", ls="", color="#999", label="rejected probe"))
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, -0.015))
fig.suptitle("Most folded stage-2 walks — pair_margin along accepted chain, bands = argmax region", y=1.005)
save_fig(fig, OUT / "v2_transect_gallery.png")

W.drop(columns="cross_pos").to_csv(OUT / "v2_walk_crossings.csv", index=False)
print("saved: v2_walk_crossings.csv")

# ---------------------------------------------------------------- noise control
# Anchors are evaluated 3x per (cell, seed): repeat-spread = eval noise floor.
anch = pd.read_parquet(ROOT / "exp100/points.parquet",
                       columns=["seed_dir", "pair_margin", "pred_label"],
                       filters=[("source", "==", "pdq_anchor")])
rep = anch.groupby("seed_dir")["pair_margin"].agg(["std", "count", "mean"])
rep = rep[rep["count"] >= 2]
sigma = rep["std"].median()
print(f"\n=== eval-noise floor from anchor repeats (n={len(rep)} seeds x3) ===")
print(f"median repeat std of pair_margin: {sigma:.4f}")
print(f"share of seeds with nonzero repeat std: {(rep['std'] > 1e-9).mean():.1%}")
print(f"90th pct repeat std: {rep['std'].quantile(0.9):.4f}")
print(f"max repeat std: {rep['std'].max():.4f}")

folded_walks = W[W.crossings > 1].walk
margins_by_walk = {w: tr[(tr.walk == w) & tr.accepted].sort_values("step").pair_margin.to_numpy()
                   for w in folded_walks}
n_all = len(W)
for band in [0.0, 0.1, 0.2, 0.38, 0.5, 1.0]:
    hc = np.asarray([hysteresis_crossings(m, band) for m in margins_by_walk.values()])
    print(f"band = +-{band:.2f} lp: of {len(hc)} folded walks, "
          f"{(hc > 1).sum()} keep >1 crossing "
          f"({(hc > 1).sum() / n_all:.1%} of all walks); "
          f"median {np.median(hc):.0f}, max {hc.max()}")

# amplitude of oscillation: peak |margin| between consecutive crossings
amps = []
for walk in W[W.crossings > 1].walk:
    g = tr[(tr.walk == walk) & tr.accepted].sort_values("step")
    m = g.pair_margin.to_numpy()
    s = np.sign(m)
    seg_start = 0
    for i in range(1, len(m)):
        if s[i] != 0 and s[seg_start] != 0 and s[i] != s[seg_start]:
            amps.append(np.abs(m[seg_start:i]).max())
            seg_start = i
amps = np.asarray(amps)
print(f"\npeak |pair_margin| between crossings: median {np.median(amps):.3f}, "
      f"q90 {np.quantile(amps, .9):.3f}, max {amps.max():.3f} "
      f"(vs noise sigma {sigma:.3f})")
