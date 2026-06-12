"""L3 escape directions: which gene-activity patterns carry stage-1 candidates
out of the junco argmax region (cat6)?

escaped := pred_label != junco (per-cell prompt; junco-anchored seeds).
Uses pdq_s1 only (anchor-centered random sampling, no path constraint).
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

apply_style()

cols = ["source", "pred_label", "n_active_img", "n_active_txt", "image_dim",
        "txt_active_mlm", "txt_active_frag", "txt_active_charnoise", "txt_active_saliency",
        "hamming_to_anchor", "target_class", "level_anchor", "level_target", "top_gap"]
s1 = pd.read_parquet(ROOT / "exp100/points.parquet", columns=cols,
                     filters=[("source", "==", "pdq_s1")])
s1 = s1[s1.target_class != "junco"].copy()
s1["escaped"] = s1.pred_label != "junco"
s1["img_frac"] = s1.n_active_img / s1.image_dim
print(f"stage-1 candidates: {len(s1)}, escape rate {s1.escaped.mean():.1%}")

GROUPS = {"txt_active_mlm": 3, "txt_active_frag": 5,
          "txt_active_charnoise": 8, "txt_active_saliency": 3}

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))

# (a) scatter img_frac x n_active_txt colored by escaped
rng = np.random.default_rng(0)
for esc, c, lbl, sz in [(True, "#C44E52", "escaped (argmax != junco)", 5),
                        (False, "#937860", "stayed junco", 9)]:
    sub = s1[s1.escaped == esc]
    axes[0].scatter(sub.img_frac, sub.n_active_txt + rng.uniform(-0.3, 0.3, len(sub)),
                    s=sz, c=c, alpha=0.35, linewidths=0, label=lbl, rasterized=True)
axes[0].set_xlabel("image activity fraction")
axes[0].set_ylabel("n_active_txt")
axes[0].set_title("(a) stage-1 candidates")
axes[0].legend(loc="lower center", fontsize=8)

# (b) escape rate vs n_active_txt, stratified by image activity — pooled cat6
# (s1 alone never samples low txt; pool all PDQ sources, note path-selection bias)
allp = pd.read_parquet(ROOT / "exp100/points.parquet",
                       columns=["pred_label", "n_active_txt", "n_active_img",
                                "image_dim", "target_class"],
                       filters=[("prompt_regime", "==", "cat6")])
allp = allp[allp.target_class != "junco"].copy()
allp["escaped"] = allp.pred_label != "junco"
allp["img_frac"] = allp.n_active_img / allp.image_dim
allp["is_boa_cell"] = allp.target_class == "boa constrictor"
strata = [("img < 10%", allp.img_frac < 0.10, "#2274A5"),
          ("img >= 60%", allp.img_frac >= 0.60, "#D64933")]
for name, mask, c in strata:
    for boa, ls in [(False, "-"), (True, "--")]:
        g = (allp[mask & (allp.is_boa_cell == boa)]
             .groupby("n_active_txt")["escaped"].agg(["mean", "size"]))
        g = g[g["size"] >= 40]
        axes[1].plot(g.index, g["mean"], ls, marker="o", color=c, ms=3, lw=1.2,
                     label=f"{name}, {'boa cells' if boa else 'other cells'}")
axes[1].set_xlabel("n_active_txt")
axes[1].set_ylabel("p(argmax != junco)")
axes[1].set_ylim(0, 1.02)
axes[1].set_xticks(range(0, 20, 2))
axes[1].set_title("(b) escape rate vs text activity (all cat6 points)")
axes[1].legend(loc="center right", fontsize=7)

# (c) text-operator-group activity: stayed vs escaped (normalized by group size)
labels, stay_v, esc_v = [], [], []
for col, size in GROUPS.items():
    labels.append(col.replace("txt_active_", ""))
    stay_v.append(s1.loc[~s1.escaped, col].mean() / size)
    esc_v.append(s1.loc[s1.escaped, col].mean() / size)
x = np.arange(len(labels))
axes[2].bar(x - 0.18, stay_v, 0.36, color="#937860", label="stayed junco")
axes[2].bar(x + 0.18, esc_v, 0.36, color="#C44E52", label="escaped")
axes[2].set_xticks(x, labels)
axes[2].set_ylabel("mean fraction of group active")
axes[2].set_title("(c) text-operator group activity")
axes[2].legend(fontsize=8)
fig.suptitle("Stage-1 escapes from the junco argmax region (cat6, pdq_s1)", y=1.04)
save_fig(fig, OUT / "v2_escape_directions.png")

# conditional analysis: text effect at fixed image activity
print("\n=== p(escape) in img_frac x n_active_txt bins ===")
s1["txt_bin"] = pd.cut(s1.n_active_txt, [-1, 5, 12, 19], labels=["txt 0-5", "txt 6-12", "txt 13-19"])
s1["img_bin"] = pd.cut(s1.img_frac, [-0.01, 0.1, 0.5, 0.85, 1.0],
                       labels=["img 0-10%", "img 10-50%", "img 50-85%", "img 85-100%"])
piv = s1.pivot_table(index="img_bin", columns="txt_bin", values="escaped",
                     aggfunc=["mean", "size"], observed=True)
print(piv.round(2).to_string())

# which group matters conditionally: escape rate by group activity at matched n_active_txt
print("\n=== escape rate vs group dominance (rows with n_active_txt in 6-12) ===")
mid = s1[(s1.n_active_txt >= 6) & (s1.n_active_txt <= 12)].copy()
for col, size in GROUPS.items():
    frac = mid[col] / size
    hi = mid[frac >= 0.67]
    lo = mid[frac <= 0.33]
    print(f"{col.replace('txt_active_', ''):>10}: high-activity escape {hi.escaped.mean():.2f} "
          f"(n={len(hi)}), low {lo.escaped.mean():.2f} (n={len(lo)})")
