"""m4_04: descent profiles — best-so-far |g_pair| vs generation, one line per
seed (120 = 40 cells x 3 seeds), grouped by wall type. Right panel: survival
curve = fraction of seeds whose best |g| has entered the eval-noise band
(|g| < tanh(0.38/2) = 0.188, from pair_margin repeat noise q90 = 0.38 lp).
"""
import sys
sys.path.insert(0, "/Users/kaiser/Projects/Masterarbeit")
sys.path.insert(0, "/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography/explore_maps")
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from analysis.core.style import apply_style, save_fig
import m4_common as M

apply_style()
NOISE_G = float(np.tanh(0.38 / 2))  # 0.188

df = M.load_smoo(None, columns=["source", "generation", "anchor_class",
                                "target_class", "level_anchor", "level_target",
                                "seed_dir", "g_pair", "d_img_sem", "d_txt_sem"])


def group_of(row):
    if row.target_class == "boa constrictor" and row.level_target == 1:
        return "wall-boa (lt=1)"
    if row.target_class == "cello" and row.level_anchor == 1:
        return "wall-cello (la=1)"
    if row.target_class in ("marimba", "green iguana", "ostrich"):
        return "easy classes"
    if row.target_class == "junco":
        return None
    return "other boa/cello"


cell_keys = df[["cell", "target_class", "level_anchor", "level_target"]].drop_duplicates()
cell_keys["group"] = cell_keys.apply(group_of, axis=1)
df = df.merge(cell_keys[["cell", "group"]], on="cell")
df = df[df.group.notna()]

GROUPS = ["wall-boa (lt=1)", "wall-cello (la=1)", "other boa/cello", "easy classes"]
GCOL = {"wall-boa (lt=1)": "#D64933", "wall-cello (la=1)": "#8B2E8B",
        "other boa/cello": "#CCB974", "easy classes": "#2274A5"}

# best-so-far |g| per (cell, seed, generation)
df["abs_g"] = df.g_pair.abs()
per_gen = (df.groupby(["group", "cell", "seed_dir", "generation"]).abs_g.min()
             .groupby(["group", "cell", "seed_dir"]).cummin().reset_index())

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2),
                              gridspec_kw={"width_ratios": [1.35, 1]})
for g in GROUPS:
    sub = per_gen[per_gen.group == g]
    for keys, line in sub.groupby(["cell", "seed_dir"]):
        ax.plot(line.generation, line.abs_g,
                color=GCOL[g], alpha=0.22, lw=0.8)
    med = sub.groupby("generation").abs_g.median()
    ax.plot(med.index, med, color=GCOL[g], lw=2.6, label=g)
ax.axhspan(0, NOISE_G, color="0.85", zorder=0)
ax.text(196, NOISE_G - 0.012, "eval-noise band: |g| < 0.188 (= tanh(0.38 lp / 2), repeat q90)",
        ha="right", va="top", fontsize=8, color="0.35")
ax.set_ylim(0, 1.02)
ax.set_xlim(0, 199)
ax.set_xlabel("generation")
ax.set_ylabel("best-so-far min |g_pair|  (distance-to-boundary proxy)")
ax.set_title("Descent profiles — one line per seed (n=120), bold = group median")
ax.legend(loc="center left", fontsize=8)

# survival: fraction of seeds that have touched the noise band by generation t
gens = np.arange(200)
for g in GROUPS:
    sub = per_gen[per_gen.group == g]
    first_touch = (sub[sub.abs_g < NOISE_G]
                   .groupby(["cell", "seed_dir"]).generation.min())
    n_seeds = sub.groupby(["cell", "seed_dir"]).ngroups
    frac = [(first_touch <= t).sum() / n_seeds for t in gens]
    ax2.plot(gens, frac, color=GCOL[g], lw=2.2,
             label=f"{g} (n={n_seeds})")
ax2.set_xlabel("generation")
ax2.set_ylabel("fraction of seeds that have touched\nthe |g|<0.188 boundary band")
ax2.set_ylim(0, 1.02)
ax2.set_xlim(0, 199)
ax2.set_title("Boundary-touch survival curve")
ax2.legend(loc="center right", fontsize=8)

fig.suptitle("Approach to the boundary over time  [smoo, pair2; junco anchor; "
             "wall groups per Exp-100 finding]", fontsize=11)
save_fig(fig, Path(M.OUT) / "m4_04_descent_profiles.png")
