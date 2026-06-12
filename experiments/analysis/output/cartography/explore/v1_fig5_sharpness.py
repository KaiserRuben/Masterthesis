"""v1 Fig 5: boundary sharpness — |margin jump| across one gene edit.
v1 Fig 6: stake hamming distance vs evolutionary hardness ranking."""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, "/Users/kaiser/Projects/Masterarbeit")
from analysis.core.style import apply_style, save_fig

apply_style(); warnings.filterwarnings("ignore")
ROOT = "/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography/exp100"
OUT = Path("/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography/explore")

S = pd.read_parquet(f"{ROOT}/straddle_pairs.parquet")
S["cell"] = (S.target_class.str.split().str[-1] + " (" + S.level_anchor.astype(str)
             + "," + S.level_target.astype(str) + ")")
S["jump"] = (S.margin_after - S.margin_before).abs()
S["grp"] = S.txt_group.fillna("img")

# which cells have zero text stakes?
txt_per_cell = S[S.gene_modality == "txt"].groupby("cell").size()
all_cells = S.cell.unique()
zero_txt = sorted(set(all_cells) - set(txt_per_cell.index))
print("cells with ZERO text stakes:", zero_txt)
print("\ntxt share, showcase cells:")
for c in ["constrictor (0,1)", "constrictor (0,0)", "lizard? n/a", "marimba (0,1)", "cello (1,1)", "cello (1,0)"]:
    sub = S[S.cell == c]
    if len(sub):
        print(f"  {c}: n={len(sub)}, txt_share={(sub.gene_modality=='txt').mean():.3f}")

WALLS = {"constrictor (0,1)", "cello (1,1)"}
HARDISH = {"constrictor (1,1)", "ostrich (1,0)", "cello (1,0)"}
S["wallcat"] = np.where(S.cell.isin(WALLS), "wall",
               np.where(S.cell.isin(HARDISH), "near-wall", "normal"))

# ---------------- Fig 5 ----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
pm = S[S.boundary_kind == "pair_margin"]

# (a) jump distribution by modality group (pair_margin stakes: jump = full sign-crossing)
ax = axes[0]
groups = ["img", "mlm", "frag", "charnoise", "saliency"]
data = [np.log10(pm[pm.grp == g].jump.clip(1e-4)) for g in groups]
bp = ax.boxplot(data, labels=groups, showfliers=False, patch_artist=True)
for p, c in zip(bp["boxes"], ["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd"]):
    p.set_facecolor(c); p.set_alpha(0.6)
ax.set_ylabel("log10 |margin_after - margin_before|")
ax.set_title("(a) margin jump across one gene edit\n(pair_margin stakes)")
meds = pm.groupby("grp").jump.median()
print("\nmedian jump by group (pair_margin):", meds.round(3).to_dict())

# (b) jump CDFs: wall vs near-wall vs normal cells
ax = axes[1]
for cat, col in [("wall", "#d62728"), ("near-wall", "#ff7f0e"), ("normal", "#1f77b4")]:
    v = np.sort(pm[pm.wallcat == cat].jump.values)
    if len(v):
        ax.plot(v, np.linspace(0, 1, len(v)), color=col, label=f"{cat} (n={len(v)}, med={np.median(v):.2f})")
v_am = np.sort(S[(S.boundary_kind == "argmax")].jump.values)
ax.plot(v_am, np.linspace(0, 1, len(v_am)), color="gray", ls="--", label=f"argmax stakes all (med={np.median(v_am):.2f})")
ax.set_xscale("log"); ax.set_xlabel("|margin jump| (log)"); ax.set_ylabel("CDF")
ax.set_title("(b) jump CDF: wall vs normal cells"); ax.legend(fontsize=7)

# (c) cliff vs slope: |margin_before| vs |margin_after| for pair_margin stakes
ax = axes[2]
hb = ax.hexbin(pm.margin_before.abs().clip(1e-3), pm.margin_after.abs().clip(1e-3),
               xscale="log", yscale="log", gridsize=35, cmap="magma", mincnt=1)
ax.plot([1e-3, 30], [1e-3, 30], color="cyan", lw=0.8)
ax.set_xlabel("|margin before| (dist to 0 before)"); ax.set_ylabel("|margin after|")
ax.set_title("(c) before vs after distance to boundary")
fig.colorbar(hb, ax=ax, shrink=0.8)
fig.suptitle("Boundary sharpness at surveyed stakes", y=1.02)
save_fig(fig, OUT / "v1_fig5_sharpness.png", tight=False)

# how cliff-like: fraction of pair_margin stakes with jump > 1 lp unit; > 2
print("frac pair_margin jump>1:", (pm.jump > 1).mean().round(3), " >2:", (pm.jump > 2).mean().round(3))
print("jump by wallcat median:", pm.groupby("wallcat").jump.median().round(3).to_dict())
print("img vs txt jump median:", pm.groupby("gene_modality").jump.median().round(3).to_dict())

# ---------------- Fig 6: stake hamming vs hardness ----------------
P = pd.read_parquet(f"{ROOT}/points.parquet", columns=[
    "source", "target_class", "level_anchor", "level_target", "g_pair", "image_dim"])
sm = P[P.source == "smoo"]
sm = sm.assign(cell=(sm.target_class.str.split().str[-1] + " (" + sm.level_anchor.astype(str)
                     + "," + sm.level_target.astype(str) + ")"))
hard = sm.groupby("cell").g_pair.agg(p05="min", frac_cross=lambda s: (s < 0).mean())
hard["min_abs_g"] = sm.groupby("cell").g_pair.apply(lambda s: s.abs().min())

S["ham_n"] = S.hamming_to_anchor_after / (S.image_dim + 19.0)
hamstats = S.groupby("cell").ham_n.agg(["min", "median", "count"])
M = hamstats.join(hard)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
order = M.sort_values("frac_cross").index
import matplotlib.cm as cm
for i, c in enumerate(order):
    sub = S[S.cell == c].ham_n
    col = "#d62728" if c in WALLS else ("#ff7f0e" if c in HARDISH else "#1f77b4")
    ax.scatter(np.full(len(sub), i) + np.random.uniform(-0.25, 0.25, len(sub)), sub,
               s=3, alpha=0.25, color=col)
    ax.scatter([i], [sub.min()], s=25, marker="_", color="k", zorder=5)
ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90, fontsize=6)
ax.set_ylabel("hamming_to_anchor / n_genes")
ax.set_title("(a) stake distance from anchor per cell\n(cells ordered by smoo frac_cross, hardest left;\nblack tick = closest approach)")

ax = axes[1]
col = ["#d62728" if c in WALLS else ("#ff7f0e" if c in HARDISH else "#1f77b4") for c in M.index]
ax.scatter(M.frac_cross.clip(1e-4), M["min"], c=col, s=28)
for c in list(WALLS | HARDISH):
    if c in M.index:
        ax.annotate(c, (max(M.loc[c, "frac_cross"], 1e-4), M.loc[c, "min"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
ax.set_xscale("log")
ax.set_xlabel("smoo frac of points crossing (pair2)  [hardness, low=hard]")
ax.set_ylabel("min stake hamming (norm)  [cat6 closest surveyed approach]")
rho = M[["frac_cross", "min"]].corr(method="spearman").iloc[0, 1]
ax.set_title(f"(b) closest surveyed boundary vs evolutionary hardness\nSpearman rho = {rho:.2f}")
fig.suptitle("Stake positions vs anchor and hardness", y=1.0)
save_fig(fig, OUT / "v1_fig6_hamming_vs_hardness.png")
print("\nSpearman(frac_cross, min ham):", round(rho, 3))
print("Spearman(frac_cross, median ham):", round(M[["frac_cross", "median"]].corr(method="spearman").iloc[0, 1], 3))
print(M.sort_values("frac_cross").head(8).round(3).to_string())
