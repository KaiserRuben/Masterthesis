"""v1 Fig 7: boundary-normal concentration — gene recurrence profiles per showcase cell.
Plus: localization-gain numbers for fig1/2 and cat6 anchor positions."""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
sys.path.insert(0, "/Users/kaiser/Projects/Masterarbeit")
from analysis.core.style import apply_style, save_fig

apply_style(); warnings.filterwarnings("ignore")
ROOT = "/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography/exp100"
OUT = Path("/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography/explore")

S = pd.read_parquet(f"{ROOT}/straddle_pairs.parquet")
S["cell"] = (S.target_class.str.split().str[-1] + " (" + S.level_anchor.astype(str)
             + "," + S.level_target.astype(str) + ")")
S["txt_offset"] = S.gene_idx - S.image_dim

SHOW = ["constrictor (0,1)", "constrictor (0,0)", "iguana (0,1)", "marimba (0,1)", "cello (1,1)"]
fig, axes = plt.subplots(2, len(SHOW), figsize=(4 * len(SHOW), 8))

for j, c in enumerate(SHOW):
    sub = S[S.cell == c]
    # top: sorted per-gene stake counts per seed (concentration profile)
    ax = axes[0, j]
    for sd, g in sub.groupby("seed_dir"):
        vc = g.gene_idx.value_counts().values
        prof = np.sort(vc)[::-1] / vc.sum()
        ax.plot(np.arange(1, len(prof) + 1), np.cumsum(prof), marker=".", ms=3,
                label=f"{sd.split('_')[1]} (n={len(g)})")
        n_genes = g.image_dim.iloc[0] + 19
        ax.plot([1, n_genes], [1 / n_genes, 1], color="gray", lw=0.6, ls="--")
    ax.set_xscale("log"); ax.set_ylim(0, 1.02)
    ax.set_xlabel("gene rank"); ax.set_title(c, fontsize=10)
    if j == 0:
        ax.set_ylabel("cumulative stake share")
    ax.legend(fontsize=6, loc="lower right")
    # bottom: text-gene x seed heatmap (cross-seed recurrence)
    ax = axes[1, j]
    t = sub[sub.gene_modality == "txt"]
    if len(t):
        H = t.groupby(["seed_dir", "txt_offset"]).size().unstack(fill_value=0).reindex(columns=range(19), fill_value=0)
        H = H.div(H.sum(axis=1), axis=0)
        ax.imshow(H.values, aspect="auto", cmap="viridis", vmin=0)
        ax.set_yticks(range(len(H))); ax.set_yticklabels([s.split("_")[1] for s in H.index], fontsize=7)
        ax.set_xticks(range(0, 19, 2))
        for off in (3, 8, 16):
            ax.axvline(off - 0.5, color="w", lw=0.7)
    else:
        ax.text(0.5, 0.5, "no text stakes", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("text gene offset")
    if j == 0:
        ax.set_ylabel("seed")
axes[1, 0].set_title("text-gene share per seed (mlm|frag|char|sal)", fontsize=8, loc="left")
fig.suptitle("Gene recurrence: cumulative concentration per seed (top), text-gene fingerprint across seeds (bottom)", y=1.0)
save_fig(fig, OUT / "v1_fig7_recurrence_profiles.png")

# top-5 share per showcase cell+seed
print("top-5 gene share (cell, seed):")
for c in SHOW:
    for sd, g in S[S.cell == c].groupby("seed_dir"):
        vc = g.gene_idx.value_counts()
        print(f"  {c} {sd.split('_')[1]}: n={len(g)} top5={vc.head(5).sum()/len(g):.2f} top1_gene={vc.index[0]} ({vc.iloc[0]})")

# ---------- localization gain + anchor positions ----------
P = pd.read_parquet(f"{ROOT}/points.parquet", columns=[
    "source", "prompt_regime", "seed_dir", "row_ref", "target_class", "level_anchor", "level_target",
    "rank_sum_img_norm", "rank_sum_txt_norm", "g_pair", "image_dim"])
p2 = P[P.source == "pdq_s2"].copy()
p2["rid"] = pd.to_numeric(p2.row_ref, errors="coerce")
SJ = S.merge(p2[["seed_dir", "rid", "rank_sum_img_norm", "rank_sum_txt_norm"]],
             left_on=["seed_dir", "call_id_after"], right_on=["seed_dir", "rid"], how="left")

def field_grid(x, y, g, nbins=36, sigma=1.3):
    xe = np.linspace(np.nanmin(x), np.nanmax(x), nbins + 1)
    ye = np.linspace(np.nanmin(y), np.nanmax(y), nbins + 1)
    num, _, _ = np.histogram2d(x, y, bins=[xe, ye], weights=g)
    den, _, _ = np.histogram2d(x, y, bins=[xe, ye])
    F = gaussian_filter(num, sigma) / gaussian_filter(den, sigma)
    F[gaussian_filter(den, sigma) < 0.05] = np.nan
    return xe, ye, F.T

CELLS = {"wall boa(0,1)": ("boa constrictor", 0, 1), "ctrl boa(0,0)": ("boa constrictor", 0, 0),
         "mid iguana(0,1)": ("green iguana", 0, 1), "easy marimba(0,1)": ("marimba", 0, 1)}
print("\nlocalization gain (rank-sum projection): med|g@stake| / med|g field|")
for name, (tc, la, lt) in CELLS.items():
    pc = P[(P.target_class == tc) & (P.level_anchor == la) & (P.level_target == lt) & (P.prompt_regime == "cat6")]
    st = SJ[(SJ.target_class == tc) & (SJ.level_anchor == la) & (SJ.level_target == lt) & (SJ.boundary_kind == "pair_margin")]
    xe, ye, F = field_grid(pc.rank_sum_img_norm.values, pc.rank_sum_txt_norm.values, pc.g_pair.values)
    ix = np.clip(np.digitize(st.rank_sum_img_norm, xe) - 1, 0, len(xe) - 2)
    iy = np.clip(np.digitize(st.rank_sum_txt_norm, ye) - 1, 0, len(ye) - 2)
    gst = np.abs(F[iy, ix])
    gall = np.abs(F[~np.isnan(F)])
    print(f"  {name}: med|g@stake|={np.nanmedian(gst):.3f} med|g field|={np.median(gall):.3f} gain={np.nanmedian(gst)/np.median(gall):.2f}")

print("\ncat6 anchor g_pair per showcase cell (pdq_anchor rows):")
pa = P[P.source == "pdq_anchor"]
for name, (tc, la, lt) in CELLS.items():
    g = pa[(pa.target_class == tc) & (pa.level_anchor == la) & (pa.level_target == lt)].g_pair
    print(f"  {name}: anchors g = {sorted(g.round(3).tolist())}")
g = pa[(pa.target_class == "cello") & (pa.level_anchor == 1) & (pa.level_target == 1)].g_pair
print(f"  wall cello(1,1): anchors g = {sorted(g.round(3).tolist())}")
