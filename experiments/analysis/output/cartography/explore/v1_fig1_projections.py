"""v1 Fig 1: surveyed stakes vs interpolated g_pair field, 3 projections, wall cell.
v1 Fig 2: 4 showcase cells, best projection, cat6 field vs pair2 (smoo) field."""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
sys.path.insert(0, "/Users/kaiser/Projects/Masterarbeit")
from analysis.core.style import apply_style, save_fig

apply_style()
warnings.filterwarnings("ignore")
ROOT = "/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography/exp100"
OUT = Path("/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography/explore")
OUT.mkdir(parents=True, exist_ok=True)

CELLS = {
    "wall: boa (0,1) 'snake'": ("boa constrictor", 0, 1),
    "control: boa (0,0) 'constrictor'": ("boa constrictor", 0, 0),
    "mid: iguana (0,1) 'lizard'": ("green iguana", 0, 1),
    "easy: marimba (0,1) 'percussion'": ("marimba", 0, 1),
}

P = pd.read_parquet(f"{ROOT}/points.parquet", columns=[
    "source", "prompt_regime", "seed_dir", "row_ref",
    "target_class", "level_anchor", "level_target",
    "n_active_img", "n_active_txt", "rank_sum_img_norm", "rank_sum_txt_norm",
    "hamming_to_anchor", "g_pair", "image_dim"])
S = pd.read_parquet(f"{ROOT}/straddle_pairs.parquet")

# stake coordinates: join after-point from pdq_s2 rows
p2 = P[P.source == "pdq_s2"].copy()
p2["rid"] = pd.to_numeric(p2.row_ref, errors="coerce")
S = S.merge(p2[["seed_dir", "rid", "n_active_img", "n_active_txt",
                "rank_sum_img_norm", "rank_sum_txt_norm",
                "hamming_to_anchor", "g_pair"]],
            left_on=["seed_dir", "call_id_after"], right_on=["seed_dir", "rid"],
            how="left", suffixes=("", "_pt"))
print("stake join match:", S.rid.notna().mean())

# normalized axes (combinable across seeds with different image_dim)
for df in (P, S):
    df["na_img_n"] = df.n_active_img / df.image_dim
    df["na_txt_n"] = df.n_active_txt / 19.0
    df["ham_n"] = df.hamming_to_anchor / (df.image_dim + 19.0)

PROJ = {
    "rank-sum (img x txt, norm)": ("rank_sum_img_norm", "rank_sum_txt_norm"),
    "n_active (img x txt, norm)": ("na_img_n", "na_txt_n"),
    "hamming x rank_sum_txt": ("ham_n", "rank_sum_txt_norm"),
}

def cellmask(df, tc, la, lt):
    return (df.target_class == tc) & (df.level_anchor == la) & (df.level_target == lt)

def field_grid(x, y, g, nbins=36, sigma=1.3):
    """nan-aware smoothed binned-mean field."""
    xe = np.linspace(np.nanmin(x), np.nanmax(x), nbins + 1)
    ye = np.linspace(np.nanmin(y), np.nanmax(y), nbins + 1)
    num, _, _ = np.histogram2d(x, y, bins=[xe, ye], weights=g)
    den, _, _ = np.histogram2d(x, y, bins=[xe, ye])
    num_s = gaussian_filter(num, sigma)
    den_s = gaussian_filter(den, sigma)
    with np.errstate(invalid="ignore"):
        F = num_s / den_s
    F[den_s < 0.05] = np.nan
    return xe, ye, F.T  # transpose for pcolormesh

def eval_field(xe, ye, F, xs, ys):
    ix = np.clip(np.digitize(xs, xe) - 1, 0, len(xe) - 2)
    iy = np.clip(np.digitize(ys, ye) - 1, 0, len(ye) - 2)
    return F[iy, ix]

# ---------------- Fig 1: wall cell, 3 projections ----------------
tc, la, lt = CELLS["wall: boa (0,1) 'snake'"]
pc = P[cellmask(P, tc, la, lt) & (P.prompt_regime == "cat6")]
st = S[cellmask(S, tc, la, lt)]
st_pm = st[st.boundary_kind == "pair_margin"]
st_am = st[st.boundary_kind == "argmax"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
stats = []
for ax, (pname, (xc, yc)) in zip(axes, PROJ.items()):
    x, y, g = pc[xc].values, pc[yc].values, pc.g_pair.values
    xe, ye, F = field_grid(x, y, g)
    pm = ax.pcolormesh(xe, ye, F, cmap="RdBu", vmin=-1, vmax=1, shading="flat")
    Xc, Yc = np.meshgrid(0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:]))
    try:
        ax.contour(Xc, Yc, F, levels=[0], colors="k", linewidths=1.6)
    except Exception:
        pass
    ax.scatter(st_pm[xc], st_pm[yc], s=22, marker="o", facecolor="yellow",
               edgecolor="k", linewidth=0.6, label=f"pair_margin stakes (n={len(st_pm)})", zorder=5)
    ax.scatter(st_am[xc], st_am[yc], s=14, marker="^", facecolor="none",
               edgecolor="green", linewidth=0.8, label=f"argmax stakes (n={len(st_am)})", zorder=4)
    gf = eval_field(xe, ye, F, st_pm[xc].values, st_pm[yc].values)
    med, p90 = np.nanmedian(np.abs(gf)), np.nanpercentile(np.abs(gf), 90)
    stats.append((pname, med, p90))
    ax.set_xlabel(xc); ax.set_ylabel(yc)
    ax.set_title(f"{pname}\n|g_field| at stakes: med={med:.3f}, p90={p90:.3f}", fontsize=10)
axes[0].legend(loc="lower left", fontsize=7)
fig.colorbar(pm, ax=axes, shrink=0.8, label="g_pair (blue=anchor, red=target)")
fig.suptitle(f"Wall cell {tc} (la=0, lt=1, 'snake'): cat6 field vs surveyed stakes — projection comparison", y=1.04)
save_fig(fig, OUT / "v1_fig1_wall_projection_comparison.png", tight=False)
for s in stats:
    print("PROJ %-28s med|g@stake|=%.3f p90=%.3f" % s)

# ---------------- Fig 2: 4 cells x (cat6 | pair2), best projection ----------------
xc, yc = PROJ["rank-sum (img x txt, norm)"]
fig, axes = plt.subplots(2, 4, figsize=(19, 8.6), sharex=True, sharey=True)
for j, (cname, (tc, la, lt)) in enumerate(CELLS.items()):
    st = S[cellmask(S, tc, la, lt)]
    st_pm = st[st.boundary_kind == "pair_margin"]
    st_am = st[st.boundary_kind == "argmax"]
    for i, regime in enumerate(["cat6", "pair2"]):
        ax = axes[i, j]
        pc = P[cellmask(P, tc, la, lt) & (P.prompt_regime == regime)]
        x, y, g = pc[xc].values, pc[yc].values, pc.g_pair.values
        xe, ye, F = field_grid(x, y, g)
        pm = ax.pcolormesh(xe, ye, F, cmap="RdBu", vmin=-1, vmax=1)
        Xc, Yc = np.meshgrid(0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:]))
        try:
            ax.contour(Xc, Yc, F, levels=[0], colors="k", linewidths=1.6)
        except Exception:
            pass
        if i == 0:
            ax.scatter(st_pm[xc], st_pm[yc], s=16, marker="o", facecolor="yellow",
                       edgecolor="k", linewidth=0.5, zorder=5)
            ax.scatter(st_am[xc], st_am[yc], s=10, marker="^", facecolor="none",
                       edgecolor="green", linewidth=0.7, zorder=4)
            gf = eval_field(xe, ye, F, st_pm[xc].values, st_pm[yc].values)
            med = np.nanmedian(np.abs(gf)) if len(st_pm) else np.nan
            ax.set_title(f"{cname}\ncat6 (n={len(pc)}), stakes pm={len(st_pm)}/am={len(st_am)}\nmed|g@stake|={med:.3f}", fontsize=9)
        else:
            ax.set_title(f"pair2 SMOO field (n={len(pc)})", fontsize=9)
        if i == 1:
            ax.set_xlabel("rank_sum_img_norm")
    axes[0, 0].set_ylabel("rank_sum_txt_norm"); axes[1, 0].set_ylabel("rank_sum_txt_norm")
fig.colorbar(pm, ax=axes, shrink=0.7, label="g_pair (blue=anchor, red=target)")
fig.suptitle("Surveyed stakes on cat6 field (top) vs pair2 SMOO field (bottom) — regime shift, rank-sum projection", y=0.99)
save_fig(fig, OUT / "v1_fig2_cells_cat6_vs_pair2.png", tight=False)
