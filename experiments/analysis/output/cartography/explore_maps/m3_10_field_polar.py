"""m3_10: Direction-resolved polar field maps of the pair2 (smoo) decision field.

Anchor genotype at origin. theta = modality mix of the perturbation:
  0 deg = pure image manipulation, 90 deg = pure text manipulation,
  theta = atan2(txt_hat, img_hat) with
    img_hat = (n_active_img / image_dim) / q99_img,
    txt_hat = (n_active_txt / 19)        / q99_txt   (q99 over all smoo junco rows).
r = ||(img_hat, txt_hat)||  (total manipulation strength, q99-scaled units).

Color = median g_pair (p_A - p_B) per (theta, r) bin; bins with n < N_MIN
masked gray (no coverage). Bold contour = flip rate P(g<0) = 0.10,
thin = 0.25. Top row: open cells, bottom row: wall cells.
"""
import sys
sys.path.insert(0, "/Users/kaiser/Projects/Masterarbeit")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from analysis.core.style import apply_style, save_fig

STORE = "/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography/exp100"
OUT = Path("/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography/explore_maps")
apply_style()

cols = ["source", "anchor_class", "target_class", "level_anchor", "level_target",
        "anchor_word", "target_word", "n_active_img", "n_active_txt",
        "g_pair", "image_dim"]
pts = pd.read_parquet(f"{STORE}/points.parquet", columns=cols)
s = pts[(pts.source == "smoo") & (pts.anchor_class == "junco")].copy()

xi = (s.n_active_img / s.image_dim).to_numpy()
yi = (s.n_active_txt / 19).to_numpy()
QX, QY = np.quantile(xi, .99), np.quantile(yi, .99)
s["theta"] = np.arctan2(yi / QY, xi / QX)          # radians, 0..pi/2
s["r"] = np.hypot(xi / QX, yi / QY)

N_MIN = 20
TH_EDGES = np.linspace(0, np.pi / 2, 13)
R_EDGES = np.linspace(0, 1.3, 14)
TH_C = 0.5 * (TH_EDGES[:-1] + TH_EDGES[1:])
R_C = 0.5 * (R_EDGES[:-1] + R_EDGES[1:])

def bin_cell(c):
    """Return (median_g, flip_rate, n) on the (theta, r) grid."""
    ti = np.digitize(c.theta, TH_EDGES) - 1
    ri = np.digitize(c.r, R_EDGES) - 1
    g = c.g_pair.to_numpy()
    nt, nr = len(TH_EDGES) - 1, len(R_EDGES) - 1
    med = np.full((nt, nr), np.nan)
    fr = np.full((nt, nr), np.nan)
    n = np.zeros((nt, nr), int)
    for t in range(nt):
        mt = ti == t
        for rr in range(nr):
            m = mt & (ri == rr)
            k = m.sum()
            n[t, rr] = k
            if k >= N_MIN:
                med[t, rr] = np.median(g[m])
                fr[t, rr] = (g[m] < 0).mean()
    return med, fr, n

CELLS = [
    ("boa constrictor", 2, 2),  # open
    ("green iguana", 2, 0),     # open
    ("marimba", 2, 1),          # open
    ("boa constrictor", 0, 1),  # wall (snake)
    ("cello", 1, 1),            # wall (songbird)
    ("ostrich", 1, 0),          # wall
]

fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.6),
                         subplot_kw=dict(projection="polar"))
TH_MESH, R_MESH = np.meshgrid(TH_EDGES, R_EDGES, indexing="ij")
cmap = plt.get_cmap("RdBu_r").copy()
cmap.set_bad("0.88")

for ax, (tc, la, lt) in zip(axes.flat, CELLS):
    c = s[(s.target_class == tc) & (s.level_anchor == la) & (s.level_target == lt)]
    med, fr, n = bin_cell(c)
    pm = ax.pcolormesh(TH_MESH, R_MESH, np.ma.masked_invalid(med),
                       cmap=cmap, vmin=-1, vmax=1, shading="flat")
    # flip-rate contours on bin centers
    TC, RC = np.meshgrid(TH_C, R_C, indexing="ij")
    fr_f = np.ma.masked_invalid(fr)
    if np.isfinite(fr).sum() > 4 and np.nanmax(fr) >= 0.10:
        ax.contour(TC, RC, fr_f, levels=[0.10], colors="k", linewidths=2.2)
    if np.isfinite(fr).sum() > 4 and np.nanmax(fr) >= 0.25:
        ax.contour(TC, RC, fr_f, levels=[0.25], colors="k", linewidths=1.0,
                   linestyles="--")
    flipfrac = (c.g_pair < 0).mean()
    ax.set_thetamin(0); ax.set_thetamax(90)
    ax.set_ylim(0, 1.3)
    ax.set_xticks(np.radians([0, 30, 60, 90]))
    ax.set_xticklabels(["0°\nimg", "30°", "60°", "90° txt"], fontsize=8)
    ax.set_yticks([0.5, 1.0])
    ax.tick_params(labelsize=7)
    kind = "WALL" if flipfrac < 0.01 else "open"
    ax.set_title(f"junco → {tc}  [{kind}]\n'{c.anchor_word.iloc[0]}' vs "
                 f"'{c.target_word.iloc[0]}'  (La{la},Lt{lt})  flips {flipfrac:.1%}",
                 fontsize=9)

cb = fig.colorbar(pm, ax=axes, shrink=0.75, pad=0.04)
cb.set_label("median g = P(anchor) − P(target)   [pair2 / smoo]")
fig.suptitle(
    "Direction-resolved boundary field around the anchor — regime pair2 (smoo), anchor junco\n"
    r"$\theta$ = modality mix: atan2(txt, img) of q99-scaled active-gene fractions "
    f"(q99: img {QX:.3f}, txt {QY:.3f});  r = vector norm.  "
    "Bold contour: flip rate P(g<0)=0.10; dashed: 0.25.  Gray = <20 points (no coverage).",
    fontsize=10)
save_fig(fig, OUT / "m3_10_field_polar.png", tight=False)
