"""m3_11: Boundary-radius curves r*(theta) — the most map-like single figure.

For each cell, per theta bin (modality mix), r*(theta) = smallest radius at
which the flip rate P(g_pair < 0) within a sliding r-window reaches 10%.
Bootstrap band (10-90 pct over point resamples). Open cells trace an arc
around the anchor; wall cells have no r* anywhere (no opening).
Light gray wedge area = sampled support (where data exists at all), so an
absent curve inside support = genuine wall, not missing data.
Regime pair2 (smoo), anchor junco.
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
rng = np.random.default_rng(7)

cols = ["source", "anchor_class", "target_class", "level_anchor", "level_target",
        "anchor_word", "target_word", "n_active_img", "n_active_txt",
        "g_pair", "image_dim"]
pts = pd.read_parquet(f"{STORE}/points.parquet", columns=cols)
s = pts[(pts.source == "smoo") & (pts.anchor_class == "junco")].copy()

xi = (s.n_active_img / s.image_dim).to_numpy()
yi = (s.n_active_txt / 19).to_numpy()
QX, QY = np.quantile(xi, .99), np.quantile(yi, .99)
s["theta"] = np.arctan2(yi / QY, xi / QX)
s["r"] = np.hypot(xi / QX, yi / QY)

TH_EDGES = np.linspace(0, np.pi / 2, 16)   # 15 bins of 6 deg
TH_C = 0.5 * (TH_EDGES[:-1] + TH_EDGES[1:])
WIN, STEP, RMAX = 0.16, 0.04, 1.3
RATE, NMIN_WIN = 0.10, 30
N_BOOT = 150

def rstar(rv, gv):
    """First sliding-window center where flip rate >= RATE (n>=NMIN_WIN)."""
    for lo in np.arange(0, RMAX - WIN, STEP):
        m = (rv >= lo) & (rv < lo + WIN)
        if m.sum() >= NMIN_WIN and (gv[m] < 0).mean() >= RATE:
            return lo + WIN / 2
    return np.nan

def cell_curve(c):
    ti = np.digitize(c.theta, TH_EDGES) - 1
    rv_all, gv_all = c.r.to_numpy(), c.g_pair.to_numpy()
    cur = np.full(len(TH_C), np.nan)
    lob = np.full(len(TH_C), np.nan)
    hib = np.full(len(TH_C), np.nan)
    sup = np.full(len(TH_C), np.nan)   # support extent: q95 of r with data
    for t in range(len(TH_C)):
        m = ti == t
        if m.sum() < 60:
            continue
        rv, gv = rv_all[m], gv_all[m]
        sup[t] = np.quantile(rv, .95)
        cur[t] = rstar(rv, gv)
        if np.isfinite(cur[t]):
            bs = []
            idx = np.arange(len(rv))
            for _ in range(N_BOOT):
                bi = rng.choice(idx, len(idx))
                bs.append(rstar(rv[bi], gv[bi]))
            bs = np.array(bs, float)
            ok = np.isfinite(bs)
            if ok.mean() > 0.5:
                lob[t], hib[t] = np.nanpercentile(bs[ok], [10, 90])
    return cur, lob, hib, sup

CELLS = [
    ("boa constrictor", 2, 2, "#D64933", "open"),
    ("green iguana", 2, 0, "#E6A817", "open"),
    ("marimba", 2, 1, "#55A868", "open"),
    ("ostrich", 0, 1, "#8172B3", "open"),
    ("boa constrictor", 0, 1, "#2274A5", "WALL"),
    ("cello", 1, 1, "#444444", "WALL"),
]

fig, ax = plt.subplots(figsize=(8.6, 8.2), subplot_kw=dict(projection="polar"))
ax.set_thetamin(0); ax.set_thetamax(90)
ax.set_ylim(0, RMAX)

sup_max = None
for tc, la, lt, col, kind in CELLS:
    c = s[(s.target_class == tc) & (s.level_anchor == la) & (s.level_target == lt)]
    cur, lob, hib, sup = cell_curve(c)
    sup_max = sup if sup_max is None else np.fmax(sup_max, sup)
    aw, tw = c.anchor_word.iloc[0], c.target_word.iloc[0]
    flips = (c.g_pair < 0).mean()
    lab = f"{kind}: '{aw}' vs '{tw}' ({tc.split()[0]}, La{la}Lt{lt}, flips {flips:.0%})"
    ok = np.isfinite(cur)
    if ok.any():
        ax.plot(TH_C[ok], cur[ok], "-o", color=col, lw=2.2, ms=4, label=lab,
                zorder=5)
        bok = ok & np.isfinite(lob)
        ax.fill_between(TH_C[bok], lob[bok], hib[bok], color=col, alpha=0.20,
                        lw=0, zorder=4)
    else:
        lab += "  — closed: searched to ×-frontier, zero opening"
        ax.plot([], [], "x", ls="none", color=col, label=lab)
    # closed frontier: theta bins with support but no crossing
    closed = np.isfinite(sup) & ~ok
    if closed.any():
        ax.plot(TH_C[closed], sup[closed], "x", color=col, ms=6, mew=1.6,
                alpha=0.85, zorder=3)

# sampled support envelope
oks = np.isfinite(sup_max)
ax.fill_between(TH_C[oks], 0, sup_max[oks], color="0.55", alpha=0.10, zorder=0)
ax.plot(TH_C[oks], sup_max[oks], color="0.55", lw=1, ls=":",
        label="sampled support (q95 of r per θ, max over cells);\n"
              "× = searched-to radius in θ-bins with no opening (per cell)")

ax.set_xticks(np.radians([0, 15, 30, 45, 60, 75, 90]))
ax.set_xticklabels(["0° pure img", "15°", "30°", "45°", "60°", "75°", "90° pure txt"])
ax.set_title(
    "Boundary radius around the anchor, by direction of perturbation — pair2 (smoo), anchor junco\n"
    r"$\theta$ = modality mix atan2(txt, img), q99-scaled active-gene fractions "
    f"(q99 img {QX:.3f}, txt {QY:.3f}); r* = first radius with flip rate ≥ 10% "
    f"(window {WIN}, n ≥ {NMIN_WIN}); band = bootstrap 10–90%",
    fontsize=9.5)
ax.legend(loc="upper left", bbox_to_anchor=(0.98, 1.0), fontsize=8)
save_fig(fig, OUT / "m3_11_radius_curves.png", tight=False)
