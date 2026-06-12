"""m5_04 (direction 5e): supervised learned chart — boundary-normal in INPUT space.

Unsupervised PCA (m5_03) fails: genotype variance is near-isotropic and
boundary-irrelevant. Here the x-axis is LEARNED from the decision signal:
ridge regression of g_pair on the z-scored within-seed genotype; x = w·z is the
"steepest-g" direction. y = PC1 of the residual genotype variation (orthogonal
complement). Boundary = binned-mean g_pair = 0 contour; if the learned normal
is real, the contour is vertical and color sorts left-right.

Honesty: fit on 70% of rows, ONLY the 30% test rows are plotted (no in-sample
gloss); test R^2 / AUC in titles. Within-seed, pair2/smoo only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, "/Users/kaiser/Projects/Masterarbeit")
from analysis.core.style import apply_style, save_fig  # noqa: E402

apply_style()
RNG = np.random.default_rng(3)
ROOT = Path("/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography")
OUT = ROOT / "explore_maps"

SEEDS = [
    ("seed_0042_1780105620", "boa lt=1 (wall, never crosses)"),
    ("seed_0061_1780475753", "boa lt=1 (wall, 2.1% cross)"),
    ("seed_0003_1779794022", "ostrich (easy, 43% cross)"),
]


def binned_contour(ax, x, y, sig, level=0.0, n_bins=40, min_n=6):
    xb = np.linspace(np.quantile(x, 0.005), np.quantile(x, 0.995), n_bins)
    yb = np.linspace(np.quantile(y, 0.005), np.quantile(y, 0.995), n_bins)
    Hs, _, _ = np.histogram2d(x, y, bins=[xb, yb], weights=sig)
    Hn, _, _ = np.histogram2d(x, y, bins=[xb, yb])
    with np.errstate(invalid="ignore"):
        f = Hs / Hn
    f[Hn < min_n] = np.nan
    if np.nanmin(f) < level < np.nanmax(f):
        ax.contour(0.5 * (xb[:-1] + xb[1:]), 0.5 * (yb[:-1] + yb[1:]), f.T,
                   levels=[level], colors="k", linewidths=1.8)
        return True
    return False


fig, axes = plt.subplots(2, 3, figsize=(15.5, 9))

for col, (seed, desc) in enumerate(SEEDS):
    df = pd.read_parquet(ROOT / "exp100/points.parquet",
                         columns=["genotype", "g_pair", "generation"],
                         filters=[("seed_dir", "==", seed), ("source", "==", "smoo")])
    G = np.stack(df.genotype.to_numpy()).astype(np.float32)
    g = df.g_pair.to_numpy()
    mu, sd = G.mean(axis=0), G.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (G - mu) / sd

    tr, te = train_test_split(np.arange(len(g)), test_size=0.3, random_state=0)
    ridge = Ridge(alpha=50.0).fit(Z[tr], g[tr])
    w = ridge.coef_ / np.linalg.norm(ridge.coef_)
    x_all = Z @ w
    r2 = r2_score(g[te], ridge.predict(Z[te]))
    auc = (roc_auc_score(g[te] < 0, -x_all[te])
           if (g[te] < 0).sum() >= 10 else np.nan)
    txt_w = float((w[-19:] ** 2).sum())

    # residual PC1 (fit on train, orthogonal to w)
    Zr = Z - np.outer(x_all, w)
    Zr_tr = Zr[tr] - Zr[tr].mean(axis=0)
    _, S, Vt = np.linalg.svd(Zr_tr, full_matrices=False)
    y_all = (Zr - Zr[tr].mean(axis=0)) @ Vt[0]
    print(f"{seed}: test R2={r2:.3f} AUC={auc if np.isnan(auc) else round(auc,3)} "
          f"txt-weight in normal={txt_w:.2%}")

    X, Y, gg = x_all[te], y_all[te], g[te]
    xlim = np.quantile(X, [0.005, 0.995]); ylim = np.quantile(Y, [0.005, 0.995])

    ax = axes[0, col]
    sc = ax.scatter(X, Y, s=4, c=gg, cmap="coolwarm_r", vmin=-0.8, vmax=0.8,
                    alpha=0.5, lw=0)
    has = binned_contour(ax, X, Y, gg, level=0.0)
    if not has:
        ax.text(0.03, 0.03, "g=0 not reached in field", transform=ax.transAxes,
                fontsize=8, color="0.3")
    plt.colorbar(sc, ax=ax, fraction=0.04, label="g_pair")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel("learned normal  w·z  (ridge on g_pair)")
    ax.set_ylabel("residual PC1")
    auc_s = "n/a" if np.isnan(auc) else f"{auc:.2f}"
    ax.set_title(f"{desc}\n{seed} [test 30% only; R²={r2:.2f}, side-AUC={auc_s}, "
                 f"txt share of normal {txt_w:.0%}]", fontsize=9)

    # bottom: 1D profile — g_pair vs learned normal, with running median
    ax = axes[1, col]
    ax.scatter(X, gg, s=3, c="0.6", alpha=0.35, lw=0)
    order = np.argsort(X)
    xb = np.array_split(order, 40)
    ax.plot([X[b].mean() for b in xb], [np.median(gg[b]) for b in xb],
            color="#D64933", lw=2, label="running median")
    ax.axhline(0, color="k", lw=1.2, ls="--")
    ax.set_xlim(*xlim)
    ax.set_xlabel("learned normal  w·z")
    ax.set_ylabel("g_pair")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title("decision profile along the learned normal", fontsize=9)

fig.suptitle("m5_04 — supervised learned chart: ridge boundary-normal × residual PC1 "
             "(within-seed, pair2/smoo; test points only)", fontsize=11, y=1.0)
save_fig(fig, OUT / "m5_04_supervised_chart.png")
