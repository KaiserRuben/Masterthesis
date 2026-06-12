"""m5_02 (direction 5b): text-subspace chart — PCA on the trailing-19 text genes.

The 19 text genes are the ONLY genotype block comparable across all seeds
(image_dim varies 222/253/276). Two encodings, because gene values are partly
categorical (MLM genes 0..25 = proposal index):
  - raw z-scored values (treats value as ordinal aggressiveness)
  - binary activity (gene != 0) — pure on/off pattern

Boundary as geometry: 0-level contour of the 2D-binned mean side-signal
(g_pair for smoo/pair2, junco-fraction-0.5 for cat6). Regimes never pooled in
panels; the chart basis is fit on smoo (optimizer-biased — stated in title) and
both regimes are projected into it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as pads

sys.path.insert(0, "/Users/kaiser/Projects/Masterarbeit")
from analysis.core.style import apply_style, save_fig  # noqa: E402

apply_style()
RNG = np.random.default_rng(7)
ROOT = Path("/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography")
OUT = ROOT / "explore_maps"

GROUPS = (["mlm"] * 3) + (["frag"] * 5) + (["charnoise"] * 8) + (["saliency"] * 3)

# ----------------------------------------------------- batched genotype scan
dset = pads.dataset(ROOT / "exp100/points.parquet")
cols = ["source", "seed_dir", "g_pair", "pred_label", "rank_sum_txt_norm",
        "n_active_txt", "row_ref", "genotype"]
T_parts, meta_parts = [], []
for batch in dset.scanner(columns=cols, batch_size=20000).to_batches():
    geno = batch.column("genotype")
    vals = geno.flatten().to_numpy(zero_copy_only=False)
    offs = geno.offsets.to_numpy()
    ends = offs[1:] - offs[0]          # rebase: flatten() returns only this slice's values
    idx = ends[:, None] - 19 + np.arange(19)[None, :]
    T_parts.append(vals[idx].astype(np.int16))
    meta_parts.append(batch.select(cols[:-1]).to_pandas())
T = np.vstack(T_parts)                      # (n, 19) trailing text genes
meta = pd.concat(meta_parts, ignore_index=True)
print(f"text-gene matrix: {T.shape}; sources: {meta.source.value_counts().to_dict()}")

is_smoo = (meta.source == "smoo").to_numpy()
is_cat6 = ~is_smoo

# side labels
side_smoo = (meta.g_pair < 0).to_numpy()                 # True = target side
side_cat6 = (meta.pred_label == "boa constrictor").to_numpy()

# ------------------------------------------------- two encodings, PCA basis fit on smoo
fit_idx = np.flatnonzero(is_smoo)
fit_idx = RNG.choice(fit_idx, size=120000, replace=False)


def pca_chart(X_all, fit_rows):
    mu = X_all[fit_rows].mean(axis=0)
    sd = X_all[fit_rows].std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X_all - mu) / sd
    _, S, Vt = np.linalg.svd(Z[fit_rows] - Z[fit_rows].mean(axis=0), full_matrices=False)
    evr = S**2 / (S**2).sum()
    return Z @ Vt[:2].T, Vt[:2], evr[:2]


XY_raw, V_raw, evr_raw = pca_chart(T.astype(np.float32), fit_idx)
XY_bin, V_bin, evr_bin = pca_chart((T != 0).astype(np.float32), fit_idx)

for name, V, evr in [("raw-z", V_raw, evr_raw), ("binary", V_bin, evr_bin)]:
    print(f"\n{name} PCA evr: {np.round(evr, 3)}")
    for k in range(2):
        top = np.argsort(-np.abs(V[k]))[:5]
        print(f"  PC{k+1} top loadings:",
              [(int(i), GROUPS[i], round(float(V[k][i]), 2)) for i in top])

# ---------------------------------------------- straddle txt-gene flip overlay
sp = pd.read_parquet(ROOT / "exp100/straddle_pairs.parquet")
sp_txt = sp[sp.gene_modality == "txt"].copy()
# join to pdq_s2 points via row_ref == call_id_after, scoped by seed_dir
pdq = meta[meta.source.isin(["pdq_s2", "pdq_s1"])].reset_index()
pdq["key"] = pdq.seed_dir + "::" + pdq.row_ref.astype(str)
sp_txt["key"] = sp_txt.seed_dir + "::" + sp_txt.call_id_after.astype(str)
j = sp_txt.merge(pdq[["key", "index"]], on="key", how="inner")
strad_rows = j["index"].to_numpy()
print(f"\ntxt-gene straddles joined to genotypes: {len(j)}/{len(sp_txt)}")


def side_contour(ax, xy, side_signal, n_bins=60, level=0.0, cmap_pts=None):
    """Binned-mean contour of side signal; returns binned field for reuse."""
    x, y = xy[:, 0], xy[:, 1]
    xb = np.linspace(np.quantile(x, 0.001), np.quantile(x, 0.999), n_bins)
    yb = np.linspace(np.quantile(y, 0.001), np.quantile(y, 0.999), n_bins)
    H_sum, _, _ = np.histogram2d(x, y, bins=[xb, yb], weights=side_signal)
    H_n, _, _ = np.histogram2d(x, y, bins=[xb, yb])
    with np.errstate(invalid="ignore"):
        field = H_sum / H_n
    field[H_n < 10] = np.nan
    Xc = 0.5 * (xb[:-1] + xb[1:]); Yc = 0.5 * (yb[:-1] + yb[1:])
    cs = ax.contour(Xc, Yc, field.T, levels=[level], colors="k", linewidths=1.6)
    return cs


# ----------------------------------------------------------------- figure
fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.5))

plot_smoo = RNG.choice(np.flatnonzero(is_smoo), size=90000, replace=False)
plot_cat = np.flatnonzero(is_cat6)
plot_cat = RNG.choice(plot_cat, size=min(40000, len(plot_cat)), replace=False)

for row, (XY, enc, evr) in enumerate([(XY_raw, "raw z-scored values", evr_raw),
                                      (XY_bin, "binary activity", evr_bin)]):
    # smoo panel, colored by g_pair
    ax = axes[row, 0]
    s = plot_smoo
    sc = ax.scatter(XY[s, 0], XY[s, 1], s=2, c=meta.g_pair.to_numpy()[s],
                    cmap="coolwarm_r", vmin=-0.8, vmax=0.8, alpha=0.3, lw=0)
    side_contour(ax, XY[is_smoo], meta.g_pair.to_numpy()[is_smoo], level=0.0)
    plt.colorbar(sc, ax=ax, fraction=0.04, label="g_pair (blue→target side)")
    ax.set_xlabel(f"text PC1 ({evr[0]:.0%})"); ax.set_ylabel(f"text PC2 ({evr[1]:.0%})")
    ax.set_title(f"pair2/smoo, all seeds+cells (90k of 726k)\n[{enc}; black: binned mean g_pair = 0]",
                 fontsize=9)

    # cat6 panel, colored by pred_label
    ax = axes[row, 1]
    c = plot_cat
    boa = side_cat6[c]
    ax.scatter(XY[c][~boa, 0], XY[c][~boa, 1], s=2, c="#937860", alpha=0.3, lw=0, label="junco")
    ax.scatter(XY[c][boa, 0], XY[c][boa, 1], s=2, c="#C44E52", alpha=0.3, lw=0, label="boa")
    side_contour(ax, XY[is_cat6], side_cat6[is_cat6].astype(float), level=0.5)
    ax.legend(loc="upper right", fontsize=7, markerscale=3)
    ax.set_xlabel("text PC1"); ax.set_ylabel("text PC2")
    ax.set_title(f"cat6/pdq s1+s2 (40k; s2 path-constrained)\n[{enc}; black: junco-fraction = 0.5]",
                 fontsize=9)

    # straddle overlay
    ax = axes[row, 2]
    ax.scatter(XY[plot_cat, 0], XY[plot_cat, 1], s=2, c="0.85", alpha=0.25, lw=0)
    ax.scatter(XY[strad_rows, 0], XY[strad_rows, 1], s=5, c="#D64933", alpha=0.6, lw=0,
               label=f"txt-gene flip after-points (n={len(strad_rows)})")
    side_contour(ax, XY[is_cat6], side_cat6[is_cat6].astype(float), level=0.5)
    ax.legend(loc="upper right", fontsize=7, markerscale=3)
    ax.set_xlabel("text PC1"); ax.set_ylabel("text PC2")
    ax.set_title(f"surveyed txt-gene crossings on the chart\n[{enc}; gray = cat6 field]",
                 fontsize=9)

fig.suptitle("m5_02 — text-subspace chart: PCA on trailing-19 text genes (cross-seed comparable; "
             "basis fit on smoo sample = optimizer-biased)", fontsize=11, y=1.0)
save_fig(fig, OUT / "m5_02_text_pca.png")

# ----------------------------------------- quick separation numbers (memo + (d) input)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

ev_idx = RNG.choice(np.flatnonzero(is_smoo), size=60000, replace=False)
y = side_smoo[ev_idx]
print(f"\nsmoo side balance in eval sample: {y.mean():.3f}")
for name, X in [
    ("rank_sum_txt_norm (1D, chosen)", meta.rank_sum_txt_norm.to_numpy()[ev_idx, None]),
    ("text PCA raw-z (2D, learned)", XY_raw[ev_idx]),
    ("text PCA binary (2D, learned)", XY_bin[ev_idx]),
    ("all 19 text genes z (full block)", ((T[ev_idx] - T[fit_idx].mean(0)) / np.where(T[fit_idx].std(0) == 0, 1, T[fit_idx].std(0)))),
]:
    clf = LogisticRegression(max_iter=2000)
    auc = cross_val_score(clf, X, y, cv=3, scoring="roc_auc").mean()
    print(f"  AUC(side|{name}): {auc:.3f}")

# save projections for reuse by m5_05 crispness benchmark
np.savez_compressed(OUT / "m5_02_text_chart.npz",
                    xy_raw=XY_raw.astype(np.float32), xy_bin=XY_bin.astype(np.float32),
                    is_smoo=is_smoo, g_pair=meta.g_pair.to_numpy(dtype=np.float32),
                    pred_is_boa=side_cat6)
meta[["source", "seed_dir"]].to_parquet(OUT / "m5_02_meta_light.parquet")
print("saved projections for m5_05")
