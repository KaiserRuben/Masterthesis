"""v3 — Robustness checks for the crispness benchmark.

(1) Sampling-bias check: in cat6, `hamming_to_anchor` wins — but pdq_s2 walks
    shrink toward the anchor and *cross* the boundary en route, so position
    along the walk axis correlates with side BY SAMPLING CONSTRUCTION.
    Re-benchmark cat6 pooled on pdq_s1 only (anchor-centered random sampling,
    no path constraint) to see whether hamming's advantage survives.

(2) Raw-genotype 2D embedding for ONE cell (cat6 wall boa(0,1), majority
    image_dim subset): PCA-2 on standardized raw genotype. Upper-context for
    how much side-information a *linear genotype* view can carry vs the
    descriptor-based views.

Outputs: v3_crispness_supplement.csv, v3_genotype_pca_wallboa01.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/kaiser/Projects/Masterarbeit")
import matplotlib.pyplot as plt  # noqa: E402
from analysis.core.style import apply_style, save_fig  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

EXPLORE = Path("/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography/explore")
sys.path.insert(0, str(EXPLORE))
from v3_benchmark_crispness import (  # noqa: E402
    BASE, load_points, seed_bounds, load_straddle_midpoints, project, crispness,
)

OUT = EXPLORE


def s1_only_benchmark(df, sp) -> list[dict]:
    rows = []
    s1 = df[(df.prompt_regime == "cat6") & (df.source == "pdq_s1")]
    print(f"pdq_s1 only: n={len(s1)}, minority={min((s1.side == 0).sum(), (s1.side == 1).sum())}")
    for proj in ["ranksum", "nactive", "semantic", "hamming", "pca2", "lda_pc"]:
        res = project(s1, None, proj)
        rec = {"regime": "cat6", "cell": "pooled (pdq_s1 only)", "projection": proj}
        if res is None:
            rec["note"] = "unavailable"
        elif isinstance(res[0], tuple):
            (xy, mxy, axes), ok = res
            rec.update(crispness(xy, s1.side.to_numpy()[ok.to_numpy()], None))
            rec["axes"] = " | ".join(axes)
        else:
            xy, mxy, axes = res
            rec.update(crispness(xy, s1.side.to_numpy(), None))
            rec["axes"] = " | ".join(axes)
        rows.append(rec)
        print(f"  s1-only {proj:8s} knn={rec.get('knn_auc', float('nan')):.3f} "
              f"logit={rec.get('logit_auc', float('nan')):.3f} "
              f"sep={rec.get('sep_1mBC', float('nan')):.3f}")
    return rows


def genotype_pca(df, sp) -> list[dict]:
    """PCA-2 of raw standardized genotype, cat6 wall boa(0,1)."""
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    t = pq.read_table(
        BASE / "points.parquet",
        columns=["genotype", "image_dim", "pair_margin", "seed_dir", "row_ref",
                 "source", "prompt_regime", "target_class", "level_anchor", "level_target"],
        filters=[("prompt_regime", "==", "cat6"), ("target_class", "==", "boa constrictor"),
                 ("level_anchor", "==", 0), ("level_target", "==", 1)])
    g = t.to_pandas()
    dim = g.image_dim.mode().iloc[0]
    g = g[g.image_dim == dim].copy()
    print(f"genotype-PCA cell: n={len(g)}, image_dim={dim}, "
          f"sides={np.bincount((g.pair_margin > 0).astype(int))}")
    X = np.stack(g.genotype.to_numpy())
    y = (g.pair_margin > 0).astype(int).to_numpy()
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-12] = 1.0
    Xz = (X - mu) / sd
    p = PCA(n_components=2, random_state=0).fit(Xz)
    XY = p.transform(Xz)

    # straddle midpoints in genotype space: after-genotype with the edited gene
    # set to the mean of before/after values
    sp_cell = sp[(sp.target_class == "boa constrictor") & (sp.level_anchor == 0)
                 & (sp.level_target == 1)].copy()
    g["ref_int"] = pd.to_numeric(g.row_ref, errors="coerce")
    gidx = g.set_index(["seed_dir", "ref_int"])
    mrows = []
    for r in sp_cell.itertuples():
        try:
            row = gidx.loc[(r.seed_dir, r.call_id_after)]
        except KeyError:
            continue
        gv = np.asarray(row.genotype, dtype=float).copy()
        gv[r.gene_idx] = 0.5 * (r.value_before + r.value_after)
        mrows.append(gv)
    MXY = p.transform((np.array(mrows) - mu) / sd) if mrows else None
    print(f"  straddle midpoints projected: {0 if MXY is None else len(MXY)}")

    rec = {"regime": "cat6", "cell": "wall boa s->snake (0,1)", "projection": "genotype_pca2",
           "axes": f"genoPC1 ({p.explained_variance_ratio_[0]:.1%} var) | "
                   f"genoPC2 ({p.explained_variance_ratio_[1]:.1%} var)"}
    rec.update(crispness(XY, y, MXY))
    print(f"  genotype_pca2 knn={rec.get('knn_auc', float('nan')):.3f} "
          f"logit={rec.get('logit_auc', float('nan')):.3f} "
          f"sep={rec.get('sep_1mBC', float('nan')):.3f} "
          f"stramb={rec.get('straddle_amb', float('nan')):.3f}")

    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for s, color, lbl, z in [(1, "#2274A5", "anchor side", 1), (0, "#D64933", "target side", 2)]:
        m = y == s
        ax.scatter(XY[m, 0], XY[m, 1], s=5, alpha=0.35, c=color,
                   label=f"{lbl} (n={m.sum()})", zorder=z, linewidths=0)
    if MXY is not None:
        ax.scatter(MXY[:, 0], MXY[:, 1], s=20, marker="x", c="black", linewidths=0.8,
                   alpha=0.85, label=f"straddle midpts (n={len(MXY)})", zorder=3)
    ax.set_xlabel(f"genotype PC1 ({p.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"genotype PC2 ({p.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title(f"Raw-genotype PCA-2 — cat6 wall boa(0,1), image_dim={dim}\n"
                 f"kNN AUC {rec.get('knn_auc', float('nan')):.2f}, "
                 f"1-BC {rec.get('sep_1mBC', float('nan')):.2f}", fontsize=10)
    leg = ax.legend(markerscale=2.5, fontsize=8)
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)
    save_fig(fig, OUT / "v3_genotype_pca_wallboa01.png")
    return [rec]


def main():
    df = load_points()
    bounds = seed_bounds(df)
    sp = load_straddle_midpoints(df, bounds)
    rows = s1_only_benchmark(df, sp) + genotype_pca(df, sp)
    pd.DataFrame(rows).to_csv(OUT / "v3_crispness_supplement.csv", index=False)
    print("wrote v3_crispness_supplement.csv")


if __name__ == "__main__":
    main()
