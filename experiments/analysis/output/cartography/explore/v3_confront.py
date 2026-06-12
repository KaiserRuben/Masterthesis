"""v3 — Visual confrontation: worst | incumbent | best projection per cell.

Side-by-side scatters (colored by side = sign(pair_margin), straddle midpoints
overlaid for cat6) so the benchmark's metric differences are visible to the eye.

Cells/projections chosen from v3_crispness_benchmark.csv results:
  cat6  mid iguana (2,0):  worst=ranksum  incumbent=ranksum -> use nactive mid, best=hamming
  cat6  wall boa  (0,1):   worst=ranksum, best=hamming (lda close second)
  pair2 ctrl boa  (0,0):   worst=nactive, incumbent=ranksum, best=semantic

Outputs: v3_confront_cat6_iguana20.png, v3_confront_cat6_wallboa01.png,
         v3_confront_pair2_ctrlboa00.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/kaiser/Projects/Masterarbeit")
import matplotlib.pyplot as plt  # noqa: E402
from analysis.core.style import apply_style, save_fig  # noqa: E402

EXPLORE = Path("/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography/explore")
sys.path.insert(0, str(EXPLORE))
from v3_benchmark_crispness import (  # noqa: E402
    load_points, seed_bounds, load_straddle_midpoints, project, CELLS,
)

OUT = EXPLORE

AXLBL = {
    "rank_sum_img_norm": "image rank-sum (norm)", "rank_sum_txt_norm": "text rank-sum (norm)",
    "n_active_img": "n active image genes", "n_active_txt": "n active text genes",
    "d_img_sem": "image semantic distance", "d_txt_sem": "text semantic distance",
    "hamming_to_anchor": "hamming distance to anchor",
}

PANELS = {  # fig name -> (regime, cell key, [projections worst->best], auc dict)
    "v3_confront_cat6_iguana20": ("cat6", "mid iguana b->iguana (2,0)",
                                  ["ranksum", "nactive", "hamming"]),
    "v3_confront_cat6_wallboa01": ("cat6", "wall boa s->snake (0,1)",
                                   ["ranksum", "nactive", "hamming"]),
    "v3_confront_pair2_ctrlboa00": ("pair2", "ctrl boa s->constr (0,0)",
                                    ["nactive", "ranksum", "semantic"]),
}


def main():
    apply_style()
    df = load_points()
    bounds = seed_bounds(df)
    sp = load_straddle_midpoints(df, bounds)
    bench = pd.read_csv(OUT / "v3_crispness_benchmark.csv")

    for figname, (regime, cell, projs) in PANELS.items():
        tc, la, lt = CELLS[cell]
        d = df[(df.prompt_regime == regime) & (df.target_class == tc)
               & (df.level_anchor == la) & (df.level_target == lt)]
        if len(d) > 9000:
            d = d.sample(9000, random_state=0)
        mids = None
        if regime == "cat6":
            mids = sp[(sp.target_class == tc) & (sp.level_anchor == la)
                      & (sp.level_target == lt)]
        y = d.side.to_numpy()

        fig, axs = plt.subplots(1, len(projs), figsize=(5.0 * len(projs), 4.4))
        for ax, proj in zip(np.atleast_1d(axs), projs):
            res = project(d, mids, proj)
            if res is None:
                ax.set_axis_off()
                continue
            if isinstance(res[0], tuple):  # semantic mask case
                (xy, mxy, axes), ok = res
                yy = y[ok.to_numpy()]
            else:
                xy, mxy, axes = res
                yy = y
            # plot anchor side first (majority), target side on top
            for s, color, lbl, z in [(1, "#2274A5", "anchor side", 1),
                                     (0, "#D64933", "target side", 2)]:
                m = yy == s
                ax.scatter(xy[m, 0], xy[m, 1], s=4, alpha=0.25, c=color,
                           label=f"{lbl} (n={m.sum()})", zorder=z, linewidths=0)
            if mxy is not None and len(mxy):
                ax.scatter(mxy[:, 0], mxy[:, 1], s=18, marker="x", c="black",
                           label=f"straddle midpts (n={len(mxy)})", zorder=3,
                           linewidths=0.8, alpha=0.8)
            row = bench[(bench.regime == regime) & (bench.cell == cell)
                        & (bench.projection == proj)]
            auc = row.knn_auc.iloc[0] if len(row) else np.nan
            sep = row.sep_1mBC.iloc[0] if len(row) else np.nan
            ax.set_title(f"{proj}  (kNN AUC {auc:.2f}, 1-BC {sep:.2f})", fontsize=10)
            ax.set_xlabel(AXLBL.get(axes[0], axes[0]), fontsize=9)
            ax.set_ylabel(AXLBL.get(axes[1], axes[1]), fontsize=9)
            leg = ax.legend(loc="best", markerscale=2.5, fontsize=7)
            for lh in leg.legend_handles:
                lh.set_alpha(1.0)
        word_row = d.iloc[0]
        fig.suptitle(f"{regime} — {tc} ({word_row.anchor_word} vs {word_row.target_word}), "
                     f"level ({la},{lt}) — worst → best projection", fontsize=11)
        save_fig(fig, OUT / f"{figname}.png")


if __name__ == "__main__":
    main()
