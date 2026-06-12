"""Chapter two: what the boundary is made of (figures 10–17).

The maps say where the boundary runs; these figures say what it consists
of — which prompt regime each wall species lives in, which genes stake it
out, how steeply it is crossed, how rough it is, what shape the junco
region takes, and how the two regimes disagree about hardness. Figure 17
is the methods appendix: which axes make the boundary crisp at all.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from analysis.core.style import PIPELINE

from .data import (TXT_GENE_GROUPS, cell_key, cell_table,
                   crispness_benchmark, hardness_order, is_wall, points,
                   straddles, transects)
from .language import CLASS_COLORS, NOISE_LP, WALL_COLOR, header
from .registry import figure


# ===========================================================================
# 10 — wall taxonomy: two wall species
# ===========================================================================

@figure(10, "wall_taxonomy")
def wall_taxonomy() -> Figure:
    """Two species of wall — the boa 'snake' wall exists only under the
    2-option prompt; the cello 'songbird' wall survives both regimes."""
    p = points(["source", "g_pair", "target_class", "level_anchor",
                "level_target", "anchor_word", "target_word"],
               anchor_class="junco")
    fig, axes = plt.subplots(2, 1, figsize=(13, 9.2), sharey=True)
    for ax, target in zip(axes, ["boa constrictor", "cello"]):
        sub = p[p.target_class == target]
        cells = sorted(sub.groupby(["level_anchor", "level_target"]).groups)
        for i, (la, lt) in enumerate(cells):
            c = sub[(sub.level_anchor == la) & (sub.level_target == lt)]
            smoo = c[c.source == "smoo"].g_pair
            s1 = c[c.source == "pdq_s1"].g_pair
            anchors = c[c.source == "pdq_anchor"].g_pair
            if len(smoo) > 3000:
                smoo = smoo.sample(3000, random_state=0)
            for vals, off, color in ((smoo, -0.18, PIPELINE["smoo"]),
                                     (s1, 0.18, PIPELINE["pdq"])):
                if len(vals) < 5:
                    continue
                vp = ax.violinplot([vals], positions=[i + off], widths=0.32,
                                   showextrema=False)
                for body in vp["bodies"]:
                    body.set_facecolor(color)
                    body.set_alpha(0.65)
            ax.scatter([i + 0.18] * len(anchors), anchors, marker="D", s=28,
                       color="black", zorder=5)
            aw = c.anchor_word.iloc[0]
            tw = c.target_word.iloc[0]
            wall = is_wall(target, la, lt)
            y0 = -1.16 if i % 2 == 0 else -1.48
            ax.annotate(f"'{aw}'\nvs '{tw}'", (i, y0), ha="center",
                        fontsize=8, annotation_clip=False,
                        fontweight="bold" if wall else "normal",
                        color=WALL_COLOR if wall else "#333333")
        ax.axhline(0, color="black", lw=1, ls="--", alpha=0.6)
        ax.set_xticks([])
        ax.set_ylim(-1.18, 1.15)
        ax.set_ylabel("g = P(anchor word) − P(target word)", fontsize=10)
        ax.set_title(f"junco → {target}", fontsize=12, fontweight="bold")
    fig.legend(handles=[
        Patch(color=PIPELINE["smoo"], alpha=0.65,
              label="2-option prompt  (evolutionary field)"),
        Patch(color=PIPELINE["pdq"], alpha=0.65,
              label="6-option prompt  (random probes, stage 1)"),
        Line2D([], [], marker="D", color="black", ls="",
               label="unmodified anchors  (6-option prompt)"),
    ], loc="lower center", ncol=3, frameon=False, fontsize=9.5,
        bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(top=0.86, bottom=0.13, hspace=0.52)
    header(
        fig,
        "Two species of wall — the boa 'snake' wall exists only in the "
        "2-option prompt; the cello 'songbird' wall survives both",
        "g distributions per label pair, junco anchor · dashed line: the boundary (g = 0) · "
        "black diamonds: the unmodified seed anchors scored under the 6-option prompt\n"
        "pair annotations show the words used in the prompt — walls in dark red · "
        "the boa anchors sit at g ≈ 0 under 6 options: that wall is a property of the prompt, not the image",
        claim_y=0.975, method_y=0.935)
    return fig


# ===========================================================================
# 11 — boundary gene anatomy
# ===========================================================================

@figure(11, "anatomy")
def anatomy() -> Figure:
    """Text genes dominate the surveyed boundary stakes — except at the
    image-only cello wall, which has zero text stakes."""
    s = straddles()
    s = s[s.anchor_class == "junco"].copy()
    s["cell"] = cell_key(s)
    order = hardness_order()
    info = cell_table(s)

    fig = plt.figure(figsize=(16, 11.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], hspace=0.85,
                          wspace=0.25)

    ax = fig.add_subplot(gs[0, 0])
    rows = []
    for kind, grp in s.groupby("boundary_kind"):
        n_txt = (grp.gene_modality == "txt").sum()
        n_img = len(grp) - n_txt
        rows.append({"kind": {"argmax": "predicted-class flip",
                              "pair_margin": "pair-margin flip"}[kind],
                     "image genes": n_img / grp.image_dim.mean(),
                     "text genes": n_txt / 19})
    r = pd.DataFrame(rows).set_index("kind")
    r.plot.bar(ax=ax, color=["#8172B3", "#DD8452"], rot=0)
    ax.set_xlabel("")
    ax.set_ylabel("boundary stakes per available gene")
    ax.legend(fontsize=9)
    ax.set_title("text genes are 2–3× over-represented\namong boundary stakes",
                 fontsize=11)

    ax = fig.add_subplot(gs[0, 1])
    ts = (s.assign(is_txt=s.gene_modality == "txt")
          .groupby("cell").agg(txt_share=("is_txt", "mean"),
                               n=("is_txt", "size")))
    ts = ts.reindex([c for c in order.index if c in ts.index])
    colors = [CLASS_COLORS[c.rsplit(" (", 1)[0]] for c in ts.index]
    ax.bar(range(len(ts)), ts.txt_share, color=colors)
    cello11 = [i for i, c in enumerate(ts.index) if c == "cello (1,1)"]
    if cello11:
        i = cello11[0]
        ax.annotate("'songbird' vs 'string instrument':\n0 text stakes in "
                    f"{int(ts.n.iloc[i])} crossings\n— an image-only wall",
                    (i, 0.02), xytext=(1, 0.3), ha="left", fontsize=9.5,
                    fontweight="bold", color=WALL_COLOR,
                    arrowprops=dict(arrowstyle="->", color=WALL_COLOR,
                                    connectionstyle="arc3,rad=-0.15"))
    ax.set_xticks(range(len(ts)))
    ax.set_xticklabels([info.loc[c, "label"] for c in ts.index],
                       rotation=90, fontsize=6.5)
    for tick, c in zip(ax.get_xticklabels(), ts.index):
        if info.loc[c, "wall"]:
            tick.set_color(WALL_COLOR)
            tick.set_fontweight("bold")
    ax.set_ylabel("text share of stakes")
    ax.set_title("modality composition per label pair\n"
                 "(ordered easy → hard, walls in dark red)", fontsize=11)

    ax = fig.add_subplot(gs[1, :])
    st = s[s.gene_modality == "txt"].copy()
    st["txt_pos"] = st.gene_idx - st.image_dim
    pivot = (st.groupby(["cell", "txt_pos"]).size().unstack(fill_value=0)
             .reindex(columns=range(19), fill_value=0))
    pivot = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0)
    pivot = pivot.reindex([c for c in order.index if c in pivot.index])
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    for name, (lo, hi) in TXT_GENE_GROUPS.items():
        ax.axvline(lo - 0.5, color="#666666", lw=0.8)
        ax.text((lo + hi - 1) / 2, -1.2, name, ha="center", fontsize=9.5)
    ax.set_xticks(range(19))
    ax.set_xticklabels(range(19), fontsize=7)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels([info.loc[c, "label"] for c in pivot.index],
                       fontsize=7.5)
    for tick, c in zip(ax.get_yticklabels(), pivot.index):
        if info.loc[c, "wall"]:
            tick.set_color(WALL_COLOR)
            tick.set_fontweight("bold")
    ax.set_xlabel("text gene position  (operator groups marked above)")
    ax.set_title("text-gene fingerprint per label pair — the MLM slots "
                 "dominate everywhere", fontsize=11, pad=26)
    fig.colorbar(im, ax=ax, shrink=0.8,
                 label="share of the cell's text stakes")
    ax.grid(False)

    fig.subplots_adjust(top=0.88, bottom=0.05)
    header(
        fig,
        "What the boundary is made of — text genes dominate the surveyed "
        "stakes, except at the image-only cello wall",
        "stakes = single-gene edits that flip a decision (surveyed boundary points from stage-2 walks, "
        "6-option prompt) · cells named by their prompt words\n"
        "predicted-class flip: the 6-way argmax changes · pair-margin flip: the "
        "anchor-vs-target margin changes sign",
        claim_y=0.97, method_y=0.94)
    return fig


# ===========================================================================
# 12 — boundary sharpness
# ===========================================================================

@figure(12, "sharpness")
def sharpness() -> Figure:
    """Crossing the boundary is a gentle slope, not a cliff — steepest in
    the MLM text directions; only ~3 % of crossings jump more than 1 lp."""
    s = straddles(kind="pair_margin")
    s["jump"] = (s.margin_after - s.margin_before).abs()
    s["group"] = np.where(s.gene_modality == "img", "image",
                          "text: " + s.txt_group.fillna("?"))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    ax = axes[0]
    order = ["image", "text: mlm", "text: frag", "text: charnoise",
             "text: saliency"]
    data = [s.loc[s.group == g, "jump"].values for g in order]
    present = [(g, d) for g, d in zip(order, data) if len(d) >= 5]
    bp = ax.boxplot([d for _, d in present],
                    tick_labels=[g for g, _ in present],
                    showfliers=False, patch_artist=True)
    for patch, (g, _) in zip(bp["boxes"], present):
        patch.set_facecolor("#8172B3" if g == "image" else "#DD8452")
        patch.set_alpha(0.7)
    ax.set_ylabel("|Δ pair margin| across the crossing edit  [lp]")
    ax.set_title("text edits cross ~2× more steeply than image edits",
                 fontsize=11)
    ax.tick_params(axis="x", rotation=15)

    ax = axes[1]
    for g, color in (("image", "#8172B3"), ("text (all groups)", "#DD8452")):
        vals = (s[s.gene_modality == "img"].jump if g == "image"
                else s[s.gene_modality == "txt"].jump)
        xs = np.sort(vals)
        ax.plot(xs, np.linspace(0, 1, len(xs)), color=color, label=g)
    ax.axvline(1.0, color="gray", ls=":", lw=1)
    ax.text(1.0, 0.04, " 1 lp", fontsize=8.5, color="gray")
    ax.set_xscale("log")
    ax.set_xlabel("|Δ pair margin|  [lp, log scale]")
    ax.set_ylabel("cumulative share of crossings")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title("only ~3 % of crossings jump more than 1 lp", fontsize=11)

    fig.subplots_adjust(top=0.78, bottom=0.14, wspace=0.24)
    header(
        fig,
        "Crossing the boundary is a gentle slope, not a cliff — steepest "
        "in the MLM text directions",
        f"pair-margin straddles: single-gene edits whose anchor-vs-target margin changes sign (n={len(s):,}, "
        "6-option prompt, stage-2 walks)\n"
        "|Δ margin| = how much the margin moves across that one edit · "
        "repeat-noise floor for a single evaluation: 0.38 lp",
        claim_y=0.96, method_y=0.9)
    return fig


# ===========================================================================
# 13 — folding along stage-2 walks
# ===========================================================================

def _crossings(margins: np.ndarray, thresh: float) -> int:
    """Debounced sign changes: a crossing only counts after the margin has
    left the ±thresh band on both sides (hysteresis)."""
    state, count = 0, 0
    for m in margins:
        if abs(m) < thresh:
            continue
        sgn = 1 if m > 0 else -1
        if state != 0 and sgn != state:
            count += 1
        state = sgn
    return count


@figure(13, "folding")
def folding() -> Figure:
    """Boundary roughness tracks attractor subdominance — the pair surface
    is rough where boa overrides it, smooth where it decides."""
    acc = transects(["seed_dir", "flip_id", "step", "pair_margin",
                     "target_class"]).sort_values("step")
    walks = {k: g.pair_margin.to_numpy()
             for k, g in acc.groupby(["seed_dir", "flip_id"])}
    target_of = {k: g.target_class.iloc[0]
                 for k, g in acc.groupby(["seed_dir", "flip_id"])}

    thresholds = [0.0, 0.1, 0.2, 0.3, 0.38, 0.5, 0.75, 1.0]
    sweep = {th: np.array([_crossings(m, th) for m in walks.values()])
             for th in thresholds}
    raw = sweep[0.0]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))

    ax = axes[0, 0]
    ax.hist(raw, bins=np.arange(0, raw.max() + 2) - 0.5, color="#4C72B0")
    ax.set_yscale("log")
    ax.set_xlabel("pair-margin sign changes per walk  (raw)")
    ax.set_ylabel("walks  (log)")
    ax.set_title(f"{np.mean(raw > 1):.1%} of {len(raw):,} walks recross the "
                 "boundary at least once", fontsize=11)

    ax = axes[0, 1]
    ax.plot(thresholds, [np.mean(sweep[th] > 1) for th in thresholds],
            marker="o", color="#C44E52")
    ax.axvline(NOISE_LP, color="gray", ls=":", lw=1)
    ax.annotate("repeat-noise q90", (NOISE_LP, 0.1), fontsize=8.5,
                rotation=90, va="bottom", ha="right", color="gray")
    ax.set_xlabel("hysteresis threshold  [lp]")
    ax.set_ylabel("share of walks with > 1 crossing")
    ax.set_title("debouncing the noise leaves a small genuinely\n"
                 "re-entrant tail on top of a thick rough shelf", fontsize=11)

    ax = axes[1, 0]
    robust = [(k, _crossings(m, NOISE_LP)) for k, m in walks.items()]
    robust = [k for k, c in sorted(robust, key=lambda x: -x[1])[:2]]
    for k, color in zip(robust, ["#55A868", "#8172B3"]):
        m = walks[k]
        ax.plot(range(len(m)), m, marker=".", lw=1.4, color=color,
                label=f"target {target_of[k]} · seed {k[0].split('_')[1]}")
    ax.axhline(0, color="black", lw=1, ls="--", alpha=0.6)
    ax.axhspan(-NOISE_LP, NOISE_LP, color="gray", alpha=0.12)
    ax.set_xlabel("accepted step along walk")
    ax.set_ylabel("pair margin  [lp]")
    ax.set_title("the two most re-entrant walks\n(grey band = noise floor)",
                 fontsize=11)
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    per_target = pd.DataFrame({
        "target": [target_of[k] for k in walks],
        "folded": [c > 1 for c in raw]}).groupby("target").folded.mean()
    per_target = per_target.reindex(
        ["boa constrictor", "ostrich", "green iguana", "cello", "marimba"])
    ax.bar(per_target.index, per_target.values,
           color=[CLASS_COLORS[c] for c in per_target.index])
    ax.set_ylabel("share of folded walks  (raw)")
    ax.set_xticks(range(len(per_target)), per_target.index,
                  rotation=20, ha="right", fontsize=9)
    ax.set_title("zero folding exactly where the pair decision is live\n"
                 "(boa cells: 0 recrossings in all walks)", fontsize=11)

    fig.subplots_adjust(top=0.85, bottom=0.09, hspace=0.42, wspace=0.24)
    header(
        fig,
        "Boundary roughness tracks attractor subdominance — the pair surface "
        "is rough where boa overrides it, smooth where it decides",
        "stage-2 shrink walks (6-option prompt, accepted steps only) · a crossing = sign change of the "
        "anchor-vs-target margin along the walk\n"
        "debounced counts require leaving a ± threshold band on both sides · "
        "repeat-noise q90 of a single evaluation = 0.38 lp",
        claim_y=0.97, method_y=0.925)
    return fig


# ===========================================================================
# 14 — junco region is a text-activity slab
# ===========================================================================

@figure(14, "junco_slab")
def junco_slab() -> Figure:
    """The junco region is a slab, not a ball — junco collapses beyond ~7
    active text genes even right next to the anchor."""
    p = points(["pred_label", "n_active_txt", "hamming_to_anchor",
                "image_dim", "target_class"],
               prompt_regime="cat6", anchor_class="junco")
    p = p[p.hamming_to_anchor >= 0].copy()
    p["ham_norm"] = p.hamming_to_anchor / (p.image_dim + 19)
    p["is_junco"] = p.pred_label == "junco"
    p["family"] = np.where(p.target_class == "boa constrictor",
                           "cells targeting boa", "cells targeting non-boa")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
    im = None
    for ax, fam in zip(axes, ["cells targeting non-boa",
                              "cells targeting boa"]):
        sub = p[p.family == fam]
        tx_bins = np.arange(0, 21) - 0.5
        hm_bins = np.linspace(0, 1, 13)
        H_n, _, _ = np.histogram2d(sub.n_active_txt, sub.ham_norm,
                                   bins=[tx_bins, hm_bins])
        H_j, _, _ = np.histogram2d(sub[sub.is_junco].n_active_txt,
                                   sub[sub.is_junco].ham_norm,
                                   bins=[tx_bins, hm_bins])
        share = np.where(H_n >= 20, H_j / np.maximum(H_n, 1), np.nan)
        im = ax.imshow(share.T, origin="lower", aspect="auto",
                       extent=[-0.5, 19.5, 0, 1], cmap="RdYlBu",
                       vmin=0, vmax=1)
        ax.set_xlabel("active text genes  (of 19)")
        ax.set_title(f"{fam}  ·  n = {len(sub):,}", fontsize=11)
        ax.grid(False)
    axes[0].set_ylabel("fraction of genes changed\n(hamming to anchor, norm.)")
    fig.subplots_adjust(top=0.76, bottom=0.13, left=0.09, right=0.84,
                        wspace=0.12)
    cax = fig.add_axes([0.865, 0.17, 0.015, 0.52])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("share of evaluations predicted junco\n"
                 "(blue = junco region, red = escaped)", fontsize=9.5)
    header(
        fig,
        "The junco region is a slab, not a ball — junco collapses beyond "
        "~7 active text genes even right next to the anchor",
        "6-option prompt, all PDQ sources pooled · per bin: share of evaluations whose 6-way argmax is junco · "
        "white = < 20 evaluations (honest coverage mask; sampling is radius-bimodal)",
        claim_y=0.95, method_y=0.875)
    return fig


# ===========================================================================
# 15 — escape probability vs active text genes
# ===========================================================================

@figure(15, "escape")
def escape() -> Figure:
    """Escape from junco is near-certain wherever ≥ 4 text genes are
    active — with a pushback tail at maximal text corruption."""
    # Stage-1 probes are text-dense by construction (uniform text init), so
    # the low-text range is only sampled by stage-2 sparse states — use both
    # sources and flag the stage-2 path bias in the method line.
    p = points(["source", "pred_label", "n_active_txt", "n_active_img",
                "image_dim", "target_class"],
               prompt_regime="cat6", anchor_class="junco")
    p = p[p.source.isin(["pdq_s1", "pdq_s2"])
          & (p.target_class != "boa constrictor")].copy()
    p["escaped"] = p.pred_label != "junco"
    p["img_frac"] = p.n_active_img / p.image_dim
    strata = [(0.0, 0.1, "< 10 % image genes active"),
              (0.1, 0.5, "10–50 % image genes active"),
              (0.5, 1.01, "> 50 % image genes active")]
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.Purples(np.linspace(0.45, 0.9, len(strata)))
    for (lo, hi, lbl), color in zip(strata, colors):
        sub = p[(p.img_frac >= lo) & (p.img_frac < hi)]
        if len(sub) < 100:
            continue
        g = sub.groupby("n_active_txt").agg(esc=("escaped", "mean"),
                                            n=("escaped", "size"))
        g = g[g.n >= 15]
        ax.plot(g.index, g.esc, marker="o", color=color,
                label=f"{lbl}  (n={int(g.n.sum()):,})")
    ax.set_xlabel("active text genes  (of 19)")
    ax.set_ylabel("P(predicted ≠ junco)")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9.5, loc="lower right",
              title="image-perturbation stratum", title_fontsize=9.5)
    fig.subplots_adjust(top=0.8, bottom=0.11)
    header(
        fig,
        "Escape from junco is near-certain wherever ≥ 4 text genes are "
        "active — with a pushback tail at maximal text corruption",
        "6-option prompt, non-boa cells, stage 1 + 2 pooled (stage 2 is path-constrained) · bins need ≥ 15 "
        "evaluations · fewer than 4 active text genes is unsampled in these cells\n"
        "the non-monotone tail: at minimal image perturbation, fully corrupted "
        "text pushes predictions back toward junco",
        claim_y=0.96, method_y=0.9)
    return fig


# ===========================================================================
# 16 — regime decoupling: near-anchor flip reach vs evolutionary hardness
# ===========================================================================

@figure(16, "flipreach_hardness")
def flipreach_hardness() -> Figure:
    """Regime decoupling — the pairs hardest under the 2-option prompt flip
    most easily near the anchor under 6 options."""
    p = points(["pair_margin", "hamming_to_anchor", "image_dim",
                "target_class", "level_anchor", "level_target",
                "anchor_word", "target_word"],
               prompt_regime="cat6", anchor_class="junco")
    p = p[p.hamming_to_anchor >= 0].copy()
    p["ham_norm"] = p.hamming_to_anchor / (p.image_dim + 19)
    near = p[p.ham_norm < 0.15].copy()
    near["cell"] = cell_key(near)
    rate = near.groupby("cell").agg(
        flip_rate=("pair_margin", lambda s: float((s < 0).mean())),
        n=("pair_margin", "size"),
        target=("target_class", "first"),
        aw=("anchor_word", "first"), tw=("target_word", "first"))
    rate = rate.join(hardness_order().rename("min_TgtBal"), how="inner")

    fig, ax = plt.subplots(figsize=(9.5, 7))
    for tgt, grp in rate.groupby("target"):
        ax.scatter(grp.min_TgtBal, grp.flip_rate, s=52,
                   color=CLASS_COLORS[tgt], label=f"target: {tgt}",
                   edgecolors="white", linewidth=0.5)
    ann = [(cell, r) for cell, r in rate.iterrows()
           if r.flip_rate > 0.5 or r.min_TgtBal > 0.05]
    for k, (cell, r) in enumerate(sorted(ann,
                                         key=lambda x: x[1].min_TgtBal)):
        dy = 5 if k % 2 == 0 else -12
        ax.annotate(f"'{r.aw}' vs '{r.tw}'", (r.min_TgtBal, r.flip_rate),
                    fontsize=7.5, xytext=(5, dy),
                    textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("hardness under the 2-option prompt — cell-median best "
                  "|P(A) − P(B)|  (log; right = harder)")
    ax.set_ylabel("near-anchor flip share under the 6-option prompt\n"
                  "(evaluations < 15 % genes changed on the target side)")
    ax.legend(fontsize=9, loc="center right")
    fig.subplots_adjust(top=0.82, bottom=0.11)
    header(
        fig,
        "Regime decoupling — the pairs hardest under the 2-option prompt "
        "flip most easily near the anchor under 6 options",
        "each point is one label pair (cell) · top-right: hard for the evolutionary search yet trivially "
        "flipped by nearby probes (boa 'snake' pairs)\n"
        "bottom-right: hard in both regimes (the cello pairs — flip share ≈ 0 "
        "everywhere) · annotations: prompt words",
        claim_y=0.96, method_y=0.92)
    return fig


# ===========================================================================
# 17 — projection crispness benchmark (methods figure)
# ===========================================================================

@figure(17, "projection_benchmark")
def projection_benchmark() -> Figure:
    """Which axes make the boundary crisp — semantic distance wins;
    rank-sum manipulation strength is never first."""
    b = crispness_benchmark().dropna(subset=["knn_auc"])
    b["col"] = (b.regime.map({"pair2": "2-option", "cat6": "6-option"})
                + " · " + b.cell)
    pivot = b.pivot_table(index="projection", columns="col", values="knn_auc")
    proj_order = ["semantic", "lda_pc", "pca2", "hamming", "nactive",
                  "ranksum"]
    pivot = pivot.reindex([p for p in proj_order if p in pivot.index])

    fig, ax = plt.subplots(figsize=(1.45 * len(pivot.columns) + 2.5,
                                    0.62 * len(pivot) + 3.2))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0.5, vmax=0.95,
                   aspect="auto")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8.5)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, fontsize=8, rotation=35, ha="right",
                       rotation_mode="anchor")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=9.5)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8,
                 label="kNN-AUC — how separable the two sides are\n"
                       "from the 2D coordinates alone")
    fig.subplots_adjust(top=0.72, bottom=0.22)
    header(
        fig,
        "Which axes make the boundary crisp — semantic distance wins; "
        "rank-sum manipulation strength is never first",
        "per (regime × cell): kNN-AUC of side-classification using only the 2D projected coordinates · "
        "green = the boundary separates cleanly in this view\n"
        "hamming's 6-option scores are stage-2 sampling artifacts (walks vary "
        "hamming monotonically) — see the v3 exploration memo",
        claim_y=0.95, method_y=0.885)
    return fig
