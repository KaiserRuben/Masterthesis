"""Chapter one: where the boundary runs (figures 01–09).

These figures project the survey into planes where the junco decision
boundary becomes a drawable object — a contour in semantic coordinates, a
coastline between class regions, a sea level in margin terrain, a radius
per direction, a straight line in the model's own output space — and then
watch the search move against it over time.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .data import (BOA_WALL, CELLO_WALL, IGUANA, MARIMBA, QUARTET, Cell,
                   evolutionary_field, points, straddles, transects)
from .grids import field, majority_rgba
from .language import (AXIS, CLASS_COLORS, EASY_COLOR, NOISE_G, NOISE_LP,
                       WALL_COLOR, header)
from .registry import figure


def _softmax(logprobs: np.ndarray) -> np.ndarray:
    p = np.exp(logprobs - logprobs.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


# ===========================================================================
# 01 — the shape of a wall
# ===========================================================================

@figure(1, "wall_shape")
def wall_shape() -> Figure:
    """An easy crossing has a boundary line; a wall is a plateau that never
    descends — pooled g-fields per cluster, flat and as terrain."""
    df = evolutionary_field()
    la, lt, tc = df.level_anchor, df.level_target, df.target_class
    clusters = [
        ("EASY CROSSING", "generic anchor word 'bird' · targets marimba / iguana",
         tc.isin(["marimba", "green iguana"]) & (la == 2), EASY_COLOR),
        ("BOA WALL", "target word 'snake'  (Lt=1, La≠1)",
         (tc == "boa constrictor") & (lt == 1) & (la != 1), WALL_COLOR),
        ("CELLO WALL", "anchor word 'songbird'  (La=1, Lt≠1)",
         (tc == "cello") & (la == 1) & (lt != 1), WALL_COLOR),
        ("DOUBLE WALL", "'songbird' vs 'snake' / 'string instr.'  (La=1, Lt=1)",
         tc.isin(["boa constrictor", "cello"]) & (la == 1) & (lt == 1),
         WALL_COLOR),
    ]

    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    cmap = plt.get_cmap("RdBu_r")

    fig = plt.figure(figsize=(19, 10.2))
    pcm = None
    for j, (name, desc, mask, col) in enumerate(clusters):
        d = df[mask]
        n_cells = len(d[["target_class", "level_anchor", "level_target"]]
                      .drop_duplicates())
        fld = field(d.d_img_sem_n, d.d_txt_sem_n, d.g_pair,
                    nbins=30, extent=(0, 1.1, 0, 1.1), min_n=25)
        img = np.ma.masked_invalid(fld.img)

        ax = fig.add_subplot(2, 4, j + 1)
        pcm = ax.pcolormesh(fld.xe, fld.ye, img, cmap=cmap, norm=norm,
                            shading="flat")
        pcm.cmap.set_bad("0.92")
        if fld.spans_zero:
            fld.boundary(ax, lw=2.8)
            ax.contour(fld.xc, fld.yc, img, levels=[-0.2, 0.2], colors="k",
                       linewidths=0.7, linestyles="--")
        ax.set_title(name, color=col, fontsize=13, fontweight="bold", pad=30)
        ax.text(0.5, 1.085, desc, transform=ax.transAxes, ha="center",
                fontsize=9, color="0.35")
        ax.text(0.5, 1.015, f"{len(d):,} evaluations · {n_cells} label pairs · "
                            f"{d.seed_dir.nunique()} seeds",
                transform=ax.transAxes, ha="center", fontsize=8, color="0.5")
        ax.set_xlabel(AXIS.img_sem, fontsize=9)
        if j == 0:
            ax.set_ylabel(AXIS.txt_sem, fontsize=9)
        ax.grid(False)

        Z = fld.img
        X, Y = np.meshgrid(fld.xc, fld.yc)
        ax3 = fig.add_subplot(2, 4, 4 + j + 1, projection="3d")
        fc = cmap(norm(np.where(np.isnan(Z), 0, Z)))
        fc[np.isnan(Z)] = (0, 0, 0, 0)
        ax3.plot_surface(X, Y, Z, facecolors=fc, rstride=1, cstride=1,
                         linewidth=0.1, edgecolor=(0, 0, 0, 0.12), shade=False)
        ax3.plot_surface(X, Y, np.zeros_like(Z), color="0.5", alpha=0.15,
                         rstride=5, cstride=5, linewidth=0)
        Zm = np.ma.masked_invalid(Z)
        if Zm.count() and Zm.min() < 0 < Zm.max():
            ax3.contour(X, Y, Zm, levels=[0.0], colors="k", linewidths=2.0)
            ax3.contour(X, Y, Zm, levels=[0.0], colors="k", linewidths=2.2,
                        offset=-1.05)
        ax3.set_zlim(-1.05, 1.05)
        ax3.set_xlabel("image dist", fontsize=8, labelpad=-2)
        ax3.set_ylabel("text dist", fontsize=8, labelpad=-2)
        ax3.set_zlabel("g", fontsize=9, labelpad=-4)
        ax3.view_init(elev=25, azim=-128)
        ax3.tick_params(labelsize=6, pad=-1)

    fig.subplots_adjust(left=0.05, right=0.9, top=0.8, bottom=0.1,
                        wspace=0.24, hspace=0.34)
    cax = fig.add_axes([0.925, 0.18, 0.013, 0.55])
    cb = fig.colorbar(pcm, cax=cax)
    cb.set_label(AXIS.g, fontsize=10)
    cb.ax.text(0.5, 1.04, "anchor\nside", transform=cb.ax.transAxes,
               ha="center", fontsize=8, color="0.35")
    cb.ax.text(0.5, -0.04, "target\nside", transform=cb.ax.transAxes,
               ha="center", va="top", fontsize=8, color="0.35")
    fig.legend(handles=[
        Line2D([], [], color="k", lw=2.8, label="decision boundary  (g = 0)"),
        Line2D([], [], color="k", lw=0.8, ls="--",
               label="near-boundary band  (|g| = 0.2)"),
        Patch(fc="0.92", ec="0.7", label="unsampled  (< 25 evaluations)"),
    ], loc="lower center", ncol=3, frameon=False, fontsize=9.5,
        bbox_to_anchor=(0.47, 0.0))
    header(
        fig,
        "The shape of a wall — an easy crossing has a boundary line; "
        "a wall is a plateau that never descends",
        "2-option prompt regime (evolutionary search field) · junco-anchored seeds, label pairs pooled per cluster · "
        "g per evaluation, median per bin\n"
        "top: flat map with the boundary drawn · bottom: the same field as terrain "
        "(grey plane = boundary level g 0)",
        claim_y=0.97, method_y=0.925)
    return fig


# ===========================================================================
# 02 — region map: junco island, boa sea
# ===========================================================================

@figure(2, "region_map")
def region_map() -> Figure:
    """Junco island, boa sea — majority-class regions with surveyed border
    stakes, in two coordinate planes."""
    pts = points(["source", "pred_label", "n_active_txt", "rank_sum_img_norm",
                  "rank_sum_txt_norm", "hamming_to_anchor", "image_dim"],
                 prompt_regime="cat6")
    pts = pts[pts.source.isin(["pdq_s1", "pdq_s2"])]
    pts["ham_norm"] = pts.hamming_to_anchor / (pts.image_dim + 19)

    stakes = straddles(kind="argmax")
    stakes["ham_norm"] = stakes.hamming_to_anchor_after / (stakes.image_dim + 19)
    junco_boa = stakes[stakes.label_after.isin(["junco", "boa constrictor"])
                       & stakes.label_before.isin(["junco", "boa constrictor"])]
    ostrich = stakes[(stakes.label_after == "ostrich")
                     | (stakes.label_before == "ostrich")]

    planes = [
        dict(x="ham_norm", y="n_active_txt",
             xl="fraction of genes changed  (hamming / genome size)",
             yl="active text genes  (of 19)",
             xlim=(0, 1.0), ylim=(-0.5, 19.5),
             xe=np.linspace(0, 1.0, 29), ye=np.arange(-0.5, 20.5, 1.0),
             sx="ham_norm", sy="m_n_active_txt"),
        dict(x="rank_sum_img_norm", y="rank_sum_txt_norm",
             xl=AXIS.img_strength, yl=AXIS.txt_strength,
             xlim=(0, 1.0), ylim=(0, 1.0),
             xe=np.linspace(0, 1.0, 29), ye=np.linspace(0, 1.0, 29),
             sx="m_rank_sum_img_norm", sy="m_rank_sum_txt_norm"),
    ]
    col_specs = [("pdq_s1", "random probes around the anchor  (stage 1)", False),
                 ("pdq_s2", "shrink walks  (stage 2, path-constrained)", False),
                 ("pdq_s2", "stage 2 + surveyed border stakes", True)]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for row, P in enumerate(planes):
        for col, (src, title, with_stakes) in enumerate(col_specs):
            ax = axes[row, col]
            sub = pts[pts.source == src]
            img = majority_rgba(sub[P["x"]], sub[P["y"]], sub.pred_label,
                                xe=P["xe"], ye=P["ye"], palette=CLASS_COLORS)
            ax.imshow(img, origin="lower", aspect="auto",
                      interpolation="nearest",
                      extent=(P["xe"][0], P["xe"][-1], P["ye"][0], P["ye"][-1]))
            if with_stakes:
                ax.scatter(junco_boa[P["sx"]], junco_boa[P["sy"]], s=3,
                           c="black", alpha=0.4, linewidths=0, zorder=3)
                ax.scatter(ostrich[P["sx"]], ostrich[P["sy"]], s=24,
                           c="#E6A817", edgecolors="black", linewidths=0.6,
                           zorder=4)
            ax.set_xlim(*P["xlim"])
            ax.set_ylim(*P["ylim"])
            ax.set_xlabel(P["xl"], fontsize=9.5)
            if col == 0:
                ax.set_ylabel(P["yl"], fontsize=10)
            if row == 0:
                ax.set_title(f"{title}\n", fontsize=11)
                ax.text(0.5, 1.02, f"n = {len(sub):,}",
                        transform=ax.transAxes, ha="center", fontsize=8.5,
                        color="0.5")
            ax.grid(False)

    handles = ([Patch(color=CLASS_COLORS[c], label=f"predicted: {c}")
                for c in ["junco", "boa constrictor", "ostrich"]]
               + [Line2D([], [], marker="o", ls="", color="black", ms=4,
                         label="junco↔boa border stake  (single-gene flip)"),
                  Line2D([], [], marker="o", ls="", mfc="#E6A817", mec="black",
                         ms=8, label="boa↔ostrich border stake")])
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(top=0.84, bottom=0.12, hspace=0.3, wspace=0.22)
    header(
        fig,
        "Junco island, boa sea — whatever target the search aims at, "
        "the territory beyond the island is boa",
        "6-option prompt regime (PDQ), all junco-anchored seeds pooled · bin color = majority predicted class, "
        "opacity = majority share · white = < 4 evaluations\n"
        "stakes = midpoints of single-gene flips that change the predicted class "
        "(exact, surveyed border points) · top/bottom row: two coordinate systems for the same territory",
        claim_y=0.975, method_y=0.93)
    return fig


# ===========================================================================
# 03 — margin relief: 'sea level' terrain
# ===========================================================================

@figure(3, "margin_relief")
def margin_relief() -> Figure:
    """The junco trench — lp(boa) − lp(junco) as terrain whose sea level is
    the decision border, draped with the predicted class."""
    pts = points(["pred_label", "logprobs", "rank_sum_img_norm",
                  "rank_sum_txt_norm"],
                 prompt_regime="cat6", source="pdq_s2")
    lp = np.stack(pts["logprobs"].to_numpy())
    pts = pts.reset_index(drop=True)
    pts["z"] = lp[:, 3] - lp[:, 0]      # lp(boa) − lp(junco)

    fld = field(pts.rank_sum_img_norm, pts.rank_sum_txt_norm, pts.z,
                stat="mean", nbins=24, extent=(0, 1, 0, 1), min_n=4)
    drape = majority_rgba(pts.rank_sum_img_norm, pts.rank_sum_txt_norm,
                          pts.pred_label, xe=fld.xe, ye=fld.ye,
                          palette=CLASS_COLORS, alpha_floor=0.35,
                          alpha_min=0.2)
    Z = fld.img

    fig = plt.figure(figsize=(14.5, 7.2))
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    X, Y = np.meshgrid(fld.xc, fld.yc)
    fc = drape.copy()
    fc[np.isnan(Z)] = (1, 1, 1, 0)
    ax3.plot_surface(X, Y, Z, facecolors=fc, rstride=1, cstride=1,
                     linewidth=0.1, edgecolor="#888888", antialiased=False,
                     shade=False)
    xx, yy = np.meshgrid([0, 1], [0, 1])
    ax3.plot_surface(xx, yy, np.zeros_like(xx), color="#4C72B0", alpha=0.15,
                     shade=False)
    ax3.set_xlabel("image strength (rank-sum, norm.)", fontsize=8.5, labelpad=2)
    ax3.set_ylabel("text strength (rank-sum, norm.)", fontsize=8.5, labelpad=2)
    ax3.set_zlabel("lp(boa) − lp(junco)", fontsize=9)
    ax3.set_title("the margin as terrain — blue plane = sea level\n"
                  "(decision border), drape = predicted class", fontsize=10.5)
    ax3.view_init(elev=28, azim=-60)

    ax = fig.add_subplot(1, 2, 2)
    vmax = np.nanquantile(np.abs(Z), 0.98)
    im = ax.imshow(Z, origin="lower", extent=(0, 1, 0, 1), aspect="auto",
                   cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   interpolation="nearest")
    fld.boundary(ax, lw=1.8)
    ax.contour(fld.xc, fld.yc, Z, levels=[-NOISE_LP, NOISE_LP],
               colors="black", linewidths=0.7, linestyles=":")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("mean  lp(boa) − lp(junco)   [red = boa above water]",
                 fontsize=9.5)
    ax.set_xlabel(AXIS.img_strength, fontsize=10)
    ax.set_ylabel(AXIS.txt_strength, fontsize=10)
    ax.set_title("the same map, flat — bold: border (margin = 0),\n"
                 "dotted: ±0.38 lp repeat-noise band", fontsize=10.5)
    ax.grid(False)

    fig.legend(handles=[Patch(color=CLASS_COLORS[c], label=f"predicted: {c}")
                        for c in ["junco", "boa constrictor", "ostrich"]],
               loc="lower center", ncol=3, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.27, 0.01))
    fig.subplots_adjust(top=0.78, bottom=0.14, wspace=0.18)
    header(
        fig,
        "The junco trench — only a small low-manipulation basin sits below "
        "the decision border; everywhere else boa is above water",
        "6-option prompt regime · stage-2 walk evaluations (path-constrained sampling — territory between walks "
        "is unsampled) · z = mean lp(boa) − lp(junco) per bin · blank = < 4 evaluations",
        claim_y=0.96, method_y=0.885)
    return fig


# ===========================================================================
# 04 — boundary compass (polar r*(θ))
# ===========================================================================

@figure(4, "compass")
def compass() -> Figure:
    """How far to the boundary, in which direction — r*(θ) over the
    image↔text mix of active genes; walls have no r* in any direction."""
    rng = np.random.default_rng(7)
    df = evolutionary_field(extra=["n_active_img", "n_active_txt",
                                   "image_dim"])
    xi = (df.n_active_img / df.image_dim).to_numpy()
    yi = (df.n_active_txt / 19).to_numpy()
    QX, QY = np.quantile(xi, .99), np.quantile(yi, .99)
    df["theta"] = np.arctan2(yi / QY, xi / QX)
    df["r"] = np.hypot(xi / QX, yi / QY)

    TH_EDGES = np.linspace(0, np.pi / 2, 16)
    TH_C = 0.5 * (TH_EDGES[:-1] + TH_EDGES[1:])
    WIN, STEP, RMAX = 0.16, 0.04, 1.3
    RATE, NMIN_WIN, N_BOOT = 0.10, 30, 150

    def rstar(rv, gv):
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
        sup = np.full(len(TH_C), np.nan)
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

    bearings = [(Cell("boa constrictor", 2, 2), "#D64933"),
                (IGUANA, "#E6A817"),
                (MARIMBA, "#55A868"),
                (Cell("ostrich", 0, 1), "#8172B3"),
                (BOA_WALL, "#2274A5"),
                (CELLO_WALL, "#444444")]

    fig, ax = plt.subplots(figsize=(12.8, 8.6),
                           subplot_kw=dict(projection="polar"))
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_ylim(0, RMAX)

    open_h, wall_h = [], []
    sup_max = None
    for cell, col in bearings:
        c = df[cell.mask(df)]
        cur, lob, hib, sup = cell_curve(c)
        sup_max = sup if sup_max is None else np.fmax(sup_max, sup)
        aw, tw = cell.words(df)
        lab = (f"'{aw}' vs '{tw}'   "
               f"({cell.target.split()[0]}, {cell.levels})")
        ok = np.isfinite(cur)
        if ok.any():
            ax.plot(TH_C[ok], cur[ok], "-o", color=col, lw=2.2, ms=4,
                    zorder=5)
            bok = ok & np.isfinite(lob)
            ax.fill_between(TH_C[bok], lob[bok], hib[bok], color=col,
                            alpha=0.20, lw=0, zorder=4)
            open_h.append(Line2D([], [], color=col, lw=2.2, marker="o",
                                 ms=4, label=lab))
        else:
            wall_h.append(Line2D([], [], color=col, marker="x", ls="",
                                 ms=7, mew=1.8, label=lab))
        closed = np.isfinite(sup) & ~ok
        if closed.any():
            ax.plot(TH_C[closed], sup[closed], "x", color=col, ms=7, mew=1.8,
                    alpha=0.9, zorder=3)

    oks = np.isfinite(sup_max)
    ax.fill_between(TH_C[oks], 0, sup_max[oks], color="0.55", alpha=0.10,
                    zorder=0)
    ax.plot(TH_C[oks], sup_max[oks], color="0.55", lw=1, ls=":")

    ax.set_xticks(np.radians([0, 15, 30, 45, 60, 75, 90]))
    ax.set_xticklabels(["0°\npure image", "15°", "30°", "45°", "60°", "75°",
                        "90°  pure text"])
    ax.text(np.radians(45), RMAX * 1.18, "direction of perturbation",
            ha="center", fontsize=10, color="0.3")
    ax.text(np.radians(-8), RMAX * 0.55, "r = total manipulation",
            ha="center", fontsize=9, color="0.3", rotation=0)

    leg1 = ax.legend(handles=open_h,
                     title="open pairs — curve: boundary radius r*(θ)\n"
                           "(shaded band: bootstrap 10–90 %)",
                     loc="upper left", bbox_to_anchor=(1.02, 1.02),
                     fontsize=9.5, title_fontsize=9.5, alignment="left")
    ax.add_artist(leg1)
    wall_h.append(Line2D([], [], color="0.55", lw=1, ls=":",
                         label="sampled territory (q95 radius per direction)"))
    ax.legend(handles=wall_h, title="WALLS — no opening found anywhere "
                                    "searched\n(× = searched-to radius per direction)",
              loc="upper left", bbox_to_anchor=(1.02, 0.52), fontsize=9.5,
              title_fontsize=9.5, alignment="left")

    fig.subplots_adjust(left=0.04, right=0.56, top=0.82, bottom=0.06)
    header(
        fig,
        "The boundary compass — how far to the boundary, in which direction",
        "2-option prompt regime (evolutionary field), junco anchor at the origin · "
        "direction = image↔text mix of active genes (q99-scaled) · \n"
        "r*(θ) = first radius where ≥ 10 % of evaluations sit on the target side "
        "(sliding window 0.16, n ≥ 30 per window) · walls: no such radius exists in any direction",
        claim_y=0.96, method_y=0.91)
    return fig


# ===========================================================================
# 05 — output-space map: the boundary as a line
# ===========================================================================

@figure(5, "output_space")
def output_space() -> Figure:
    """Seen from the model's own coordinates the junco↔boa boundary is a
    straight line — barycentric map plus the unrolled margin view."""
    rng = np.random.default_rng(42)
    df = points(["source", "logprobs", "pred_label", "hamming_to_anchor",
                 "target_class"],
                prompt_regime="cat6", anchor_class="junco")
    L = np.stack(df.logprobs.to_numpy()).astype(np.float64)
    P = _softmax(L)
    p_j, p_b = P[:, 0], P[:, 3]
    p_rest = 1.0 - p_j - p_b
    SQ3 = np.sqrt(3) / 2
    df = df.assign(bx=p_b + 0.5 * p_rest, by=SQ3 * p_rest, u=L[:, 0] - L[:, 3])

    n_vec = np.zeros(6)
    n_vec[0], n_vec[3] = 1 / np.sqrt(2), -1 / np.sqrt(2)
    Lc = L - L.mean(axis=0, keepdims=True)
    res = Lc - np.outer(Lc @ n_vec, n_vec)
    fit = rng.choice(len(res), size=min(40000, len(res)), replace=False)
    _, _, Vt = np.linalg.svd(res[fit] - res[fit].mean(axis=0),
                             full_matrices=False)
    df["pc1"] = res @ Vt[0]

    keep = (df.source != "pdq_s2").to_numpy()
    s2 = np.flatnonzero(~keep)
    keep[rng.choice(s2, size=min(20000, len(s2)), replace=False)] = True
    sub = df[keep]
    anchors = sub[sub.source == "pdq_anchor"]

    stakes = straddles(kind="argmax")
    Lm = 0.5 * (np.stack(stakes.logprobs_before.to_numpy()).astype(np.float64)
                + np.stack(stakes.logprobs_after.to_numpy()).astype(np.float64))
    Pm = _softmax(Lm)
    sm_bx = Pm[:, 3] + 0.5 * (1 - Pm[:, 0] - Pm[:, 3])
    sm_by = SQ3 * (1 - Pm[:, 0] - Pm[:, 3])

    def ternary_frame(ax):
        ax.plot([0, 1, 0.5, 0], [0, 0, SQ3, 0], color="0.3", lw=1)
        ax.plot([0.5, 0.5], [0, SQ3], color="k", lw=1.4, ls="--", zorder=5)
        ax.text(0.49, SQ3 / 2, "decision border  P(junco) = P(boa)",
                fontsize=8, rotation=90, va="center", ha="right", color="k")
        ax.text(-0.02, -0.04, "all junco", ha="right", fontsize=10,
                color=CLASS_COLORS["junco"], fontweight="bold")
        ax.text(1.02, -0.04, "all boa", ha="left", fontsize=10,
                color=CLASS_COLORS["boa constrictor"], fontweight="bold")
        ax.text(0.47, SQ3 + 0.04, "all other classes",
                ha="right", fontsize=9, color="0.35")
        ax.set_xlim(-0.14, 1.14)
        ax.set_ylim(-0.12, SQ3 + 0.12)
        ax.set_aspect("equal")
        ax.axis("off")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11.5))

    ax = axes[0, 0]
    h = sub[sub.hamming_to_anchor >= 0]
    sc = ax.scatter(h.bx, h.by, s=3, c=h.hamming_to_anchor, cmap="viridis",
                    alpha=0.35, lw=0, vmin=0,
                    vmax=np.nanquantile(h.hamming_to_anchor, 0.98))
    cb = plt.colorbar(sc, ax=ax, fraction=0.04)
    cb.set_label("genes changed vs anchor (hamming)", fontsize=9)
    ax.scatter(anchors.bx, anchors.by, s=45, marker="*", c="black", zorder=6,
               label="unmodified anchors")
    ternary_frame(ax)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("input distance organizes along the output map —\n"
                 "few edits: junco corner · many edits: across the border",
                 fontsize=10.5)

    ax = axes[0, 1]
    ax.scatter(sub.bx, sub.by, s=2, c="0.82", alpha=0.25, lw=0)
    for mod, col, lab in [("img", "#8172B3", "image-gene flip"),
                          ("txt", "#DD8452", "text-gene flip")]:
        m = (stakes.gene_modality == mod).to_numpy()
        ax.scatter(sm_bx[m], sm_by[m], s=7, c=col, alpha=0.6, lw=0,
                   label=f"{lab}  (n={m.sum():,})")
    ternary_frame(ax)
    ax.legend(loc="upper right", fontsize=9, markerscale=2.5)
    ax.set_title("surveyed crossings sit on the border —\n"
                 "midpoints of single-gene flips that change the prediction",
                 fontsize=10.5)

    ax = axes[1, 0]
    for lab in ["junco", "boa constrictor"]:
        s = sub[sub.pred_label == lab]
        ax.scatter(s.u, s.pc1, s=3, c=CLASS_COLORS[lab], alpha=0.3, lw=0,
                   label=f"predicted {lab}")
    ax.axvline(0, color="k", lw=1.4, ls="--")
    ax.axvspan(-NOISE_LP, NOISE_LP, color="0.5", alpha=0.18, lw=0)
    ax.set_xlabel("lp(junco) − lp(boa)    [boundary = 0 · grey: repeat-noise "
                  "±0.38 lp]", fontsize=10)
    ax.set_ylabel("residual PC1  (largest logprob variation ⊥ boundary)",
                  fontsize=10)
    ax.legend(loc="upper left", fontsize=9, markerscale=2.5)
    ax.set_title("unrolled: the boundary as a straight line —\n"
                 "x is the exact margin, y the largest remaining variation",
                 fontsize=10.5)

    ax = axes[1, 1]
    for tc in ["ostrich", "green iguana", "boa constrictor", "cello",
               "marimba"]:
        s = sub[sub.target_class == tc]
        if len(s):
            ax.scatter(s.u, s.pc1, s=3, c=CLASS_COLORS[tc], alpha=0.3, lw=0,
                       label=f"search target: {tc}")
    ax.axvline(0, color="k", lw=1.4, ls="--")
    ax.set_xlabel("lp(junco) − lp(boa)", fontsize=10)
    ax.set_ylabel("residual PC1", fontsize=10)
    ax.legend(loc="upper left", fontsize=8.5, markerscale=2.5)
    ax.set_title("same chart by experiment cell —\n"
                 "each label pair probes a different segment of the border",
                 fontsize=10.5)

    fig.subplots_adjust(top=0.86, bottom=0.06, hspace=0.3, wspace=0.24)
    header(
        fig,
        "Seen from the model's own coordinates, the junco↔boa boundary "
        "is a straight line",
        "6-option prompt regime · softmax of the 6 class logprobs · top: barycentric map on (junco, boa, "
        "rest = ostrich+iguana+cello+marimba) — every evaluation is a point in the model's output space\n"
        "bottom: the same data unrolled along the boundary normal · stage-2 rows "
        "subsampled to 20k (path-constrained sampling)",
        claim_y=0.97, method_y=0.935)
    return fig


# ===========================================================================
# 06 — conquest map
# ===========================================================================

@figure(6, "conquest")
def conquest() -> Figure:
    """When did the search first reach each region, and when did it first
    see the far side? Easy pairs cross within ~20 generations; walls never."""
    df = evolutionary_field()
    NBINS, EXT = 18, (0, 1.25, 0, 1.25)

    fig, axes = plt.subplots(2, 4, figsize=(16, 9), sharex=True, sharey=True)
    pm = None
    levels = [0, 1, 5, 20, 50, 100, 200]
    norm = BoundaryNorm(levels, 256)
    for c, cell in enumerate(QUARTET):
        d = df[cell.mask(df)].copy()
        d.generation = d.generation.astype(float)
        aw, tw = cell.words(df)
        gfield = field(d.d_img_sem_n, d.d_txt_sem_n, d.g_pair,
                       nbins=NBINS, extent=EXT, min_n=8)
        crossed = d.g_pair < -NOISE_G
        visit = field(d.d_img_sem_n, d.d_txt_sem_n, d.generation,
                      stat="min", nbins=NBINS, extent=EXT)
        cross = field(d.d_img_sem_n[crossed], d.d_txt_sem_n[crossed],
                      d.generation[crossed], stat="min", nbins=NBINS,
                      extent=EXT)
        n_cross = int(np.isfinite(cross.values).sum())

        for r, fld in enumerate([visit, cross]):
            ax = axes[r, c]
            ax.set_facecolor("0.9")
            pm = ax.pcolormesh(fld.xe, fld.ye, fld.img, cmap="viridis",
                               norm=norm)
            gfield.boundary(ax)
            ax.plot(0, 0, marker="*", ms=15, color="white", mec="black",
                    mew=0.8)
            if r == 0:
                ax.set_title(f"{cell.tag}\n'{aw}' vs '{tw}'", color=cell.color,
                             fontsize=10.5, fontweight="bold", pad=20)
                ax.text(0.5, 1.025, f"junco → {cell.target}  ({cell.levels})",
                        transform=ax.transAxes, ha="center", fontsize=8,
                        color="0.45")
            if r == 1:
                ax.set_xlabel("image distance from seed", fontsize=9)
                never = n_cross == 0
                ax.text(0.04, 0.96,
                        "never crossed — 0 bins" if never
                        else f"{n_cross} bins crossed",
                        transform=ax.transAxes, va="top",
                        fontsize=9.5 if never else 8.5,
                        fontweight="bold" if never else "normal",
                        color=WALL_COLOR if never else "0.2",
                        bbox=dict(fc="white", alpha=0.85, ec="none"))
            if c == 0:
                row_lab = ["first visit", "first crossing\n(beyond noise)"][r]
                ax.set_ylabel(f"{row_lab}\n\ntext distance from seed",
                              fontsize=9.5)
            ax.set_xlim(*EXT[:2])
            ax.set_ylim(*EXT[2:])
            ax.grid(False)

    fig.subplots_adjust(top=0.82, bottom=0.09, right=0.88, hspace=0.16,
                        wspace=0.12)
    cax = fig.add_axes([0.905, 0.2, 0.013, 0.5])
    cb = fig.colorbar(pm, cax=cax, ticks=levels)
    cb.set_label("earliest generation  (dark = early, yellow = late, "
                 "grey = never)", fontsize=9.5)
    header(
        fig,
        "The conquest map — easy pairs see the far side of the boundary "
        "within ~20 generations; walls never do",
        "2-option prompt regime · 3 seeds pooled per label pair · axes: semantic distance from seed, per-seed "
        "q99-normalized · top row: earliest generation each map bin was visited\n"
        "bottom row: earliest generation a beyond-noise crossing (g < −0.19) was observed there · black contour: "
        "boundary (g = 0) of the all-generation field · star: anchor · grey: never reached / never crossed",
        claim_y=0.97, method_y=0.92)
    return fig


# ===========================================================================
# 07 — field over time
# ===========================================================================

@figure(7, "field_over_time")
def field_over_time() -> Figure:
    """The boundary does not move — the search's picture of it does.
    Median-g fields per generation window for the canonical quartet."""
    windows = [(0, 20), (40, 60), (90, 110), (160, 200)]
    NBINS, MIN_N = 14, 8
    df = evolutionary_field()

    fig, axes = plt.subplots(4, 4, figsize=(13.5, 13.8), sharex=True,
                             sharey=True)
    pm = None
    for r, cell in enumerate(QUARTET):
        d = df[cell.mask(df)]
        aw, tw = cell.words(df)
        for c, (lo, hi) in enumerate(windows):
            ax = axes[r, c]
            w = d[(d.generation >= lo) & (d.generation < hi)]
            fld = field(w.d_img_sem_n, w.d_txt_sem_n, w.g_pair,
                        nbins=NBINS, extent=(0, 1.25, 0, 1.25), min_n=MIN_N)
            pm = ax.pcolormesh(fld.xe, fld.ye, fld.img, cmap="RdBu_r",
                               vmin=-1, vmax=1)
            ax.set_facecolor("0.88")
            fld.boundary(ax)
            frac = (w.g_pair < 0).mean()
            ax.text(0.03, 0.97,
                    f"{int((fld.counts >= MIN_N).sum())} bins mapped\n"
                    f"{frac:.0%} on target side",
                    transform=ax.transAxes, va="top", fontsize=8,
                    bbox=dict(fc="white", alpha=0.75, ec="none"))
            if r == 0:
                ax.set_title(f"generations {lo}–{hi - 1}", fontsize=11)
            if c == 0:
                ax.text(-0.34, 0.5, f"{cell.tag}\n'{aw}' vs '{tw}'",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=10, fontweight="bold", color=cell.color,
                        rotation=90)
                ax.set_ylabel("text distance from seed", fontsize=8.5)
            if r == 3:
                ax.set_xlabel("image distance from seed", fontsize=8.5)
            ax.set_xlim(0, 1.25)
            ax.set_ylim(0, 1.25)
            ax.grid(False)

    fig.subplots_adjust(top=0.87, bottom=0.06, left=0.12, right=0.87,
                        hspace=0.14, wspace=0.1)
    cax = fig.add_axes([0.895, 0.3, 0.014, 0.36])
    cb = fig.colorbar(pm, cax=cax)
    cb.set_label(AXIS.g + "\n(red = anchor side · blue = target side)",
                 fontsize=9)
    header(
        fig,
        "The boundary does not move — the search's picture of it does",
        "2-option prompt regime · binned median g per generation window, 3 seeds pooled per label pair · axes: "
        "semantic distance from seed, per-seed q99-normalized · black contour: boundary (g = 0) · grey: < 8 evaluations\n"
        "mapped territory concentrates over time; in the WALL rows a boundary line never appears",
        claim_y=0.975, method_y=0.945)
    return fig


# ===========================================================================
# 08 — descent profiles + boundary-touch survival
# ===========================================================================

@figure(8, "descent")
def descent() -> Figure:
    """Approach to the boundary over generations — easy pairs touch the
    noise band in ~25 generations; the boa wall stalls at 1 seed in 3."""
    df = evolutionary_field()

    def group_of(row):
        if row.target_class == "boa constrictor" and row.level_target == 1:
            return "boa wall — target word 'snake'"
        if row.target_class == "cello" and row.level_anchor == 1:
            return "cello wall — anchor word 'songbird'"
        if row.target_class in ("marimba", "green iguana", "ostrich"):
            return "easy targets (marimba / iguana / ostrich)"
        if row.target_class == "junco":
            return None
        return "other boa / cello pairs"

    keys = df[["target_class", "level_anchor", "level_target"]].drop_duplicates()
    keys["group"] = keys.apply(group_of, axis=1)
    df = df.merge(keys, on=["target_class", "level_anchor", "level_target"])
    df = df[df.group.notna()]

    groups = ["boa wall — target word 'snake'",
              "cello wall — anchor word 'songbird'",
              "other boa / cello pairs",
              "easy targets (marimba / iguana / ostrich)"]
    gcol = dict(zip(groups, ["#D64933", "#8B2E8B", "#CCB974", "#2274A5"]))

    df["abs_g"] = df.g_pair.abs()
    df["cell"] = (df.target_class + "|" + df.level_anchor.astype(str)
                  + df.level_target.astype(str))
    per_gen = (df.groupby(["group", "cell", "seed_dir", "generation"])
               .abs_g.min()
               .groupby(["group", "cell", "seed_dir"]).cummin().reset_index())

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.8),
                                  gridspec_kw={"width_ratios": [1.35, 1]})
    for g in groups:
        sub = per_gen[per_gen.group == g]
        for _, line in sub.groupby(["cell", "seed_dir"]):
            ax.plot(line.generation, line.abs_g, color=gcol[g], alpha=0.2,
                    lw=0.8)
        med = sub.groupby("generation").abs_g.median()
        ax.plot(med.index, med, color=gcol[g], lw=2.8, label=g)
    ax.axhspan(0, NOISE_G, color="0.85", zorder=0)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 199)
    ax.set_xlabel("generation")
    ax.set_ylabel("closest approach so far   min |g|")
    ax.set_title("descent profiles — one line per seed, bold = group median",
                 fontsize=11)
    handles, _ = ax.get_legend_handles_labels()
    handles.append(Patch(fc="0.85",
                         label="boundary band — |g| < 0.19 "
                               "(repeat-noise floor)"))
    ax.legend(handles=handles, loc="center left", fontsize=9,
              bbox_to_anchor=(0.18, 0.72))

    gens = np.arange(200)
    for g in groups:
        sub = per_gen[per_gen.group == g]
        first = (sub[sub.abs_g < NOISE_G]
                 .groupby(["cell", "seed_dir"]).generation.min())
        n_seeds = sub.groupby(["cell", "seed_dir"]).ngroups
        ax2.plot(gens, [(first <= t).sum() / n_seeds for t in gens],
                 color=gcol[g], lw=2.4, label=f"{g}  (n={n_seeds})")
    ax2.set_xlabel("generation")
    ax2.set_ylabel("fraction of seeds that have\ntouched the boundary band")
    ax2.set_ylim(0, 1.02)
    ax2.set_xlim(0, 199)
    ax2.set_title("boundary-touch survival", fontsize=11)
    ax2.legend(loc="lower right", fontsize=8.5)

    fig.subplots_adjust(top=0.78, bottom=0.12, wspace=0.24)
    header(
        fig,
        "Approach to the boundary — easy pairs touch the boundary band in "
        "~25 generations; the boa wall stalls at 1 seed in 3",
        "2-option prompt regime · |g| of the best individual so far, per seed · boundary band: |g| < 0.19 "
        "(= repeat-noise q90 of 0.38 lp) · junco-target control cells excluded\n"
        "a touch is a single best evaluation, not a crossed population — "
        "wall-cello touches late while its population median stays ≈ 0.9",
        claim_y=0.97, method_y=0.9)
    return fig


# ===========================================================================
# 09 — PDQ shrink-walk flow
# ===========================================================================

@figure(9, "walk_flow")
def walk_flow() -> Figure:
    """Walking back toward the anchor — wall walks never recross the
    boundary; at easy pairs one in five weaves through it."""
    t = transects(["seed_dir", "flip_id", "step", "pair_margin",
                   "target_class", "level_anchor", "level_target",
                   "anchor_word", "target_word"])

    COL = {"cross": "#D64933", "stay": "0.55"}

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.8), sharey=True,
                             sharex=True)
    for k, cell in enumerate(QUARTET):
        ax = axes.flat[k]
        d = t[cell.mask(t)]
        aw, tw = cell.words(t)
        n_walks, n_cross = 0, 0
        for _, w in d.groupby(["seed_dir", "flip_id"]):
            w = w.sort_values("step")
            crosses = (w.pair_margin.min() < -NOISE_LP) and \
                      (w.pair_margin.max() > NOISE_LP)
            n_walks += 1
            n_cross += crosses
            ax.plot(w.step, w.pair_margin,
                    color=COL["cross" if crosses else "stay"],
                    lw=1.0 if crosses else 0.6,
                    alpha=0.65 if crosses else 0.3,
                    zorder=2 if crosses else 1)
        med = d.groupby("step").pair_margin.median()
        ax.plot(med.index, med.values, color="black", lw=2.4, zorder=3)
        ax.axhline(0, color="black", lw=1.0, ls="--", zorder=3)
        ax.axhspan(-NOISE_LP, NOISE_LP, color="#F2E5B8", alpha=0.7, zorder=0)
        ax.set_title(f"{cell.tag} · '{aw}' vs '{tw}'", color=cell.color,
                     fontsize=12, fontweight="bold", pad=24)
        ax.text(0.5, 1.02, f"junco → {cell.target}  ({cell.levels})   ·   "
                           f"{n_walks} walks, {n_cross} recross",
                transform=ax.transAxes, ha="center", fontsize=9, color="0.45")
        if k >= 2:
            ax.set_xlabel("step within walk   (0 = at the flip  →  29 = "
                          "closest to the anchor)")
        if k % 2 == 0:
            ax.set_ylabel("pair margin [lp]   (junco side − target side)")

    fig.legend(handles=[
        Line2D([], [], color=COL["cross"], lw=1.6,
               label="walk recrosses the boundary (beyond noise, debounced)"),
        Line2D([], [], color=COL["stay"], lw=1.6,
               label="walk stays one-sided"),
        Line2D([], [], color="black", lw=2.4, label="per-cell median"),
        Patch(fc="#F2E5B8", label="repeat-noise band  ±0.38 lp"),
    ], loc="lower center", ncol=4, fontsize=9.5, frameon=False,
        bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(top=0.82, bottom=0.13, hspace=0.3, wspace=0.08)
    header(
        fig,
        "Walking back toward the anchor — wall walks never recross the "
        "boundary; at easy pairs one in five weaves through it",
        "6-option prompt regime · stage-2 shrink walks (gene-by-gene, accepted steps only) · "
        "crossing requires leaving the ±0.38 lp noise band on both sides (hysteresis)\n"
        "caveat: walks were steered by the Exp-100 6-class argmax criterion "
        "(pre-fix), so they minimize toward that criterion, not the pair margin",
        claim_y=0.97, method_y=0.92)
    return fig
