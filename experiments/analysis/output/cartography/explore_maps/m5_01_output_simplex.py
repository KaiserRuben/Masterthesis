"""m5_01 (direction 5a): OUTPUT-space map — the model's own logprob simplex as territory.

cat6 regime only (6-option concrete prompt). Two charts:
  1. Barycentric ternary on (p_junco, p_boa, p_rest=sum of other 4) after softmax.
     The junco/boa argmax boundary is EXACTLY the vertical bisector x=0.5
     (argmax is always junco or boa in this data, so p_j=p_b is the full story).
  2. Boundary-normal chart: x = lp_junco - lp_boa (exact signed margin, boundary
     = line x=0, eval-noise band +-0.38 lp), y = PC1 of the residual logprob
     component orthogonal to the boundary normal ("what else moves").

Honesty notes baked into titles: pdq_s2 is path-constrained (95% of rows) ->
stratified subsample and per-source panels; ternary compresses near vertices.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/kaiser/Projects/Masterarbeit")
from analysis.core.style import apply_style, save_fig  # noqa: E402

apply_style()
RNG = np.random.default_rng(42)
ROOT = Path("/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography")
OUT = ROOT / "explore_maps"

CAT6 = ["junco", "ostrich", "green iguana", "boa constrictor", "cello", "marimba"]
CLASS_COLORS = {"junco": "#937860", "ostrich": "#E6A817", "green iguana": "#55A868",
                "boa constrictor": "#C44E52", "cello": "#4C72B0", "marimba": "#CCB974"}
SRC_COLORS = {"pdq_anchor": "#000000", "pdq_s1": "#2274A5", "pdq_s2": "#D64933"}

# ---------------------------------------------------------------- load cat6
cols = ["source", "seed_dir", "logprobs", "pred_label", "hamming_to_anchor",
        "n_active_txt", "n_active_img", "target_class", "level_anchor", "level_target"]
df = pd.read_parquet(ROOT / "exp100/points.parquet", columns=cols,
                     filters=[("prompt_regime", "==", "cat6"),
                              ("anchor_class", "==", "junco")])
L = np.stack(df.logprobs.to_numpy()).astype(np.float64)        # (n, 6) raw logprobs
P = np.exp(L - L.max(axis=1, keepdims=True))
P /= P.sum(axis=1, keepdims=True)
p_j, p_b = P[:, 0], P[:, 3]
p_rest = 1.0 - p_j - p_b

# barycentric: junco vertex (0,0), boa (1,0), rest (0.5, sqrt3/2)
SQ3 = np.sqrt(3) / 2
bx = p_b + 0.5 * p_rest
by = SQ3 * p_rest

# boundary-normal coordinate (exact): margin in logprob space
u = L[:, 0] - L[:, 3]                                          # lp_junco - lp_boa
# residual: remove the (e_j - e_b)/sqrt2 component, then PC1 of what's left
n_vec = np.zeros(6); n_vec[0], n_vec[3] = 1 / np.sqrt(2), -1 / np.sqrt(2)
Lc = L - L.mean(axis=0, keepdims=True)
res = Lc - np.outer(Lc @ n_vec, n_vec)
# PCA via SVD on a subsample for speed, project all
fit_idx = RNG.choice(len(res), size=min(40000, len(res)), replace=False)
_, _, Vt = np.linalg.svd(res[fit_idx] - res[fit_idx].mean(axis=0), full_matrices=False)
pc1 = res @ Vt[0]
evr = None  # explained variance share of residual PC1
res_var = (res[fit_idx] - res[fit_idx].mean(axis=0)).var(axis=0).sum()
evr = ((res[fit_idx] - res[fit_idx].mean(axis=0)) @ Vt[0]).var() / res_var
print(f"residual PC1 explains {evr:.2%} of residual variance")
print("residual PC1 loadings (class order j/ostr/igua/boa/cello/mar):",
      np.round(Vt[0], 3))

df = df.assign(bx=bx, by=by, u=u, pc1=pc1, p_rest=p_rest)

# stratified subsample for scatter: all anchors + all s1 + 20k of s2
keep = np.zeros(len(df), dtype=bool)
keep[(df.source != "pdq_s2").to_numpy()] = True
s2_idx = np.flatnonzero((df.source == "pdq_s2").to_numpy())
keep[RNG.choice(s2_idx, size=20000, replace=False)] = True
sub = df[keep]
print(f"scatter subsample: {sub.source.value_counts().to_dict()}")

# straddle midpoints (argmax flips = surveyed boundary points)
sp = pd.read_parquet(ROOT / "exp100/straddle_pairs.parquet")
am = sp[sp.boundary_kind == "argmax"].copy()
Lb = np.stack(am.logprobs_before.to_numpy()).astype(np.float64)
La = np.stack(am.logprobs_after.to_numpy()).astype(np.float64)
Lm = 0.5 * (Lb + La)
Pm = np.exp(Lm - Lm.max(axis=1, keepdims=True)); Pm /= Pm.sum(axis=1, keepdims=True)
sm_bx = Pm[:, 3] + 0.5 * (1 - Pm[:, 0] - Pm[:, 3])
sm_by = SQ3 * (1 - Pm[:, 0] - Pm[:, 3])
print(f"argmax straddle midpoints: {len(am)}; |lp_j-lp_b| at midpoint "
      f"q50={np.median(np.abs(Lm[:,0]-Lm[:,3])):.3f}")


def ternary_frame(ax):
    """Triangle outline + labeled vertices + junco=boa bisector."""
    vx = [0, 1, 0.5, 0]; vy = [0, 0, SQ3, 0]
    ax.plot(vx, vy, color="0.3", lw=1)
    ax.plot([0.5, 0.5], [0, SQ3], color="k", lw=1.4, ls="--", zorder=5)
    ax.text(0.5, SQ3 / 2, " argmax boundary\n p(junco)=p(boa)", fontsize=7,
            rotation=90, va="center", ha="right", color="k")
    ax.text(-0.02, -0.03, "junco", ha="right", fontsize=9,
            color=CLASS_COLORS["junco"], fontweight="bold")
    ax.text(1.02, -0.03, "boa", ha="left", fontsize=9,
            color=CLASS_COLORS["boa constrictor"], fontweight="bold")
    ax.text(0.5, SQ3 + 0.03, "rest (ostrich+iguana+cello+marimba)", ha="center",
            fontsize=8, color="0.35")
    ax.set_xlim(-0.12, 1.12); ax.set_ylim(-0.10, SQ3 + 0.10)
    ax.set_aspect("equal"); ax.axis("off")


# ------------------------------------------------------------------ figure 1
fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))

# (a) by source
ax = axes[0, 0]
for src in ["pdq_s2", "pdq_s1", "pdq_anchor"]:
    s = sub[sub.source == src]
    ax.scatter(s.bx, s.by, s=(28 if src == "pdq_anchor" else 3),
               c=SRC_COLORS[src], alpha=(0.9 if src == "pdq_anchor" else 0.25),
               marker=("*" if src == "pdq_anchor" else "o"),
               lw=0, label=f"{src} (n={len(s)})")
ternary_frame(ax)
ax.legend(loc="upper left", fontsize=7, markerscale=2)
ax.set_title("(a) softmax ternary, by source\n[cat6, junco-anchored; s2 subsampled 20k, path-constrained]",
             fontsize=9)

# (b) by hamming_to_anchor (input radius)
ax = axes[0, 1]
h = sub[sub.hamming_to_anchor.notna()]
sc = ax.scatter(h.bx, h.by, s=3, c=h.hamming_to_anchor, cmap="viridis",
                alpha=0.35, lw=0, vmin=0, vmax=np.nanquantile(h.hamming_to_anchor, 0.98))
plt.colorbar(sc, ax=ax, fraction=0.04, label="hamming to anchor")
ternary_frame(ax)
ax.set_title("(b) colored by input radius (hamming)\n[does input distance organize on the output map?]",
             fontsize=9)

# (c) straddle midpoints
ax = axes[0, 2]
ax.scatter(sub.bx, sub.by, s=2, c="0.8", alpha=0.2, lw=0)
for mod, col in [("img", "#8172B3"), ("txt", "#DD8452")]:
    m = (am.gene_modality == mod).to_numpy()
    ax.scatter(sm_bx[m], sm_by[m], s=6, c=col, alpha=0.6, lw=0,
               label=f"{mod}-gene flip (n={m.sum()})")
ternary_frame(ax)
ax.legend(loc="upper left", fontsize=7, markerscale=2)
ax.set_title("(c) surveyed crossings: hamming-1 argmax-flip midpoints\n[lp-midpoint of before/after; gray = field]",
             fontsize=9)

# (d) boundary-normal chart by pred_label
ax = axes[1, 0]
for lab in ["junco", "boa constrictor"]:
    s = sub[sub.pred_label == lab]
    ax.scatter(s.u, s.pc1, s=3, c=CLASS_COLORS[lab], alpha=0.3, lw=0,
               label=f"argmax={lab}")
ax.axvline(0, color="k", lw=1.4, ls="--")
ax.axvspan(-0.38, 0.38, color="0.5", alpha=0.18, lw=0)
ax.set_xlabel("u = lp(junco) − lp(boa)   [boundary = 0; shaded: eval-noise q90 ±0.38]")
ax.set_ylabel(f"residual PC1 ({evr:.0%} of ⊥ variance)")
ax.legend(loc="upper left", fontsize=7, markerscale=2)
ax.set_title("(d) boundary-normal × residual-PC1 (logprob space)\n[x is exact margin — boundary is a line by construction]",
             fontsize=9)

# (e) same chart colored by text activity
ax = axes[1, 1]
sc = ax.scatter(sub.u, sub.pc1, s=3, c=sub.n_active_txt, cmap="magma",
                alpha=0.35, lw=0, vmin=0, vmax=12)
plt.colorbar(sc, ax=ax, fraction=0.04, label="n active text genes")
ax.axvline(0, color="k", lw=1.4, ls="--")
ax.set_xlabel("u = lp(junco) − lp(boa)")
ax.set_ylabel("residual PC1")
ax.set_title("(e) same chart, colored by text-gene activity\n[does input structure organize along/across the boundary?]",
             fontsize=9)

# (f) same chart colored by target_class of the cell
ax = axes[1, 2]
for tc, col in [("ostrich", CLASS_COLORS["ostrich"]),
                ("green iguana", CLASS_COLORS["green iguana"]),
                ("boa constrictor", CLASS_COLORS["boa constrictor"]),
                ("cello", CLASS_COLORS["cello"]),
                ("marimba", CLASS_COLORS["marimba"])]:
    s = sub[sub.target_class == tc]
    if len(s):
        ax.scatter(s.u, s.pc1, s=3, c=col, alpha=0.3, lw=0, label=f"cell target={tc} (n={len(s)})")
ax.axvline(0, color="k", lw=1.4, ls="--")
ax.set_xlabel("u = lp(junco) − lp(boa)")
ax.set_ylabel("residual PC1")
ax.legend(loc="upper left", fontsize=6, markerscale=2)
ax.set_title("(f) colored by experiment cell (target class)\n[do different cells probe different boundary segments?]",
             fontsize=9)

fig.suptitle("m5_01 — OUTPUT-space map: cat6 logprob simplex as territory "
             "(projection: barycentric junco/boa/rest + exact boundary-normal)",
             fontsize=11, y=1.0)
save_fig(fig, OUT / "m5_01_output_simplex.png")

# ------------------------------------------------------------- figure 2: 3D funnel
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

fig = plt.figure(figsize=(13, 6))
h = sub[sub.hamming_to_anchor.notna()]
hsub = h.iloc[RNG.choice(len(h), size=min(15000, len(h)), replace=False)]

for k, (col_by, title) in enumerate([
        ("pred_label", "colored by argmax"),
        ("source", "colored by source")]):
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    if col_by == "pred_label":
        for lab in ["junco", "boa constrictor"]:
            s = hsub[hsub.pred_label == lab]
            ax.scatter(s.bx, s.by, s.hamming_to_anchor, s=2, c=CLASS_COLORS[lab],
                       alpha=0.25, lw=0, label=lab)
    else:
        for src in ["pdq_s2", "pdq_s1"]:
            s = hsub[hsub.source == src]
            ax.scatter(s.bx, s.by, s.hamming_to_anchor, s=2, c=SRC_COLORS[src],
                       alpha=0.25, lw=0, label=src)
    # bisector plane x=0.5
    zz = np.linspace(0, float(np.nanquantile(hsub.hamming_to_anchor, 0.98)), 2)
    yy = np.linspace(0, SQ3, 2)
    Y, Z = np.meshgrid(yy, zz)
    ax.plot_surface(np.full_like(Y, 0.5), Y, Z, alpha=0.15, color="k")
    # anchors at z=0
    anc = sub[sub.source == "pdq_anchor"]
    ax.scatter(anc.bx, anc.by, np.zeros(len(anc)), s=30, c="k", marker="*",
               label="anchors (z=0)")
    ax.set_xlabel("bary x (junco↔boa)"); ax.set_ylabel("bary y (rest mass)")
    ax.set_zlabel("hamming to anchor")
    ax.legend(loc="upper left", fontsize=7, markerscale=3)
    ax.set_title(f"({chr(97+k)}) {title}", fontsize=9)
    ax.view_init(elev=18, azim=-70)

fig.suptitle("m5_01b — anchor→boundary funnel: barycentric (x,y) + input radius (z); "
             "plane = junco/boa bisector [cat6, 15k stratified subsample]", fontsize=10)
save_fig(fig, OUT / "m5_01b_output_funnel3d.png")

# quick numbers for the memo
print("\nfraction of s1 points with p_rest > 0.5:",
      (df[df.source == 'pdq_s1'].p_rest > 0.5).mean().round(3))
print("fraction within noise band |u|<0.38:", (np.abs(df.u) < 0.38).mean().round(4))
corr = df[df.hamming_to_anchor.notna()][["u", "hamming_to_anchor", "p_rest"]].corr()
print("corr(u, hamming), corr(p_rest, hamming):",
      corr.loc['u', 'hamming_to_anchor'].round(3), corr.loc['p_rest', 'hamming_to_anchor'].round(3))
