"""Prototype d — does 3D help? z = lp_boa - lp_junco relief with argmax-region drape
vs the same field as a flat 2D map with zero-contour. s2 rank-sum plane (cat6).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

sys.path.insert(0, "/Users/kaiser/Projects/Masterarbeit")
from analysis.core.style import apply_style, save_fig

ROOT = Path("/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography")
OUT = ROOT / "explore_maps"
CAT6 = ["junco", "ostrich", "green iguana", "boa constrictor", "cello", "marimba"]
CLASS_COLORS = {
    "junco": "#937860", "ostrich": "#E6A817", "green iguana": "#55A868",
    "boa constrictor": "#C44E52", "cello": "#4C72B0", "marimba": "#CCB974",
}
apply_style()

cols = ["source", "pred_label", "logprobs", "rank_sum_img_norm", "rank_sum_txt_norm"]
pts = pd.read_parquet(ROOT / "exp100/points.parquet", columns=cols,
                      filters=[("prompt_regime", "==", "cat6"), ("source", "==", "pdq_s2")])
lp = np.stack(pts["logprobs"].to_numpy())
pts = pts.reset_index(drop=True)
pts["z"] = lp[:, 3] - lp[:, 0]  # lp_boa - lp_junco

N = 24
xe = np.linspace(0, 1, N + 1); ye = np.linspace(0, 1, N + 1)
xm = 0.5 * (xe[:-1] + xe[1:]); ym = 0.5 * (ye[:-1] + ye[1:])
ix = np.clip(np.digitize(pts.rank_sum_img_norm, xe) - 1, 0, N - 1)
iy = np.clip(np.digitize(pts.rank_sum_txt_norm, ye) - 1, 0, N - 1)
df = pd.DataFrame({"ix": ix, "iy": iy, "z": pts.z, "lbl": pts.pred_label})

Z = np.full((N, N), np.nan)
C = np.zeros((N, N, 4))
for (bx, by), sub in df.groupby(["ix", "iy"]):
    if len(sub) < 4:
        continue
    Z[by, bx] = sub.z.mean()
    vc = sub.lbl.value_counts()
    share = vc.iloc[0] / len(sub)
    r, g, b = to_rgb(CLASS_COLORS[vc.index[0]])
    C[by, bx] = (r, g, b, np.clip(0.35 + 0.65 * (share - 1 / 3) / (2 / 3), 0.2, 1))

fig = plt.figure(figsize=(14, 6.2))
ax3 = fig.add_subplot(1, 2, 1, projection="3d")
X, Y = np.meshgrid(xm, ym)
Zp = np.where(np.isnan(Z), np.nan, Z)
fc = C.copy()
fc[np.isnan(Z)] = (1, 1, 1, 0)
ax3.plot_surface(X, Y, Zp, facecolors=fc[:-0 or None, :], rstride=1, cstride=1,
                 linewidth=0.1, edgecolor="#888888", antialiased=False, shade=False)
# sea level z=0 plane
xx, yy = np.meshgrid([0, 1], [0, 1])
ax3.plot_surface(xx, yy, np.zeros_like(xx), color="#4C72B0", alpha=0.15, shade=False)
ax3.set_xlabel("rank_sum_img (norm.)"); ax3.set_ylabel("rank_sum_txt (norm.)")
ax3.set_zlabel("lp(boa) - lp(junco)")
ax3.set_title("3D relief: boa-junco margin, argmax color drape\n(blue plane = sea level / decision border)", fontsize=10)
ax3.view_init(elev=28, azim=-60)

ax = fig.add_subplot(1, 2, 2)
vmax = np.nanquantile(np.abs(Z), 0.98)
im = ax.imshow(Z, origin="lower", extent=(0, 1, 0, 1), aspect="auto",
               cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
cs = ax.contour(xm, ym, Z, levels=[0], colors="black", linewidths=1.6)
ax.contour(xm, ym, Z, levels=[-0.38, 0.38], colors="black", linewidths=0.7, linestyles=":")
fig.colorbar(im, ax=ax, shrink=0.85, label="mean lp(boa) - lp(junco)")
ax.set_xlabel("rank_sum_img (norm.)"); ax.set_ylabel("rank_sum_txt (norm.)")
ax.set_title("same field flat: zero contour = argmax border (junco<->boa)\ndotted = +-0.38 lp repeat-noise band", fontsize=10)
ax.grid(False)

handles = [Line2D([], [], marker="s", ls="", color=CLASS_COLORS[c], label=c, markersize=9)
           for c in ["junco", "boa constrictor", "ostrich"]]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.27, -0.01))
fig.suptitle("cat6 prompt, pdq_s2 (path-constrained) — boa-junco margin relief, rank-sum plane", fontsize=12)
save_fig(fig, OUT / "m2_05_3d_relief.png")
print("z range:", np.nanmin(Z), np.nanmax(Z), " bins:", np.sum(~np.isnan(Z)))
