"""v1 Fig 3: boundary gene anatomy — modality/group shares, positional histograms.
v1 Fig 4: boundary fingerprint per cell (text-gene heatmap + concentration)."""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, "/Users/kaiser/Projects/Masterarbeit")
from analysis.core.style import apply_style, save_fig

apply_style(); warnings.filterwarnings("ignore")
ROOT = "/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography/exp100"
OUT = Path("/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography/explore")

S = pd.read_parquet(f"{ROOT}/straddle_pairs.parquet")
S["cell"] = (S.target_class.str.split().str[-1] + " (" + S.level_anchor.astype(str)
             + "," + S.level_target.astype(str) + ")")
S["txt_offset"] = S.gene_idx - S.image_dim          # 0..18 for text genes
S["img_frac"] = S.gene_idx / S.image_dim            # 0..1 for image genes

# per-seed image_dim check
dim_per_cell = S.groupby("cell").image_dim.nunique()
print("cells with mixed image_dim across seeds:", (dim_per_cell > 1).sum(), "/", len(dim_per_cell))

TXT_GENES = {  # offset -> group  (mlm -19..-17, frag -16..-12, charnoise -11..-4, saliency -3..-1)
    **{i: "mlm" for i in range(0, 3)}, **{i: "frag" for i in range(3, 8)},
    **{i: "charnoise" for i in range(8, 16)}, **{i: "saliency" for i in range(16, 19)},
}
GROUP_SIZE = {"mlm": 3, "frag": 5, "charnoise": 8, "saliency": 3}
GROUP_COL = {"mlm": "#d62728", "frag": "#ff7f0e", "charnoise": "#2ca02c", "saliency": "#9467bd"}

# ---------------- Fig 3 ----------------
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# (a) modality share by boundary_kind, normalized per gene
ax = axes[0, 0]
rows = []
for bk, g in S.groupby("boundary_kind"):
    n_img_genes = g.groupby("seed_dir").image_dim.first().mean()  # avg image gene count
    img_per_gene = (g.gene_modality == "img").sum() / n_img_genes
    txt_per_gene = (g.gene_modality == "txt").sum() / 19.0
    rows.append((bk, (g.gene_modality == "txt").mean(), img_per_gene, txt_per_gene))
W = pd.DataFrame(rows, columns=["bk", "txt_share", "img_per_gene", "txt_per_gene"]).set_index("bk")
x = np.arange(2); w = 0.35
ax.bar(x - w/2, W.img_per_gene, w, label="img", color="#1f77b4")
ax.bar(x + w/2, W.txt_per_gene, w, label="txt", color="#d62728")
ax.set_xticks(x); ax.set_xticklabels(W.index); ax.legend()
ax.set_ylabel("stakes per gene"); ax.set_title("(a) stakes per gene by modality\n(txt raw share: %s)"
    % ", ".join(f"{i}={v:.1%}" for i, v in W.txt_share.items()))

# (b) txt_group per-gene rate by boundary_kind
ax = axes[0, 1]
t = S[S.gene_modality == "txt"]
rates = (t.groupby(["boundary_kind", "txt_group"]).size().unstack()
         .div(pd.Series(GROUP_SIZE))).fillna(0)
rates[["mlm", "frag", "charnoise", "saliency"]].plot.bar(
    ax=ax, color=[GROUP_COL[c] for c in ["mlm", "frag", "charnoise", "saliency"]], rot=0)
ax.set_ylabel("stakes per gene"); ax.set_title("(b) text-group stakes per gene")

# (c) positional histogram, text genes (offset from start of text block)
ax = axes[0, 2]
for bk, m in [("pair_margin", "o"), ("argmax", "^")]:
    cnt = t[t.boundary_kind == bk].txt_offset.value_counts().reindex(range(19), fill_value=0)
    ax.plot(cnt.index, cnt.values, marker=m, label=bk)
for off, grp in [(0, "mlm"), (3, "frag"), (8, "charnoise"), (16, "saliency")]:
    ax.axvline(off - 0.5, color="gray", lw=0.6, ls="--")
    ax.text(off + 0.1, ax.get_ylim()[1] * 0.95, grp, fontsize=8, color=GROUP_COL[grp])
ax.set_xlabel("text gene offset (0..18)"); ax.set_ylabel("stake count")
ax.set_title("(c) text-gene positional histogram"); ax.legend()

# (d) image-gene fractional position histogram
ax = axes[1, 0]
im = S[S.gene_modality == "img"]
for bk in ["pair_margin", "argmax"]:
    ax.hist(im[im.boundary_kind == bk].img_frac, bins=40, histtype="step", label=bk, density=True)
ax.set_xlabel("gene_idx / image_dim"); ax.set_ylabel("density")
ax.set_title("(d) image-gene position (fractional)"); ax.legend()

# (e) per-cell txt share (per-gene normalized ratio txt:img)
ax = axes[1, 1]
cell_mod = S.groupby(["cell", "gene_modality"]).size().unstack(fill_value=0)
dims = S.groupby("cell").image_dim.mean()
ratio = (cell_mod["txt"] / 19.0) / (cell_mod["img"] / dims)
ratio = ratio.sort_values()
ax.barh(range(len(ratio)), ratio.values, color="#555")
ax.set_yticks(range(len(ratio))); ax.set_yticklabels(ratio.index, fontsize=6)
ax.axvline(1, color="red", lw=1)
ax.set_xlabel("txt per-gene rate / img per-gene rate"); ax.set_title("(e) per-cell modality bias")

# (f) recurrence concentration: per cell+seed, share of stakes on top-5 genes
ax = axes[1, 2]
conc = []
for (cell, sd), g in S.groupby(["cell", "seed_dir"]):
    if len(g) < 10:
        continue
    vc = g.gene_idx.value_counts()
    conc.append((cell, sd, len(g), vc.head(5).sum() / len(g), g.gene_idx.nunique() / len(g)))
C = pd.DataFrame(conc, columns=["cell", "seed", "n", "top5_share", "uniq_frac"])
ax.scatter(C.n, C.top5_share, s=14, alpha=0.6)
ax.set_xlabel("stakes in (cell,seed)"); ax.set_ylabel("top-5 gene share")
ax.set_title("(f) gene recurrence per (cell,seed)\nmedian top-5 share = %.2f" % C.top5_share.median())
# expected under uniform: 5/(n_genes) * something — annotate uniform baseline for n stakes over ~241 genes
ax.axhline(5 / 241, color="red", lw=1, ls="--", label="uniform baseline (5/241)")
ax.legend()

fig.suptitle("Boundary gene anatomy (straddle stakes, Exp-100)", y=1.0)
save_fig(fig, OUT / "v1_fig3_gene_anatomy.png")

print("\nraw txt share overall:", (S.gene_modality == "txt").mean().round(3))
print("per-gene rates txt groups:\n", rates.round(1))
print("recurrence: median top5 share=%.2f  (n cells*seeds=%d)" % (C.top5_share.median(), len(C)))
print("uniq gene frac median:", C.uniq_frac.median().round(2))

# ---------------- Fig 4: fingerprint heatmap (text genes x cell) ----------------
t = S[S.gene_modality == "txt"].copy()
H = t.groupby(["cell", "txt_offset"]).size().unstack(fill_value=0).reindex(columns=range(19), fill_value=0)
# normalize by total stakes in cell (img+txt) -> share of cell's boundary attributable to each text gene
tot = S.groupby("cell").size()
Hn = H.div(tot, axis=0)
# order cells by smoo hardness? order by target class then level
order = sorted(Hn.index)
Hn = Hn.loc[order]

fig, axes = plt.subplots(1, 2, figsize=(15, 9), gridspec_kw={"width_ratios": [3, 1]})
ax = axes[0]
imh = ax.imshow(Hn.values, aspect="auto", cmap="viridis")
ax.set_xticks(range(19))
ax.set_xticklabels([f"{i}\n{TXT_GENES[i][:4]}" for i in range(19)], fontsize=7)
ax.set_yticks(range(len(Hn))); ax.set_yticklabels(Hn.index, fontsize=7)
ax.set_xlabel("text gene offset (group)"); ax.set_title("share of cell's stakes on each text gene")
fig.colorbar(imh, ax=ax, shrink=0.8)
# right: txt total share per cell
ax = axes[1]
txt_share = t.groupby("cell").size().reindex(order).fillna(0) / tot.reindex(order)
ax.barh(range(len(order)), txt_share.values, color="#d62728")
ax.set_yticks([]); ax.invert_yaxis()
ax.set_xlabel("total txt stake share"); ax.set_title("txt share of all stakes")
ax.set_ylim(len(order) - 0.5, -0.5)
fig.suptitle("Boundary fingerprint per cell — text genes", y=0.995)
save_fig(fig, OUT / "v1_fig4_fingerprint_txt_genes.png")
print("\ntop text-gene cells:")
print(Hn.max(axis=1).sort_values(ascending=False).head(8).round(3).to_string())
