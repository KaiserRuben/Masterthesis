"""Diagnostic figures for Exp-100 partial PDQ analysis."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/exp100_partial"
arch = pd.read_parquet(OUT + "/_combined_archive_final.parquet")
anchors = pd.read_parquet(OUT + "/_combined_anchors.parquet")
pheno = pd.read_parquet(OUT + "/_anchor_pheno_uniq.parquet")
ag = pd.read_parquet(OUT + "/_combined_anchor_genos.parquet")

# ---------------------------------------------------------------- fig 1
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axes[0]
ct = pd.crosstab(arch["target_class"], arch["label_flipped"])
ctn = ct.div(ct.sum(axis=1), axis=0)
im = ax.imshow(ctn.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(ctn.columns)), ctn.columns, rotation=20, ha="right")
ax.set_yticks(range(len(ctn.index)), ctn.index)
for i in range(ctn.shape[0]):
    for j in range(ctn.shape[1]):
        ax.text(j, i, f"{ctn.values[i, j]:.2f}\n(n={ct.values[i, j]})",
                ha="center", va="center", fontsize=8,
                color="white" if ctn.values[i, j] < 0.6 else "black")
ax.set_ylabel("evolutionary target class (seed)")
ax.set_xlabel("label_flipped")
ax.set_title("Where flips land vs evo target\n(row-normalized)")

ax = axes[1]
ct2 = pd.crosstab(arch["anchor_6cat"], arch["label_flipped"])
im = ax.imshow(ct2.values, cmap="magma", aspect="auto")
ax.set_xticks(range(len(ct2.columns)), ct2.columns, rotation=20, ha="right")
ax.set_yticks(range(len(ct2.index)), ct2.index)
for i in range(ct2.shape[0]):
    for j in range(ct2.shape[1]):
        ax.text(j, i, f"{ct2.values[i, j]}", ha="center", va="center",
                fontsize=11, color="white")
ax.set_ylabel("anchor 6-cat argmax label")
ax.set_xlabel("label_flipped")
ax.set_title("Flip transitions in 6-cat space\n(diagonal = label-space artifact, not a flip)")

ax = axes[2]
vals = {"boa constrictor": 70690, "junco": 12365, "green iguana": 0,
        "ostrich": 0, "cello": 0, "marimba": 0}
ax.bar(range(len(vals)), list(vals.values()), color=["#b40426", "#3b4cc0"] + ["#999999"] * 4)
ax.set_xticks(range(len(vals)), list(vals.keys()), rotation=25, ha="right")
ax.set_ylabel("count")
ax.set_title("top-1 label, ALL 83,055 PDQ SUT calls\n(only 2 basins ever observed)")
fig.suptitle("Exp-100 PoC (LLaVA-1.6-mistral-7B INT8): boa constrictor attractor", y=1.02)
fig.tight_layout()
fig.savefig(OUT + "/pdq_flip_transitions.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- fig 2
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
ax = axes[0]
ops = arch["found_by"].value_counts().index.tolist()
data = [arch.loc[arch["found_by"] == o, "hamming_min"] for o in ops]
ax.hist(data, bins=np.arange(0, 295, 10), stacked=True, label=ops)
ax.set_yscale("log")
ax.set_xlabel("hamming_min (genes changed vs anchor in minimal flip)")
ax.set_ylabel("flips (log)")
ax.legend(fontsize=8, title="found_by")
ax.set_title("Minimal-flip size, all 2730 refined flips\n(bimodal: text-modality flips shrink to ~0-5 genes)")

ax = axes[1]
sub = arch[arch["hamming_min"] <= 30].copy()
sub["diff_mod"] = np.select(
    [(sub["img_diff_min"] > 0) & (sub["txt_diff_min"] > 0),
     (sub["img_diff_min"] > 0), (sub["txt_diff_min"] > 0)],
    ["mixed", "image-only", "text-only"], default="none (artifact)")
for mod, c in [("none (artifact)", "#888888"), ("text-only", "#3b4cc0"),
               ("image-only", "#b40426"), ("mixed", "#7b3294")]:
    s = sub[sub["diff_mod"] == mod]
    ax.hist(s["hamming_min"], bins=np.arange(-0.5, 31.5, 1), histtype="step",
            lw=2, label=f"{mod} (n={len(s)})", color=c)
ax.set_xlabel("hamming_min")
ax.set_ylabel("flips")
ax.legend(fontsize=8)
ax.set_title("Sparse minimal flips (hamming_min<=30): modality of changed genes\n19 text genes vs 222 image genes")
fig.tight_layout()
fig.savefig(OUT + "/pdq_hamming_min_modality.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- fig 3
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
ax = axes[0]
capped = arch["stage2_sut_calls"] >= 30
ax.scatter(arch.loc[capped, "hamming_flipped"], arch.loc[capped, "hamming_min"],
           s=6, alpha=0.25, label=f"budget-capped 30 calls (n={capped.sum()})", color="#b40426")
ax.scatter(arch.loc[~capped, "hamming_flipped"], arch.loc[~capped, "hamming_min"],
           s=6, alpha=0.4, label=f"finished early (n={(~capped).sum()})", color="#3b4cc0")
lim = [0, 300]
ax.plot(lim, lim, "k--", lw=0.8, label="no reduction")
ax.plot(lim, [max(0, v - 30) for v in lim], "k:", lw=0.8, label="max possible w/ 30 accepts")
ax.set_xlabel("hamming_flipped (at stage-1 discovery)")
ax.set_ylabel("hamming_min (after stage-2 shrink)")
ax.legend(fontsize=8)
ax.set_title("Stage-2 shrink: 30-call budget binds for dense flips\n(median 233 -> 204; would need ~200 more calls)")

ax = axes[1]
arch["ham_red_rel"] = (arch["hamming_flipped"] - arch["hamming_min"]) / arch["hamming_flipped"].clip(lower=1)
bins = [0, 30, 60, 120, 180, 240, 300]
arch["hf_bin"] = pd.cut(arch["hamming_flipped"], bins)
g = arch.groupby("hf_bin", observed=True)["ham_red_rel"].agg(["mean", "count"])
ax.bar(range(len(g)), g["mean"], color="#7b3294")
for i, (m, n) in enumerate(zip(g["mean"], g["count"])):
    ax.text(i, m + 0.01, f"n={n}", ha="center", fontsize=8)
ax.set_xticks(range(len(g)), [str(i) for i in g.index], rotation=20, ha="right")
ax.set_ylabel("mean relative hamming reduction")
ax.set_xlabel("hamming_flipped bin")
ax.set_title("Relative shrink vs starting density\n(accept rate 95.2%; budget is the limiter, not rejections)")
fig.tight_layout()
fig.savefig(OUT + "/pdq_shrink_efficacy.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- fig 4
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
ax = axes[0]
ax.hist(anchors["p_a"], bins=30, color="#3b4cc0")
ax.axvspan(0.4, 0.6, color="orange", alpha=0.2, label="near-boundary [0.4,0.6]")
ax.set_xlabel("anchor p_a (pair softmax)")
ax.set_ylabel("anchors")
ax.legend(fontsize=8)
ax.set_title("Anchor p_a, 357 anchors\n(88 in [0.4,0.6])")

ax = axes[1]
vc = pheno["n_unique_pheno"].value_counts().sort_index()
ax.bar(vc.index.astype(str), vc.values, color=["#b40426", "#e8855d", "#3b4cc0"])
for i, v in enumerate(vc.values):
    ax.text(i, v + 1, str(v), ha="center")
ax.set_xlabel("unique rendered phenotypes among 3 anchors")
ax.set_ylabel("seeds")
ax.set_title("Anchor phenotype duplication per seed\n(66/119 seeds: all 3 anchors identical)")

ax = axes[2]
a6 = arch.groupby(["seed_dir", "pareto_idx"])["anchor_6cat"].first().value_counts()
ax.bar(a6.index, a6.values, color=["#b40426", "#3b4cc0"])
for i, v in enumerate(a6.values):
    ax.text(i, v + 2, str(v), ha="center")
ax.set_ylabel("anchors")
ax.set_title("Anchor label in 6-cat space\n(302/357 'near-boundary' anchors already boa)")
fig.tight_layout()
fig.savefig(OUT + "/pdq_anchor_quality.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- fig 5
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
ax = axes[0]
att = {"dense_uniform": 1785, "bituniform_density": 1071, "modality_text": 771,
       "sparsity_sweep": 56, "max_rank": 48, "modality_image": 16}
fl = {"dense_uniform": 1391, "bituniform_density": 836, "modality_text": 470,
      "sparsity_sweep": 10, "max_rank": 8, "modality_image": 15}
ops = list(att)
x = np.arange(len(ops))
ax.bar(x - 0.2, [att[o] for o in ops], width=0.4, label="stage-1 attempts", color="#999999")
ax.bar(x + 0.2, [fl[o] for o in ops], width=0.4, label="flips", color="#b40426")
for i, o in enumerate(ops):
    ax.text(i + 0.2, fl[o] + 20, f"{fl[o]/att[o]:.0%}", ha="center", fontsize=8)
ax.set_xticks(x, ops, rotation=25, ha="right")
ax.set_ylabel("count")
ax.legend(fontsize=8)
ax.set_title("Stage-1 strategy attempts vs flips (yield %)")

ax = axes[1]
data, labels = [], []
for o in ops:
    s = arch.loc[arch["found_by"] == o, "hamming_min"]
    if len(s) > 0:
        data.append(s)
        labels.append(f"{o}\n(n={len(s)})")
ax.boxplot(data, tick_labels=labels, showfliers=False)
ax.set_ylabel("hamming_min")
ax.set_title("Minimal-flip size by discovering strategy\n(modality_text -> near-zero minimal flips)")
plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=7)
fig.tight_layout()
fig.savefig(OUT + "/pdq_stage1_discovery.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("saved 5 figures")
