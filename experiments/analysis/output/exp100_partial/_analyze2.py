"""Tasks 2/3 (diff-based), 4, 5, 6."""
import pandas as pd
import numpy as np

OUT = "/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/exp100_partial"
arch = pd.read_parquet(OUT + "/_combined_archive_enriched.parquet")
diff = pd.read_parquet(OUT + "/_combined_diffgeom.parquet")
s1 = pd.read_parquet(OUT + "/_combined_stage1.parquet")
anchors = pd.read_parquet(OUT + "/_combined_anchors.parquet")
ag = pd.read_parquet(OUT + "/_combined_anchor_genos.parquet")
meta = pd.read_parquet(OUT + "/_combined_meta.parquet")
pd.set_option("display.width", 250)

arch = arch.merge(diff.drop(columns=["pareto_idx"]), on=["seed_dir", "flip_id"], how="left")
assert arch["hamming_min"].notna().all()

print("=" * 70)
print("TASK 2b: MINIMAL-FLIP GEOMETRY, diff-to-anchor semantics")
print("=" * 70)
print("\nhamming_min (genes changed vs anchor in minimal flip):")
print(arch["hamming_min"].describe().round(2).to_string())
print("quantiles:", arch["hamming_min"].quantile([.05, .1, .25, .5, .75, .95]).round(1).to_dict())
print("hamming_min == 0 cases:", (arch["hamming_min"] == 0).sum())
print("hamming_min <= 5:", (arch["hamming_min"] <= 5).sum(), "<=10:", (arch["hamming_min"] <= 10).sum(),
      "<=20:", (arch["hamming_min"] <= 20).sum())

arch["diff_modality_min"] = np.select(
    [(arch["img_diff_min"] > 0) & (arch["txt_diff_min"] > 0),
     (arch["img_diff_min"] > 0),
     (arch["txt_diff_min"] > 0)],
    ["mixed", "image-only", "text-only"], default="none")
print("\nmodality of the min DIFF (which genes changed vs anchor):")
mc = arch["diff_modality_min"].value_counts()
print(pd.concat([mc, (mc / len(arch)).rename("frac").round(3)], axis=1).to_string())

# split sparse vs dense flips (found_by sparse strategies end small)
arch["sparse_min"] = arch["hamming_min"] <= 30
print("\nsparse_min (hamming_min<=30) count:", arch["sparse_min"].sum(), "/", len(arch))
print("\nmodality among SPARSE minimal flips (hamming_min<=30):")
sub = arch[arch["sparse_min"]]
mc = sub["diff_modality_min"].value_counts()
print(pd.concat([mc, (mc / len(sub)).rename("frac").round(3)], axis=1).to_string())
print("\nmean img/txt diff genes among sparse minimal flips:")
print(sub[["hamming_min", "img_diff_min", "txt_diff_min"]].describe().round(2).to_string())
print("\ntxt share of diff among sparse minimal flips:")
print((sub["txt_diff_min"] / sub["hamming_min"].clip(lower=1)).describe().round(3).to_string())

print("\nmodality x found_by:")
print(pd.crosstab(arch["found_by"], arch["diff_modality_min"]).to_string())
print("\nhamming_min by found_by:")
print(arch.groupby("found_by")["hamming_min"].describe().round(1).to_string())

print("\nhamming_min by target class:")
print(arch.groupby("target_class")["hamming_min"].agg(["mean", "median", "count"]).round(1).to_string())
print("\nhamming_min by cell:")
print(arch.groupby(["level_anchor", "level_target"])["hamming_min"].agg(["mean", "median"]).round(1).to_string())
print("\nhamming_min by label_flipped:")
print(arch.groupby("label_flipped")["hamming_min"].agg(["mean", "median", "count"]).round(1).to_string())

print()
print("=" * 70)
print("TASK 3b: STAGE-2 SHRINK, diff semantics")
print("=" * 70)
arch["ham_red_abs"] = arch["hamming_flipped"] - arch["hamming_min"]
arch["ham_red_rel"] = arch["ham_red_abs"] / arch["hamming_flipped"].clip(lower=1)
arch["l1_red_rel"] = (arch["l1_diff_flipped"] - arch["l1_diff_min"]) / arch["l1_diff_flipped"].clip(lower=1)
print("\nhamming reduction abs:")
print(arch["ham_red_abs"].describe().round(1).to_string())
print("hamming reduction rel:")
print(arch["ham_red_rel"].describe().round(3).to_string())
print("L1-diff reduction rel:")
print(arch["l1_red_rel"].describe().round(3).to_string())
print("\nby budget capped (stage2_sut_calls>=30):")
print(arch.groupby("budget_capped")[["hamming_flipped", "hamming_min", "ham_red_rel", "l1_red_rel"]]
      .agg(["mean", "median", "count"]).round(2).to_string())
print("\nby found_by:")
print(arch.groupby("found_by")[["hamming_flipped", "hamming_min", "ham_red_rel", "stage2_sut_calls"]]
      .mean().round(1).to_string())

print()
print("=" * 70)
print("TASK 4: OUTPUT DISTANCES vs INPUT COST")
print("=" * 70)
for c in ["d_o_label_mismatch", "d_o_label_edit", "d_o_label_embedding", "d_o_wordnet_path", "d_i_primary", "pdq"]:
    print(f"\n{c}:")
    print(arch[c].describe().round(4).to_string())
print("\nd_o_* by label pair (label_anchor -> label_flipped):")
arch["pair_lab"] = arch["label_anchor"] + " -> " + arch["label_flipped"]
print(arch.groupby("pair_lab")[["d_o_label_edit", "d_o_label_embedding", "d_o_wordnet_path"]]
      .agg(["mean", "std", "count"]).round(3).to_string())

print("\nINPUT cost by label pair:")
print(arch.groupby("pair_lab")[["hamming_min", "sparsity_min", "rank_sum_min", "l1_diff_min",
                                "image_pixel_L2_min", "text_cosine_sum_min"]].agg(["mean", "median"]).round(1).to_string())

print("\ncorrelations (pearson & spearman), input cost vs output distance, all flips:")
inp = ["hamming_min", "l1_diff_min", "sparsity_min", "rank_sum_min", "image_pixel_L2_min", "text_cosine_sum_min"]
outd = ["d_o_label_edit", "d_o_label_embedding", "d_o_wordnet_path"]
sub2 = arch[inp + outd].dropna()
print("n =", len(sub2))
print("\npearson:")
print(sub2.corr(method="pearson").loc[inp, outd].round(3).to_string())
print("\nspearman:")
print(sub2.corr(method="spearman").loc[inp, outd].round(3).to_string())

# restricted to junco-labeled anchors only (clean comparison)
ja = arch[arch["label_anchor"] == "junco"]
print("\nsame, junco-anchored flips only (all land on boa) -> no output variance; "
      "n =", len(ja), "; d_o_wordnet_path unique:", ja["d_o_wordnet_path"].nunique())
print("\nd_o unique value counts overall:")
for c in outd:
    print(c, arch[c].value_counts().head(6).to_dict())

print()
print("=" * 70)
print("TASK 5: STAGE-1 DISCOVERY")
print("=" * 70)
print("\nfound_by (archive) counts:")
print(arch["found_by"].value_counts().to_string())
print("\nstage1_flips operation counts:")
print(s1["operation"].value_counts().to_string())
# flip yield per strategy needs attempts: candidates files
print("\ndiscovery_sut_call distribution:")
print(s1["discovery_sut_call"].describe().round(1).to_string())
print("\ndiscovery call by operation:")
print(s1.groupby("operation")["discovery_sut_call"].agg(["mean", "median", "count"]).round(1).to_string())
print("\nL_target by operation:")
print(pd.crosstab(s1["operation"], s1["L_target"]).to_string())
print("\nis_first_for_target:", s1["is_first_for_target"].sum(), "/", len(s1))
print("\nn_distinct_targets per anchor (manifest):")
print(anchors["n_distinct_targets"].value_counts().to_string())
print("\nstage1 flips per anchor by anchor label:")
arch_anch = arch.groupby(["seed_dir", "pareto_idx"]).agg(
    label_anchor=("label_anchor", "first"), n=("flip_id", "size"),
    n_targets=("label_flipped", "nunique"))
print(arch_anch.groupby("label_anchor")[["n", "n_targets"]].mean().round(2).to_string())
print("\nanchors flipping to >1 distinct label (archive-level):",
      (arch_anch["n_targets"] > 1).sum(), "/", len(arch_anch))

print()
print("=" * 70)
print("TASK 6: ANCHOR QUALITY / DUPLICATION")
print("=" * 70)
# identical p_a within seed
pa = anchors.groupby("seed_dir")["p_a"].nunique()
print("seeds where all 3 anchors share IDENTICAL p_a:", (pa == 1).sum(), "/", len(pa))
print("seeds with 2 unique p_a:", (pa == 2).sum(), "; 3 unique:", (pa == 3).sum())
# genotype distinctness
gn = ag.groupby("seed_dir")["geno_hash"].nunique()
print("\nseeds where anchor GENOTYPES are all identical:", (gn == 1).sum(), "/", len(gn))
print("seeds with 2 unique genotypes:", (gn == 2).sum(), "; 3 unique:", (gn == 3).sum())
# cross: identical p_a but distinct genotypes?
both = pd.concat([pa.rename("n_pa"), gn.rename("n_geno")], axis=1)
print("\ncrosstab unique p_a x unique genotypes (seed counts):")
print(pd.crosstab(both["n_pa"], both["n_geno"]).to_string())
print("\nanchor p_a distribution (all anchors):")
print(anchors["p_a"].describe().round(3).to_string())
print("anchors with p_a in [0.4,0.6] (near-boundary):",
      ((anchors["p_a"] >= .4) & (anchors["p_a"] <= .6)).sum(), "/", len(anchors))
print("\nanchor_active genes (evolutionary Pareto anchors):")
print(ag["anchor_active"].describe().round(1).to_string())

arch.to_parquet(OUT + "/_combined_archive_final.parquet")
