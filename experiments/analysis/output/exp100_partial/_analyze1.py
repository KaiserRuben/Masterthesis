"""Tasks 1-3: flip landscape, minimal-flip geometry, stage-2 shrink efficacy."""
import pandas as pd
import numpy as np

OUT = "/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/exp100_partial"
arch = pd.read_parquet(OUT + "/_combined_archive.parquet")
s1 = pd.read_parquet(OUT + "/_combined_stage1.parquet")
s2 = pd.read_parquet(OUT + "/_combined_stage2.parquet")
anchors = pd.read_parquet(OUT + "/_combined_anchors.parquet")
meta = pd.read_parquet(OUT + "/_combined_meta.parquet")

pd.set_option("display.width", 250)

print("=" * 70)
print("TASK 1: FLIP LANDSCAPE")
print("=" * 70)

# flips per seed
fps = arch.groupby("seed_dir").size()
print(f"\nflips per seed: mean={fps.mean():.2f} median={fps.median()} min={fps.min()} max={fps.max()}")
print(fps.describe().to_string())
# stage1 candidates: 3 anchors x 10 = 30 per seed; flips found
print("\nflip rate: stage1 candidates per seed = 30 (3 anchors x 10)")
print(f"stage1 flip yield overall: {len(s1)} / {30*len(meta)} = {len(s1)/(30*len(meta)):.1%}")

# anchors-level
print("\nper-anchor stage1 flips (from manifests):")
print(anchors["n_stage1_flips"].describe().to_string())
print("anchors with 0 flips:", (anchors["n_stage1_flips"] == 0).sum(), "/", len(anchors))

# label_flipped distribution overall
print("\nlabel_flipped overall (refined flips):")
lf = arch["label_flipped"].value_counts()
print(pd.concat([lf, (lf / len(arch)).rename("frac").round(3)], axis=1).to_string())

print("\nlabel_min overall:")
lm = arch["label_min"].value_counts()
print(pd.concat([lm, (lm / len(arch)).rename("frac").round(3)], axis=1).to_string())

# does label_min ever differ from label_flipped?
print("\nlabel_min != label_flipped:", (arch["label_min"] != arch["label_flipped"]).sum(), "/", len(arch))

# fraction landing on evo target
print(f"\nflips landing on evo target class: {arch['flip_on_evo_target'].sum()} / {len(arch)} = {arch['flip_on_evo_target'].mean():.1%}")

# cross-tab evo-target x label_flipped
ct = pd.crosstab(arch["target_class"], arch["label_flipped"])
print("\ncrosstab: evo target_class (rows) x label_flipped (cols), counts:")
print(ct.to_string())
ctn = pd.crosstab(arch["target_class"], arch["label_flipped"], normalize="index").round(3)
print("\nrow-normalized:")
print(ctn.to_string())

# per-target on-target fraction
ot = arch.groupby("target_class")["flip_on_evo_target"].agg(["mean", "sum", "count"])
print("\non-evo-target fraction per target class:")
print(ot.round(3).to_string())

# per abstraction cell
cell = arch.groupby(["level_anchor", "level_target"]).agg(
    n_flips=("flip_on_evo_target", "size"),
    on_target_frac=("flip_on_evo_target", "mean"),
)
print("\nper abstraction cell (level_anchor, level_target):")
print(cell.round(3).to_string())

# attractor: per cell, label_flipped distribution
ct2 = pd.crosstab([arch["level_anchor"], arch["level_target"]], arch["label_flipped"], normalize="index").round(3)
print("\nlabel_flipped distribution per abstraction cell (row-norm):")
print(ct2.to_string())

# seeds-with-zero-flips
zero_seeds = set(meta["seed_dir"]) - set(arch["seed_dir"])
print("\nseeds with ZERO refined flips:", len(zero_seeds), sorted(zero_seeds)[:10])

print()
print("=" * 70)
print("TASK 2: MINIMAL-FLIP GEOMETRY")
print("=" * 70)
print("\nsparsity_min distribution:")
print(arch["sparsity_min"].describe().round(2).to_string())
print("quantiles:", arch["sparsity_min"].quantile([.05, .25, .5, .75, .95]).round(1).to_dict())

print("\nimg vs txt active genes in genotype_min:")
print(arch[["img_active_min", "txt_active_min"]].describe().round(2).to_string())

arch["modality_min"] = np.select(
    [(arch["img_active_min"] > 0) & (arch["txt_active_min"] > 0),
     (arch["img_active_min"] > 0),
     (arch["txt_active_min"] > 0)],
    ["mixed", "image-only", "text-only"], default="empty")
print("\nmodality of minimal flip genotype:")
mc = arch["modality_min"].value_counts()
print(pd.concat([mc, (mc / len(arch)).rename("frac").round(3)], axis=1).to_string())

print("\ntext gene fraction of active genes (txt/(img+txt)) in min genotype:")
arch["txt_frac_min"] = arch["txt_active_min"] / arch["sparsity_min"].clip(lower=1)
print(arch["txt_frac_min"].describe().round(3).to_string())
# baseline: 19/241 = 0.079
print("baseline if random: 19/241 =", round(19 / 241, 3))

print("\nmodality split per label_flipped:")
print(pd.crosstab(arch["label_flipped"], arch["modality_min"], normalize="index").round(3).to_string())
print("\nmean active img/txt genes per label_flipped:")
print(arch.groupby("label_flipped")[["sparsity_min", "img_active_min", "txt_active_min", "txt_frac_min"]].mean().round(2).to_string())

print("\nmodality per target class (evo):")
print(pd.crosstab(arch["target_class"], arch["modality_min"], normalize="index").round(3).to_string())
print("\nsparsity_min per target class:")
print(arch.groupby("target_class")["sparsity_min"].describe().round(1).to_string())
print("\nsparsity_min per abstraction cell:")
print(arch.groupby(["level_anchor", "level_target"])["sparsity_min"].agg(["mean", "median", "count"]).round(1).to_string())

print("\nrank_sum_min / image_pixel_L2_min / text_cosine_sum_min:")
print(arch[["rank_sum_min", "image_pixel_L2_min", "text_cosine_sum_min"]].describe().round(3).to_string())

print()
print("=" * 70)
print("TASK 3: STAGE-2 SHRINK EFFICACY")
print("=" * 70)
arch["sparsity_red_abs"] = arch["sparsity_flipped"] - arch["sparsity_min"]
arch["sparsity_red_rel"] = arch["sparsity_red_abs"] / arch["sparsity_flipped"].clip(lower=1)
arch["rank_red_abs"] = arch["rank_sum_flipped"] - arch["rank_sum_min"]
arch["rank_red_rel"] = arch["rank_red_abs"] / arch["rank_sum_flipped"].clip(lower=1)
print("\nsparsity_flipped (start):")
print(arch["sparsity_flipped"].describe().round(1).to_string())
print("\nsparsity reduction absolute:")
print(arch["sparsity_red_abs"].describe().round(1).to_string())
print("\nsparsity reduction relative:")
print(arch["sparsity_red_rel"].describe().round(3).to_string())
print("\nrank_sum reduction relative:")
print(arch["rank_red_rel"].describe().round(3).to_string())

# trajectories
print("\nstage2 trajectory steps per flip:")
spf = s2.groupby(["seed_dir", "flip_id"]).size()
print(spf.describe().round(1).to_string())
print("\naccepted rate overall:", round(s2["accepted"].mean(), 3))
print("still_flipped rate overall:", round(s2["still_flipped"].mean(), 3))
print("rejected because flip lost (accepted=False & still_flipped=False):",
      ((~s2["accepted"]) & (~s2["still_flipped"])).sum())
print("rejected but still flipped (accepted=False & still_flipped=True):",
      ((~s2["accepted"]) & (s2["still_flipped"])).sum())
print("\npass_name breakdown:")
print(s2.groupby("pass_name").agg(steps=("accepted", "size"), accept_rate=("accepted", "mean"),
                                  flip_kept=("still_flipped", "mean")).round(3).to_string())

print("\nstage2_sut_calls per flip (archive):")
print(arch["stage2_sut_calls"].describe().round(1).to_string())
print("flips hitting exactly 30 calls:", (arch["stage2_sut_calls"] >= 30).sum(), "/", len(arch))
print("stage2_sut_calls value counts (top):")
print(arch["stage2_sut_calls"].value_counts().head(10).to_string())

# is budget binding: residual sparsity at end vs budget
print("\nresidual sparsity_min by stage2_sut_calls==30 vs <30:")
arch["budget_capped"] = arch["stage2_sut_calls"] >= 30
print(arch.groupby("budget_capped")[["sparsity_min", "sparsity_red_rel"]].agg(["mean", "median", "count"]).round(2).to_string())

arch.to_parquet(OUT + "/_combined_archive_enriched.parquet")
