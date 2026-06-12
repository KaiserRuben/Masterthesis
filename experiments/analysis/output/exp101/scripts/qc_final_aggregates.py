"""Final aggregates for the QC report from qc_per_run.csv."""
import pandas as pd

df = pd.read_csv("/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/exp101/qc_per_run.csv")

print("tgtbal_lp_maxerr: max =", df["tgtbal_lp_maxerr"].max())
print("tgtbal_min over all runs:", df["tgtbal_min"].min())
print("missing_files any:", df["missing_files"].notna().sum())
print("name_idx_match all:", bool(df["name_idx_match"].all()))
print("dup_keys total:", df["dup_keys"].sum())
print("nan totals:", df[["nan_fitness_TgtBal", "nan_p_class_a", "nan_p_class_b"]].sum().to_dict())

# cache counter pattern: cumulative per worker?
tot = (df["cache_hits_stats"] + df["cache_misses_stats"]).sort_values()
vc = tot.value_counts().sort_index()
print("hits+misses value -> count:", vc.to_dict())

# per-run trace cache hit rate
print("trace cache_hit rate: ", df["cache_hit_rate_trace"].describe().round(4).to_dict())

# runtime
print("runtime describe:", df["runtime_s"].describe().round(1).to_dict())

# image_dim per run with seed_idx
print("\nimage_dim by run:")
for _, r in df.sort_values("seed_idx").iterrows():
    pass
print(df.groupby("image_dim")["seed_idx"].apply(lambda s: sorted(s.tolist())).to_string())

# n bounds >= 1000 (wide-cone positions) summary
print("\nimg_bounds_n_ge_1000 describe:", df["img_bounds_n_ge_1000"].describe().round(1).to_dict())
print("img_bounds_max describe:", df["img_bounds_max"].describe().round(0).to_dict())

# n_pareto
print("n_pareto describe:", df["n_pareto"].describe().round(1).to_dict())
