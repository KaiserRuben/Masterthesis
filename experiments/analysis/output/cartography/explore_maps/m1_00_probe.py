"""m1_00: probe schema + per-cell support for the smoo (pair2) field."""
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

P = "/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography/exp100/points.parquet"

pf = pq.ParquetFile(P)
print("rows:", pf.metadata.num_rows)

cols = ["source", "prompt_regime", "anchor_class", "target_class",
        "level_anchor", "level_target", "seed_dir", "seed_idx_in_class",
        "d_img_sem", "d_txt_sem", "g_pair",
        "rank_sum_img_norm", "rank_sum_txt_norm", "n_active_img", "n_active_txt"]
df = pq.read_table(P, columns=cols).to_pandas()

print("\nsources:", df.source.value_counts().to_dict())
print("regimes:", df.prompt_regime.value_counts().to_dict())

smoo = df[df.source == "smoo"]
print("\nsmoo rows:", len(smoo))
print("anchor classes:", smoo.anchor_class.unique())

cell = smoo.groupby(["target_class", "level_anchor", "level_target"]).agg(
    n=("g_pair", "size"),
    n_seeds=("seed_dir", "nunique"),
    g_med=("g_pair", "median"),
    g_min=("g_pair", "min"),
    g_max=("g_pair", "max"),
    frac_gpos=("g_pair", lambda s: (s > 0).mean()),
)
print("\nper-cell (smoo, junco anchor):")
print(cell.to_string())

print("\ng_pair sign convention check: g = p_A - p_B; anchor=junco is A?")
print("d_img_sem describe (smoo):")
print(smoo.d_img_sem.describe().to_string())
print("\nd_txt_sem describe (smoo):")
print(smoo.d_txt_sem.describe().to_string())
print("\nd_txt_sem == 0 fraction:", (smoo.d_txt_sem == 0).mean())
print("d_txt_sem quantiles:", smoo.d_txt_sem.quantile([.5, .9, .99, .999]).to_dict())
print("d_img_sem quantiles:", smoo.d_img_sem.quantile([.5, .9, .99, .999]).to_dict())
print("\nNaN: d_img_sem", smoo.d_img_sem.isna().mean(), " d_txt_sem", smoo.d_txt_sem.isna().mean(),
      " g_pair", smoo.g_pair.isna().mean())

# per-seed q99 of axes (for pooled normalization)
g = smoo.groupby("seed_dir")[["d_img_sem", "d_txt_sem"]].quantile(0.99)
print("\nper-seed q99 spread: d_img_sem", g.d_img_sem.min(), g.d_img_sem.max(),
      "| d_txt_sem", g.d_txt_sem.min(), g.d_txt_sem.max())

# pdq_s1 availability for contrast
s1 = df[df.source == "pdq_s1"]
print("\npdq_s1 rows:", len(s1), " d_img_sem NaN:", s1.d_img_sem.isna().mean())
cell1 = s1.groupby(["target_class", "level_anchor", "level_target"]).size()
print(cell1.to_string())
