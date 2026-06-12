"""m4_00: probe the TIME dimension — generation coverage (smoo), step coverage (pdq transects)."""
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

BASE = "/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography/exp100"

# --- smoo time ---
cols = ["source", "generation", "anchor_class", "target_class", "level_anchor",
        "level_target", "seed_dir", "g_pair", "d_img_sem", "d_txt_sem"]
df = pq.read_table(f"{BASE}/points.parquet", columns=cols).to_pandas()
smoo = df[(df.source == "smoo") & (df.anchor_class == "junco")].copy()
print("smoo rows:", len(smoo))
print("generation describe:")
print(smoo.generation.describe().to_string())
print("\ngenerations per seed (min/max/nuniq):")
g = smoo.groupby("seed_dir").generation.agg(["min", "max", "nunique"])
print(g.describe().to_string())

smoo["cell"] = (smoo.target_class.astype(str) + " L" +
                smoo.level_anchor.astype(str) + "-" + smoo.level_target.astype(str))
cell = smoo.groupby("cell").agg(
    n=("g_pair", "size"), n_seeds=("seed_dir", "nunique"),
    gen_max=("generation", "max"),
    g_med=("g_pair", "median"),
    frac_cross=("g_pair", lambda s: (s < 0).mean()),
)
print("\nper-cell:")
print(cell.to_string())

# earliest generation of crossing per seed (g_pair < 0 = target side?)
smoo["crossed"] = smoo.g_pair < 0
first_cross = smoo[smoo.crossed].groupby(["cell", "seed_dir"]).generation.min()
print("\nfirst-crossing generation per cell (median over seeds):")
print(first_cross.groupby("cell").median().to_string())

# how many individuals per generation
per_gen = smoo.groupby(["seed_dir", "generation"]).size()
print("\nindividuals per (seed, gen):", per_gen.describe()[["mean", "min", "max"]].to_dict())

# --- pdq transects time ---
tf = pq.ParquetFile(f"{BASE}/transects.parquet")
print("\ntransects rows:", tf.metadata.num_rows)
print("transects schema:", [f.name for f in tf.schema_arrow])
t = pq.read_table(f"{BASE}/transects.parquet").to_pandas()
print(t.head(3).to_string())
print("\nsteps per walk:")
wl = t.groupby(["seed_dir", "flip_id"]).step.agg(["max", "size"])
print(wl.describe().to_string())
print("\naccepted fraction:", t.accepted.mean())
print("still_flipped fraction:", t.still_flipped.mean() if "still_flipped" in t else "n/a")
print("\ntransect columns of interest:")
for c in ["hamming_to_anchor", "pair_margin"]:
    print(c, t[c].describe().to_dict())
# cells in transects
if "target_class" in t.columns:
    t["cell"] = (t.target_class.astype(str) + " L" +
                 t.level_anchor.astype(str) + "-" + t.level_target.astype(str))
    print("\ntransect walks per cell:")
    print(t.groupby("cell").apply(lambda d: d.groupby(["seed_dir", "flip_id"]).ngroups).to_string())
