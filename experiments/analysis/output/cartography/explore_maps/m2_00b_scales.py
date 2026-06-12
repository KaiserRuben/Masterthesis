"""Check seed sharing across cells + semantic axis scales per seed (pdq_s1)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path("/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography")

cols = ["source", "seed_dir", "seed_idx", "seed_idx_in_class", "image_dim",
        "d_img_sem", "d_txt_sem", "target_class", "level_anchor", "level_target"]
pts = pd.read_parquet(ROOT / "exp100/points.parquet", columns=cols,
                      filters=[("prompt_regime", "==", "cat6"), ("source", "==", "pdq_s1")])

print("seed_dirs:", pts.seed_dir.nunique())
print("seed_idx_in_class values:", sorted(pts.seed_idx_in_class.unique()))
print("image_dim by seed_idx_in_class:")
print(pts.groupby("seed_idx_in_class").image_dim.unique())

print("\nd_img_sem quantiles per seed_idx_in_class:")
print(pts.groupby("seed_idx_in_class").d_img_sem.describe(percentiles=[.5, .9, .99]).round(0))
print("\nd_txt_sem quantiles per seed_idx_in_class:")
print(pts.groupby("seed_idx_in_class").d_txt_sem.describe(percentiles=[.5, .9]).round(3))

# does each cell have all 3 seed images?
chk = pts.groupby(["target_class", "level_anchor", "level_target"]).seed_idx_in_class.nunique()
print("\nseeds per cell min/max:", chk.min(), chk.max())
