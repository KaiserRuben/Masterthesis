"""Census of cat6 data for region-map prototypes: sources, cells, labels, axes coverage."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography")

cols = ["source", "seed_dir", "pred_label", "top_gap", "logprobs",
        "n_active_img", "n_active_txt", "rank_sum_img_norm", "rank_sum_txt_norm",
        "hamming_to_anchor", "image_dim", "d_img_sem", "d_txt_sem",
        "anchor_class", "target_class", "level_anchor", "level_target"]
pts = pd.read_parquet(ROOT / "exp100/points.parquet", columns=cols,
                      filters=[("prompt_regime", "==", "cat6")])
print(f"cat6 rows: {len(pts)}")
print("\nby source:")
print(pts.source.value_counts())
print("\npred_label x source:")
print(pts.groupby(["source", "pred_label"]).size().unstack(fill_value=0))

print("\ncells (target_class x levels), pdq_s1 rows:")
s1 = pts[pts.source == "pdq_s1"]
cell = s1.groupby(["target_class", "level_anchor", "level_target"]).size()
print(cell)

print("\nn seeds per cell (s1):")
print(s1.groupby(["target_class", "level_anchor", "level_target"]).seed_dir.nunique())

print("\nsemantic axes coverage s1: d_img_sem notna %.3f  d_txt_sem notna %.3f" % (
    s1.d_img_sem.notna().mean(), s1.d_txt_sem.notna().mean()))
print("d_img_sem range:", s1.d_img_sem.min(), s1.d_img_sem.max())
print("d_txt_sem range:", s1.d_txt_sem.min(), s1.d_txt_sem.max())

# runner-up census on s1
lp = np.stack(s1["logprobs"].to_numpy())
order = np.argsort(-lp, axis=1)
CAT6 = ["junco", "ostrich", "green iguana", "boa constrictor", "cello", "marimba"]
runner = np.array(CAT6)[order[:, 1]]
print("\nrunner-up class distribution (s1):")
print(pd.Series(runner).value_counts())
top = np.array(CAT6)[order[:, 0]]
print("\ntop x runner crosstab (s1):")
print(pd.crosstab(pd.Series(top, name="top"), pd.Series(runner, name="runner")))

# check logprob column order assumption: does pred_label == CAT6[argmax]?
match = (pd.Series(top, index=s1.index) == s1.pred_label).mean()
print(f"\npred_label == CAT6[argmax(logprobs)]: {match:.4f}")

# straddle pairs
sp = pd.read_parquet(ROOT / "exp100/straddle_pairs.parquet")
print(f"\nstraddle rows: {len(sp)}; columns: {list(sp.columns)}")
print("\nboundary_kind:")
print(sp.boundary_kind.value_counts())
print("\nargmax flips labels_before -> labels_after:")
am = sp[sp.boundary_kind == "argmax"]
print(am.groupby(["label_before", "label_after"]).size() if "label_before" in sp.columns else
      sp.filter(like="label").columns.tolist())
