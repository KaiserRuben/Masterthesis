"""Exp-101 vs Exp-100 matched-cell comparison (junco-anchored cells)."""
import numpy as np
import pandas as pd

OUT = "/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/exp101"
cell = pd.read_csv(f"{OUT}/exp101_per_cell.csv")
e100 = pd.read_parquet(
    "/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/"
    "exp100_poc_aggregate.parquet")

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
pd.set_option("display.float_format", lambda x: f"{x:.4g}")

print("exp100 'run' values:", e100["run"].unique())
print("exp100 anchors:", e100["anchor_class_concrete"].unique())
print("exp100 n rows:", len(e100))

# exp100 per-cell aggregate
g = e100.groupby(["anchor_class_concrete", "target_class_concrete",
                  "level_anchor", "level_target",
                  "anchor_label_in_prompt", "target_label_in_prompt"])
e100c = g.agg(n100=("min_TgtBal", "size"),
              min_TgtBal_100=("min_TgtBal", "min"),
              med_TgtBal_100=("min_TgtBal", "median"),
              max_TgtBal_100=("min_TgtBal", "max"),
              min_at_gen_med=("min_TgtBal_at_gen", "median"),
              ).reset_index()
e100c = e100c.rename(columns={
    "anchor_class_concrete": "anchor", "target_class_concrete": "target",
    "level_anchor": "la", "level_target": "lt",
    "anchor_label_in_prompt": "aw100", "target_label_in_prompt": "tw100"})

m = cell.merge(e100c, on=["anchor", "target", "la", "lt"], how="inner")
m["words_match"] = (m.anchor_word == m.aw100) & (m.target_word == m.tw100)
m["stuck100_all"] = m["min_TgtBal_100"] > 0.1   # even best seed stuck
m["agree_stuck"] = m["stuck_any"] == m["stuck100_all"]

print("\nMATCHED CELLS (Exp-101 cell x Exp-100 cell):", len(m))
cols = ["cell_id", "anchor_word", "target_word", "words_match", "n",
        "min_tgtbal_50", "stuck_any", "crossed_50_any",
        "n100", "min_TgtBal_100", "med_TgtBal_100", "max_TgtBal_100",
        "min_at_gen_med", "stuck100_all", "agree_stuck"]
print(m[cols].to_string(index=False))

print("\nqualitative agreement on stuck (Exp-101 stuck_any vs Exp-100 best-seed"
      " min>0.1):", m["agree_stuck"].sum(), "/", len(m))
disagree = m[~m["agree_stuck"]]
if len(disagree):
    print("\ndisagreements:")
    print(disagree[cols].to_string(index=False))

# Spearman of cell-level min TgtBal across the two experiments
from scipy.stats import spearmanr
rho, p = spearmanr(np.log10(m["min_tgtbal_50"]),
                   np.log10(m["med_TgtBal_100"]))
print(f"\nSpearman log10(min_tgtbal_50 Exp101) vs log10(median min_TgtBal "
      f"Exp100) over {len(m)} matched cells: rho={rho:.3f}, p={p:.4f}")
rho2, p2 = spearmanr(np.log10(m["min_tgtbal_50"]),
                     np.log10(m["min_TgtBal_100"]))
print(f"Spearman vs Exp100 best-seed min: rho={rho2:.3f}, p={p2:.4f}")

# how often did Exp-100 reach its min only after gen 50 (budget caveat)
print("\nExp-100 median min_TgtBal_at_gen per matched cell (budget caveat: "
      "Exp-101 stops at 50):")
print(m[["cell_id", "min_at_gen_med"]].to_string(index=False))
m.to_csv(f"{OUT}/exp100_matched_cells.csv", index=False)
print(f"\nwrote {OUT}/exp100_matched_cells.csv")
