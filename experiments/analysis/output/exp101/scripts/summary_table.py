#!/usr/bin/env python
import os
import pandas as pd

OUT_DIR = "/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/exp101"
cell = pd.read_csv(os.path.join(OUT_DIR, "exp101_per_cell.csv"))
seed = pd.read_csv(os.path.join(OUT_DIR, "exp101_per_seed.csv"))

c = cell.sort_values("probe", ascending=False).reset_index(drop=True)

print("| cell_id | words | stratum | probe | dex50 | crossed | gfc | stuck |")
print("|---|---|---|---|---|---|---|---|")
for _, r in c.iterrows():
    words = f"{r['anchor_word']} vs {r['target_word']}"
    gfc = "" if pd.isna(r["gen_first_cross_mean"]) else f"{r['gen_first_cross_mean']:.0f}"
    crossed = f"{r['crossed_50_frac']:.2f}"
    stuck = f"{r['stuck_frac']:.2f}"
    print(f"| {r['cell_id']} | {words} | {r['stratum']} | {r['probe']:.3f} | {r['dex_eroded_50']:.3f} | {crossed} | {gfc} | {stuck} |")

print()
print("=== sanity ===")
print("strata (per-cell):")
print(cell["stratum"].value_counts().to_dict())
print("strata (per-seed):")
print(seed["stratum"].value_counts().to_dict())
print("non-junco anchor cells:", (cell["anchor"] != "junco").sum())
print("non-junco anchor seeds:", (seed["anchor"] != "junco").sum())
print("crossed_50 seeds:", int(seed["crossed_50"].sum()), "/", len(seed))
print("stuck seeds:", int(seed["stuck"].sum()), "/", len(seed))
print("probe range:", round(seed["probe"].min(), 4), "-", round(seed["probe"].max(), 4))
print("dex_eroded_50 range:", round(seed["dex_eroded_50"].min(), 4), "-", round(seed["dex_eroded_50"].max(), 4))
print("n_in_cell distribution:", seed["n_in_cell"].value_counts().to_dict())
print("anchor classes:", sorted(seed["anchor"].unique()))
