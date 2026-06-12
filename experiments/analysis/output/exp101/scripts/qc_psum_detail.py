"""Quantify p_class_a + p_class_b excess over 1.0 across all Exp-101 runs."""
from pathlib import Path
import numpy as np
import pandas as pd

RUNS = Path("/Users/kaiser/Projects/Masterarbeit/runs/Exp-101")

max_excess = -np.inf
n_gt = {1e-9: 0, 1e-7: 0, 1e-5: 0, 1e-3: 0}
n_rows = 0
sum_exact_1 = 0
for d in sorted(RUNS.glob("exp101_margin_predictor_seed_*")):
    tr = pd.read_parquet(d / "trace.parquet", columns=["p_class_a", "p_class_b"])
    s = (tr["p_class_a"] + tr["p_class_b"]).to_numpy()
    n_rows += len(s)
    excess = s - 1.0
    max_excess = max(max_excess, excess.max())
    for tol in n_gt:
        n_gt[tol] += int((excess > tol).sum())
    sum_exact_1 += int((s == 1.0).sum())

print(f"rows total: {n_rows}")
print(f"max excess over 1.0: {max_excess:.3e}")
for tol, n in n_gt.items():
    print(f"rows with p_a+p_b - 1 > {tol:g}: {n}")
print(f"rows with p_a+p_b == 1.0 exactly: {sum_exact_1}")
print(f"min sum: dataset-wide check (pair-renormalized -> expect ~1):")
# distribution of sums
mins = []
for d in sorted(RUNS.glob("exp101_margin_predictor_seed_*")):
    tr = pd.read_parquet(d / "trace.parquet", columns=["p_class_a", "p_class_b"])
    mins.append((tr["p_class_a"] + tr["p_class_b"]).min())
print(f"min p_a+p_b over runs: {min(mins):.6f}")
