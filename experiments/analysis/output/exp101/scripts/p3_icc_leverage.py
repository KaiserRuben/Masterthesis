"""Leverage of the junco->ostrich(0,0) outlier pair on ICC(1) and within-cell SS."""
import numpy as np
import pandas as pd
from scipy import stats

OUT = "/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/exp101"
ps = pd.read_csv(f"{OUT}/exp101_per_seed.csv")
ps["log10_min50"] = np.log10(ps["min_tgtbal_50"])


def icc1(df, col):
    groups = [g[col].values for _, g in df.groupby("cell_id")]
    k, N = len(groups), sum(len(g) for g in groups)
    grand = df[col].mean()
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    dfb, dfw = k - 1, N - k
    msb, msw = ssb / dfb, ssw / dfw
    n0 = (N - sum(len(g) ** 2 for g in groups) / N) / (k - 1)
    icc = (msb - msw) / (msb + (n0 - 1) * msw)
    F = msb / msw
    return icc, msw, dfw, F, stats.f.sf(F, dfb, dfw), ssw


for col in ["log10_min50", "probe", "dex_eroded_50"]:
    # per-pair contribution to SSW
    n2 = ps.groupby("cell_id").filter(lambda g: len(g) == 2)
    contrib = n2.groupby("cell_id")[col].apply(
        lambda g: ((g - g.mean()) ** 2).sum()).sort_values(ascending=False)
    icc_all, msw, dfw, F, p, ssw = icc1(ps, col)
    print(f"{col}: SSW total={ssw:.3f}; per-pair contributions:")
    for c, v in contrib.items():
        print(f"    {c}: {v:.3f} ({v / ssw:.1%})")
    # drop the junco->ostrich(0,0) pair entirely (both runs)
    sub = ps[ps.cell_id != "junco->ostrich(0,0)"]
    icc_x, msw_x, dfw_x, F_x, p_x, ssw_x = icc1(sub, col)
    print(f"  ICC(1) all 46 runs = {icc_all:.3f} (MSW={msw:.3f}, df={dfw}); "
          f"excluding junco->ostrich(0,0) cell = {icc_x:.3f} "
          f"(MSW={msw_x:.3f}, df={dfw_x}, F={F_x:.1f}, p={p_x:.3g})\n")
