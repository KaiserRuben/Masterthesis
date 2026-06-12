"""Supplemental skeptic checks:
 S1 floor effect among non-stuck cells (is the null rho there explained by saturation?)
 S2 P3: within-cell concordance for the 6 n=2 cells
 S3 graded banding: crossing rate by probe band
 S4 is the 30-call probe necessary? single gen-0 individual AUC
 S5 seed-level rho for Exp-100 comparison
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats as st

OUT = "/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/exp101"
RUNS = "/Users/kaiser/Projects/Masterarbeit/runs/Exp-101"

seed = pd.read_csv(f"{OUT}/exp101_per_seed.csv")
ext = pd.read_csv(f"{OUT}/skeptic_run_extract.csv")
seed = seed.merge(ext, on="seed_idx", validate="1:1")
cell = pd.read_csv(f"{OUT}/exp101_per_cell.csv")
cell["log_min50"] = np.log10(cell["min_tgtbal_50"])

print("[S1] floor effect among non-stuck cells")
nons = cell[~cell.stuck_any]
print(f"  non-stuck cells n={len(nons)}: log10(min50) min={nons.log_min50.min():.2f} "
      f"median={nons.log_min50.median():.2f} max={nons.log_min50.max():.2f}")
print(f"  fraction of non-stuck cells with min50 < 0.01 (deep floor): "
      f"{(nons.min_tgtbal_50 < .01).mean():.2f}")
print(f"  IQR of log10(min50) non-stuck: "
      f"{nons.log_min50.quantile(.75)-nons.log_min50.quantile(.25):.2f} dex "
      f"vs stuck cells: "
      f"{cell[cell.stuck_any].log_min50.quantile(.75)-cell[cell.stuck_any].log_min50.quantile(.25):.2f} dex")

print()
print("[S2] P3 within-cell check: the 6 n=2 cells")
two = seed[seed.n_in_cell == 2].sort_values(["cell_id", "seed_idx"])
conc_min, conc_dex = [], []
for cid, g in two.groupby("cell_id"):
    g = g.sort_values("probe")
    dprobe = g.probe.iloc[1] - g.probe.iloc[0]
    dmin = g.min_tgtbal_50.iloc[1] - g.min_tgtbal_50.iloc[0]
    ddex = g.dex_eroded_50.iloc[1] - g.dex_eroded_50.iloc[0]
    conc_min.append(np.sign(dmin) > 0)
    conc_dex.append(np.sign(ddex) < 0)
    print(f"  {cid:32s} probes [{g.probe.iloc[0]:.2f},{g.probe.iloc[1]:.2f}] "
          f"min50 [{g.min_tgtbal_50.iloc[0]:.4f},{g.min_tgtbal_50.iloc[1]:.4f}] "
          f"dex50 [{g.dex_eroded_50.iloc[0]:.2f},{g.dex_eroded_50.iloc[1]:.2f}] "
          f"stuck [{g.stuck.iloc[0]},{g.stuck.iloc[1]}]")
print(f"  higher-probe seed has higher min50 in {sum(conc_min)}/6 cells "
      f"(binomial two-sided p={st.binomtest(sum(conc_min),6,.5).pvalue:.3f})")
print(f"  higher-probe seed has lower dex50 in {sum(conc_dex)}/6 cells")
# within-cell deltas pooled
dp = []
dm = []
for cid, g in two.groupby("cell_id"):
    g = g.sort_values("seed_idx")
    dp.append(g.probe.iloc[1] - g.probe.iloc[0])
    dm.append(np.log10(g.min_tgtbal_50.iloc[1]) - np.log10(g.min_tgtbal_50.iloc[0]))
r, p = st.spearmanr(dp, dm)
print(f"  Spearman of within-cell deltas (d_probe vs d_log_min50): rho={r:+.3f} p={p:.3f} n=6")
print(f"  within-cell |d_probe| median {np.median(np.abs(dp)):.2f} nats vs "
      f"between-cell probe SD {cell.probe.std():.2f} nats")

print()
print("[S3] crossing rate by probe band (cell level)")
bands = [(0, 1.5), (1.5, 3.0), (3.0, 99)]
for lo, hi in bands:
    g = cell[(cell.probe >= lo) & (cell.probe < hi)]
    print(f"  probe [{lo},{hi}): n={len(g):2d} cells, crossed_any={int((g.crossed_50_frac>0).sum())}, "
          f"stuck_any={int(g.stuck_any.sum())}, "
          f"median log10(min50)={g.log_min50.median():+.2f}")

print()
print("[S4] is the 30-call median necessary? single gen-0 individual as probe")
g0 = {}
for d in sorted(glob.glob(os.path.join(RUNS, "exp101_margin_predictor_seed_*"))):
    sidx = json.load(open(os.path.join(d, "stats.json")))["seed_idx"]
    tr = pd.read_parquet(os.path.join(d, "trace.parquet"),
                         columns=["generation", "individual", "fitness_TgtBal"])
    v = tr[tr.generation == 0].sort_values("individual").fitness_TgtBal.to_numpy()
    g0[sidx] = v
seed = seed.sort_values("seed_idx")
y = (~seed.crossed_50).to_numpy()
mat = np.vstack([g0[s] for s in seed.seed_idx])  # 46 x 30


def auc(score, label):
    a, b = score[label], score[~label]
    u, _ = st.mannwhitneyu(a, b, alternative="two-sided")
    return u / (len(a) * len(b))


aucs = [auc(mat[:, i], y) for i in range(30)]
print(f"  single-individual AUC(not crossed): mean {np.mean(aucs):.3f}, "
      f"min {np.min(aucs):.3f}, max {np.max(aucs):.3f}  (median-of-30 probe AUC = "
      f"{auc(seed.probe.to_numpy(), y):.3f})")
sub_aucs = []
rng = np.random.default_rng(7)
for _ in range(2000):
    idx = rng.choice(30, 5, replace=False)
    sub_aucs.append(auc(np.median(mat[:, idx], axis=1), y))
print(f"  median-of-5 random gen-0 individuals: AUC mean {np.mean(sub_aucs):.3f} "
      f"[{np.quantile(sub_aucs,.025):.3f},{np.quantile(sub_aucs,.975):.3f}]")

print()
print("[S5] seed-level Spearman (Exp-100 comparison: rho=0.53 there)")
r, p = st.spearmanr(seed.probe, seed.dex_eroded_50)
print(f"  probe vs dex_eroded_50, all 46 seeds: rho={r:+.3f} p={p:.5f}")
r, p = st.spearmanr(seed.probe, np.log10(seed.min_tgtbal_50))
print(f"  probe vs log10(min50), all 46 seeds: rho={r:+.3f} p={p:.5f}")
nj = seed[seed.anchor != "junco"]
r, p = st.spearmanr(nj.probe, nj.dex_eroded_50)
print(f"  probe vs dex_eroded_50, non-junco 35 seeds: rho={r:+.3f} p={p:.5f}")
