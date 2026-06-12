"""Exp-101 pre-registered confirmatory analysis: P1 (probe vs dex_eroded) and P2 (high-margin
cells rarely cross). Cell-level, using canonical per-cell / per-seed tables."""
import numpy as np
import pandas as pd
from scipy import stats

rng = np.random.default_rng(101)
OUT = "/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/exp101"
pc = pd.read_csv(f"{OUT}/exp101_per_cell.csv")
ps = pd.read_csv(f"{OUT}/exp101_per_seed.csv")

N_BOOT = 20000


def spearman_with_boot(x, y, n_boot=N_BOOT):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    rho, p = stats.spearmanr(x, y)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        # degenerate resamples (all-equal) give nan; resample again
        r = stats.spearmanr(x[idx], y[idx]).statistic
        while np.isnan(r):
            idx = rng.integers(0, n, n)
            r = stats.spearmanr(x[idx], y[idx]).statistic
        boots[b] = r
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return rho, p, lo, hi, n


# ---------------- P1 ----------------
subsets = {
    "all_40_cells": pc,
    "non_junco_31_cells": pc[pc.anchor != "junco"],
    "junco_anchor_9_cells": pc[pc.anchor == "junco"],
    "stratum=within_bucket": pc[pc.stratum == "within_bucket"],
    "stratum=wall_repl": pc[pc.stratum == "wall_repl"],
    "stratum=cross_breadth": pc[pc.stratum == "cross_breadth"],
}
outcomes = ["dex_eroded_50", "dex_eroded_min_50", "dex_eroded_20"]

rows = []
print("=" * 100)
print(f"P1: Spearman(probe, outcome) at cell level, bootstrap 95 pct CI ({N_BOOT} draws, resample cells)")
print("=" * 100)
for oc in outcomes:
    for name, df in subsets.items():
        rho, p, lo, hi, n = spearman_with_boot(df["probe"], df[oc])
        rows.append(dict(outcome=oc, subset=name, n=n, rho=rho, p=p, ci_lo=lo, ci_hi=hi))
        print(f"{oc:20s} {name:24s} n={n:3d}  rho={rho:+.4f}  p={p:.5f}  CI95=[{lo:+.4f},{hi:+.4f}]")
p1 = pd.DataFrame(rows)
p1.to_csv(f"{OUT}/p1_spearman_results.csv", index=False)

# falsification verdict
fr = p1[(p1.outcome == "dex_eroded_50") & (p1.subset == "non_junco_31_cells")].iloc[0]
print("\nP1 FALSIFICATION LINE: |rho| < 0.3 on non-junco subset (dex_eroded_50)")
print(f"  observed rho = {fr.rho:+.4f}  -> {'FALSIFIED' if abs(fr.rho) < 0.3 else 'NOT falsified'}")
print(f"  sign: {'NEGATIVE as predicted' if fr.rho < 0 else 'POSITIVE (wrong sign!)'}")

# seed-level for reference (Exp-100 reported seed-level rho)
for sub, name in [(ps, "all_46_seeds"), (ps[ps.anchor != "junco"], "non_junco_seeds")]:
    rho, p = stats.spearmanr(sub["probe"], sub["dex_eroded_50"])
    print(f"  [reference seed-level] {name:18s} n={len(sub)}  rho={rho:+.4f}  p={p:.5f}")

# ---------------- P2 ----------------
print("\n" + "=" * 100)
print("P2: high-margin cells (probe >= 3 nats, cell mean) rarely cross within budget")
print("=" * 100)
pc["high_margin"] = pc.probe >= 3.0
for flag, grp in pc.groupby("high_margin"):
    lab = "probe>=3" if flag else "probe<3 "
    print(f"{lab}: n_cells={len(grp):2d}  crossed_50_any rate={grp.crossed_50_any.mean():.3f} "
          f"({int(grp.crossed_50_any.sum())}/{len(grp)})  mean crossed_50_frac={grp.crossed_50_frac.mean():.3f}")

# gen_first_cross distribution per group (seed level, only crossing seeds)
ps["high_margin_cell"] = ps.cell_id.map(pc.set_index("cell_id").high_margin)
for flag, grp in ps.groupby("high_margin_cell"):
    g = grp.gen_first_cross.dropna()
    lab = "probe>=3" if flag else "probe<3 "
    if len(g):
        print(f"{lab}: crossing seeds n={len(g)}  gen_first_cross min/med/max = "
              f"{g.min():.0f}/{g.median():.1f}/{g.max():.0f}  quartiles=({g.quantile(.25):.1f},{g.quantile(.75):.1f})")
    else:
        print(f"{lab}: crossing seeds n=0 (no gen_first_cross distribution)")

# threshold sweep
print("\nThreshold sweep (cell level, crossed_50_any):")
sweep_rows = []
for thr in np.arange(2.0, 5.01, 0.25):
    hi_g = pc[pc.probe >= thr]
    lo_g = pc[pc.probe < thr]
    hr = hi_g.crossed_50_any.mean() if len(hi_g) else np.nan
    lr = lo_g.crossed_50_any.mean() if len(lo_g) else np.nan
    sweep_rows.append(dict(threshold=thr, n_high=len(hi_g), cross_rate_high=hr,
                           n_low=len(lo_g), cross_rate_low=lr, gap=lr - hr if len(hi_g) and len(lo_g) else np.nan))
    print(f"  thr={thr:4.2f}  high: n={len(hi_g):2d} cross={hr if not np.isnan(hr) else float('nan'):.3f}   "
          f"low: n={len(lo_g):2d} cross={lr:.3f}   gap(low-high)={lr-hr if len(hi_g) else float('nan'):+.3f}")
pd.DataFrame(sweep_rows).to_csv(f"{OUT}/p2_threshold_sweep.csv", index=False)

# AUC: probe predicting NOT crossed_50 at cell level
y = (~pc.crossed_50_any.astype(bool)).astype(int).values  # 1 = did not cross
x = pc.probe.values
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y, x)
# bootstrap CI for AUC
aucs = []
n = len(pc)
b = 0
while b < 10000:
    idx = rng.integers(0, n, n)
    if len(np.unique(y[idx])) < 2:
        continue
    aucs.append(roc_auc_score(y[idx], x[idx]))
    b += 1
alo, ahi = np.percentile(aucs, [2.5, 97.5])
print(f"\nAUC(probe -> NOT crossed_50, cell level): {auc:.4f}  bootstrap CI95=[{alo:.4f},{ahi:.4f}]")
print(f"  base rate NOT crossed: {y.mean():.3f} ({y.sum()}/{n})")
# Mann-Whitney probe in crossed vs not
mw = stats.mannwhitneyu(x[y == 1], x[y == 0], alternative="greater")
print(f"  Mann-Whitney (probe higher in non-crossing cells): U={mw.statistic:.1f} p={mw.pvalue:.5f}")
print(f"  probe median: non-crossing cells {np.median(x[y==1]):.3f}  crossing cells {np.median(x[y==0]):.3f}")

# non-junco only AUC (robustness)
m = (pc.anchor != "junco").values
if len(np.unique(y[m])) == 2:
    auc_nj = roc_auc_score(y[m], x[m])
    print(f"  AUC on non-junco 31 cells: {auc_nj:.4f}  (base NOT-crossed rate {y[m].mean():.3f})")

# ---------------- Stuck-seed tercile check ----------------
print("\n" + "=" * 100)
print("Stuck-seed check (Exp-100 replication): stuck = min_tgtbal_50 > 0.1")
print("=" * 100)
ps["probe_tercile"] = pd.qcut(ps.probe, 3, labels=["bottom", "middle", "top"])
ct = pd.crosstab(ps.probe_tercile, ps.stuck)
print(ct)
stuck = ps[ps.stuck]
print(f"\nn_stuck={len(stuck)}/46. Tercile membership of stuck runs: "
      f"{stuck.probe_tercile.value_counts().to_dict()}")
print(f"All stuck in top tercile? {(stuck.probe_tercile=='top').all()}")
print(f"Stuck-rate by tercile: {ps.groupby('probe_tercile', observed=True).stuck.mean().round(3).to_dict()}")
print(f"Probe range of stuck runs: [{stuck.probe.min():.3f}, {stuck.probe.max():.3f}]; "
      f"non-stuck: [{ps[~ps.stuck].probe.min():.3f}, {ps[~ps.stuck].probe.max():.3f}]")
auc_stuck = roc_auc_score(ps.stuck.astype(int), ps.probe)
print(f"AUC(probe -> stuck, seed level): {auc_stuck:.4f}")

# ---------------- Scatter PNG ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8.5, 6.5))
colors = {"within_bucket": "#1f77b4", "wall_repl": "#d62728", "cross_breadth": "#2ca02c"}
# per-seed points so n=2 cells show as both points
for stratum, grp in ps.groupby("stratum"):
    junco = grp.anchor == "junco"
    ax.scatter(grp.probe[~junco], grp.dex_eroded_50[~junco], c=colors[stratum], s=55,
               alpha=0.85, label=f"{stratum} (non-junco anchor)", edgecolors="none")
    if junco.any():
        ax.scatter(grp.probe[junco], grp.dex_eroded_50[junco], facecolors="none",
                   edgecolors=colors[stratum], s=80, linewidths=1.8,
                   label=f"{stratum} (junco anchor)")
# connect n=2 cell pairs with thin lines
for cid, grp in ps.groupby("cell_id"):
    if len(grp) == 2:
        ax.plot(grp.probe, grp.dex_eroded_50, color="gray", lw=0.7, alpha=0.6, zorder=0)
rho_all = p1[(p1.outcome == "dex_eroded_50") & (p1.subset == "all_40_cells")].iloc[0]
ax.axvline(3.0, color="k", ls=":", lw=1, alpha=0.5)
ax.text(3.05, ax.get_ylim()[1] * 0.97, "P2 threshold (3 nats)", fontsize=8, va="top")
ax.set_xlabel("probe = gen-0 median fitness_TgtBal (nats)")
ax.set_ylabel("dex_eroded_50 = log10(probe / min TgtBal @ gen 50)")
ax.set_title(f"Exp-101 P1: gen-0 margin vs erosion (46 runs, 40 cells)\n"
             f"cell-level Spearman: all rho={rho_all.rho:+.3f}, "
             f"non-junco rho={fr.rho:+.3f} [CI {fr.ci_lo:+.2f},{fr.ci_hi:+.2f}]")
ax.legend(fontsize=8, loc="best")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f"{OUT}/p1_scatter.png", dpi=150)
print(f"\nSaved {OUT}/p1_scatter.png")
