"""P3 within-cell null analysis for Exp-101.

Variance decomposition: within-cell vs between-cell signal of the gen-0
margin probe, using the 6 design cells with n=2 seeds, plus a
seed_idx_in_class check across all 46 runs.
"""
import numpy as np
import pandas as pd
from scipy import stats

OUT = "/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/exp101"
ps = pd.read_csv(f"{OUT}/exp101_per_seed.csv")
pc = pd.read_csv(f"{OUT}/exp101_per_cell.csv")

print(f"per-seed rows: {len(ps)}, per-cell rows: {len(pc)}")
ps["log10_min50"] = np.log10(ps["min_tgtbal_50"])
ps["log10_probe"] = np.log10(ps["probe"])

# ---------------------------------------------------------------- 1. n=2 cells
counts = ps.groupby("cell_id").size()
n2_cells = sorted(counts[counts == 2].index.tolist())
print(f"\n=== 1. Cells with n=2 ({len(n2_cells)}) ===")
for c in n2_cells:
    sub = ps[ps.cell_id == c]
    print(f"  {c}: seed_idx={sub.seed_idx.tolist()}, "
          f"seed_idx_in_class={sub.seed_idx_in_class.tolist()}, "
          f"stratum={sub.stratum.iloc[0]}")
design = ["junco->ostrich(0,0)", "green iguana->boa constrictor(1,1)",
          "cello->marimba(0,0)", "junco->boa constrictor(0,1)",
          "boa constrictor->junco(0,1)", "green iguana->cello(0,0)"]
print("  design-expected match:", sorted(design) == n2_cells or
      (set(n2_cells), set(design)))

# ---------------------------------------------------------- 2. within-pair deltas
print("\n=== 2. Within-pair deltas (seed_idx_in_class 1 minus 0) ===")
rows = []
for c in n2_cells:
    sub = ps[ps.cell_id == c].sort_values("seed_idx_in_class")
    a, b = sub.iloc[0], sub.iloc[1]  # a: idx_in_class 0, b: idx_in_class 1
    d_probe = b.probe - a.probe
    d_dex = b.dex_eroded_50 - a.dex_eroded_50
    d_logmin = b.log10_min50 - a.log10_min50
    # higher-probe seed: does it erode less (lower dex_eroded_50)?
    hi = b if b.probe > a.probe else a
    lo = a if b.probe > a.probe else b
    concord_dex = hi.dex_eroded_50 < lo.dex_eroded_50      # expected if within-cell signal
    concord_min = hi.min_tgtbal_50 > lo.min_tgtbal_50      # higher probe -> worse floor
    rows.append(dict(cell=c, probe_0=a.probe, probe_1=b.probe,
                     dex50_0=a.dex_eroded_50, dex50_1=b.dex_eroded_50,
                     log10min50_0=a.log10_min50, log10min50_1=b.log10_min50,
                     d_probe=d_probe, d_dex_eroded_50=d_dex,
                     d_log10_min50=d_logmin,
                     hi_probe_erodes_less=bool(concord_dex),
                     hi_probe_higher_floor=bool(concord_min),
                     crossed_0=a.crossed_50, crossed_1=b.crossed_50,
                     stuck_0=a.stuck, stuck_1=b.stuck,
                     abs_d_log10_min50=abs(d_logmin)))
pairs = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(pairs[["cell", "d_probe", "d_dex_eroded_50", "d_log10_min50",
             "hi_probe_erodes_less", "hi_probe_higher_floor"]].to_string(index=False))
pairs.to_csv(f"{OUT}/exp101_p3_n2_pairs.csv", index=False)

k_less = int(pairs.hi_probe_erodes_less.sum())
k_floor = int(pairs.hi_probe_higher_floor.sum())
print(f"\nSign concordance (higher probe -> erodes LESS, dex_eroded_50): "
      f"{k_less}/6, binomial two-sided p={stats.binomtest(k_less, 6, 0.5).pvalue:.4f}")
print(f"Sign concordance (higher probe -> HIGHER floor min_tgtbal_50): "
      f"{k_floor}/6, binomial two-sided p={stats.binomtest(k_floor, 6, 0.5).pvalue:.4f}")

# within-cell residuals pooled over the 12 runs of n=2 cells
sub12 = ps[ps.cell_id.isin(n2_cells)].copy()
for col in ["probe", "dex_eroded_50", "log10_min50"]:
    sub12[f"res_{col}"] = sub12[col] - sub12.groupby("cell_id")[col].transform("mean")
for ycol in ["res_dex_eroded_50", "res_log10_min50"]:
    r_p, p_p = stats.pearsonr(sub12["res_probe"], sub12[ycol])
    r_s, p_s = stats.spearmanr(sub12["res_probe"], sub12[ycol])
    print(f"Within-cell residual corr res_probe vs {ycol} (n=12): "
          f"Pearson r={r_p:.3f} (p={p_p:.3f}), Spearman rho={r_s:.3f} (p={p_s:.3f})")
print("NOTE: residuals within an n=2 pair are exactly +/-delta/2; the 12-point "
      "correlation has only 6 independent pairs (effective df ~ 4-5).")

# cross/stuck concordance within pairs
agree_cross = int((pairs.crossed_0 == pairs.crossed_1).sum())
agree_stuck = int((pairs.stuck_0 == pairs.stuck_1).sum())
print(f"Binary outcome agreement within pairs: crossed_50 {agree_cross}/6, "
      f"stuck {agree_stuck}/6")

# ------------------------------------------------- 3. ICC / magnitude framing
print("\n=== 3. Magnitude framing: within vs between spread ===")


def icc1(df, col):
    groups = [g[col].values for _, g in df.groupby("cell_id")]
    k = len(groups)
    N = sum(len(g) for g in groups)
    grand = df[col].mean()
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    dfb, dfw = k - 1, N - k
    msb = ssb / dfb
    msw = ssw / dfw if dfw > 0 else np.nan
    n0 = (N - sum(len(g) ** 2 for g in groups) / N) / (k - 1)
    icc = (msb - msw) / (msb + (n0 - 1) * msw)
    F = msb / msw
    p = stats.f.sf(F, dfb, dfw)
    return icc, msb, msw, dfb, dfw, n0, F, p


for col, label in [("log10_min50", "log10(min_tgtbal_50) [dex]"),
                   ("probe", "probe [nats]"),
                   ("dex_eroded_50", "dex_eroded_50 [dex]")]:
    icc, msb, msw, dfb, dfw, n0, F, p = icc1(ps, col)
    print(f"{label}: ICC(1)={icc:.3f}  MSB={msb:.3f} (df={dfb})  "
          f"MSW={msw:.3f} (df={dfw})  n0={n0:.3f}  F={F:.2f} p={p:.3g}")
    sd_w = np.sqrt(msw)
    sd_b_cells = ps.groupby("cell_id")[col].mean().std(ddof=1)
    print(f"    within-cell SD={sd_w:.3f}; SD of cell means={sd_b_cells:.3f}; "
          f"full range across 46 runs={ps[col].max() - ps[col].min():.3f}")

rng = pairs["abs_d_log10_min50"]
print(f"\nWithin-cell range on min_tgtbal_50 (dex), 6 pairs: "
      f"median={rng.median():.3f}, mean={rng.mean():.3f}, "
      f"min={rng.min():.3f}, max={rng.max():.3f}")
print("Per-pair |d log10 min50|:")
print(pairs[["cell", "abs_d_log10_min50"]].to_string(index=False))
cellmeans = ps.groupby("cell_id")["log10_min50"].mean()
print(f"Between-cell range of cell-mean log10(min_tgtbal_50): "
      f"{cellmeans.max() - cellmeans.min():.3f} dex "
      f"(IQR={cellmeans.quantile(0.75) - cellmeans.quantile(0.25):.3f})")
print(f"Exp-100 reference: ICC(1)=0.52, median within-cell range 1.08 dex at n=3")

# probe within-pair spread vs between-cell spread
rng_probe = (pairs.probe_1 - pairs.probe_0).abs()
cm_probe = ps.groupby("cell_id")["probe"].mean()
print(f"\nProbe within-pair |delta| (nats): median={rng_probe.median():.3f}, "
      f"max={rng_probe.max():.3f}; between-cell probe range="
      f"{cm_probe.max() - cm_probe.min():.3f} nats, SD={cm_probe.std(ddof=1):.3f}")

# ------------------------------------ 4. seed_idx_in_class across all 46 runs
print("\n=== 4. seed_idx_in_class 0 vs 1, all 46 runs ===")
print(ps.seed_idx_in_class.value_counts().to_string())
g0 = ps[ps.seed_idx_in_class == 0]
g1 = ps[ps.seed_idx_in_class == 1]
for col in ["probe", "dex_eroded_50", "log10_min50"]:
    u, p = stats.mannwhitneyu(g0[col], g1[col], alternative="two-sided")
    print(f"{col}: idx0 n={len(g0)} median={g0[col].median():.3f}  "
          f"idx1 n={len(g1)} median={g1[col].median():.3f}  "
          f"Mann-Whitney U={u:.0f} p={p:.3f} (unpaired, cell-confounded)")
print(f"crossed_50: idx0 {g0.crossed_50.mean():.2%} ({int(g0.crossed_50.sum())}/{len(g0)}), "
      f"idx1 {g1.crossed_50.mean():.2%} ({int(g1.crossed_50.sum())}/{len(g1)})")
print(f"stuck:      idx0 {g0.stuck.mean():.2%} ({int(g0.stuck.sum())}/{len(g0)}), "
      f"idx1 {g1.stuck.mean():.2%} ({int(g1.stuck.sum())}/{len(g1)})")

print("\nPaired (cell-controlled) test on the 6 n=2 cells, idx1 minus idx0:")
for dcol in ["d_probe", "d_dex_eroded_50", "d_log10_min50"]:
    d = pairs[dcol]
    try:
        w, pw = stats.wilcoxon(d)
    except ValueError:
        w, pw = np.nan, np.nan
    npos = int((d > 0).sum())
    print(f"  {dcol}: mean={d.mean():+.3f} median={d.median():+.3f} "
          f"signs +{npos}/-{6 - npos}  Wilcoxon W={w} p={pw:.3f}  "
          f"sign-test p={stats.binomtest(npos, 6, 0.5).pvalue:.3f}")

# paired probe correlation between the two anchor images (consistency check)
p0 = pairs.probe_0.values
p1 = pairs.probe_1.values
r, p = stats.pearsonr(p0, p1)
rs, prs = stats.spearmanr(p0, p1)
print(f"\nProbe(idx0) vs Probe(idx1) across the 6 paired cells: "
      f"Pearson r={r:.3f} (p={p:.3f}), Spearman rho={rs:.3f} (p={prs:.3f})")
r2, p2 = stats.pearsonr(pairs.log10min50_0, pairs.log10min50_1)
rs2, prs2 = stats.spearmanr(pairs.log10min50_0, pairs.log10min50_1)
print(f"log10(min50)(idx0) vs (idx1): Pearson r={r2:.3f} (p={p2:.3f}), "
      f"Spearman rho={rs2:.3f} (p={prs2:.3f})")
