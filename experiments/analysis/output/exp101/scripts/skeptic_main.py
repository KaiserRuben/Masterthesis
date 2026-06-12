"""Adversarial audit of the Exp-101 gen-0 margin predictor (P1/P2/P3).

Attack lines:
 1 mechanical coupling (shared-term arithmetic + truncation), decoupled outcomes
 2 confounds (anchor class, levels, two-cluster wall structure)
 3 probe validity (identity with p_class_a under pair renormalisation; option order)
 4 budget truncation (are 'stuck' runs still improving at gen 49?)
 5 tercile-claim replication and base rates
"""
import numpy as np
import pandas as pd
from scipy import stats as st

OUT = "/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/exp101"
rng = np.random.default_rng(101)

seed = pd.read_csv(f"{OUT}/exp101_per_seed.csv")
cell = pd.read_csv(f"{OUT}/exp101_per_cell.csv")
ext = pd.read_csv(f"{OUT}/skeptic_run_extract.csv")
seed = seed.merge(ext, on="seed_idx", validate="1:1")

# cell-level aggregation of new decoupled outcomes (mean within cell, like prior agent)
cext = seed.groupby("cell_id").agg(
    search_min_1_49=("search_min_1_49", "mean"),
    slope_log10_last10=("slope_log10_last10", "mean"),
).reset_index()
cell = cell.merge(cext, on="cell_id", validate="1:1")
cell["log_min50"] = np.log10(cell["min_tgtbal_50"])
cell["log_searchmin"] = np.log10(cell["search_min_1_49"])
seed["log_min50"] = np.log10(seed["min_tgtbal_50"])
seed["log_searchmin"] = np.log10(seed["search_min_1_49"])

nj = cell[cell.anchor != "junco"].copy()
nj_seed = seed[seed.anchor != "junco"].copy()


def sp(x, y, label):
    r, p = st.spearmanr(x, y)
    print(f"  {label:68s} rho={r:+.3f} p={p:.4f} n={len(x)}")
    return r, p


def auc_mw(score, binary):
    """AUC of score for predicting binary==True, with Mann-Whitney p."""
    a = np.asarray(score)[np.asarray(binary)]
    b = np.asarray(score)[~np.asarray(binary)]
    u, p = st.mannwhitneyu(a, b, alternative="two-sided")
    return u / (len(a) * len(b)), p, len(a), len(b)


print("=" * 100)
print("BASELINE (claimed P1 form, cell level)")
print("=" * 100)
sp(cell.probe, cell.dex_eroded_50, "ALL 40 cells: probe vs dex_eroded_50")
sp(nj.probe, nj.dex_eroded_50, "NON-JUNCO 31 cells: probe vs dex_eroded_50  [pre-registered P1]")
sp(nj.probe, nj.dex_eroded_min_50, "NON-JUNCO: probe vs dex_eroded_min_50 (sensitivity)")

print()
print("=" * 100)
print("ATTACK 1 — MECHANICAL COUPLING")
print("=" * 100)
print("[1a] outcomes with NO shared term with probe (cell level):")
sp(cell.probe, cell.log_min50, "ALL: probe vs log10(min_tgtbal_50)   (truncation min<=probe remains)")
sp(nj.probe, nj.log_min50, "NON-JUNCO: probe vs log10(min_tgtbal_50)")
sp(cell.probe, cell.log_searchmin, "ALL: probe vs log10(search_min gens1-49)  (no gen-0 rows at all)")
sp(nj.probe, nj.log_searchmin, "NON-JUNCO: probe vs log10(search_min gens1-49)")

print()
print("[1a-sim] shuffle nulls for Spearman(probe, dex_eroded_50), non-junco cells, 10000 perms")
probe = nj.probe.to_numpy()
min50 = nj.min_tgtbal_50.to_numpy()
mingen0 = nj.min_gen0.to_numpy()
obs, _ = st.spearmanr(probe, np.log10(probe / min50))
nullA, nullB = [], []
for _ in range(10000):
    perm = rng.permutation(len(probe))
    mp = min50[perm]
    nullA.append(st.spearmanr(probe, np.log10(probe / mp))[0])  # pure shared-term
    mpc = np.minimum(mp, mingen0)                                # + truncation clip
    nullB.append(st.spearmanr(probe, np.log10(probe / mpc))[0])
nullA, nullB = np.array(nullA), np.array(nullB)
print(f"  observed rho = {obs:+.3f}")
print(f"  null A (permute min50, shared probe term only):  mean {nullA.mean():+.3f}, "
      f"95% [{np.quantile(nullA,.025):+.3f},{np.quantile(nullA,.975):+.3f}], "
      f"P(null<=obs) = {np.mean(nullA<=obs):.5f}")
print(f"  null B (+ truncation min<=min_gen0 enforced):    mean {nullB.mean():+.3f}, "
      f"95% [{np.quantile(nullB,.025):+.3f},{np.quantile(nullB,.975):+.3f}], "
      f"P(null<=obs) = {np.mean(nullB<=obs):.5f}")
print("  NOTE: coupling direction is POSITIVE (probe in numerator) while P1 predicts NEGATIVE;")
print("        mechanical coupling therefore works AGAINST P1, it cannot manufacture it.")

print()
print("[1b] probe vs gen_first_cross (survival flavour)")
cr = seed[seed.crossed_50]
r, p = st.spearmanr(cr.probe, cr.gen_first_cross)
print(f"  crossers only (seed level): rho={r:+.3f} p={p:.4f} n={len(cr)} (censored runs excluded)")
crc = cell[cell.crossed_50_any]
r, p = st.spearmanr(crc.probe, crc.gen_first_cross_mean)
print(f"  crossers only (cell level): rho={r:+.3f} p={p:.4f} n={len(crc)}")
# logistic crossed ~ probe, seed level (manual Newton via statsmodels if present)
try:
    import statsmodels.api as sm
    for tag, df in [("ALL seeds", seed), ("NON-JUNCO seeds", nj_seed)]:
        X = sm.add_constant(df.probe.to_numpy())
        m = sm.Logit(df.crossed_50.astype(float).to_numpy(), X).fit(disp=0)
        print(f"  logit crossed_50 ~ probe ({tag}): beta={m.params[1]:+.3f} "
              f"p={m.pvalues[1]:.4f} (per-nat odds ratio {np.exp(m.params[1]):.3f}) n={len(df)}")
except Exception as e:
    print("  statsmodels unavailable:", e)

print()
print("[1c] probe vs crossed_50 (coupling-free binary outcome)")
for tag, df in [("ALL seeds", seed), ("NON-JUNCO seeds", nj_seed)]:
    auc, p, n1, n0 = auc_mw(df.probe, df.crossed_50.to_numpy())
    rpb, ppb = st.pointbiserialr(df.crossed_50.astype(int), df.probe)
    print(f"  {tag}: AUC(probe -> crossed)={auc:.3f} MW-p={p:.4f} "
          f"point-biserial r={rpb:+.3f} p={ppb:.4f} (crossed {n1}/{n1+n0})")
for tag, df in [("ALL cells", cell), ("NON-JUNCO cells", nj)]:
    auc, p, n1, n0 = auc_mw(df.probe, (df.crossed_50_frac > 0).to_numpy())
    print(f"  {tag}: AUC(probe -> any-cross)={auc:.3f} MW-p={p:.4f} (crossed {n1}/{n1+n0})")

print()
print("=" * 100)
print("ATTACK 2 — CONFOUNDS")
print("=" * 100)
print("[2a] per-anchor-class Spearman (cell level), probe vs outcome:")
for a, g in cell.groupby("anchor"):
    r1 = st.spearmanr(g.probe, g.dex_eroded_50)
    r2 = st.spearmanr(g.probe, g.log_min50)
    print(f"  anchor={a:16s} n={len(g):2d}  dex50: rho={r1.statistic:+.3f} p={r1.pvalue:.3f} | "
          f"log10(min50): rho={r2.statistic:+.3f} p={r2.pvalue:.3f}")
print("  pooled within-anchor (residualised on anchor-class means, cell level):")
for ycol in ["dex_eroded_50", "log_min50"]:
    d = cell.copy()
    d["px"] = d.probe - d.groupby("anchor").probe.transform("mean")
    d["py"] = d[ycol] - d.groupby("anchor")[ycol].transform("mean")
    sp(d.px, d.py, f"  partial (anchor demeaned): probe vs {ycol}")

print()
print("[2b] level composition:")
sp(cell.probe, cell["lt"], "probe vs level_target (cell)")
sp(cell.probe, cell["la"], "probe vs level_anchor (cell)")
sp(cell["lt"], cell.log_min50, "level_target vs log10(min50) (cell)")
diag = cell[cell["la"] == cell["lt"]]
sp(diag.probe, diag.dex_eroded_50, f"diagonal-only cells (la==lt): probe vs dex50")
sp(diag.probe, diag.log_min50, f"diagonal-only cells: probe vs log10(min50)")
diag_nj = diag[diag.anchor != "junco"]
sp(diag_nj.probe, diag_nj.log_min50, f"diagonal-only NON-JUNCO: probe vs log10(min50)")
# partial out levels
d = cell.copy()
for c in ["probe", "dex_eroded_50", "log_min50"]:
    d[c + "_r"] = d[c] - d.groupby(["la", "lt"])[c].transform("mean")
sp(d.probe_r, d.dex_eroded_50_r, "partial (level-pair demeaned): probe vs dex50")
sp(d.probe_r, d.log_min50_r, "partial (level-pair demeaned): probe vs log10(min50)")

print()
print("[2c] two-cluster / wall-detector check:")
stuckc = cell[cell.stuck_any]
nons = cell[~cell.stuck_any]
print(f"  stuck cells n={len(stuckc)}, non-stuck cells n={len(nons)}")
sp(nons.probe, nons.dex_eroded_50, "NON-STUCK cells only: probe vs dex50")
sp(nons.probe, nons.log_min50, "NON-STUCK cells only: probe vs log10(min50)")
nons_nj = nons[nons.anchor != "junco"]
sp(nons_nj.probe, nons_nj.log_min50, "NON-STUCK NON-JUNCO: probe vs log10(min50)")
cr_nons = nons.dropna(subset=["gen_first_cross_mean"])
sp(cr_nons.probe, cr_nons.gen_first_cross_mean, "NON-STUCK cells: probe vs gen_first_cross")
sp(stuckc.probe, stuckc.dex_eroded_50, "STUCK cells only: probe vs dex50")
sp(stuckc.probe, stuckc.log_min50, "STUCK cells only: probe vs log10(min50)")
print("  by stratum (probe vs log10(min50)):")
for s, g in cell.groupby("stratum"):
    r = st.spearmanr(g.probe, g.log_min50)
    print(f"    {s:14s} n={len(g):2d} rho={r.statistic:+.3f} p={r.pvalue:.3f}")
wb_cb = cell[cell.stratum != "wall_repl"]
sp(wb_cb.probe, wb_cb.log_min50, "EXCLUDING wall_repl stratum: probe vs log10(min50)")
sp(wb_cb.probe, wb_cb.dex_eroded_50, "EXCLUDING wall_repl stratum: probe vs dex50")

print()
print("[P2-grade] graded vs step: cells with probe<3 nats only:")
low = cell[cell.probe < 3.0]
low_nj = low[low.anchor != "junco"]
sp(low.probe, low.log_min50, f"probe<3 cells: probe vs log10(min50)")
sp(low.probe, low.dex_eroded_50, f"probe<3 cells: probe vs dex50")
sp(low_nj.probe, low_nj.log_min50, f"probe<3 NON-JUNCO: probe vs log10(min50)")
hi = cell[cell.probe >= 3.0]
print(f"  P2 table: probe>=3: {int((hi.crossed_50_frac>0).sum())}/{len(hi)} cells crossed; "
      f"probe<3: {int((low.crossed_50_frac>0).sum())}/{len(low)} crossed")

print()
print("=" * 100)
print("ATTACK 3 — PROBE VALIDITY")
print("=" * 100)
print("[3a] probs are PAIR-RENORMALIZED (p_a+p_b=1) => TgtBal = |logit(p_a)|;")
print("     probe is a deterministic monotone transform of median p_class_a:")
r, p = st.spearmanr(seed.probe, seed.pop0_p_class_a_median)
print(f"  seed-level Spearman probe vs pop0_p_class_a_median: rho={r:+.4f} p={p:.2e}")
ident = np.abs(seed.probe - np.abs(np.log(seed.pop0_p_class_a_median)
                                   - np.log(1 - seed.pop0_p_class_a_median)))
print(f"  max |probe - |logit(median p_a)|| = {ident.max():.4f} "
      f"(median {ident.median():.4f}) -> near-identity (median/logit commute approx)")
print("  => probe IS pair-renormalized anchor-option confidence; BUT it is pair-specific:")
spread = seed.groupby("anchor").probe.agg(["min", "max", "median", "count"])
spread["range"] = spread["max"] - spread["min"]
print(spread.round(2).to_string())
print("  -> same anchor class spans wide probe range across targets; if probe were pure")
print("     anchor-image confidence it would be ~constant per anchor class.")
print(f"  roster_min_anchor_confidence=3.5 applies to ORIGIN seed; gen-0 probe<3.5 in "
      f"{int((seed.probe<3.5).sum())}/46 runs (perturbation erodes margin), no clip at 3.5 visible.")
print("[3b] option order: anchor word == class_a == first prompt option in 46/46 runs.")
print("     Order is perfectly confounded with anchor role; order bias untestable in-design.")

print()
print("=" * 100)
print("ATTACK 4 — BUDGET TRUNCATION (are stuck runs merely slow?)")
print("=" * 100)
stk = seed[seed.stuck].copy()
print(f"  stuck runs: {len(stk)}/46")
imp = stk[stk.slope_raw_last10 < -1e-6]
print(f"  best-so-far still improving over gens 40-49 (raw slope < -1e-6): {len(imp)}/{len(stk)}")
print(f"  last_improve_gen distribution (stuck runs): "
      f"median {stk.last_improve_gen.median():.0f}, "
      f">=40: {(stk.last_improve_gen>=40).sum()}, "
      f"30-39: {((stk.last_improve_gen>=30)&(stk.last_improve_gen<40)).sum()}, "
      f"<30: {(stk.last_improve_gen<30).sum()}")
# extrapolation to gen 200 (EXTRAPOLATION, clearly labeled)
lin_cross = ((stk.bsf_49 + stk.slope_raw_last10 * (200 - 49)) <= 0.1) & (stk.slope_raw_last10 < 0)
log_cross = ((np.log10(stk.bsf_49) + stk.slope_log10_last10 * (200 - 49)) <= -1.0) \
    & (stk.slope_log10_last10 < 0)
print(f"  EXTRAPOLATION to gen 200 (last-10-gen slope held constant):")
print(f"    linear-in-TgtBal:   {int(lin_cross.sum())}/{len(stk)} stuck runs would reach <=0.1")
print(f"    linear-in-log10:    {int(log_cross.sum())}/{len(stk)} stuck runs would reach <=0.1")
print(f"  stuck runs with bsf_49 within 2x of wall (0.1<bsf<=0.2): "
      f"{int(((stk.bsf_49>0.1)&(stk.bsf_49<=0.2)).sum())}")
print(f"  stuck-run bsf_49: min {stk.bsf_49.min():.3f} median {stk.bsf_49.median():.3f} "
      f"max {stk.bsf_49.max():.3f}")
hi_imp = stk.sort_values("slope_log10_last10").head(8)[
    ["seed_idx", "anchor", "target", "la", "lt", "probe", "bsf_49",
     "slope_raw_last10", "slope_log10_last10", "last_improve_gen"]]
print("  steepest still-improving stuck runs (log10 slope):")
print(hi_imp.round(4).to_string(index=False))

print()
print("=" * 100)
print("ATTACK 5 — TERCILE CLAIM REPLICATION")
print("=" * 100)
for tag, df in [("ALL 46 runs", seed), ("NON-JUNCO 35 runs", nj_seed)]:
    df = df.copy()
    terc = df.probe.quantile([1 / 3, 2 / 3]).to_numpy()
    df["terc"] = np.digitize(df.probe, terc)  # 0,1,2
    n_stuck = int(df.stuck.sum())
    top = df[df.terc == 2]
    stuck_in_top = int((df.stuck & (df.terc == 2)).sum())
    n_top = len(top)
    # hypergeometric: P(X >= stuck_in_top) drawing n_top from population with n_stuck successes
    pval = st.hypergeom.sf(stuck_in_top - 1, len(df), n_stuck, n_top)
    print(f"  {tag}: stuck={n_stuck}/{len(df)} (base rate {n_stuck/len(df):.2f}); "
          f"top tercile n={n_top}, stuck in top tercile {stuck_in_top}/{n_top} "
          f"({stuck_in_top/n_top:.2f}); stuck captured by top tercile {stuck_in_top}/{n_stuck}")
    print(f"    'all stuck in top tercile' replicates: "
          f"{'YES' if stuck_in_top==n_stuck else 'NO'} (arithmetically "
          f"{'possible' if n_stuck<=n_top else f'IMPOSSIBLE: {n_stuck} stuck > {n_top} top-tercile slots'})")
    print(f"    hypergeom P(>= {stuck_in_top} stuck in top tercile) = {pval:.4f}")
    by_t = df.groupby("terc").agg(n=("stuck", "size"), stuck=("stuck", "sum"),
                                  probe_lo=("probe", "min"), probe_hi=("probe", "max"))
    print(by_t.round(2).to_string())

print()
print("AUC comparison: probe vs alternative trivial predictors of NOT-crossing (seed level):")
for col in ["probe", "pop0_p_class_a_median", "min_gen0"]:
    auc, p, _, _ = auc_mw(seed[col], (~seed.crossed_50).to_numpy())
    print(f"  AUC({col} -> not crossed) = {auc:.3f} (MW p={p:.4f})")

# scatter artifact
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for a, (ycol, ylab) in zip(ax, [("log_min50", "log10(min TgtBal @50)"),
                                ("log_searchmin", "log10(min TgtBal gens 1-49)")]):
    for stuckval, c, m in [(True, "crimson", "s"), (False, "steelblue", "o")]:
        g = cell[cell.stuck_any == stuckval]
        a.scatter(g.probe, g[ycol], c=c, marker=m, s=38,
                  label=f"stuck_any={stuckval} (n={len(g)})", alpha=.8)
    j = cell[cell.anchor == "junco"]
    a.scatter(j.probe, j[ycol], facecolors="none", edgecolors="k", s=90,
              label="junco anchor")
    a.axhline(-1, ls=":", c="gray")
    a.axvline(3, ls=":", c="gray")
    a.set_xlabel("probe (gen-0 median TgtBal, nats)")
    a.set_ylabel(ylab)
    a.legend(fontsize=7)
fig.suptitle("Exp-101 skeptic: probe vs decoupled outcomes (cell level)")
fig.tight_layout()
fig.savefig(f"{OUT}/skeptic_probe_vs_decoupled_outcomes.png", dpi=140)
print("\nwrote skeptic_probe_vs_decoupled_outcomes.png")
