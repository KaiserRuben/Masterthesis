"""Exp-100 PoC boundary_pair partial-slice analysis (junco anchor, 119 seeds).
Tasks 1-3 + 5(aggregate part): per-cell summary, H1-H4, asymmetry, variance decomposition.
"""
import numpy as np
import pandas as pd

pd.set_option('display.width', 300)
OUT = '/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/exp100_partial'

df = pd.read_parquet('/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/exp100_poc_aggregate.parquet')
d = df[df['run'] == 'poc_boundary_pair'].copy()
d['target'] = d['target_class_concrete']
d['la'] = d['level_anchor']
d['lt'] = d['level_target']
d['cell'] = list(zip(d['target'], d['la'], d['lt']))
d['log_mtb'] = np.log10(d['min_TgtBal'].clip(lower=1e-9))
TARGETS = ['ostrich', 'green iguana', 'boa constrictor', 'cello', 'marimba']
ANIMAL = ['green iguana', 'boa constrictor']
ARTIFACT = ['cello', 'marimba']

print('=== global min_TgtBal stats ===')
print(d['min_TgtBal'].describe().to_string())
print('min positive value:', d['min_TgtBal'].min())
print('final_pareto_min_TgtBal == min_TgtBal for all?', (d['final_pareto_min_TgtBal'] == d['min_TgtBal']).all())

# ---------- Task 1: per-cell summary ----------
g = d.groupby(['target', 'la', 'lt'])
cell = g.agg(
    n=('min_TgtBal', 'size'),
    med_mtb=('min_TgtBal', 'median'),
    min_mtb=('min_TgtBal', 'min'),
    max_mtb=('min_TgtBal', 'max'),
    med_gen=('min_TgtBal_at_gen', 'median'),
    med_dimg=('d_img_at_min_TgtBal', 'median'),
    med_dtext=('d_text_at_min_TgtBal', 'median'),
).reset_index()
cell['range_mtb'] = cell['max_mtb'] - cell['min_mtb']
cell['log_range'] = np.log10(cell['max_mtb'].clip(lower=1e-9)) - np.log10(cell['min_mtb'].clip(lower=1e-9))
cell['target'] = pd.Categorical(cell['target'], TARGETS)
cell = cell.sort_values(['target', 'la', 'lt'])
cell.to_csv(f'{OUT}/per_cell_summary.csv', index=False)
print('\n=== Task 1: per-cell summary (saved per_cell_summary.csv) ===')
print(cell.to_string(index=False, float_format=lambda x: f'{x:.5g}'))


def cliffs(x, y):
    """Cliff's delta: P(x>y) - P(x<y). Positive => x stochastically larger."""
    x = np.asarray(x)[:, None]
    y = np.asarray(y)[None, :]
    return float(((x > y).sum() - (x < y).sum()) / (x.size if False else (x.shape[0] * y.shape[1])))


def summ(s):
    q1, q2, q3 = np.percentile(s, [25, 50, 75])
    return f'n={len(s)} median={q2:.5g} IQR=[{q1:.5g},{q3:.5g}]'


# ---------- H1: within (ostrich) vs across ----------
print('\n=== H1: within-bucket (junco-ostrich, c=2) vs across (c=None) ===')
w = d[d['cell_kind'] == 'within']
a = d[d['cell_kind'] == 'across']
# fair comparison: restrict across to the cells ostrich has: la,lt in {0,1}
a_restr = a[(a['la'] <= 1) & (a['lt'] <= 1)]
for name, col in [('min_TgtBal', 'min_TgtBal'), ('min_TgtBal_at_gen', 'min_TgtBal_at_gen')]:
    print(f'-- {name} --')
    print('  within        :', summ(w[col]))
    print('  across (all)  :', summ(a[col]))
    print('  across (la,lt<=1):', summ(a_restr[col]))
    print(f"  Cliff's delta within vs across(all):   {cliffs(w[col], a[col]):+.3f}")
    print(f"  Cliff's delta within vs across(restr): {cliffs(w[col], a_restr[col]):+.3f}")

# ---------- H2: animal vs artifact targets (across only) ----------
print('\n=== H2: across-bucket animal vs artifact targets ===')
an = a[a['target'].isin(ANIMAL)]
ar = a[a['target'].isin(ARTIFACT)]
for col in ['min_TgtBal', 'min_TgtBal_at_gen']:
    print(f'-- {col} --')
    print('  animal  :', summ(an[col]))
    print('  artifact:', summ(ar[col]))
    print(f"  Cliff's delta animal vs artifact: {cliffs(an[col], ar[col]):+.3f}")
print('per-target medians (min_TgtBal):')
print(a.groupby('target')['min_TgtBal'].median().reindex(TARGETS[1:]).to_string(float_format=lambda x: f'{x:.5g}'))

# ---------- H3: abstraction on diagonal ----------
print('\n=== H3: diagonal (la==lt) abstraction trend ===')
diag = d[d['la'] == d['lt']].copy()
piv = diag.groupby(['target', 'la'])['min_TgtBal'].median().unstack()
piv = piv.reindex(TARGETS)
print('median min_TgtBal per target x diagonal level:')
print(piv.to_string(float_format=lambda x: f'{x:.5g}'))
pooled = diag[diag['cell_kind'] == 'across'].groupby('la')['min_TgtBal'].agg(['median', 'count'])
print('pooled across-targets diag medians:\n', pooled.to_string(float_format=lambda x: f'{x:.5g}'))
from scipy import stats as ss
da = diag[diag['cell_kind'] == 'across']
rho, p = ss.spearmanr(da['la'], da['log_mtb'])
print(f'Spearman(level, log10 min_TgtBal), across-diag pooled (n={len(da)}): rho={rho:+.3f} (p={p:.3f}, descriptive only)')
for t in TARGETS[1:]:
    sub = da[da['target'] == t]
    rho_t, _ = ss.spearmanr(sub['la'], sub['log_mtb'])
    print(f'  {t:16s} rho={rho_t:+.3f} (n={len(sub)})')
# pairwise level deltas
for l0, l1 in [(0, 1), (1, 2), (0, 2)]:
    x = da[da['la'] == l1]['min_TgtBal']
    y = da[da['la'] == l0]['min_TgtBal']
    print(f"  Cliff's delta diag level {l1} vs {l0}: {cliffs(x, y):+.3f}")

# ---------- H4 partial: abstraction trend animal vs artifact ----------
print('\n=== H4 partial: diagonal trend by subkind ===')
for grp, ts in [('animal', ANIMAL), ('artifact', ARTIFACT)]:
    sub = da[da['target'].isin(ts)]
    med = sub.groupby('la')['min_TgtBal'].median()
    rho_g, _ = ss.spearmanr(sub['la'], sub['log_mtb'])
    print(f'{grp:8s}: medians L0={med.get(0, np.nan):.5g} L1={med.get(1, np.nan):.5g} L2={med.get(2, np.nan):.5g}  rho={rho_g:+.3f} (n={len(sub)})')

# ---------- off-diagonal asymmetry ----------
print('\n=== Off-diagonal asymmetry (across targets only) ===')
off = a[a['la'] != a['lt']].copy()
up_anchor = off[off['la'] > off['lt']]   # anchor label more abstract
up_target = off[off['la'] < off['lt']]   # target label more abstract
print('  la>lt (anchor more abstract):', summ(up_anchor['min_TgtBal']))
print('  la<lt (target more abstract):', summ(up_target['min_TgtBal']))
print(f"  Cliff's delta (la>lt) vs (la<lt): {cliffs(up_anchor['min_TgtBal'], up_target['min_TgtBal']):+.3f}")
print('marginal medians of min_TgtBal over across-targets:')
print('  by la:', a.groupby('la')['min_TgtBal'].median().round(6).to_dict())
print('  by lt:', a.groupby('lt')['min_TgtBal'].median().round(6).to_dict())
print('full 3x3 median heatmap values per target:')
for t in TARGETS:
    sub = d[d['target'] == t]
    pv = sub.groupby(['la', 'lt'])['min_TgtBal'].median().unstack()
    print(f'-- {t} --')
    print(pv.to_string(float_format=lambda x: f'{x:.5g}'))

# ---------- Task 3: variance decomposition ----------
print('\n=== Task 3: seed-level vs cell-level variance (log10 min_TgtBal) ===')
y = d['log_mtb']
grand = y.mean()
groups = d.groupby('cell')['log_mtb']
ssb = sum(len(g_) * (g_.mean() - grand) ** 2 for _, g_ in groups)
ssw = sum(((g_ - g_.mean()) ** 2).sum() for _, g_ in groups)
k = groups.ngroups
N = len(d)
msb = ssb / (k - 1)
msw = ssw / (N - k)
nbar = (N - sum(len(g_) ** 2 for _, g_ in groups) / N) / (k - 1)  # ANOVA n0
icc = (msb - msw) / (msb + (nbar - 1) * msw)
print(f'cells k={k}, N={N}, SS_between={ssb:.2f} ({ssb/(ssb+ssw)*100:.1f}%), SS_within={ssw:.2f} ({ssw/(ssb+ssw)*100:.1f}%)')
print(f'MS_between={msb:.3f}, MS_within={msw:.3f}, F={msb/msw:.2f}, ICC(1)={icc:.3f}')
wr = cell['log_range']
cm = np.log10(cell['med_mtb'].clip(lower=1e-9))
print(f'median within-cell log10-range: {wr.median():.2f} dex (IQR [{wr.quantile(.25):.2f},{wr.quantile(.75):.2f}], max {wr.max():.2f})')
print(f'between-cell spread of cell medians: {cm.max() - cm.min():.2f} dex (IQR of cell medians {cm.quantile(.75)-cm.quantile(.25):.2f} dex)')
print(f'cells whose 3 seeds span >1 dex: {(wr > 1).sum()}/{k}; >2 dex: {(wr > 2).sum()}/{k}')
# bimodality check: how many seeds never get below 0.1?
print(f"seeds with min_TgtBal > 0.1 ('failed to approach boundary'): {(d['min_TgtBal'] > 0.1).sum()}/{N}")
print(d[d['min_TgtBal'] > 0.1][['seed_dir', 'target', 'la', 'lt', 'min_TgtBal', 'anchor_min_p_gap']].to_string(index=False))

# ---------- Task 5 (aggregate part): when do improvements stop ----------
print('\n=== min_TgtBal_at_gen distribution (last improvement generation) ===')
print(d['min_TgtBal_at_gen'].describe().to_string())
print('seeds with last improvement after gen 150:', (d['min_TgtBal_at_gen'] > 150).sum(), '/', N)
print('seeds with last improvement after gen 180:', (d['min_TgtBal_at_gen'] > 180).sum(), '/', N)

# (1,1) anomaly check
print('\n=== (1,1) cells vs other cells ===')
c11 = d[(d['la'] == 1) & (d['lt'] == 1)]
rest = d[~((d['la'] == 1) & (d['lt'] == 1))]
print('  (1,1)  :', summ(c11['min_TgtBal']))
print('  others :', summ(rest['min_TgtBal']))
print(f"  Cliff's delta (1,1) vs rest: {cliffs(c11['min_TgtBal'], rest['min_TgtBal']):+.3f}")
print('  per-target (1,1) medians:', c11.groupby('target')['min_TgtBal'].median().round(6).to_dict())
print('  prompt labels at (1,1): anchor=songbird; targets:',
      c11[['target', 'target_label_in_prompt']].drop_duplicates().to_dict('records'))
# all level-1 usages
l1a = d[d['la'] == 1]
l1t = d[d['lt'] == 1]
print('  median when la==1 (anchor=songbird):', f"{l1a['min_TgtBal'].median():.6f}", 'n=', len(l1a))
print('  median when lt==1:', f"{l1t['min_TgtBal'].median():.6f}", 'n=', len(l1t))
print('  median when la!=1 and lt!=1:', f"{d[(d['la'] != 1) & (d['lt'] != 1)]['min_TgtBal'].median():.6f}")
