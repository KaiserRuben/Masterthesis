"""Supplement: log-scale convergence timing + magnitude of late-generation gains."""
import os
import numpy as np
import pandas as pd

RUNS = '/Users/kaiser/Projects/Masterarbeit/runs/Exp-100/poc_boundary_pair'
df = pd.read_parquet('/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/exp100_poc_aggregate.parquet')
d = df[df['run'] == 'poc_boundary_pair'].copy()
d['target'] = d['target_class_concrete']

rows = []
for _, r in d.iterrows():
    c = pd.read_parquet(os.path.join(RUNS, r['seed_dir'], 'evolutionary', 'convergence.parquet'))
    tb = c['pareto_min_TgtBal'].clip(lower=1e-9).to_numpy()
    l0, lf = np.log10(tb[0]), np.log10(tb[-1])
    if l0 - lf > 0:
        thr90 = lf + 0.1 * (l0 - lf)
        g90log = int(np.argmax(np.log10(tb) <= thr90))
    else:
        g90log = 0
    ratio_150_199 = tb[150] / tb[-1]
    ratio_100_199 = tb[100] / tb[-1]
    rows.append(dict(seed_dir=r['seed_dir'], target=r['target'],
                     la=r['level_anchor'], lt=r['level_target'], mtb=r['min_TgtBal'],
                     g90log=g90log, ratio_150_199=ratio_150_199, ratio_100_199=ratio_100_199,
                     dex_gain_after_150=np.log10(ratio_150_199), dex_gain_after_100=np.log10(ratio_100_199)))
s = pd.DataFrame(rows)
s['stuck'] = s['mtb'] > 0.1
print('log-scale gen90 (90% of log10 improvement):')
print('  pooled median=%d IQR=[%d,%d], >150: %d/119, >180: %d/119' % (
    s['g90log'].median(), s['g90log'].quantile(.25), s['g90log'].quantile(.75),
    (s['g90log'] > 150).sum(), (s['g90log'] > 180).sum()))
print('improvement still gained after gen 100 (dex = orders of magnitude):')
print('  median %.2f dex, IQR [%.2f,%.2f], seeds gaining >0.5 dex after gen100: %d/119' % (
    s['dex_gain_after_100'].median(), s['dex_gain_after_100'].quantile(.25),
    s['dex_gain_after_100'].quantile(.75), (s['dex_gain_after_100'] > 0.5).sum()))
print('improvement still gained after gen 150:')
print('  median %.2f dex, IQR [%.2f,%.2f], seeds gaining >0.5 dex after gen150: %d/119, >1 dex: %d' % (
    s['dex_gain_after_150'].median(), s['dex_gain_after_150'].quantile(.25),
    s['dex_gain_after_150'].quantile(.75), (s['dex_gain_after_150'] > 0.5).sum(),
    (s['dex_gain_after_150'] > 1).sum()))
print('\nby stuck status, median dex gain after gen 100 / 150:')
print(s.groupby('stuck')[['dex_gain_after_100', 'dex_gain_after_150', 'g90log']].median().to_string())
print('\nstuck seeds: did any move at all in last 50 gens? ratio_150_199 stats:')
print(s[s['stuck']][['seed_dir', 'target', 'la', 'lt', 'mtb', 'ratio_150_199']].to_string(index=False))
