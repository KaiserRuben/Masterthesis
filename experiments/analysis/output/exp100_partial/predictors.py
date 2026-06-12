"""Quick predictors of final hardness: initial TgtBal (gen0) and preflight anchor_min_p_gap."""
import numpy as np
import pandas as pd
from scipy import stats as ss

OUT = '/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/exp100_partial'
cv = pd.read_csv(f'{OUT}/per_seed_convergence_stats.csv')
df = pd.read_parquet('/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/exp100_poc_aggregate.parquet')
d = df[df['run'] == 'poc_boundary_pair'][['seed_dir', 'anchor_min_p_gap']]
m = cv.merge(d, on='seed_dir')
m['log_final'] = np.log10(m['min_TgtBal'].clip(lower=1e-9))
m['log_tb0'] = np.log10(m['tb0'].clip(lower=1e-9))
rho1, p1 = ss.spearmanr(m['log_tb0'], m['log_final'])
rho2, p2 = ss.spearmanr(m['anchor_min_p_gap'], m['log_final'])
print(f'Spearman(log10 tb0 [gen0 TgtBal], log10 final min_TgtBal): rho={rho1:+.3f} (p={p1:.2g})')
print(f'Spearman(anchor_min_p_gap [preflight], log10 final):       rho={rho2:+.3f} (p={p2:.2g})')
print('tb0 by stuck status:')
print(m.groupby(m['min_TgtBal'] > 0.1)['tb0'].describe()[['count', '25%', '50%', '75%']].to_string())
# stuck rate by tb0 tercile
m['tb0_tercile'] = pd.qcut(m['tb0'], 3, labels=['low', 'mid', 'high'])
print('stuck rate by tb0 tercile:')
print(m.groupby('tb0_tercile', observed=True).apply(lambda g: f"{(g['min_TgtBal'] > 0.1).sum()}/{len(g)}", include_groups=False).to_string())
