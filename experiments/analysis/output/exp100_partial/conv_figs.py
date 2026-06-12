"""Exp-100 partial: convergence-shape analysis (Task 4-5) + figures (Task 6)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

OUT = '/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/exp100_partial'
RUNS = '/Users/kaiser/Projects/Masterarbeit/runs/Exp-100/poc_boundary_pair'
TARGETS = ['ostrich', 'green iguana', 'boa constrictor', 'cello', 'marimba']
FLOOR = 1e-9

df = pd.read_parquet('/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/exp100_poc_aggregate.parquet')
d = df[df['run'] == 'poc_boundary_pair'].copy()
d['target'] = d['target_class_concrete']
d['la'] = d['level_anchor']
d['lt'] = d['level_target']

# ---------- load all convergence curves ----------
curves = {}
rows = []
for _, r in d.iterrows():
    p = os.path.join(RUNS, r['seed_dir'], 'evolutionary', 'convergence.parquet')
    c = pd.read_parquet(p)
    tb = c['pareto_min_TgtBal'].to_numpy()
    curves[r['seed_dir']] = c
    c0, cf = tb[0], tb[-1]
    imp = c0 - cf
    def first_gen(frac):
        if imp <= 0:
            return 0
        thresh = cf + (1 - frac) * imp
        return int(np.argmax(tb <= thresh))
    g50, g90, g99 = first_gen(.5), first_gen(.9), first_gen(.99)
    frac100 = (c0 - tb[100]) / imp if imp > 0 else 1.0
    rows.append(dict(seed_dir=r['seed_dir'], target=r['target'], la=r['la'], lt=r['lt'],
                     cell_kind=r['cell_kind'], cross_subkind=r['cross_subkind'],
                     min_TgtBal=r['min_TgtBal'], tb0=c0, tb_final=cf,
                     gen50=g50, gen90=g90, gen99=g99, frac_imp_by_gen100=frac100,
                     last_improve_gen=r['min_TgtBal_at_gen']))
cv = pd.DataFrame(rows)
cv['stuck'] = cv['min_TgtBal'] > 0.1
cv['is11'] = (cv['la'] == 1) & (cv['lt'] == 1)

print('=== Task 5: budget sizing (gen at X% of final TgtBal improvement) ===')
print('pooled: gen50 med=%d  gen90 med=%d IQR=[%d,%d]  gen99 med=%d  frac_by_gen100 med=%.3f' % (
    cv['gen50'].median(), cv['gen90'].median(), cv['gen90'].quantile(.25), cv['gen90'].quantile(.75),
    cv['gen99'].median(), cv['frac_imp_by_gen100'].median()))


def grp_summ(name, sub):
    print(f'{name:28s} n={len(sub):3d}  gen90 med={sub["gen90"].median():5.0f} IQR=[{sub["gen90"].quantile(.25):.0f},{sub["gen90"].quantile(.75):.0f}]  '
          f'gen99 med={sub["gen99"].median():5.0f}  last_improve med={sub["last_improve_gen"].median():5.0f}  '
          f'gen90>150: {(sub["gen90"] > 150).sum()}')


grp_summ('within (ostrich)', cv[cv['cell_kind'] == 'within'])
grp_summ('across animal', cv[cv['cross_subkind'] == 'animal-animal'])
grp_summ('across artifact', cv[cv['cross_subkind'] == 'animal-artifact'])
grp_summ('(1,1) cells', cv[cv['is11']])
grp_summ('stuck (min_TgtBal>0.1)', cv[cv['stuck']])
grp_summ('converged (<=0.1)', cv[~cv['stuck']])
print('seeds with gen90 > 180:', (cv['gen90'] > 180).sum(), '/', len(cv))
print('seeds with gen99 > 180:', (cv['gen99'] > 180).sum(), '/', len(cv))
cv.to_csv(f'{OUT}/per_seed_convergence_stats.csv', index=False)

# ---------- Task 4: representative cells ----------
cellmed = d.groupby(['target', 'la', 'lt'])['min_TgtBal'].median()
easiest = cellmed.idxmin()
hardest = cellmed.idxmax()
print('\n=== Task 4: representative cells ===')
print('easiest cell:', easiest, 'median', f'{cellmed.min():.2e}')
print('hardest cell:', hardest, 'median', f'{cellmed.max():.2e}')
rep_cells = [
    ('easiest: ' + f'{easiest[0]} ({easiest[1]},{easiest[2]})', easiest, 'tab:green'),
    ('hardest: ' + f'{hardest[0]} ({hardest[1]},{hardest[2]})', hardest, 'tab:red'),
    ('boa constrictor (1,1)', ('boa constrictor', 1, 1), 'tab:purple'),
    ('cello (1,1)', ('cello', 1, 1), 'tab:orange'),
    ('green iguana (1,1)', ('green iguana', 1, 1), 'tab:blue'),
]
for label, (t, la, lt), _c in rep_cells:
    sel = d[(d['target'] == t) & (d['la'] == la) & (d['lt'] == lt)]
    print(f'-- {label} --')
    for _, r in sel.iterrows():
        c = curves[r['seed_dir']]
        tb = c['pareto_min_TgtBal'].to_numpy()
        bi = int(np.argmax(tb <= tb[-1] + 1e-15))  # first gen reaching final value
        di = c['pareto_atbest_TgtBal_MatrixDistance_fro'].iloc[-1]
        dt = c['pareto_atbest_TgtBal_TextDist'].iloc[-1]
        print(f"   {r['seed_dir']}: tb0={tb[0]:.3f} final={tb[-1]:.2e} reached_at_gen={bi} "
              f"d_img@best={di:.4f} d_text@best={dt:.3f}")

# drift paid at best point, by group
print('\ndrift at best-TgtBal point (medians):')
for name, sub in [('all', d), ('stuck', d[d['min_TgtBal'] > 0.1]), ('converged', d[d['min_TgtBal'] <= 0.1])]:
    print(f'  {name:10s} d_img={sub["d_img_at_min_TgtBal"].median():.4f}  d_text={sub["d_text_at_min_TgtBal"].median():.3f}  n={len(sub)}')

# ============ FIGURES ============
plt.rcParams.update({'figure.dpi': 130, 'font.size': 9})

# --- Fig 1: heatmaps median min_TgtBal per target ---
fig, axes = plt.subplots(1, 5, figsize=(16, 3.4))
vmin, vmax = 1e-5, 1.0
for ax, t in zip(axes, TARGETS):
    sub = d[d['target'] == t]
    pv = sub.groupby(['la', 'lt'])['min_TgtBal'].median().unstack()
    im = ax.imshow(pv.to_numpy(), norm=LogNorm(vmin=vmin, vmax=vmax), cmap='viridis_r', origin='lower')
    for i in pv.index:
        for j in pv.columns:
            v = pv.loc[i, j]
            ax.text(j, i, f'{v:.1e}', ha='center', va='center',
                    color='white' if v > 3e-3 else 'black', fontsize=7)
    ax.set_xticks(range(len(pv.columns)), pv.columns)
    ax.set_yticks(range(len(pv.index)), pv.index)
    ax.set_xlabel('level_target')
    if t == TARGETS[0]:
        ax.set_ylabel('level_anchor')
    kind = 'within' if t == 'ostrich' else 'across'
    ax.set_title(f'{t}\n({kind})', fontsize=9)
fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01, label='median min TgtBal (log)')
fig.suptitle('Exp-100 partial (anchor=junco): median min_TgtBal per cell — lower = closer to boundary', y=1.04)
fig.savefig(f'{OUT}/evo_heatmap_median_min_TgtBal.png', bbox_inches='tight')
plt.close(fig)

# --- Fig 2: diagonal abstraction boxplot (H3/H4) ---
diag = d[d['la'] == d['lt']].copy()
diag['subkind'] = np.where(diag['cell_kind'] == 'within', 'within (ostrich)',
                           np.where(diag['target'].isin(['green iguana', 'boa constrictor']),
                                    'across animal', 'across artifact'))
fig, ax = plt.subplots(figsize=(7, 4.2))
groups = ['within (ostrich)', 'across animal', 'across artifact']
colors = {'within (ostrich)': 'tab:gray', 'across animal': 'tab:blue', 'across artifact': 'tab:orange'}
width = 0.25
pos_map = {}
for gi, gname in enumerate(groups):
    for lv in [0, 1, 2]:
        vals = diag[(diag['subkind'] == gname) & (diag['la'] == lv)]['min_TgtBal'].clip(lower=1e-7)
        if len(vals) == 0:
            continue
        x = lv + (gi - 1) * width
        bp = ax.boxplot([vals], positions=[x], widths=width * 0.85, patch_artist=True,
                        showfliers=False, medianprops=dict(color='black'))
        bp['boxes'][0].set_facecolor(colors[gname])
        bp['boxes'][0].set_alpha(0.6)
        ax.scatter(np.full(len(vals), x) + np.random.default_rng(0).uniform(-0.04, 0.04, len(vals)),
                   vals, s=14, color=colors[gname], edgecolor='k', linewidth=0.4, zorder=3)
ax.set_yscale('log')
ax.set_xticks([0, 1, 2], ['level 0\n(fine)', 'level 1\n(mid)', 'level 2\n(super)'])
ax.set_ylabel('min TgtBal (log)')
ax.set_title('H3/H4: diagonal cells (la==lt) — abstraction level vs boundary proximity')
handles = [plt.Rectangle((0, 0), 1, 1, fc=colors[g], alpha=0.6) for g in groups]
ax.legend(handles, groups, loc='upper right', fontsize=8)
ax.axhline(0.1, color='red', ls=':', lw=0.8)
ax.text(2.35, 0.12, 'stuck threshold', color='red', fontsize=7)
fig.tight_layout()
fig.savefig(f'{OUT}/evo_diag_abstraction_boxplot.png', bbox_inches='tight')
plt.close(fig)

# --- Fig 3: convergence overlays for representative cells ---
fig, ax = plt.subplots(figsize=(8, 4.8))
for label, (t, la, lt), col in rep_cells:
    sel = d[(d['target'] == t) & (d['la'] == la) & (d['lt'] == lt)]
    for k, (_, r) in enumerate(sel.iterrows()):
        tb = curves[r['seed_dir']]['pareto_min_TgtBal'].clip(lower=1e-7)
        ax.plot(tb.index, tb, color=col, alpha=0.85, lw=1.1, label=label if k == 0 else None)
ax.set_yscale('log')
ax.set_xlabel('generation')
ax.set_ylabel('pareto min TgtBal (log)')
ax.set_title('Convergence of pareto-min TgtBal — representative cells (3 seeds each)')
ax.legend(fontsize=8)
ax.axhline(0.1, color='gray', ls=':', lw=0.8)
fig.tight_layout()
fig.savefig(f'{OUT}/evo_convergence_representative_cells.png', bbox_inches='tight')
plt.close(fig)

# --- Fig 4: budget — ECDF of gen50/gen90/gen99 + last improvement ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax = axes[0]
for colname, c in [('gen50', 'tab:green'), ('gen90', 'tab:blue'), ('gen99', 'tab:purple'),
                   ('last_improve_gen', 'tab:red')]:
    v = np.sort(cv[colname].to_numpy())
    ax.step(v, np.arange(1, len(v) + 1) / len(v), color=c, label=colname)
ax.axvline(200, color='k', lw=0.8)
ax.set_xlabel('generation')
ax.set_ylabel('ECDF over 119 seeds')
ax.set_title('When is improvement reached? (budget = 200 gens)')
ax.legend(fontsize=8)
ax = axes[1]
for name, mask, c in [('converged', ~cv['stuck'], 'tab:blue'), ('stuck>0.1', cv['stuck'], 'tab:red')]:
    sub = cv[mask]
    ax.scatter(sub['gen90'], sub['min_TgtBal'].clip(lower=1e-7), s=16, color=c, alpha=0.7, label=f'{name} (n={len(sub)})')
ax.set_yscale('log')
ax.set_xlabel('gen90 (90% of improvement reached)')
ax.set_ylabel('final min TgtBal (log)')
ax.set_title('gen90 vs achieved boundary proximity')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f'{OUT}/evo_budget_gen90_ecdf.png', bbox_inches='tight')
plt.close(fig)

# --- Fig 5: per-cell seed spread (variance structure) ---
cellstats = d.groupby(['target', 'la', 'lt'])['min_TgtBal'].median().sort_values()
order = list(cellstats.index)
fig, ax = plt.subplots(figsize=(12, 4.5))
for i, key in enumerate(order):
    t, la, lt = key
    vals = d[(d['target'] == t) & (d['la'] == la) & (d['lt'] == lt)]['min_TgtBal'].clip(lower=1e-7)
    col = {'ostrich': 'tab:gray', 'green iguana': 'tab:blue', 'boa constrictor': 'tab:cyan',
           'cello': 'tab:orange', 'marimba': 'tab:red'}[t]
    ax.scatter(np.full(len(vals), i), vals, s=18, color=col, edgecolor='k', linewidth=0.3, zorder=3)
    ax.plot([i - 0.3, i + 0.3], [cellstats[key]] * 2, color='k', lw=1.4, zorder=4)
ax.set_yscale('log')
ax.set_xticks(range(len(order)),
              [f'{t.split()[0][:4]}({la},{lt})' for t, la, lt in order], rotation=90, fontsize=6.5)
ax.set_ylabel('min TgtBal (log)')
ax.axhline(0.1, color='red', ls=':', lw=0.8)
ax.set_title('Per-cell seed spread, cells sorted by median (dots = seeds, bar = cell median)')
handles = [plt.Line2D([], [], marker='o', ls='', color=c, label=t) for t, c in
           [('ostrich', 'tab:gray'), ('green iguana', 'tab:blue'), ('boa constrictor', 'tab:cyan'),
            ('cello', 'tab:orange'), ('marimba', 'tab:red')]]
ax.legend(handles=handles, fontsize=7, ncols=5, loc='upper left')
fig.tight_layout()
fig.savefig(f'{OUT}/evo_cell_seed_spread.png', bbox_inches='tight')
plt.close(fig)

print('\nfigures saved:')
for f in sorted(os.listdir(OUT)):
    if f.endswith('.png'):
        print(' ', os.path.join(OUT, f))
