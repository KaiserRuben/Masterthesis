import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = '/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/exp100_partial'
ps = pd.read_parquet(f'{OUT}/backfill_pairflips_per_seed.parquet')
pc = pd.read_parquet(f'{OUT}/backfill_pairflip_candidates.parquet')
cell = pd.read_csv(f'{OUT}/backfill_cell_summary.csv')

main = ps[ps.anchor == 'junco']
# boundary-in-neighborhood: both signs sampled in seed's logged calls
both_sides = main[(main.n_pairflips > 0) & (main.n_pairflips < main.n_calls)]
print(f"seeds with both margin signs sampled (boundary inside logged neighborhood): {len(both_sides)}/{len(main)}")

targets = ['ostrich', 'boa constrictor', 'green iguana', 'cello', 'marimba']
las = [0, 1, 2]; lts = [0, 1, 2]
combos = [(la, lt) for la in las for lt in lts]
xlabels = [f"{la},{lt}" for la, lt in combos]

def grid(col):
    M = np.full((len(targets), len(combos)), np.nan)
    for i, t in enumerate(targets):
        for j, (la, lt) in enumerate(combos):
            r = cell[(cell['target'] == t) & (cell['la'] == la) & (cell['lt'] == lt)]
            if len(r): M[i, j] = r[col].iloc[0]
    return M

fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), constrained_layout=True)
panels = [('cross_rate', 'Evolutionary: seed crossing rate\n(pair space, 2-option prompt)'),
          ('pf_seed_rate', 'PDQ backfill: raw pair-flip seed rate\n(lp[target]>lp[anchor], 6-option prompt)'),
          ('gf_seed_rate', 'PDQ backfill: GENUINE pair-flip seed rate\n(anchor on junco side under same prompt)')]
for ax, (colname, title) in zip(axes, panels):
    M = grid(colname)
    Mm = np.ma.masked_invalid(M)
    cmap = plt.cm.viridis.copy(); cmap.set_bad('0.85')
    im = ax.imshow(Mm, vmin=0, vmax=1, cmap=cmap, aspect='auto')
    ax.set_xticks(range(len(combos))); ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_yticks(range(len(targets))); ax.set_yticklabels(targets, fontsize=9)
    ax.set_xlabel('(level_anchor, level_target)')
    ax.set_title(title, fontsize=9)
    for i in range(len(targets)):
        for j in range(len(combos)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha='center', va='center',
                        color='white' if M[i, j] < 0.6 else 'black', fontsize=7)
fig.colorbar(im, ax=axes, shrink=0.8, label='fraction of seeds')
fig.suptitle('Exp-100 backfill: where is the junco↔target pair boundary reachable? (grey = cell absent/no data)', fontsize=11)
fig.savefig(f'{OUT}/backfill_heatmap_pairflip_vs_evo.png', dpi=150)
plt.close(fig)

# --- Fig 2: minimal pair-flip hamming distribution ---
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
rng = np.random.default_rng(0)
ax = axes[0]
for k, t in enumerate(targets):
    sub = main[(main.target == t)]
    raw = sub.min_pf_hamming.dropna()
    gen = sub.min_gf_hamming.dropna()
    x0 = k - 0.18 + rng.uniform(-0.08, 0.08, len(raw))
    x1 = k + 0.18 + rng.uniform(-0.08, 0.08, len(gen))
    ax.scatter(x0, raw + 0.5, s=18, alpha=0.6, c='tab:gray', label='raw' if k == 0 else None)
    ax.scatter(x1, gen + 0.5, s=22, alpha=0.85, c='tab:red', label='genuine' if k == 0 else None)
ax.axhline(175, color='tab:blue', ls='--', lw=1)
ax.text(4.45, 175, '6-cat junco→boa p25≈175', fontsize=7, color='tab:blue', va='bottom', ha='right')
ax.axhspan(1, 6, color='tab:green', alpha=0.15)
ax.text(4.45, 2.2, '6-cat boa→junco range 1–6', fontsize=7, color='tab:green', ha='right')
ax.set_yscale('log'); ax.set_ylabel('min pair-flip hamming (+0.5 offset, log)')
ax.set_xticks(range(len(targets))); ax.set_xticklabels(targets, fontsize=8)
ax.set_title('Per-seed minimal pair-flip cost (upper bounds:\nsearch steered by wrong 6-cat criterion)', fontsize=9)
ax.legend(fontsize=8, loc='center left')

ax = axes[1]
gd = main.min_gf_hamming.dropna()
ax.hist(gd, bins=np.arange(0, 245, 10), color='tab:red', alpha=0.75)
ax.set_xlabel('min genuine pair-flip hamming'); ax.set_ylabel('seeds')
ax.set_title(f'Bimodal: sparse crossings (≤30 genes, n={(gd<=30).sum()})\nvs dense-only (~170–230, n={(gd>30).sum()}) of {len(gd)} seeds', fontsize=9)
fig.savefig(f'{OUT}/backfill_min_hamming_dist.png', dpi=150)
plt.close(fig)

# --- Fig 3: basin overlay ---
fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
pcm = pc[pc.seed.isin(main.seed)]
agg = pcm.groupby('target').agg(n=('call_id', 'count'),
                                boa=('cat6_is_boa', 'sum'),
                                tgt=('cat6_is_target', 'sum')).reindex(targets)
x = np.arange(len(targets))
boa_only = agg.boa - np.where(np.array(targets) == 'boa constrictor', agg.tgt, 0)
boa_only = agg.n - np.where(np.array(targets) == 'boa constrictor', agg.n, 0)
vals_boa = [agg.loc[t, 'n'] if t != 'boa constrictor' else 0 for t in targets]
vals_tgt = [agg.loc[t, 'n'] if t == 'boa constrictor' else 0 for t in targets]
ax.bar(x, vals_boa, color='tab:brown', label='pair-flip, 6-cat argmax = boa (≠ target)')
ax.bar(x, vals_tgt, color='tab:green', label='pair-flip, 6-cat argmax = target (= boa)')
for i, t in enumerate(targets):
    ax.text(i, agg.loc[t, 'n'] + 100, f"{agg.loc[t,'n']}", ha='center', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(targets, fontsize=8)
ax.set_ylabel('pair-flip calls (raw)')
ax.set_title('Basin overlay: 6-cat argmax of ALL 15,477 pair-flip calls is "boa constrictor".\nFor non-boa targets the pair boundary is fully submerged in the boa basin (0% argmax=target).', fontsize=9)
ax.legend(fontsize=8)
fig.savefig(f'{OUT}/backfill_basin_overlay.png', dpi=150)
plt.close(fig)
print('figures written')

# extra summary numbers for report
print(f"\nstage2 steps total {ps.n_stage2_steps.sum()}, accepted&non-pairflip {ps.s2_accepted_nonpairflip.sum()} "
      f"({ps.s2_accepted_nonpairflip.sum()/ps.n_stage2_steps.sum():.1%})")
sp = pcm[(pcm.genuine) & (pcm.hamming > 0) & (pcm.hamming <= 30)]
print("sparse genuine flips stage split:", sp.stage.value_counts().to_dict())
print("genuine flips stage split:", pcm[pcm.genuine].stage.value_counts().to_dict())
