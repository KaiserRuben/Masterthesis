import json
import numpy as np
import pandas as pd

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 60)
pd.set_option('display.max_rows', 100)

OUT = '/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/exp100_partial'
ps = pd.read_parquet(f'{OUT}/backfill_pairflips_per_seed.parquet')
pc = pd.read_parquet(f'{OUT}/backfill_pairflip_candidates.parquet')
evo = pd.read_csv(f'{OUT}/cell_summary.csv')

print("=== GLOBAL ===")
print(f"seeds: {len(ps)}, calls: {ps.n_calls.sum()}, pair-flip calls: {ps.n_pairflips.sum()} ({ps.n_pairflips.sum()/ps.n_calls.sum():.1%})")
print(f"seeds with >=1 pair-flip: {(ps.n_pairflips>0).sum()}/{len(ps)} = {(ps.n_pairflips>0).mean():.1%}")
print(f"anchors pair-flipped at anchor call (6-opt prompt): {ps.n_anchor_pairflipped.sum()} of {ps.n_anchors.sum()}")
print(f"targets: {sorted(ps.target.unique())}")
print(f"cells: {ps.groupby(['target','la','lt']).ngroups}")

# === PER-CELL ===
cell = ps.groupby(['target','la','lt']).agg(
    n_seeds=('seed','count'),
    n_seeds_pf=('n_pairflips', lambda s: int((s>0).sum())),
    total_calls=('n_calls','sum'),
    total_pf=('n_pairflips','sum'),
    med_min_ham=('min_pf_hamming','median'),
    min_min_ham=('min_pf_hamming','min'),
    med_anchor_margin=('anchor_margin_mean','median'),
).reset_index()
cell['pf_seed_rate'] = cell.n_seeds_pf / cell.n_seeds
cell['pf_call_frac'] = cell.total_pf / cell.total_calls
cell = cell.merge(evo[['target','la','lt','cross_rate','n_crossed','n']], on=['target','la','lt'], how='outer')
print("\n=== PER-CELL (pdq pair-flip recovery vs evo crossing) ===")
print(cell.to_string(index=False))

# agreement
both = cell.dropna(subset=['pf_seed_rate','cross_rate'])
agree = ((both.pf_seed_rate>0) == (both.cross_rate>0)).mean()
print(f"\ncell-level agreement (any-pf vs any-evo-cross): {agree:.1%} of {len(both)} cells")
print("cells evo crossed but NO pdq pair-flip:")
print(both[(both.cross_rate>0)&(both.pf_seed_rate==0)][['target','la','lt','cross_rate','total_calls']].to_string(index=False))
print("cells pdq pair-flip but evo NEVER crossed:")
print(both[(both.cross_rate==0)&(both.pf_seed_rate>0)][['target','la','lt','pf_seed_rate','total_pf','total_calls','med_min_ham']].to_string(index=False))
corr = both[['pf_call_frac','cross_rate']].corr().iloc[0,1]
corr_s = both[['pf_seed_rate','cross_rate']].corr(method='spearman').iloc[0,1]
print(f"corr(pf_call_frac, cross_rate) pearson: {corr:.3f}; spearman(pf_seed_rate, cross_rate): {corr_s:.3f}")

# === MIN HAMMING ===
print("\n=== MINIMAL PAIR-FLIP HAMMING ===")
mh = ps.dropna(subset=['min_pf_hamming'])
print(f"seeds with pair-flips: {len(mh)}; min_pf_hamming quartiles: p25={mh.min_pf_hamming.quantile(.25):.0f} p50={mh.min_pf_hamming.quantile(.5):.0f} p75={mh.min_pf_hamming.quantile(.75):.0f} min={mh.min_pf_hamming.min():.0f} max={mh.min_pf_hamming.max():.0f}")
print("\nby target:")
print(mh.groupby('target').min_pf_hamming.describe()[['count','min','25%','50%','75%','max']].to_string())
sparse = mh[mh.min_pf_hamming <= 30]
print(f"\nseeds with min pair-flip hamming <= 30: {len(sparse)}")
if len(sparse):
    print(sparse[['seed','target','la','lt','min_pf_hamming','min_pf_img_diff','min_pf_txt_diff','min_pf_margin']].to_string(index=False))
# modality split for sparse pair-flip CALLS
spc = pc[pc.hamming <= 30]
print(f"\nsparse pair-flip CALLS (hamming<=30): {len(spc)} across {spc.seed.nunique()} seeds")
if len(spc):
    print(f"  mean img_diff {spc.img_diff.mean():.1f}, mean txt_diff {spc.txt_diff.mean():.1f}; txt share of diff: {spc.txt_diff.sum()/(spc.hamming.sum()):.1%}")
    print(spc.groupby('target')[['hamming','img_diff','txt_diff']].agg(['count','mean']).to_string())

# === BASIN OVERLAY ===
print("\n=== BASIN OVERLAY (6-cat argmax of pair-flip calls) ===")
print(pc.top1_label.value_counts().to_string())
print(f"\npair-flip AND 6cat==target: {pc.cat6_is_target.sum()} ({pc.cat6_is_target.mean():.1%})")
print(f"pair-flip AND 6cat==boa:    {pc.cat6_is_boa.sum()} ({pc.cat6_is_boa.mean():.1%})")
bas = pc.groupby('target').agg(n=('call_id','count'),
                               cat6_target=('cat6_is_target','mean'),
                               cat6_boa=('cat6_is_boa','mean')).reset_index()
bas['cat6_other'] = 1 - bas.cat6_target - bas.cat6_boa
print(bas.to_string(index=False))

# === STAGE-2 WASTE ===
print("\n=== STAGE-2 WASTE ===")
print(f"total stage2 steps: {ps.n_stage2_steps.sum()}")
print(f"stage2 steps that were pair-flips: {ps.s2_pairflip_steps.sum()}")
print(f"  of which rejected by shrink (accepted=False): {ps.s2_pairflip_rejected.sum()}")
print(f"  of which 6-cat said 'flip broken' (still_flipped=False): {ps.s2_pairflip_broken6cat.sum()}")
print(f"stage2 ACCEPTED steps that were NOT pair-flips (shrink preserved wrong thing): {ps.s2_accepted_nonpairflip.sum()}")
print(f"archive flips total: {ps.n_archive_flips.sum()}")
print(f"  final genotype_min is a pair-flip: {ps.n_min_is_pairflip.sum()} ({ps.n_min_is_pairflip.sum()/max(ps.n_archive_flips.sum(),1):.1%})")
print(f"  stage1 discovery (genotype_flipped) was pair-flip: {ps.n_discovery_is_pairflip.sum()} ({ps.n_discovery_is_pairflip.sum()/max(ps.n_archive_flips.sum(),1):.1%})")

# stage distribution of pair-flips
print(f"\npair-flips by stage: stage1={ps.n_pf_stage1.sum()}, stage2={ps.n_pf_stage2.sum()}")

cell.to_csv(f'{OUT}/backfill_cell_summary.csv', index=False)
print(f"\nwrote {OUT}/backfill_cell_summary.csv")
