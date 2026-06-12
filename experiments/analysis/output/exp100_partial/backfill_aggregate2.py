import numpy as np
import pandas as pd

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 60)
pd.set_option('display.max_rows', 120)

OUT = '/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/exp100_partial'
ps = pd.read_parquet(f'{OUT}/backfill_pairflips_per_seed.parquet')
pc = pd.read_parquet(f'{OUT}/backfill_pairflip_candidates.parquet')
evo = pd.read_csv(f'{OUT}/cell_summary.csv')

main = ps[ps.anchor == 'junco']  # exclude reversed-pair seed_0120
pcm = pc[pc.seed.isin(main.seed)]

print("=== GENUINE vs RAW ===")
print(f"junco-anchored seeds: {len(main)}")
print(f"raw pair-flip calls: {main.n_pairflips.sum()}, genuine (anchor on junco side, 6-opt prompt): {main.n_genuine.sum()}")
print(f"seeds with >=1 raw pf: {(main.n_pairflips>0).sum()}, with >=1 GENUINE pf: {(main.n_genuine>0).sum()} of {len(main)}")
print(f"anchor segments pair-flipped at anchor call: {main.n_anchor_pairflipped.sum()}/{main.n_anchors.sum()} = {main.n_anchor_pairflipped.sum()/main.n_anchors.sum():.1%}")
print(f"genuine flip calls total: {pcm.genuine.sum()} of {len(pcm)} pair-flip calls")

cell = main.groupby(['target','la','lt']).agg(
    n_seeds=('seed','count'),
    n_seeds_pf=('n_pairflips', lambda s: int((s>0).sum())),
    n_seeds_gf=('n_genuine', lambda s: int((s>0).sum())),
    total_calls=('n_calls','sum'),
    total_pf=('n_pairflips','sum'),
    total_gf=('n_genuine','sum'),
    med_min_gf_ham=('min_gf_hamming','median'),
    min_min_gf_ham=('min_gf_hamming','min'),
    n_anch_flipped=('n_anchor_pairflipped','sum'),
).reset_index()
cell['pf_seed_rate'] = cell.n_seeds_pf/cell.n_seeds
cell['gf_seed_rate'] = cell.n_seeds_gf/cell.n_seeds
cell = cell.merge(evo[['target','la','lt','cross_rate']], on=['target','la','lt'], how='left')
print("\n=== PER-CELL (genuine) ===")
print(cell.to_string(index=False))

both = cell.dropna(subset=['cross_rate'])
agree_g = ((both.gf_seed_rate>0) == (both.cross_rate>0)).mean()
print(f"\ncell agreement (genuine-pf vs evo-cross): {agree_g:.1%}")
print("evo crossed, NO genuine pf:", both[(both.cross_rate>0)&(both.gf_seed_rate==0)][['target','la','lt','cross_rate']].values.tolist())
print("genuine pf, evo NEVER crossed:", both[(both.cross_rate==0)&(both.gf_seed_rate>0)][['target','la','lt','gf_seed_rate','total_gf']].values.tolist())
rs = both[['gf_seed_rate','cross_rate']].corr(method='spearman').iloc[0,1]
print(f"spearman(gf_seed_rate, cross_rate): {rs:.3f}")

print("\n=== GENUINE MIN HAMMING by target ===")
mg = main.dropna(subset=['min_gf_hamming'])
print(mg.groupby('target').min_gf_hamming.describe()[['count','min','25%','50%','75%','max']].to_string())
print(f"\noverall genuine min-hamming quartiles: p25={mg.min_gf_hamming.quantile(.25):.0f} p50={mg.min_gf_hamming.quantile(.5):.0f} p75={mg.min_gf_hamming.quantile(.75):.0f}")
zero = mg[mg.min_gf_hamming == 0]
print(f"genuine flips at hamming==0 (should be impossible): {len(zero)}")

gsp = pcm[(pcm.genuine) & (pcm.hamming <= 30) & (pcm.hamming > 0)]
print(f"\nsparse GENUINE flip calls (0<ham<=30): {len(gsp)} across {gsp.seed.nunique()} seeds")
print(f"  txt share of diff genes: {gsp.txt_diff.sum()/gsp.hamming.sum():.1%} (img {gsp.img_diff.sum()}, txt {gsp.txt_diff.sum()})")
print(gsp.groupby('target')[['hamming']].agg(['count','mean','min']).to_string())

print("\n=== BASIN (genuine flips) ===")
g = pcm[pcm.genuine]
print(f"genuine flips with 6-cat argmax boa: {(g.top1_label=='boa constrictor').mean():.1%}")
print(f"genuine flips with 6-cat argmax == target (non-boa targets): "
      f"{(g[g.target!='boa constrictor'].top1_label == g[g.target!='boa constrictor'].target).mean():.2%}")
print(g.groupby('target').top1_label.value_counts().to_string())

# check overlap min/discovery pair-flip
print("\n=== archive consistency ===")
print(f"n_min_is_pairflip == n_discovery_is_pairflip per seed: {(ps.n_min_is_pairflip == ps.n_discovery_is_pairflip).all()}")

cell.to_csv(f'{OUT}/backfill_cell_summary.csv', index=False)
print("rewrote backfill_cell_summary.csv (genuine columns included)")
