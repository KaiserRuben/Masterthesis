import json, glob
import pandas as pd
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 60)

d = glob.glob('/Users/kaiser/Projects/Masterarbeit/runs/Exp-100/poc_boundary_pair/seed_0006_*')[0]

s2 = pd.read_parquet(f'{d}/pdq/stage2_trajectories.parquet')
print("STAGE2_TRAJ", s2.shape)
print(list(s2.columns))
print(s2.head(5))
print("dtypes:\n", s2.dtypes)

sc = pd.read_parquet(f'{d}/pdq/sut_calls.parquet')
# segment structure: anchor calls positions
anchor_pos = sc.index[sc.stage == 'anchor'].tolist()
print("\nanchor call row positions:", anchor_pos)
print("call_id monotonic:", sc.call_id.is_monotonic_increasing)

# do stage2 sut_call candidate_ids appear in stage2_traj?
if 'sut_call_id' in s2.columns:
    s2ids = set(s2['sut_call_id'])
    sc2 = sc[sc.stage == 'stage2']
    print("stage2 sut_calls covered by traj sut_call_id:", sc2['call_id'].isin(s2ids).mean())
if 'candidate_id' in s2.columns:
    s2c = set(s2['candidate_id'])
    sc2 = sc[sc.stage == 'stage2']
    print("stage2 sut_calls candidate_id coverage:", sc2['candidate_id'].isin(s2c).mean())

# prior evo reference
import os
base = '/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/exp100_partial'
for f in ['per_cell_summary.csv', 'cell_summary.csv', 'seed_summary.csv']:
    p = os.path.join(base, f)
    if os.path.exists(p):
        df = pd.read_csv(p)
        print(f"\n{f}: {df.shape}\n cols: {list(df.columns)}")
        print(df.head(4))
