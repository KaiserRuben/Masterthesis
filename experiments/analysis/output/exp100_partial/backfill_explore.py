import json, glob
import pandas as pd
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 50)

d = glob.glob('/Users/kaiser/Projects/Masterarbeit/runs/Exp-100/poc_boundary_pair/seed_0006_*')[0]

stats = json.load(open(f'{d}/evolutionary/stats.json'))
print("STATS keys:", list(stats.keys()))
print("seed_metadata:", json.dumps(stats.get('seed_metadata', {}), indent=1)[:800])

man = json.load(open(f'{d}/manifest.json'))
print("\nMANIFEST keys:", list(man.keys()))
print(json.dumps(man, indent=1)[:1500])

sc = pd.read_parquet(f'{d}/pdq/sut_calls.parquet')
print("\nSUT_CALLS", sc.shape, "\ncols:", list(sc.columns))
print(sc.head(3))
print("categories example:", sc['categories'].iloc[0])
print("logprobs example:", sc['logprobs'].iloc[0])
print("stage values:", sc['stage'].value_counts().to_dict())

ca = pd.read_parquet(f'{d}/pdq/candidates.parquet')
print("\nCANDIDATES", ca.shape, "\ncols:", list(ca.columns))
print(ca.head(3))
print("stage values:", ca['stage'].value_counts().to_dict())
print("hamming_to_anchor nulls:", ca['hamming_to_anchor'].isna().sum())
print("genotype len:", len(ca['genotype'].iloc[0]))

ar = pd.read_parquet(f'{d}/pdq/archive.parquet')
print("\nARCHIVE", ar.shape, "\ncols:", list(ar.columns))
print(ar.head(3))

s1 = pd.read_parquet(f'{d}/pdq/stage1_flips.parquet')
print("\nSTAGE1_FLIPS", s1.shape, list(s1.columns))
s2 = pd.read_parquet(f'{d}/pdq/stage2_trajectories.parquet')
print("STAGE2_TRAJ", s2.shape, list(s2.columns))
print(s2.head(3))
