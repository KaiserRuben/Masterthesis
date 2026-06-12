"""m5_00: probe for learned-coordinate prototypes — sklearn, genotype encoding,
text-gene block stats, cat6 logprob geometry, straddle columns."""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import sklearn
    print("sklearn:", sklearn.__version__)
except ImportError:
    print("sklearn: MISSING")

ROOT = "/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography/exp100"

# --- light columns first ---
cols = ["source", "prompt_regime", "seed_dir", "target_class", "level_anchor",
        "level_target", "anchor_class", "image_dim", "g_pair", "pred_label",
        "pair_margin", "generation", "hamming_to_anchor", "n_active_txt"]
df = pd.read_parquet(f"{ROOT}/points.parquet", columns=cols)
print("\nrows:", len(df))
print(df.groupby(["prompt_regime", "source"]).size())

print("\nimage_dim per seed_dir (unique count):",
      df.groupby("seed_dir").image_dim.nunique().max())
print("image_dim values:", sorted(df.image_dim.dropna().unique()))

# smoo cells with seed counts
sm = df[df.source == "smoo"]
print("\nsmoo cells:")
print(sm.groupby(["target_class", "level_anchor", "level_target"]).agg(
    n=("g_pair", "size"), seeds=("seed_dir", "nunique"),
    frac_cross=("g_pair", lambda s: (s < 0).mean())).to_string())

# pick representative seeds: boa wall (boa, lt==1) and an easy cell
for tc, la, lt in [("boa constrictor", None, 1), ("ostrich", None, None)]:
    sub = sm[sm.target_class == tc]
    if lt is not None:
        sub = sub[sub.level_target == lt]
    per_seed = sub.groupby("seed_dir").agg(
        n=("g_pair", "size"), gmin=("g_pair", "min"),
        frac_cross=("g_pair", lambda s: (s < 0).mean()),
        dim=("image_dim", "first"))
    print(f"\n{tc} lt={lt} per-seed:")
    print(per_seed.to_string())

# --- genotype: one seed, smoo, inspect encoding ---
seed0 = sm.seed_dir.iloc[0]
gt = pd.read_parquet(f"{ROOT}/points.parquet",
                     columns=["seed_dir", "source", "genotype", "image_dim"],
                     filters=[("seed_dir", "==", seed0), ("source", "==", "smoo")])
G = np.stack(gt.genotype.to_numpy())
print(f"\nseed {seed0}: genotype matrix {G.shape}, image_dim={gt.image_dim.iloc[0]}")
print("gene value range:", G.min(), G.max())
print("frac zero:", (G == 0).mean())
txt = G[:, -19:]
print("text block value range:", txt.min(), txt.max())
print("text block frac zero:", (txt == 0).mean())
print("text block per-gene max:", txt.max(axis=0))
print("text block per-gene nonzero frac:", np.round((txt != 0).mean(axis=0), 3))

# cat6 genotype text block — same encoding?
ct = pd.read_parquet(f"{ROOT}/points.parquet",
                     columns=["seed_dir", "source", "genotype"],
                     filters=[("seed_dir", "==", seed0), ("source", "==", "pdq_s1")])
if len(ct):
    Gc = np.stack(ct.genotype.to_numpy())
    print(f"\npdq_s1 same seed: {Gc.shape}, txt range {Gc[:, -19:].min()}..{Gc[:, -19:].max()}")

# --- cat6 logprob geometry quick look ---
lp6 = pd.read_parquet(f"{ROOT}/points.parquet",
                      columns=["source", "logprobs", "pred_label"],
                      filters=[("prompt_regime", "==", "cat6")])
L = np.stack(lp6.logprobs.to_numpy())
print("\ncat6 logprobs shape:", L.shape, "row sums (should not be 1):", L[:3].sum(axis=1))
P = np.exp(L - L.max(axis=1, keepdims=True))
P /= P.sum(axis=1, keepdims=True)
print("softmax mass on junco/boa (cols 0,3):",
      np.round(P[:, [0, 3]].sum(axis=1).mean(), 4))
print("mean prob per class:", np.round(P.mean(axis=0), 4))
print("q05/q50/q95 of rest-mass:",
      np.round(np.quantile(1 - P[:, 0] - P[:, 3], [0.05, 0.5, 0.95]), 4))

# straddle pairs columns
sp = pd.read_parquet(f"{ROOT.rsplit('/',0)[0]}/straddle_pairs.parquet".replace("exp100/", "exp100/"))
print("\nstraddle columns:", list(sp.columns))
print(sp.boundary_kind.value_counts().to_dict())
