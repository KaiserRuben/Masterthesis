"""Follow-up: config diffs, no-improvement trigger check, anchor hamming, gaps."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/kaiser/Projects/Masterarbeit")
RUN = ROOT / "runs/Exp-100/poc_boundary_pair"
OUT = ROOT / "experiments/analysis/output/exp100_partial"

df = pd.read_csv(OUT / "per_seed_audit.csv")

# ---- 1. config diffs ------------------------------------------------
def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = json.dumps(v)
    return out

reps = {}
for h, grp in df.groupby("config_hash"):
    first = grp.sort_values("dir_ts").iloc[0]
    cfg = json.loads((RUN / first.seed / "config.json").read_text())
    reps[h] = (first.seed, int(grp.dir_ts.min()), len(grp), flatten(cfg))

hashes = sorted(reps, key=lambda h: reps[h][1])
print("config variants in launch order:")
for h in hashes:
    s, ts, n, _ = reps[h]
    print(f"  {h}  n={n:3d}  first={s}  start={pd.to_datetime(ts, unit='s')}")

base_h = hashes[0]
base = reps[base_h][3]
for h in hashes[1:]:
    other = reps[h][3]
    keys = sorted(set(base) | set(other))
    diffs = [(k, base.get(k), other.get(k)) for k in keys if base.get(k) != other.get(k)]
    print(f"\n--- {base_h} (v1) vs {h} ---")
    for k, a, b in diffs:
        print(f"  {k}: {a} -> {b}")

# ---- 2. no-improvement trigger check --------------------------------
# Trigger should fire at gen >= warmup(30) if min TgtBal at that gen has
# not strictly improved on gen 0. Count seeds where that condition held.
WARMUP = 30
n_should_fire = 0
first_improve = []
for _, r in df[df.has_manifest == True].iterrows():  # noqa: E712
    cv = pd.read_parquet(RUN / r.seed / "evolutionary/convergence.parquet",
                         columns=["pareto_min_TgtBal"])
    v = cv.pareto_min_TgtBal.to_numpy()
    imp = np.where(v < v[0])[0]
    fi = int(imp[0]) if len(imp) else None
    first_improve.append(fi)
    if fi is None or fi > WARMUP:
        n_should_fire += 1
fi_arr = np.array([f for f in first_improve if f is not None])
print(f"\nno-improvement audit over {len(first_improve)} complete seeds:")
print(f"  seeds never improving on gen0: {sum(1 for f in first_improve if f is None)}")
print(f"  seeds with first improvement AFTER gen {WARMUP} "
      f"(no_improvement should have fired at gen {WARMUP}): {n_should_fire}")
print(f"  first-improvement gen percentiles: "
      f"{np.percentile(fi_arr, [10, 50, 90]).tolist()}")

# ---- 3. anchor genotype hamming -------------------------------------
rows = []
for _, r in df[df.has_manifest == True].iterrows():  # noqa: E712
    genos = []
    for aj in sorted((RUN / r.seed / "pdq/anchors").glob("anchor_*.json")):
        genos.append(np.asarray(json.loads(aj.read_text())["genotype"]))
    if len(genos) < 2:
        continue
    hd = []
    for i in range(len(genos)):
        for j in range(i + 1, len(genos)):
            hd.append(int((genos[i] != genos[j]).sum()))
    g0_active = int((genos[0] != 0).sum())
    rows.append({"seed": r.seed, "min_hamming": min(hd), "max_hamming": max(hd),
                 "mean_hamming": float(np.mean(hd)), "genome_len": len(genos[0]),
                 "anchor0_active_genes": g0_active})
hg = pd.DataFrame(rows)
print(f"\nanchor pairwise hamming over {len(hg)} seeds "
      f"(genome len {hg.genome_len.iloc[0]}):")
print("  min-hamming percentiles:", np.percentile(hg.min_hamming, [0, 10, 50, 90, 100]).tolist())
print("  mean-hamming percentiles:", np.percentile(hg.mean_hamming, [0, 10, 50, 90, 100]).tolist())
print("  seeds with some anchor pair hamming <= 2:", int((hg.min_hamming <= 2).sum()))
print("  active genes in anchor0, percentiles:",
      np.percentile(hg.anchor0_active_genes, [0, 50, 100]).tolist())
hg.to_csv(OUT / "anchor_hamming.csv", index=False)

# ---- 4. timeline gaps (merged streams) -------------------------------
ts = df.sort_values("dir_ts")[["seed_idx", "dir_ts"]].to_numpy()
gaps = np.diff(ts[:, 1])
print("\nglobal start-time gaps > 4h (run pauses/restarts):")
for i, g in enumerate(gaps):
    if g > 4 * 3600:
        print(f"  after seed_{int(ts[i,0]):04d} @ {pd.to_datetime(ts[i,1], unit='s')} "
          f"-> seed_{int(ts[i+1,0]):04d} @ {pd.to_datetime(ts[i+1,1], unit='s')} "
          f"gap {g/3600:.1f} h")
total_gap = sum(g - 4*3600 for g in gaps if g > 4*3600)
print(f"total idle in gaps>4h (excess over 4h): {total_gap/86400:.2f} days")

# config hash by seed-idx range
print("\nconfig hash per contiguous seed_idx block:")
d2 = df.sort_values(["dir_ts"])[["seed_idx", "config_hash", "dir_ts"]]
prev = None
for _, r in d2.iterrows():
    if r.config_hash != prev:
        print(f"  from start {pd.to_datetime(r.dir_ts, unit='s')} seed_{int(r.seed_idx):04d}: {r.config_hash}")
        prev = r.config_hash
