"""Extract decoupled outcomes from raw traces for the adversarial audit.

Per run:
- search_min_1_49: min fitness_TgtBal over generations 1..49 (excludes gen 0
  entirely -> no arithmetic term shared with the gen-0 probe)
- best-so-far min TgtBal per generation (gens 0..49) -> slope diagnostics
- last generation at which best-so-far improved
"""
import glob
import json
import os

import numpy as np
import pandas as pd

RUNS = "/Users/kaiser/Projects/Masterarbeit/runs/Exp-101"
OUT = "/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/exp101"

rows = []
bsf_rows = []
for d in sorted(glob.glob(os.path.join(RUNS, "exp101_margin_predictor_seed_*"))):
    stats = json.load(open(os.path.join(d, "stats.json")))
    sidx = stats["seed_idx"]
    tr = pd.read_parquet(os.path.join(d, "trace.parquet"),
                         columns=["generation", "fitness_TgtBal"])
    gmin = tr.groupby("generation")["fitness_TgtBal"].min()
    gmin = gmin.sort_index()
    assert len(gmin) == 50, (d, len(gmin))
    bsf = gmin.cummin()
    search_min_1_49 = float(tr.loc[tr.generation >= 1, "fitness_TgtBal"].min())
    # slope of best-so-far over last 10 generations (gens 40..49)
    g = np.arange(40, 50)
    y = bsf.loc[40:49].to_numpy()
    slope_raw = float(np.polyfit(g, y, 1)[0])
    slope_log = float(np.polyfit(g, np.log10(y), 1)[0])
    improved = bsf.to_numpy()
    last_improve = int(np.max(np.where(np.diff(improved) < -1e-12)[0]) + 1) \
        if np.any(np.diff(improved) < -1e-12) else 0
    rows.append(dict(seed_idx=sidx,
                     search_min_1_49=search_min_1_49,
                     bsf_49=float(bsf.iloc[-1]),
                     bsf_40=float(bsf.loc[40]),
                     slope_raw_last10=slope_raw,
                     slope_log10_last10=slope_log,
                     last_improve_gen=last_improve))
    for gen, v in bsf.items():
        bsf_rows.append(dict(seed_idx=sidx, generation=int(gen), bsf=float(v)))

pd.DataFrame(rows).to_csv(os.path.join(OUT, "skeptic_run_extract.csv"), index=False)
pd.DataFrame(bsf_rows).to_csv(os.path.join(OUT, "skeptic_bsf_series.csv"), index=False)
print("wrote", len(rows), "runs")
