"""Pass 1b: trajectories + text-gene histograms + best examples (column-pruned)."""
import glob
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RUNS = "/Users/kaiser/Projects/Masterarbeit/runs/Exp-100/poc_boundary_pair"
OUT = "/tmp/exp100_state"
IMG_DIM, NB_K = 222, 50

traj_min_tgtbal, traj_max_pb, hist_nb, hist_all = {}, {}, {}, {}
best_examples = []

for d in sorted(glob.glob(os.path.join(RUNS, "seed_*"))):
    tp = os.path.join(d, "evolutionary", "trace.parquet")
    sp = os.path.join(d, "evolutionary", "stats.json")
    if not (os.path.exists(tp) and os.path.exists(sp)):
        continue
    pf = pq.ParquetFile(tp)
    if pf.metadata.num_rows < 6000:
        continue
    with open(sp) as f:
        stats = json.load(f)
    meta = stats["seed_metadata"]
    seed_idx = stats["seed_idx"]
    df = pf.read(columns=["generation", "genotype", "decoded_text",
                          "p_class_a", "p_class_b", "fitness_TgtBal"]).to_pandas()
    txt = np.stack(df["genotype"].to_numpy())[:, IMG_DIM:]
    tgtbal = df["fitness_TgtBal"].to_numpy()
    pb = df["p_class_b"].to_numpy()
    gen = df["generation"].to_numpy()

    nb_idx = np.argsort(tgtbal, kind="stable")[:NB_K]
    hist_nb[seed_idx] = (txt[nb_idx] != 0).sum(axis=0)
    hist_all[seed_idx] = (txt != 0).sum(axis=0)

    g = pd.DataFrame({"gen": gen, "tb": tgtbal, "pb": pb}).groupby("gen")
    traj_min_tgtbal[seed_idx] = g["tb"].min().to_numpy()
    traj_max_pb[seed_idx] = g["pb"].max().to_numpy()

    bi = int(np.argmin(tgtbal))
    zero_txt = np.where((txt != 0).sum(axis=1) == 0)[0]
    best_examples.append(dict(
        seed_idx=seed_idx, target=meta["target_class_concrete"],
        la=meta["level_anchor"], lt=meta["level_target"],
        label_a=stats["class_a"], label_b=stats["class_b"],
        tgtbal=float(tgtbal[bi]), p_b=float(pb[bi]), gen=int(gen[bi]),
        baseline=(df["decoded_text"].iloc[zero_txt[0]] if len(zero_txt)
                  else stats["prompt_template"]),
        mutated=df["decoded_text"].iloc[bi]))
    print("done", seed_idx, flush=True)

import faulthandler
import sys
faulthandler.enable()
print("loop finished, n seeds:", len(traj_min_tgtbal), flush=True)
lens = {len(v) for v in traj_min_tgtbal.values()}
print("trajectory lengths:", lens, flush=True)
sidx = np.array(sorted(traj_min_tgtbal))
try:
    np.savez_compressed(
        os.path.join(OUT, "trajectories.npz"),
        seed_idx=sidx,
        min_tgtbal=np.stack([traj_min_tgtbal[i] for i in sidx]),
        max_pb=np.stack([traj_max_pb[i] for i in sidx]),
        txt_hist_nb=np.stack([hist_nb[i] for i in sidx]),
        txt_hist_all=np.stack([hist_all[i] for i in sidx]),
    )
    print("npz saved", flush=True)
    with open(os.path.join(OUT, "best_examples.json"), "w") as f:
        json.dump(best_examples, f, indent=1)
    print("saved", len(sidx), flush=True)
except BaseException as e:
    print("SAVE FAILED:", type(e).__name__, e, flush=True)
    sys.exit(2)
