"""Pass 1 (fixed): per-seed aggregation over Exp-100 poc_boundary_pair traces.

Genotype length varies per seed (image_dim 222 or 276 + text_dim 19), so the
text block is sliced as the LAST 19 genes, image block as the rest.
Intermediate state goes to /tmp/exp100_state; deliverables to exp100_partial.
"""
import difflib
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr

RUNS = "/Users/kaiser/Projects/Masterarbeit/runs/Exp-100/poc_boundary_pair"
OUT = "/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/exp100_partial"
STATE = "/tmp/exp100_state"
TXT_DIM, NB_K = 19, 50

COLS = ["generation", "genotype", "decoded_text", "predicted_class",
        "p_class_a", "p_class_b", "fitness_TgtBal"]

rows = []
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
    img_dim = stats["image_dim"]
    assert stats["text_dim"] == TXT_DIM

    df = pf.read(columns=COLS).to_pandas()
    G = np.stack(df["genotype"].to_numpy())
    assert G.shape[1] == img_dim + TXT_DIM, (seed_idx, G.shape)
    txt = G[:, -TXT_DIM:]
    img = G[:, :-TXT_DIM]
    n_act_img = (img != 0).sum(axis=1)
    n_act_txt = (txt != 0).sum(axis=1)

    tgtbal = df["fitness_TgtBal"].to_numpy()
    pa = df["p_class_a"].to_numpy()
    pb = df["p_class_b"].to_numpy()
    gen = df["generation"].to_numpy()

    crossed_mask = pa < pb
    first_gen_crossed = int(gen[crossed_mask].min()) if crossed_mask.any() else -1

    nb_idx = np.argsort(tgtbal, kind="stable")[:NB_K]
    rho_img, p_img = spearmanr(tgtbal, n_act_img)
    rho_txt, p_txt = spearmanr(tgtbal, n_act_txt)

    hist_nb[seed_idx] = (txt[nb_idx] != 0).sum(axis=0)
    hist_all[seed_idx] = (txt != 0).sum(axis=0)

    zero_txt = np.where(n_act_txt == 0)[0]
    baseline_text = df["decoded_text"].iloc[zero_txt[0]] if len(zero_txt) else None

    bi = int(np.argmin(tgtbal))
    best_text = df["decoded_text"].iloc[bi]
    ref = baseline_text if baseline_text is not None else stats["prompt_template"]
    sm = difflib.SequenceMatcher(a=ref.split(), b=best_text.split())
    n_edit = sum(max(i2 - i1, j2 - j1)
                 for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")

    cell = (meta["target_class_concrete"], meta["level_anchor"], meta["level_target"])
    rows.append(dict(
        seed_idx=seed_idx, run_dir=os.path.basename(d),
        target=cell[0], la=cell[1], lt=cell[2],
        label_a=stats["class_a"], label_b=stats["class_b"],
        common_ancestor_level=meta.get("common_ancestor_level"),
        seed_idx_in_class=meta.get("seed_idx_in_class"),
        image_dim=img_dim,
        crossed=bool(crossed_mask.any()),
        frac_crossed=float(crossed_mask.mean()),
        first_gen_crossed=first_gen_crossed,
        max_p_class_b=float(pb.max()),
        min_tgtbal=float(tgtbal.min()),
        median_tgtbal=float(np.median(tgtbal)),
        nb_mean_img=float(n_act_img[nb_idx].mean()),
        nb_med_img=float(np.median(n_act_img[nb_idx])),
        nb_mean_txt=float(n_act_txt[nb_idx].mean()),
        nb_med_txt=float(np.median(n_act_txt[nb_idx])),
        nb_mean_img_frac=float((n_act_img[nb_idx] / img_dim).mean()),
        all_mean_img=float(n_act_img.mean()),
        all_mean_txt=float(n_act_txt.mean()),
        all_mean_img_frac=float((n_act_img / img_dim).mean()),
        rho_tgtbal_img=float(rho_img), p_rho_img=float(p_img),
        rho_tgtbal_txt=float(rho_txt), p_rho_txt=float(p_txt),
        best_tgtbal=float(tgtbal[bi]), best_gen=int(gen[bi]),
        best_p_a=float(pa[bi]), best_p_b=float(pb[bi]),
        best_pred=df["predicted_class"].iloc[bi],
        best_n_act_img=int(n_act_img[bi]), best_n_act_txt=int(n_act_txt[bi]),
        best_token_edits=int(n_edit),
        baseline_text=baseline_text, best_text=best_text,
        n_zero_txt_rows=int(len(zero_txt)),
    ))

    g = pd.DataFrame({"gen": gen, "tb": tgtbal, "pb": pb}).groupby("gen")
    traj_min_tgtbal[seed_idx] = g["tb"].min().to_numpy()
    traj_max_pb[seed_idx] = g["pb"].max().to_numpy()

    best_examples.append(dict(
        seed_idx=seed_idx, target=cell[0], la=cell[1], lt=cell[2],
        label_a=stats["class_a"], label_b=stats["class_b"],
        tgtbal=float(tgtbal[bi]), p_b=float(pb[bi]), gen=int(gen[bi]),
        n_txt=int(n_act_txt[bi]), n_img=int(n_act_img[bi]),
        baseline=ref, mutated=best_text))
    print("done", seed_idx, flush=True)

summary = pd.DataFrame(rows).sort_values("seed_idx")
summary.to_csv(os.path.join(STATE, "seed_summary.csv"), index=False)
summary.to_csv(os.path.join(OUT, "seed_summary.csv"), index=False)

sidx = summary["seed_idx"].to_numpy()
try:
    np.savez_compressed(
        os.path.join(STATE, "trajectories.npz"),
        seed_idx=sidx,
        min_tgtbal=np.stack([traj_min_tgtbal[i] for i in sidx]),
        max_pb=np.stack([traj_max_pb[i] for i in sidx]),
        txt_hist_nb=np.stack([hist_nb[i] for i in sidx]),
        txt_hist_all=np.stack([hist_all[i] for i in sidx]),
    )
    with open(os.path.join(STATE, "best_examples.json"), "w") as f:
        json.dump(best_examples, f, indent=1)
    with open(os.path.join(OUT, "best_examples.json"), "w") as f:
        json.dump(best_examples, f, indent=1)
    print("saved", len(sidx), flush=True)
except BaseException as e:
    print("SAVE FAILED:", type(e).__name__, e, flush=True)
    sys.exit(2)
