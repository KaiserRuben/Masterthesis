"""Second pass: diff-to-anchor geometry from raw genotypes + anchor duplication check."""
import json
import glob
import os
import numpy as np
import pandas as pd

ROOT = "/Users/kaiser/Projects/Masterarbeit/runs/Exp-100/poc_boundary_pair"
OUT = "/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/exp100_partial"

rows = []
anchor_geno_rows = []
for d in sorted(glob.glob(os.path.join(ROOT, "seed_*"))):
    ap = os.path.join(d, "pdq", "archive.parquet")
    if not os.path.exists(ap):
        continue
    seed_name = os.path.basename(d)
    a = pd.read_parquet(ap, columns=["flip_id", "pareto_idx", "genotype_anchor",
                                     "genotype_flipped", "genotype_min"])
    seen_anchor = {}
    for _, r in a.iterrows():
        ga = np.asarray(r["genotype_anchor"])
        gf = np.asarray(r["genotype_flipped"])
        gm = np.asarray(r["genotype_min"])
        dm = gm != ga
        df_ = gf != ga
        rows.append({
            "seed_dir": seed_name,
            "flip_id": r["flip_id"],
            "pareto_idx": r["pareto_idx"],
            "anchor_active": int((ga != 0).sum()),
            "hamming_flipped": int(df_.sum()),
            "hamming_min": int(dm.sum()),
            "img_diff_min": int(dm[:222].sum()),
            "txt_diff_min": int(dm[222:].sum()),
            "img_diff_flipped": int(df_[:222].sum()),
            "txt_diff_flipped": int(df_[222:].sum()),
            "l1_diff_flipped": int(np.abs(gf - ga).sum()),
            "l1_diff_min": int(np.abs(gm - ga).sum()),
        })
        if r["pareto_idx"] not in seen_anchor:
            seen_anchor[r["pareto_idx"]] = tuple(ga.tolist())
    for pidx, g in seen_anchor.items():
        anchor_geno_rows.append({"seed_dir": seed_name, "pareto_idx": pidx,
                                 "geno_hash": hash(g), "anchor_active": int(sum(v != 0 for v in g))})

diff = pd.DataFrame(rows)
diff.to_parquet(os.path.join(OUT, "_combined_diffgeom.parquet"))
ag = pd.DataFrame(anchor_geno_rows)
ag.to_parquet(os.path.join(OUT, "_combined_anchor_genos.parquet"))
print("diff rows:", len(diff), "anchor genotypes:", len(ag))
print(diff[["anchor_active", "hamming_flipped", "hamming_min", "img_diff_min", "txt_diff_min"]].describe().round(2).to_string())
