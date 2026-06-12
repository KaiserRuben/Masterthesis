"""Load all Exp-100 PoC pdq data joined with seed metadata -> combined parquets."""
import json
import glob
import os
import numpy as np
import pandas as pd

ROOT = "/Users/kaiser/Projects/Masterarbeit/runs/Exp-100/poc_boundary_pair"
OUT = "/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/exp100_partial"

seed_dirs = sorted(glob.glob(os.path.join(ROOT, "seed_*")))
arch_rows, s1_rows, s2_rows, anchor_rows, meta_rows = [], [], [], [], []

for d in seed_dirs:
    ap = os.path.join(d, "pdq", "archive.parquet")
    if not os.path.exists(ap):
        continue
    seed_name = os.path.basename(d)
    sm = json.load(open(os.path.join(d, "evolutionary", "stats.json")))["seed_metadata"]
    man = json.load(open(os.path.join(d, "manifest.json")))
    meta = {
        "seed_dir": seed_name,
        "seed_idx": man["seed_idx"],
        "anchor_class": sm["anchor_class_concrete"],
        "target_class": sm["target_class_concrete"],
        "level_anchor": sm["level_anchor"],
        "level_target": sm["level_target"],
        "common_ancestor_level": sm["common_ancestor_level"],
        "seed_idx_in_class": sm["seed_idx_in_class"],
        "anchor_label_in_prompt": sm["anchor_label_in_prompt"],
        "target_label_in_prompt": sm["target_label_in_prompt"],
        "n_pareto": man["n_pareto"],
    }
    meta_rows.append(meta)

    for a in man["anchors"]:
        anchor_rows.append({**meta, **a})

    arch = pd.read_parquet(ap)
    for k, v in meta.items():
        arch[k] = v
    arch_rows.append(arch)

    s1 = pd.read_parquet(os.path.join(d, "pdq", "stage1_flips.parquet"))
    for k, v in meta.items():
        s1[k] = v
    s1_rows.append(s1)

    s2 = pd.read_parquet(os.path.join(d, "pdq", "stage2_trajectories.parquet"))
    for k, v in meta.items():
        s2[k] = v
    s2_rows.append(s2)

arch = pd.concat(arch_rows, ignore_index=True)
s1 = pd.concat(s1_rows, ignore_index=True)
s2 = pd.concat(s2_rows, ignore_index=True)
anchors = pd.DataFrame(anchor_rows)
meta_df = pd.DataFrame(meta_rows)

# modality split of genotype_min: genes 0..221 image, 222..240 text
def modality(g):
    g = np.asarray(g)
    img = int((g[:222] != 0).sum())
    txt = int((g[222:] != 0).sum())
    return img, txt

mods = arch["genotype_min"].apply(modality)
arch["img_active_min"] = [m[0] for m in mods]
arch["txt_active_min"] = [m[1] for m in mods]

modsf = arch["genotype_flipped"].apply(modality)
arch["img_active_flipped"] = [m[0] for m in modsf]
arch["txt_active_flipped"] = [m[1] for m in modsf]

# evo-target hit
arch["flip_on_evo_target"] = arch["label_flipped"] == arch["target_class"]
arch["min_on_evo_target"] = arch["label_min"] == arch["target_class"]

# numeric coercion for object-typed distance cols
for c in ["d_o_label_embedding", "d_o_wordnet_path", "evolutionary_gen"]:
    arch[c] = pd.to_numeric(arch[c], errors="coerce")

# drop heavy cols not needed downstream (keep genotype hashes for anchor dup check)
arch["genotype_anchor_hash"] = arch["genotype_anchor"].apply(lambda g: hash(tuple(np.asarray(g).tolist())))
arch_slim = arch.drop(columns=["genotype_anchor", "genotype_flipped", "genotype_min",
                               "logprobs_anchor", "logprobs_flipped", "logprobs_min"])
arch_slim.to_parquet(os.path.join(OUT, "_combined_archive.parquet"))
s1.drop(columns=["genotype_flipped"]).to_parquet(os.path.join(OUT, "_combined_stage1.parquet"))
s2.to_parquet(os.path.join(OUT, "_combined_stage2.parquet"))
anchors.to_parquet(os.path.join(OUT, "_combined_anchors.parquet"))
meta_df.to_parquet(os.path.join(OUT, "_combined_meta.parquet"))

print("seeds:", len(meta_df))
print("archive rows (refined flips):", len(arch))
print("stage1 flips:", len(s1))
print("stage2 trajectory steps:", len(s2))
print("anchors:", len(anchors))
print("\ntarget_class counts (seeds):")
print(meta_df["target_class"].value_counts().to_string())
print("\nabstraction cells (level_anchor, level_target) seed counts:")
print(meta_df.groupby(["level_anchor", "level_target"]).size().to_string())
print("\nvalidity:", arch["validity"].value_counts().to_dict())
print("anchor_source:", arch["anchor_source"].value_counts().to_dict())
