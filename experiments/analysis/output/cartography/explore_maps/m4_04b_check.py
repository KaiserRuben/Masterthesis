"""Sanity-check descent/survival numbers per group."""
import sys
sys.path.insert(0, "/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography/explore_maps")
import numpy as np
import m4_common as M

NOISE_G = float(np.tanh(0.38 / 2))
df = M.load_smoo(None, columns=["source", "generation", "anchor_class",
                                "target_class", "level_anchor", "level_target",
                                "seed_dir", "g_pair", "d_img_sem", "d_txt_sem"])

def group_of(row):
    if row.target_class == "boa constrictor" and row.level_target == 1:
        return "wall-boa"
    if row.target_class == "cello" and row.level_anchor == 1:
        return "wall-cello"
    if row.target_class in ("marimba", "green iguana", "ostrich"):
        return "easy"
    if row.target_class == "junco":
        return None
    return "other"

ck = df[["cell", "target_class", "level_anchor", "level_target"]].drop_duplicates()
ck["group"] = ck.apply(group_of, axis=1)
df = df.merge(ck[["cell", "group"]], on="cell")
df = df[df.group.notna()]
df["abs_g"] = df.g_pair.abs()

per_gen = (df.groupby(["group", "cell", "seed_dir", "generation"]).abs_g.min()
             .groupby(["group", "cell", "seed_dir"]).cummin().reset_index())

final = per_gen[per_gen.generation == 199]
for g, sub in final.groupby("group"):
    print(f"\n{g}: n={len(sub)}")
    print("  final best |g| per seed:", np.sort(sub.abs_g.values).round(3))
    print("  frac touched noise band:", (sub.abs_g < NOISE_G).mean().round(3))
# wall-cello detail
wc = per_gen[per_gen.group == "wall-cello"]
ft = wc[wc.abs_g < NOISE_G].groupby(["cell", "seed_dir"]).generation.min()
print("\nwall-cello first-touch gens:")
print(ft.to_string())
