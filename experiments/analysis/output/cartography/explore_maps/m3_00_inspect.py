"""m3_00: inspect data ranges relevant for polar (direction-resolved) maps."""
import pandas as pd
import numpy as np

STORE = "/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography/exp100"

cols = ["source", "prompt_regime", "anchor_class", "target_class", "level_anchor",
        "level_target", "n_active_img", "n_active_txt", "rank_sum_img_norm",
        "rank_sum_txt_norm", "hamming_to_anchor", "d_img_sem", "d_txt_sem",
        "g_pair", "pair_margin", "pred_label", "image_dim",
        "txt_active_mlm", "txt_active_frag", "txt_active_charnoise", "txt_active_saliency"]
pts = pd.read_parquet(f"{STORE}/points.parquet", columns=cols)
print("points:", len(pts))
print(pts.groupby(["prompt_regime", "source"]).size())
print("\ncells (anchor=junco):")
j = pts[pts.anchor_class == "junco"]
print(j.groupby(["target_class", "level_anchor", "level_target"]).size())

print("\nrank_sum_img_norm / txt_norm ranges by source:")
print(pts.groupby("source")[["rank_sum_img_norm", "rank_sum_txt_norm"]].describe().T)

smoo = pts[pts.source == "smoo"]
print("\nsmoo g_pair stats:", smoo.g_pair.describe())
print("smoo d_img_sem:", smoo.d_img_sem.describe())
print("smoo d_txt_sem nonzero frac:", (smoo.d_txt_sem > 0).mean(), "describe:", smoo.d_txt_sem.describe())
print("\nsmoo n_active_img/img_dim:", (smoo.n_active_img / smoo.image_dim).describe())
print("smoo n_active_txt/19:", (smoo.n_active_txt / 19).describe())

s1 = pts[pts.source == "pdq_s1"]
print("\npdq_s1 pair_margin:", s1.pair_margin.describe())
print("pdq_s1 d_img_sem:", s1.d_img_sem.describe())
print("pdq_s1 d_txt_sem:", s1.d_txt_sem.describe())

# straddles
st = pd.read_parquet(f"{STORE}/straddle_pairs.parquet")
print("\nstraddles:", len(st), "cols:", list(st.columns))
print(st.boundary_kind.value_counts())
print(st.gene_modality.value_counts())
print(st.txt_group.value_counts(dropna=False))
mcols = [c for c in st.columns if c.startswith("m_")]
print("midpoint cols:", mcols)
print(st[["m_rank_sum_img_norm", "m_rank_sum_txt_norm"]].describe())
stj = st[st.anchor_class == "junco"]
print("\nstraddle cells:")
print(stj.groupby(["target_class", "level_anchor", "level_target"]).size())
