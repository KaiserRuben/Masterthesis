"""m4_01: per-seed semantic axis scale check for the chosen contrast cells."""
import pyarrow.parquet as pq
import pandas as pd

BASE = "/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography/exp100"
cols = ["source", "generation", "anchor_class", "target_class", "level_anchor",
        "level_target", "seed_dir", "g_pair", "d_img_sem", "d_txt_sem",
        "n_active_img", "n_active_txt"]
df = pq.read_table(f"{BASE}/points.parquet", columns=cols).to_pandas()
smoo = df[(df.source == "smoo") & (df.anchor_class == "junco")].copy()
smoo["cell"] = (smoo.target_class.astype(str) + " L" +
                smoo.level_anchor.astype(str) + "-" + smoo.level_target.astype(str))

CELLS = ["boa constrictor L0-1", "cello L1-1", "marimba L2-1", "green iguana L2-0"]
for c in CELLS:
    d = smoo[smoo.cell == c]
    print("\n===", c, "===")
    q = d.groupby("seed_dir")[["d_img_sem", "d_txt_sem"]].quantile([0.5, 0.99]).unstack()
    print(q.to_string())
    print("d_txt_sem==0 frac:", (d.d_txt_sem == 0).mean().round(3),
          "| g_pair med by gen-window:")
    gw = d.groupby(pd.cut(d.generation, [0, 50, 100, 150, 200], right=False),
                   observed=True).g_pair.median()
    print(gw.to_string())
