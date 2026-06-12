"""m3_02: per-cell flip fraction (g<0) and angular location of flips, smoo pair2."""
import pandas as pd
import numpy as np

STORE = "/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography/exp100"
cols = ["source", "anchor_class", "target_class", "level_anchor", "level_target",
        "anchor_word", "target_word", "n_active_img", "n_active_txt",
        "g_pair", "image_dim", "d_img_sem", "d_txt_sem"]
pts = pd.read_parquet(f"{STORE}/points.parquet", columns=cols)
s = pts[(pts.source == "smoo") & (pts.anchor_class == "junco")].copy()

xi = (s.n_active_img / s.image_dim).to_numpy()
yi = (s.n_active_txt / 19).to_numpy()
qx, qy = np.quantile(xi, .99), np.quantile(yi, .99)
s["theta"] = np.degrees(np.arctan2(yi / qy, xi / qx))
s["r"] = np.hypot(xi / qx, yi / qy)

rows = []
for (tc, la, lt), c in s.groupby(["target_class", "level_anchor", "level_target"]):
    neg = c[c.g_pair < 0]
    rows.append({
        "cell": f"{tc[:12]:12s} La{la} Lt{lt}",
        "aw": c.anchor_word.iloc[0], "tw": c.target_word.iloc[0],
        "n": len(c), "n_neg": len(neg),
        "frac_neg": len(neg) / len(c),
        "g_q05": c.g_pair.quantile(.05),
        "g_med": c.g_pair.median(),
        "th_neg_med": neg.theta.median() if len(neg) else np.nan,
        "th_neg_iqr": (neg.theta.quantile(.75) - neg.theta.quantile(.25)) if len(neg) > 4 else np.nan,
        "r_neg_med": neg.r.median() if len(neg) else np.nan,
    })
df = pd.DataFrame(rows).sort_values("frac_neg", ascending=False)
pd.set_option("display.width", 200)
print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
