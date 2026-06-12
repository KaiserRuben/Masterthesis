"""m3_03: probe straddle midpoint axes for the stake compass."""
import pandas as pd
import numpy as np

STORE = "/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography/exp100"
st = pd.read_parquet(f"{STORE}/straddle_pairs.parquet")
st = st[st.anchor_class == "junco"]
print("junco straddles:", len(st))

for tag, (x, y) in {
    "raw rank_sum": (st.m_rank_sum_img_norm, st.m_rank_sum_txt_norm),
    "n_active frac": (st.m_n_active_img / st.image_dim, st.m_n_active_txt / 19),
}.items():
    x, y = x.to_numpy(), y.to_numpy()
    th = np.degrees(np.arctan2(y, x))
    print(f"{tag}: theta q5/25/50/75/95 =", np.round(np.percentile(th, [5,25,50,75,95]),1),
          "frac>85:", round((th>85).mean(),3), " x q99=%.3f y q99=%.3f" % (np.quantile(x,.99), np.quantile(y,.99)))
    xh, yh = x/np.quantile(x,.99), y/np.quantile(y,.99)
    th2 = np.degrees(np.arctan2(yh, xh))
    print("   q99-scaled: q5/25/50/75/95 =", np.round(np.percentile(th2,[5,25,50,75,95]),1),
          "frac>85:", round((th2>85).mean(),3))

print("\nboundary_kind x gene_modality:")
print(st.groupby(["boundary_kind", "gene_modality"]).size())

cells = [("boa constrictor",2,2),("green iguana",2,0),("marimba",2,1),
         ("boa constrictor",0,1),("cello",1,1),("ostrich",1,0)]
for tc, la, lt in cells:
    c = st[(st.target_class==tc)&(st.level_anchor==la)&(st.level_target==lt)]
    print(f"{tc:16s} La{la}Lt{lt}: n={len(c):4d} ", dict(c.boundary_kind.value_counts()),
          dict(c.gene_modality.value_counts()))
# hamming radius alternative
print("\nhamming_to_anchor_after describe:", st.hamming_to_anchor_after.describe().round(1).to_dict())
