"""m3_01: probe candidate (theta, r) definitions for polar boundary maps.

Question 1: under which per-axis scaling does smoo's (img, txt) strength give a
usable angular spread (not collapsed to 90 deg)?
Question 2: per cell, do per-theta-bin median-g(r) profiles actually cross 0?
"""
import pandas as pd
import numpy as np

STORE = "/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography/exp100"

cols = ["source", "anchor_class", "target_class", "level_anchor", "level_target",
        "anchor_word", "target_word", "n_active_img", "n_active_txt",
        "rank_sum_img_norm", "rank_sum_txt_norm", "d_img_sem", "d_txt_sem",
        "g_pair", "image_dim"]
pts = pd.read_parquet(f"{STORE}/points.parquet", columns=cols)
smoo = pts[(pts.source == "smoo") & (pts.anchor_class == "junco")].copy()
print("smoo junco rows:", len(smoo))

# words per cell
w = smoo.groupby(["target_class", "level_anchor", "level_target"])[
    ["anchor_word", "target_word"]].agg(lambda s: s.iloc[0])
print(w)

def theta_spread(x, y, name):
    th = np.degrees(np.arctan2(y, x))
    print(f"{name}: theta deg quantiles 5/25/50/75/95 =",
          np.round(np.percentile(th[np.isfinite(th)], [5, 25, 50, 75, 95]), 1),
          " frac theta>85deg =", round((th > 85).mean(), 3))

# candidate scalings
for tag, (xi, yi) in {
    "raw rank_sum": (smoo.rank_sum_img_norm, smoo.rank_sum_txt_norm),
    "n_active frac": (smoo.n_active_img / smoo.image_dim, smoo.n_active_txt / 19),
    "rank_sum/q99": (smoo.rank_sum_img_norm / smoo.rank_sum_img_norm.quantile(.99),
                     smoo.rank_sum_txt_norm / smoo.rank_sum_txt_norm.quantile(.99)),
    "n_active/q99": ((smoo.n_active_img / smoo.image_dim) / (smoo.n_active_img / smoo.image_dim).quantile(.99),
                     (smoo.n_active_txt / 19) / (smoo.n_active_txt / 19).quantile(.99)),
    "sem/q99": (smoo.d_img_sem / smoo.d_img_sem.quantile(.99),
                smoo.d_txt_sem / smoo.d_txt_sem.quantile(.99)),
    "rank_sum sqrt(x/q99)": (np.sqrt(smoo.rank_sum_img_norm / smoo.rank_sum_img_norm.quantile(.99)),
                             np.sqrt(smoo.rank_sum_txt_norm / smoo.rank_sum_txt_norm.quantile(.99))),
}.items():
    theta_spread(xi.values, yi.values, tag)
    print("   q99 values: x=%.4g y=%.4g" % (xi.quantile(.99) if not tag.endswith("q99") else 1,
                                            yi.quantile(.99) if not tag.endswith("q99") else 1))

print("\nq99 raw: img=%.4g txt=%.4g  sem img=%.4g txt=%.4g" % (
    smoo.rank_sum_img_norm.quantile(.99), smoo.rank_sum_txt_norm.quantile(.99),
    smoo.d_img_sem.quantile(.99), smoo.d_txt_sem.quantile(.99)))

# --- Question 2: g(r) crossing per theta bin, two theta definitions ---
defs = {}
# combinatorial: n_active fractions / q99
xi = (smoo.n_active_img / smoo.image_dim).to_numpy()
yi = (smoo.n_active_txt / 19).to_numpy()
defs["n_active/q99"] = (xi / np.quantile(xi, .99), yi / np.quantile(yi, .99))
# semantic: d_sem / q99
xs = smoo.d_img_sem.to_numpy()
ys = smoo.d_txt_sem.to_numpy()
defs["sem/q99"] = (xs / np.quantile(xs, .99), ys / np.quantile(ys, .99))

TH_BINS = np.linspace(0, 90, 10)   # 9 bins of 10 deg

cells = [("boa constrictor", 0, 1, "WALL boa/snake"),
         ("cello", 1, 0, "WALL songbird/cello"),
         ("marimba", 0, 0, "easy junco/marimba"),
         ("green iguana", 0, 0, "easy junco/iguana"),
         ("ostrich", 0, 0, "easy junco/ostrich"),
         ("boa constrictor", 0, 0, "junco/boa concrete")]

for dname, (X, Y) in defs.items():
    theta = np.degrees(np.arctan2(Y, X))
    r = np.hypot(X, Y)
    print(f"\n=== theta def: {dname} ===  r q5/50/95/99:",
          np.round(np.percentile(r, [5, 50, 95, 99]), 3))
    R_BINS = np.linspace(0, np.quantile(r, .995), 13)
    g = smoo.g_pair.to_numpy()
    for tc, la, lt, tag in cells:
        m_cell = ((smoo.target_class == tc) & (smoo.level_anchor == la)
                  & (smoo.level_target == lt)).to_numpy()
        gi = np.digitize(theta[m_cell], TH_BINS) - 1
        ri = np.digitize(r[m_cell], R_BINS) - 1
        gc = g[m_cell]
        n_cross = 0
        n_th_with_data = 0
        for t in range(len(TH_BINS) - 1):
            med = []
            for rr in range(len(R_BINS) - 1):
                m = (gi == t) & (ri == rr)
                med.append(np.median(gc[m]) if m.sum() >= 8 else np.nan)
            med = np.array(med)
            if np.isfinite(med).sum() >= 2:
                n_th_with_data += 1
                if np.nanmin(med) < 0 < np.nanmax(med):
                    n_cross += 1
        frac_neg = (gc < 0).mean()
        print(f"{tag:26s} n={m_cell.sum():6d} fracg<0={frac_neg:.3f} "
              f"theta-bins with data={n_th_with_data}/9, crossing={n_cross}")
