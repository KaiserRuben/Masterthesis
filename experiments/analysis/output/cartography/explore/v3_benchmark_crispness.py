"""v3 — Which 2D projection makes the decision boundary crispest?

Benchmarks candidate 2D projections of the cartography point cloud by how well
the two boundary sides (side = sign(pair_margin)) separate *in the projected
plane*. The boundary is codim-1 in ~241-dim gene space; every 2D map smears it
into a band — these metrics measure band crispness.

Crispness metrics (complementary by construction):
  M1 knn_auc      5-fold CV AUC of kNN(k=25) on the 2 standardized coords.
                  Nonparametric: total side-information retained by the view,
                  regardless of boundary shape. Upper-bounds any frontier you
                  could draw on the map.
  M2 logit_auc    Same CV, logistic regression. Separability by ONE straight
                  cut. Gap M1-M2 = curvature/folding of the boundary in this
                  view (a map you can read with a single monotone frontier has
                  a small gap).
  M3 sep_1mBC     1 - Bhattacharyya coefficient of the side-conditional 2D
                  histograms (32x32, sides equally weighted). Classifier-free
                  distribution overlap; directly penalizes band thickness.
  M4 straddle_amb mean 2|p_hat - 0.5| where p_hat = kNN side-probability of
                  the view evaluated at hamming-1 straddle MIDPOINTS (surveyed
                  boundary points). LOWER = crisper: ground-truth boundary
                  points should land in the map's own ambiguous zone.
                  cat6 only (straddles come from PDQ stage 2).

Projections:
  ranksum   rank_sum_img_norm x rank_sum_txt_norm   (incumbent)
  nactive   n_active_img      x n_active_txt
  semantic  d_img_sem         x d_txt_sem           (smoo + pdq_s1 only)
  hamming   hamming_to_anchor x rank_sum_txt_norm   (cat6 only; const for smoo)
  pca2      PCA-2 of the standardized 8-feature combinatorial descriptor block
  lda_pc    LDA(side) axis x PC1 of the orthogonal residual (supervised upper
            bound for linear descriptor-based views; mild optimism, 8 features)

Cells: wall boa(0,1)=sparrow->snake, control boa(0,0)=sparrow->constrictor,
easy marimba(0,1)=sparrow->percussion, mid iguana(2,0)=bird->iguana, + pooled.
Regimes pair2 / cat6 are benchmarked separately (different boundaries).

Outputs (this dir): v3_crispness_benchmark.csv, v3_benchmark_heatmap.png,
v3_axis_correlations.csv, v3_axis_corr.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/kaiser/Projects/Masterarbeit")
import matplotlib.pyplot as plt  # noqa: E402
from analysis.core.style import apply_style, save_fig  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402

BASE = Path("/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography/exp100")
OUT = Path("/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography/explore")
RNG = np.random.default_rng(0)
K = 25
MAX_N = 12000
MAX_N_POOLED = 24000
MIN_MINORITY = 40

FEATURES = ["n_active_img", "n_active_txt", "rank_sum_img_norm", "rank_sum_txt_norm",
            "txt_active_mlm", "txt_active_frag", "txt_active_charnoise", "txt_active_saliency"]

CELLS = {
    "wall boa s->snake (0,1)":      ("boa constrictor", 0, 1),
    "ctrl boa s->constr (0,0)":     ("boa constrictor", 0, 0),
    "easy marimba s->perc (0,1)":   ("marimba", 0, 1),
    "mid iguana b->iguana (2,0)":   ("green iguana", 2, 0),
    "pooled":                        None,
}

PROJECTIONS = ["ranksum", "nactive", "semantic", "hamming", "pca2", "lda_pc"]


# ---------------------------------------------------------------- data loading

def load_points() -> pd.DataFrame:
    cols = ["source", "prompt_regime", "target_class", "level_anchor", "level_target",
            "anchor_word", "target_word", "pair_margin", "seed_dir", "row_ref",
            "n_active_img", "n_active_txt", "rank_sum_img", "rank_sum_txt",
            "rank_sum_img_norm", "rank_sum_txt_norm", "hamming_to_anchor",
            "d_img_sem", "d_txt_sem",
            "txt_active_mlm", "txt_active_frag", "txt_active_charnoise", "txt_active_saliency"]
    df = pd.read_parquet(BASE / "points.parquet", columns=cols)
    df["side"] = (df.pair_margin > 0).astype(int)  # 1 = anchor side
    return df


def seed_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Recover per-seed rank-sum normalization bounds (rank_sum / norm)."""
    gi = df[df.rank_sum_img > 0].groupby("seed_dir").apply(
        lambda s: (s.rank_sum_img / s.rank_sum_img_norm).median(), include_groups=False)
    gt = df[df.rank_sum_txt > 0].groupby("seed_dir").apply(
        lambda s: (s.rank_sum_txt / s.rank_sum_txt_norm).median(), include_groups=False)
    return pd.DataFrame({"bound_img": gi, "bound_txt": gt})


def load_straddle_midpoints(df: pd.DataFrame, bounds: pd.DataFrame) -> pd.DataFrame:
    """Hamming-1 straddle pairs whose evo-pair lp margin flips sign -> exact
    midpoint combinatorial descriptors, reconstructed from the joined after-
    point plus the single gene edit (store lacks midpoint columns)."""
    sp = pd.read_parquet(BASE / "straddle_pairs.parquet")
    sp = sp[np.sign(sp.margin_before) != np.sign(sp.margin_after)].copy()

    pdq = df[df.source.str.startswith("pdq")].copy()
    pdq["ref_int"] = pd.to_numeric(pdq.row_ref, errors="coerce")
    after = pdq.set_index(["seed_dir", "ref_int"])
    sp = sp.join(after[["n_active_img", "n_active_txt", "rank_sum_img", "rank_sum_txt",
                        "hamming_to_anchor", *FEATURES[4:]]],
                 on=["seed_dir", "call_id_after"], how="inner")

    d_active = (sp.value_before != 0).astype(int) - (sp.value_after != 0).astype(int)
    d_rank = (sp.value_before - sp.value_after).astype(float)
    is_img = (sp.gene_modality == "img").astype(int)
    is_txt = 1 - is_img

    sp["m_n_active_img"] = sp.n_active_img + 0.5 * d_active * is_img
    sp["m_n_active_txt"] = sp.n_active_txt + 0.5 * d_active * is_txt
    rs_img = sp.rank_sum_img + 0.5 * d_rank * is_img
    rs_txt = sp.rank_sum_txt + 0.5 * d_rank * is_txt
    sp = sp.join(bounds, on="seed_dir")
    sp["m_rank_sum_img_norm"] = rs_img / sp.bound_img
    sp["m_rank_sum_txt_norm"] = rs_txt / sp.bound_txt
    sp["m_hamming_to_anchor"] = sp.hamming_to_anchor  # +-1 resolution floor
    for grp in ["mlm", "frag", "charnoise", "saliency"]:
        col = f"txt_active_{grp}"
        sp[f"m_{col}"] = sp[col] + 0.5 * d_active * is_txt * (sp.txt_group == grp).astype(int)
    return sp


# ---------------------------------------------------------------- projections

def project(sub: pd.DataFrame, mids: pd.DataFrame | None, proj: str):
    """Return (XY points, XY midpoints or None, axis labels) or None if N/A."""
    def cols2(c1, c2, m1, m2):
        xy = sub[[c1, c2]].to_numpy(float)
        mxy = mids[[m1, m2]].to_numpy(float) if mids is not None and len(mids) else None
        return xy, mxy

    if proj == "ranksum":
        xy, mxy = cols2("rank_sum_img_norm", "rank_sum_txt_norm",
                        "m_rank_sum_img_norm", "m_rank_sum_txt_norm")
        return xy, mxy, ("rank_sum_img_norm", "rank_sum_txt_norm")
    if proj == "nactive":
        xy, mxy = cols2("n_active_img", "n_active_txt", "m_n_active_img", "m_n_active_txt")
        return xy, mxy, ("n_active_img", "n_active_txt")
    if proj == "semantic":
        ok = sub.d_img_sem.notna() & sub.d_txt_sem.notna()
        if ok.sum() < 200:
            return None
        xy = sub.loc[ok, ["d_img_sem", "d_txt_sem"]].to_numpy(float)
        return (xy, None, ("d_img_sem", "d_txt_sem")), ok  # special: row mask
    if proj == "hamming":
        if sub.hamming_to_anchor.std() < 1e-9:
            return None
        xy, mxy = cols2("hamming_to_anchor", "rank_sum_txt_norm",
                        "m_hamming_to_anchor", "m_rank_sum_txt_norm")
        return xy, mxy, ("hamming_to_anchor", "rank_sum_txt_norm")
    if proj in ("pca2", "lda_pc"):
        if proj == "lda_pc" and sub.side.value_counts().reindex([0, 1]).fillna(0).min() < MIN_MINORITY:
            return None  # LDA undefined / unstable with a (near-)absent side
        X = sub[FEATURES].to_numpy(float)
        mu, sd = X.mean(0), X.std(0)
        sd[sd < 1e-12] = 1.0
        Xz = (X - mu) / sd
        Xm = None
        if mids is not None and len(mids):
            Xm = (mids[[f"m_{f}" for f in FEATURES]].to_numpy(float) - mu) / sd
        if proj == "pca2":
            p = PCA(n_components=2, random_state=0).fit(Xz)
            return p.transform(Xz), (p.transform(Xm) if Xm is not None else None), \
                tuple(_pc_label(p.components_[i], FEATURES) for i in range(2))
        lda = LinearDiscriminantAnalysis(n_components=1).fit(Xz, sub.side.to_numpy())
        w = lda.coef_[0] / np.linalg.norm(lda.coef_[0])
        resid = Xz - np.outer(Xz @ w, w)
        p1 = PCA(n_components=1, random_state=0).fit(resid)
        v = p1.components_[0]
        xy = np.column_stack([Xz @ w, Xz @ v])
        mxy = np.column_stack([Xm @ w, Xm @ v]) if Xm is not None else None
        return xy, mxy, (_pc_label(w, FEATURES, "LDA"), _pc_label(v, FEATURES, "rPC1"))
    raise ValueError(proj)


def _pc_label(vec, names, tag="PC"):
    i = np.argsort(-np.abs(vec))[:2]
    return f"{tag}[" + ", ".join(f"{vec[j]:+.2f}{names[j]}" for j in i) + "]"


# ------------------------------------------------------------------- metrics

def crispness(xy: np.ndarray, y: np.ndarray, mxy: np.ndarray | None) -> dict:
    out = {"n": len(y), "n_minority": int(min((y == 0).sum(), (y == 1).sum()))}
    if out["n_minority"] < MIN_MINORITY:
        return out
    mu, sd = xy.mean(0), xy.std(0)
    sd[sd < 1e-12] = 1.0
    Z = (xy - mu) / sd

    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    auc_knn, auc_log = [], []
    for tr, te in skf.split(Z, y):
        kn = KNeighborsClassifier(n_neighbors=K).fit(Z[tr], y[tr])
        auc_knn.append(roc_auc_score(y[te], kn.predict_proba(Z[te])[:, 1]))
        lg = LogisticRegression(max_iter=1000).fit(Z[tr], y[tr])
        auc_log.append(roc_auc_score(y[te], lg.predict_proba(Z[te])[:, 1]))
    out["knn_auc"] = float(np.mean(auc_knn))
    out["knn_auc_sd"] = float(np.std(auc_knn))
    out["logit_auc"] = float(np.mean(auc_log))

    # Bhattacharyya coefficient on 32x32 histograms, sides equally weighted
    lo = np.percentile(Z, 0.5, axis=0)
    hi = np.percentile(Z, 99.5, axis=0)
    hi = np.maximum(hi, lo + 1e-9)
    bins = [np.linspace(lo[d], hi[d], 33) for d in range(2)]
    Zc = np.clip(Z, lo, hi - 1e-12)
    h0, _, _ = np.histogram2d(Zc[y == 0, 0], Zc[y == 0, 1], bins=bins)
    h1, _, _ = np.histogram2d(Zc[y == 1, 0], Zc[y == 1, 1], bins=bins)
    bc = float(np.sum(np.sqrt((h0 / h0.sum()) * (h1 / h1.sum()))))
    out["sep_1mBC"] = 1.0 - bc

    if mxy is not None and len(mxy) >= 30:
        Zm = (mxy - mu) / sd
        kn = KNeighborsClassifier(n_neighbors=K).fit(Z, y)
        p = kn.predict_proba(Zm)[:, 1]
        out["straddle_amb"] = float(np.mean(2 * np.abs(p - 0.5)))
        out["straddle_n"] = len(mxy)
    return out


def subsample(sub: pd.DataFrame, cap: int) -> pd.DataFrame:
    if len(sub) <= cap:
        return sub
    # stratified by side, keep all minority up to cap/2
    parts = []
    for s, grp in sub.groupby("side"):
        frac = cap * len(grp) / len(sub)
        n = int(min(len(grp), max(frac, min(len(grp), cap // 4))))
        parts.append(grp.sample(n=n, random_state=0))
    return pd.concat(parts)


# ---------------------------------------------------------------------- main

def main():
    apply_style()
    df = load_points()
    bounds = seed_bounds(df)
    sp = load_straddle_midpoints(df, bounds)
    print(f"points {len(df)}, straddle midpoints (margin-flip, joined) {len(sp)}")

    rows = []
    for regime in ["pair2", "cat6"]:
        dreg = df[df.prompt_regime == regime]
        spreg = sp  # straddles are cat6-surveyed; only used for cat6 below
        for cell_name, key in CELLS.items():
            if key is None:
                sub, mids = dreg, (spreg if regime == "cat6" else None)
                cap = MAX_N_POOLED
            else:
                tc, la, lt = key
                m = (dreg.target_class == tc) & (dreg.level_anchor == la) & (dreg.level_target == lt)
                sub = dreg[m]
                mids = None
                if regime == "cat6":
                    mm = (spreg.target_class == tc) & (spreg.level_anchor == la) & (spreg.level_target == lt)
                    mids = spreg[mm]
                cap = MAX_N
            sub = subsample(sub, cap)
            y_all = sub.side.to_numpy()
            for proj in PROJECTIONS:
                res = project(sub, mids, proj)
                rec = {"regime": regime, "cell": cell_name, "projection": proj}
                if res is None:
                    rec.update({"n": len(sub), "note": "axis unavailable/degenerate"})
                elif isinstance(res[0], tuple):           # semantic row-mask case
                    (xy, mxy, axes), ok = res
                    rec.update(crispness(xy, y_all[ok.to_numpy()], mxy))
                    rec["axes"] = " | ".join(axes)
                    rec["note"] = f"semantic rows only ({ok.sum()}/{len(sub)})"
                else:
                    xy, mxy, axes = res
                    rec.update(crispness(xy, y_all, mxy))
                    rec["axes"] = " | ".join(axes)
                rows.append(rec)
                print(f"{regime:5s} {cell_name:28s} {proj:8s} "
                      f"knn={rec.get('knn_auc', float('nan')):.3f} "
                      f"logit={rec.get('logit_auc', float('nan')):.3f} "
                      f"sep={rec.get('sep_1mBC', float('nan')):.3f} "
                      f"stramb={rec.get('straddle_amb', float('nan')):.3f}")

    bench = pd.DataFrame(rows)
    bench.to_csv(OUT / "v3_crispness_benchmark.csv", index=False)
    print("wrote v3_crispness_benchmark.csv")

    # ---------------- heatmap figure ----------------
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    metrics = [("knn_auc", "kNN AUC (M1, higher=crisper)"),
               ("sep_1mBC", "1 - Bhattacharyya (M3, higher=crisper)")]
    for j, regime in enumerate(["pair2", "cat6"]):
        for i, (met, title) in enumerate(metrics):
            ax = axs[i, j]
            piv = bench[bench.regime == regime].pivot_table(
                index="cell", columns="projection", values=met)
            piv = piv.reindex(index=list(CELLS), columns=PROJECTIONS)
            im = ax.imshow(piv.to_numpy(), cmap="viridis", aspect="auto",
                           vmin=(0.5 if met == "knn_auc" else 0), vmax=1)
            ax.set_xticks(range(len(piv.columns)), piv.columns, rotation=30, ha="right")
            ax.set_yticks(range(len(piv.index)), piv.index, fontsize=8)
            ax.set_title(f"{regime} — {title}", fontsize=10)
            ax.grid(False)
            for (r, c), v in np.ndenumerate(piv.to_numpy()):
                if np.isfinite(v):
                    ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if v < (0.8 if met == "knn_auc" else 0.55) else "black")
                else:
                    ax.text(c, r, "—", ha="center", va="center", fontsize=8, color="grey")
            fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Projection crispness benchmark — side separability in the 2D view", y=1.0)
    save_fig(fig, OUT / "v3_benchmark_heatmap.png")

    # ---------------- axis correlation matrices per source ----------------
    axes_cols = ["n_active_img", "n_active_txt", "rank_sum_img_norm", "rank_sum_txt_norm",
                 "d_img_sem", "d_txt_sem", "hamming_to_anchor", "pair_margin"]
    corr_frames = []
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, src in zip(axs, ["smoo", "pdq_s1", "pdq_s2"]):
        s = df[df.source == src]
        if len(s) > 60000:
            s = s.sample(60000, random_state=0)
        cm = s[axes_cols].corr(method="spearman")
        cf = cm.copy()
        cf["source"] = src
        corr_frames.append(cf)
        im = ax.imshow(cm.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(axes_cols)), axes_cols, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(axes_cols)), axes_cols, fontsize=7)
        ax.set_title(f"{src} (n={len(s)})", fontsize=10)
        ax.grid(False)
        for (r, c), v in np.ndenumerate(cm.to_numpy()):
            if np.isfinite(v):
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, ax=axs, shrink=0.75)
    fig.suptitle("Candidate-axis Spearman correlations per source")
    save_fig(fig, OUT / "v3_axis_corr.png", tight=False)
    pd.concat(corr_frames).to_csv(OUT / "v3_axis_correlations.csv")
    print("wrote v3_axis_correlations.csv")


if __name__ == "__main__":
    main()
