"""Codebook homogeneity analysis for the f8-16384 VQ-GAN preset.

Prior to bumping ``n_candidates`` from 25 to 16383 (full codebook), we need
to know whether deep neighbor swaps (large k) visit different regions of
latent space, or whether the codebook is so homogeneous that any k yields
essentially the same cosine shift.

Produces three figures under ``analysis/exp09/codebook/`` and prints a
<=500 word report to stdout.

NumPy-only, CPU-only, no changes to pipeline code.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make ``src`` importable without editable install metadata drift.
ROOT = Path("/Users/kaiser/Projects/Masterarbeit")
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.manipulator.image.loading import load_vqgan

OUT = ROOT / "analysis" / "exp09" / "codebook"
OUT.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(0)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n


def cosine_distance_matrix(unit: np.ndarray) -> np.ndarray:
    """Return full NxN cosine distance (1 - cos sim) given unit-norm rows."""
    sim = unit @ unit.T
    np.clip(sim, -1.0, 1.0, out=sim)
    return 1.0 - sim


def main() -> None:
    print("[exp09] Loading f8-16384 VQ-GAN codebook ...", flush=True)
    model = load_vqgan("f8-16384")
    codebook = model.quantize.embedding.weight.detach().cpu().numpy().astype(np.float64)
    N, D = codebook.shape
    print(f"[exp09] Codebook shape: ({N}, {D})")

    unit = l2_normalize(codebook)

    # ------------------------------------------------------------------
    # Analysis 1: pairwise cosine distance distribution
    # ------------------------------------------------------------------
    # Full NxN cosine sim/dist matrix. At N=16384 this is ~2 GB in float64,
    # so use float32 to keep memory manageable.
    print("[exp09] Computing full NxN cosine distance matrix ...", flush=True)
    unit32 = unit.astype(np.float32)
    sim = unit32 @ unit32.T  # (N, N), ~1 GB float32
    np.clip(sim, -1.0, 1.0, out=sim)
    # Upper triangle without diagonal -> all unique pairs
    iu = np.triu_indices(N, k=1)
    pair_dist = 1.0 - sim[iu]
    del iu

    stats = {
        "count": pair_dist.size,
        "mean": float(pair_dist.mean()),
        "std": float(pair_dist.std()),
        "min": float(pair_dist.min()),
        "p5": float(np.percentile(pair_dist, 5)),
        "p50": float(np.percentile(pair_dist, 50)),
        "p95": float(np.percentile(pair_dist, 95)),
        "max": float(pair_dist.max()),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pair_dist, bins=200, color="#3b6ea5", edgecolor="none")
    for q, lbl, color in [
        (stats["p5"], "5%", "#888"),
        (stats["p50"], "50%", "#111"),
        (stats["p95"], "95%", "#888"),
        (stats["mean"], "mean", "#c0392b"),
    ]:
        ax.axvline(q, color=color, linestyle=("--" if lbl != "mean" else "-"), lw=1, label=f"{lbl}={q:.3f}")
    ax.set_xlabel("Cosine distance")
    ax.set_ylabel("Count (pairs)")
    ax.set_title(f"Pairwise cosine distance, f8-16384 codebook (N={N}, D={D})")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "pairwise_distances.png", dpi=150)
    plt.close(fig)

    # Free pair_dist array — keep ``sim`` around for k-NN analysis
    del pair_dist

    # ------------------------------------------------------------------
    # Analysis 2: k-th nearest neighbor curve
    # ------------------------------------------------------------------
    # For each row, sort cosine distance ascending. Row i has self-distance
    # 0 at position 0, so the k-th nearest neighbor is index k (1-based
    # counting of neighbors = position k in a 0-indexed sorted distance row).
    print("[exp09] Sorting distance rows for k-NN analysis ...", flush=True)
    dist = 1.0 - sim
    del sim
    # Sort each row ascending. Fills >4 GB in float32 — do in place to save.
    dist.sort(axis=1)  # (N, N)

    k_values = np.array([1, 25, 100, 250, 500, 1000, 2500, 5000, 10000, 16383])
    # position in sorted row = k (since index 0 is self, distance 0)
    kth = dist[:, k_values]  # (N, len(k_values))

    k_mean = kth.mean(axis=0)
    k_p5 = np.percentile(kth, 5, axis=0)
    k_p95 = np.percentile(kth, 95, axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(k_values, k_p5, k_p95, color="#3b6ea5", alpha=0.25, label="5–95% band")
    ax.plot(k_values, k_mean, color="#3b6ea5", lw=2, marker="o", label="Mean")
    for k_mark, color, lbl in [
        (25, "#c0392b", "planned: k=25"),
        (250, "#d68910", "k=250"),
        (2500, "#7d3c98", "k=2500"),
        (16383, "#117864", "k=16383"),
    ]:
        ax.axvline(k_mark, color=color, linestyle="--", lw=0.8, alpha=0.7)
        idx = int(np.where(k_values == k_mark)[0][0])
        ax.scatter([k_mark], [k_mean[idx]], color=color, zorder=5, label=f"{lbl}: µ={k_mean[idx]:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("k (rank of nearest neighbor)")
    ax.set_ylabel("Cosine distance to k-th NN")
    ax.set_title("k-th nearest neighbor distance vs k, averaged over 16384 codewords")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "kth_nn_curve.png", dpi=150)
    plt.close(fig)

    # Key table entries at the planned experiment k values
    planned_k = [25, 250, 2500, 16383]
    planned_summary = {}
    for k in planned_k:
        idx = int(np.where(k_values == k)[0][0])
        planned_summary[k] = {
            "mean": float(k_mean[idx]),
            "p5": float(k_p5[idx]),
            "p95": float(k_p95[idx]),
        }

    # Free dist matrix before clustering
    del dist, kth

    # ------------------------------------------------------------------
    # Analysis 3: cluster structure
    # ------------------------------------------------------------------
    print("[exp09] Clustering: hierarchical (1000 sample) + k-means silhouette ...", flush=True)

    sample_idx = RNG.choice(N, size=1000, replace=False)
    sample = unit[sample_idx]  # already unit-norm

    # Hierarchical on cosine distance (using scipy condensed form)
    # Condensed pdist: 1000*999/2 ≈ 5e5 entries, fine.
    from scipy.spatial.distance import pdist
    cond = pdist(sample, metric="cosine")
    Z = linkage(cond, method="average")

    # K-means silhouette on the full codebook with cosine via unit-norm points.
    # sklearn KMeans uses Euclidean. For unit-norm vectors, Euclidean²=2(1-cos).
    # Monotonic in cosine, so k-means on unit vectors approximates spherical
    # k-means. Compute silhouette using cosine metric directly.
    ks = [2, 5, 10, 50]
    sil_scores = {}
    # Use a subset of ~5000 points for silhouette (O(n²) metric otherwise).
    sil_idx = RNG.choice(N, size=5000, replace=False)
    sil_data = unit[sil_idx]
    for k in ks:
        km = KMeans(n_clusters=k, n_init=5, random_state=0)
        labels = km.fit_predict(unit)  # fit on full codebook
        sub_labels = labels[sil_idx]
        # If any cluster has <2 samples in the subset, silhouette fails.
        uniq, cnt = np.unique(sub_labels, return_counts=True)
        if (cnt < 2).any() or uniq.size < 2:
            sil_scores[k] = float("nan")
            continue
        score = silhouette_score(sil_data, sub_labels, metric="cosine", sample_size=None)
        sil_scores[k] = float(score)
        print(f"  k={k:3d}: silhouette={sil_scores[k]:+.4f}")

    # Figure: dendrogram + silhouette bar chart
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    dendrogram(
        Z,
        no_labels=True,
        color_threshold=0,
        above_threshold_color="#3b6ea5",
        ax=axes[0],
    )
    axes[0].set_title("Average-linkage dendrogram\n(1000-codeword sample, cosine)")
    axes[0].set_xlabel("Codewords (unlabeled)")
    axes[0].set_ylabel("Cosine-distance merge height")

    k_lbl = list(sil_scores.keys())
    k_val = [sil_scores[k] for k in k_lbl]
    bars = axes[1].bar([str(k) for k in k_lbl], k_val, color="#3b6ea5")
    for bar, v in zip(bars, k_val):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v,
                     f"{v:+.3f}", ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=9)
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_xlabel("k (number of k-means clusters)")
    axes[1].set_ylabel("Silhouette (cosine)")
    axes[1].set_title("K-means silhouette score vs k")
    axes[1].set_ylim(min(-0.05, min(k_val) - 0.05), max(0.25, max(k_val) + 0.05))
    fig.tight_layout()
    fig.savefig(OUT / "clustering.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    def fmt_k(k):
        s = planned_summary[k]
        return f"k={k:>5}: mean={s['mean']:.4f} (5% {s['p5']:.4f}, 95% {s['p95']:.4f})"

    print()
    print("=" * 72)
    print("CODEBOOK HOMOGENEITY REPORT — f8-16384")
    print("=" * 72)
    print(f"Preset            : f8-16384 (ommer-lab vq-f8)")
    print(f"Codebook shape    : ({N}, {D}) — {N} codewords in R^{D}")
    print(f"Pairs analysed    : {stats['count']:,} (full upper triangle)")
    print()
    print("Pairwise cosine distance distribution")
    print("-" * 72)
    print(f"  mean   = {stats['mean']:.4f}   std = {stats['std']:.4f}")
    print(f"  min    = {stats['min']:.4f}   max = {stats['max']:.4f}")
    print(f"  p5     = {stats['p5']:.4f}")
    print(f"  p50    = {stats['p50']:.4f}")
    print(f"  p95    = {stats['p95']:.4f}")
    print()
    print("k-th nearest neighbor mean distance (averaged over 16384 codewords)")
    print("-" * 72)
    for k in planned_k:
        print("  " + fmt_k(k))
    print()
    print("Cluster analysis")
    print("-" * 72)
    for k in ks:
        print(f"  k-means k={k:3d}  silhouette (cosine) = {sil_scores[k]:+.4f}")
    print()
    print("Figures saved:")
    print(f"  {OUT / 'pairwise_distances.png'}")
    print(f"  {OUT / 'kth_nn_curve.png'}")
    print(f"  {OUT / 'clustering.png'}")
    print()

    # ------------------------------------------------------------------
    # Verdict paragraph (printed + copy for report writer)
    # ------------------------------------------------------------------
    # Determine sharpness of the k-NN curve
    d25 = planned_summary[25]["mean"]
    d250 = planned_summary[250]["mean"]
    d2500 = planned_summary[2500]["mean"]
    d16383 = planned_summary[16383]["mean"]
    ratio_250_25 = d250 / max(d25, 1e-12)
    ratio_2500_25 = d2500 / max(d25, 1e-12)
    ratio_max_25 = d16383 / max(d25, 1e-12)

    print("Gene-value sweep ratios (mean cosine distance, relative to k=25)")
    print("-" * 72)
    print(f"  d(250) / d(25)   = {ratio_250_25:.2f}x")
    print(f"  d(2500) / d(25)  = {ratio_2500_25:.2f}x")
    print(f"  d(16383) / d(25) = {ratio_max_25:.2f}x")
    print()


if __name__ == "__main__":
    main()
