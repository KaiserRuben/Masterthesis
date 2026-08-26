"""Statistics for HS-01: clustered bootstrap, effect sizes, agreement."""

from __future__ import annotations

from itertools import combinations

import krippendorff
import numpy as np
import pandas as pd
from scipy import stats

BOOT = 5000
SEED = 42


def boot_ci(df: pd.DataFrame, col: str, cluster: str = "item_id",
            n: int = BOOT, seed: int = SEED) -> tuple[float, float, float]:
    """Point estimate + 95% CI for a mean, bootstrap resampling clusters.

    Clustering by item accounts for the 11-12 non-independent ratings each
    item receives (items are photo-disjoint, so item == photo cluster).
    """
    rng = np.random.default_rng(seed)
    clusters = [g[col].to_numpy(float) for _, g in df.groupby(cluster)]
    k = len(clusters)
    reps = [np.concatenate([clusters[i] for i in rng.integers(0, k, k)]).mean()
            for _ in range(n)]
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return float(df[col].mean()), float(lo), float(hi)


def cliffs_delta(x, y) -> tuple[float, float]:
    """Cliff's delta (via Mann-Whitney U) + two-sided MW p-value."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
    return 2 * u / (len(x) * len(y)) - 1, float(p)


def kripp_alpha(df: pd.DataFrame, value_col: str, level: str) -> float:
    """Krippendorff's alpha on a raters x items reliability matrix."""
    mat = df.pivot_table(index="participant", columns="item_id",
                         values=value_col, aggfunc="first")
    return float(krippendorff.alpha(reliability_data=mat.values,
                                    level_of_measurement=level))


def pairwise_agreement(df: pd.DataFrame, value_col: str) -> float:
    """Observed agreement: share of agreeing rater pairs within each item."""
    agree = total = 0
    for _, g in df.groupby("item_id"):
        for a, b in combinations(g[value_col].values, 2):
            agree += a == b
            total += 1
    return agree / total if total else float("nan")


def gwet_ac1(df: pd.DataFrame, value_col: str, categories: list) -> tuple[float, float]:
    """Gwet's AC1 (nominal) — robust to prevalence skew, unlike alpha.

    Chance agreement uses the pooled category shares (ratings are near-
    balanced across items, so this matches the per-item-average formula).
    Returns (AC1, observed pairwise agreement).
    """
    pa = pairwise_agreement(df, value_col)
    pi = df[value_col].value_counts(normalize=True).reindex(categories).fillna(0).to_numpy()
    pe = float((pi * (1 - pi)).sum() / (len(categories) - 1))
    return (pa - pe) / (1 - pe), pa


def split_half(df: pd.DataFrame, value_col: str, n: int = 2000,
               seed: int = 7) -> tuple[float, float]:
    """Item-level split-half reliability over random rater halves.

    Returns (median Spearman rho between half item-means, Spearman-Brown
    corrected full-panel reliability).
    """
    rng = np.random.default_rng(seed)
    raters = df.participant.unique()
    cors = []
    for _ in range(n):
        half = set(rng.choice(raters, len(raters) // 2, replace=False))
        a = df[df.participant.isin(half)].groupby("item_id")[value_col].mean()
        b = df[~df.participant.isin(half)].groupby("item_id")[value_col].mean()
        j = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
        if len(j) > 5:
            cors.append(stats.spearmanr(j.a, j.b).statistic)
    rho = float(np.median(cors))
    return rho, 2 * rho / (1 + rho)


def spearman(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    """Spearman rho, p, n over pairwise-complete observations."""
    j = pd.concat([x, y], axis=1).dropna()
    if len(j) < 3:
        return float("nan"), float("nan"), len(j)
    r = stats.spearmanr(j.iloc[:, 0], j.iloc[:, 1])
    return float(r.statistic), float(r.pvalue), len(j)
