"""Stage 1 flip-discovery search.

Each strategy is a pure generator function:

    strategy(anchor_geno, gene_bounds, image_dim, budget, rng, strategy_cfg)
        -> Iterator[np.ndarray]

Generators yield up to *budget* candidate genotypes.  They never call the
SUT — that happens in ``run_stage1`` via ``score_candidate``.

``run_stage1`` distributes the global SUT budget proportionally across
strategies, runs them sequentially, and applies early-stopping rules from
``EarlyStopConfig``.
"""

from __future__ import annotations

import logging
from typing import Callable, Iterator

import numpy as np

from ..config import EarlyStopConfig, StrategyConfig
from .base import Candidate, ScoredCandidate, score_candidate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy generators
# ---------------------------------------------------------------------------


def _dense_uniform(
    anchor_geno: np.ndarray,
    gene_bounds: np.ndarray,
    image_dim: int,
    budget: int,
    rng: np.random.Generator,
    strategy_cfg: StrategyConfig,
) -> Iterator[np.ndarray]:
    """Uniformly random genotype with per-gene independent sampling.

    Each gene is independently set to a uniformly random value in
    ``[0, bounds[i])``.  The ``density`` parameter (default 1.0) is the
    probability that any given gene is non-zero; set it below 1.0 to
    produce sparse genotypes while keeping each active gene fully random.

    :param density: Probability each gene is non-zero (default 1.0).
    """
    n = len(gene_bounds)
    density = strategy_cfg.density  # default 1.0
    for _ in range(budget):
        g = np.zeros(n, dtype=np.int64)
        for i in range(n):
            if gene_bounds[i] > 1 and (density >= 1.0 or rng.random() < density):
                g[i] = rng.integers(1, gene_bounds[i])
        yield g


def _sparsity_sweep(
    anchor_geno: np.ndarray,
    gene_bounds: np.ndarray,
    image_dim: int,
    budget: int,
    rng: np.random.Generator,
    strategy_cfg: StrategyConfig,
) -> Iterator[np.ndarray]:
    """Sweep sparsity from low to high, cycling through ``densities``.

    Generates candidates at each density fraction in order, cycling until
    the budget is exhausted.  Useful for probing the boundary at multiple
    perturbation strengths.

    :param densities: Active-gene fractions to sweep (default 0.2…1.0).
    """
    densities = strategy_cfg.densities or (0.2, 0.4, 0.6, 0.8, 1.0)
    n = len(gene_bounds)
    mutable_genes = np.where(gene_bounds > 1)[0]
    count = 0
    while count < budget:
        for d in densities:
            if count >= budget:
                break
            g = np.zeros(n, dtype=np.int64)
            n_active = max(1, int(d * len(mutable_genes)))
            chosen = rng.choice(mutable_genes, size=min(n_active, len(mutable_genes)), replace=False)
            for i in chosen:
                g[i] = rng.integers(1, gene_bounds[i])
            yield g
            count += 1


def _max_rank(
    anchor_geno: np.ndarray,
    gene_bounds: np.ndarray,
    image_dim: int,
    budget: int,
    rng: np.random.Generator,
    strategy_cfg: StrategyConfig,
) -> Iterator[np.ndarray]:
    """Set a random subset of genes to their maximum rank value.

    Cycles through ``subset_fractions`` — at each fraction, a randomly
    chosen subset of genes are maximally perturbed (set to their upper
    bound minus 1).  Tests whether extreme local perturbations produce flips.

    :param subset_fractions: Fractions of genes to set to max (default 0.1…1.0).
    """
    subset_fractions = strategy_cfg.subset_fractions or (0.1, 0.25, 0.5, 1.0)
    n = len(gene_bounds)
    mutable_genes = np.where(gene_bounds > 1)[0]
    count = 0
    while count < budget:
        for frac in subset_fractions:
            if count >= budget:
                break
            g = np.zeros(n, dtype=np.int64)
            n_active = max(1, int(frac * len(mutable_genes)))
            chosen = rng.choice(mutable_genes, size=min(n_active, len(mutable_genes)), replace=False)
            for i in chosen:
                g[i] = gene_bounds[i] - 1  # maximum rank
            yield g
            count += 1


def _modality_image(
    anchor_geno: np.ndarray,
    gene_bounds: np.ndarray,
    image_dim: int,
    budget: int,
    rng: np.random.Generator,
    strategy_cfg: StrategyConfig,
) -> Iterator[np.ndarray]:
    """Perturb only image genes; text genes remain at zero.

    Isolates the image modality to check whether image-only perturbations
    can cross the boundary without any text change.

    :param density: Probability each image gene is non-zero (default 0.6).
    """
    n = len(gene_bounds)
    density = strategy_cfg.density  # default 0.6
    for _ in range(budget):
        g = np.zeros(n, dtype=np.int64)
        for i in range(image_dim):
            if gene_bounds[i] > 1 and rng.random() < density:
                g[i] = rng.integers(1, gene_bounds[i])
        yield g


def _modality_text(
    anchor_geno: np.ndarray,
    gene_bounds: np.ndarray,
    image_dim: int,
    budget: int,
    rng: np.random.Generator,
    strategy_cfg: StrategyConfig,
) -> Iterator[np.ndarray]:
    """Perturb only text genes; image genes remain at zero.

    Isolates the text modality to check whether prompt-only perturbations
    can cross the boundary without any visual change.

    :param density: Probability each text gene is non-zero (default 1.0).
    """
    n = len(gene_bounds)
    density = strategy_cfg.density  # default 1.0
    for _ in range(budget):
        g = np.zeros(n, dtype=np.int64)
        for i in range(image_dim, n):
            if gene_bounds[i] > 1 and rng.random() < density:
                g[i] = rng.integers(1, gene_bounds[i])
        yield g


def _bituniform_density(
    anchor_geno: np.ndarray,
    gene_bounds: np.ndarray,
    image_dim: int,
    budget: int,
    rng: np.random.Generator,
    strategy_cfg: StrategyConfig,
) -> Iterator[np.ndarray]:
    """Per-gene independent uniform sampling from the full value range.

    Each gene is independently and uniformly sampled from ``[0, bounds[i])``.
    Unlike ``dense_uniform``, there is no density gating — each gene
    takes any value with equal probability, so the expected sparsity is
    ``n * (1 - 1/mean_bounds)``.
    """
    n = len(gene_bounds)
    for _ in range(budget):
        g = np.array(
            [rng.integers(0, max(1, int(b))) for b in gene_bounds],
            dtype=np.int64,
        )
        yield g


def _sparse_small(
    anchor_geno: np.ndarray,
    gene_bounds: np.ndarray,
    image_dim: int,
    budget: int,
    rng: np.random.Generator,
    strategy_cfg: StrategyConfig,
) -> Iterator[np.ndarray]:
    """Very few non-zero genes at low rank — the PDQ-friendly regime.

    Explicitly targets the genotype neighbourhood where input distance is
    minimal (small k, small r), maximising the chance that PDQ is large
    for any flip found.  The other strategies explore broadly; this one
    probes aggressively near the anchor.

    Reuses existing :class:`StrategyConfig` fields to avoid new config
    additions:

    - ``densities`` — integer k values (non-zero gene counts to cycle through).
      Default ``(1, 2, 3, 4, 5)``.  Interpreted as integers even if stored
      as floats by the YAML parser.
    - ``subset_fractions`` — integer r values (rank values to assign to each
      active gene).  Default ``(1, 2)``.  For each active gene, a rank is
      drawn uniformly from this set and clamped to ``bounds[i] - 1``.

    :param densities: k values (non-zero gene counts); default (1,2,3,4,5).
    :param subset_fractions: r values (rank per active gene); default (1,2).
    """
    k_values: list[int] = (
        [max(1, int(d)) for d in strategy_cfg.densities]
        if strategy_cfg.densities
        else [1, 2, 3, 4, 5]
    )
    r_values: list[int] = (
        [max(1, int(f)) for f in strategy_cfg.subset_fractions]
        if strategy_cfg.subset_fractions
        else [1, 2]
    )

    n = len(gene_bounds)
    mutable = np.where(gene_bounds > 1)[0]
    if len(mutable) == 0:
        return

    for _ in range(budget):
        g = np.zeros(n, dtype=np.int64)
        k = int(rng.choice(k_values))
        chosen = rng.choice(mutable, size=min(k, len(mutable)), replace=False)
        for i in chosen:
            r = int(rng.choice(r_values))
            g[i] = min(r, int(gene_bounds[i]) - 1)
        yield g


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: dict[
    str, Callable[
        [np.ndarray, np.ndarray, int, int, np.random.Generator, StrategyConfig],
        Iterator[np.ndarray],
    ]
] = {
    "dense_uniform": _dense_uniform,
    "sparsity_sweep": _sparsity_sweep,
    "max_rank": _max_rank,
    "modality_image": _modality_image,
    "modality_text": _modality_text,
    "bituniform_density": _bituniform_density,
    "sparse_small": _sparse_small,
}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_stage1(
    anchor_geno: np.ndarray,
    anchor_label: str,
    anchor_image_arr: np.ndarray,
    gene_bounds: np.ndarray,
    image_dim: int,
    text_candidate_distances: tuple[np.ndarray, ...],
    seed_idx: int,
    strategies: tuple[StrategyConfig, ...],
    budget: int,
    rng: np.random.Generator,
    sut_call_fn: Callable[[np.ndarray], tuple[list[float], int, object, str, float]],
    input_distance_fn: Callable[[np.ndarray, np.ndarray], float],
    output_distance_fn: Callable[[str, str], float],
    categories: tuple[str, ...],
    early_stop_cfg: EarlyStopConfig,
    max_flips: int,
    max_distinct_targets: int,
) -> list[ScoredCandidate]:
    """Run Stage-1 flip discovery across all configured strategies.

    Distributes the global SUT *budget* proportionally across strategies
    by weight, then runs them sequentially.  A single global counter
    ensures the total never exceeds *budget* even with rounding.

    Early stopping fires after ``early_stop_cfg.min_calls_before_stop``
    calls if ``on_flips_complete`` and we have ≥ *max_flips* flips, or if
    ``on_targets_complete`` and we've seen ≥ *max_distinct_targets* labels.

    :param anchor_geno: Zero genotype (identity; used for hamming/distances).
    :param anchor_label: VLM label on the zero-genotype anchor.
    :param anchor_image_arr: Pixel array of the anchor (for pixel L2).
    :param gene_bounds: Per-gene upper bounds (exclusive).
    :param image_dim: Number of image genes (split point).
    :param text_candidate_distances: Precomputed text cosine distances.
    :param seed_idx: 0-based seed index for ``Candidate`` bookkeeping.
    :param strategies: Ordered strategy configs (weights used for budget split).
    :param budget: Maximum total SUT calls for Stage 1.
    :param rng: Seeded random generator.
    :param sut_call_fn: ``(genotype) → (logprobs, call_id, image, text, t_cum)``.
    :param input_distance_fn: ``(g, anchor) → float``.
    :param output_distance_fn: ``(label_a, label_b) → float``.
    :param categories: Category tuple (same order as logprobs).
    :param early_stop_cfg: Early-stopping thresholds.
    :param max_flips: Stop early when this many flips are found.
    :param max_distinct_targets: Stop early when this many distinct target
        labels have been flipped to.
    :returns: All evaluated candidates in evaluation order.
    """
    if not strategies:
        return []

    # -- Budget split -------------------------------------------------------
    total_weight = sum(s.weight for s in strategies)
    strategy_budgets: list[int] = []
    allocated = 0
    for i, s in enumerate(strategies):
        if i == len(strategies) - 1:
            b = max(1, budget - allocated)
        else:
            b = max(1, round(budget * s.weight / total_weight))
            allocated += b
        strategy_budgets.append(b)

    # -- Search loop --------------------------------------------------------
    results: list[ScoredCandidate] = []
    total_calls = 0
    n_flips = 0
    distinct_targets: set[str] = set()

    for s_cfg, s_budget in zip(strategies, strategy_budgets):
        if total_calls >= budget:
            break

        gen_fn = STRATEGY_REGISTRY[s_cfg.name]
        generator = gen_fn(anchor_geno, gene_bounds, image_dim, s_budget, rng, s_cfg)

        for gen_step, genotype in enumerate(generator):
            if total_calls >= budget:
                break

            cand = Candidate(
                genotype=genotype,
                strategy=s_cfg.name,
                seed_idx=seed_idx,
                gen_step=gen_step,
            )
            sc = score_candidate(
                cand=cand,
                anchor_label=anchor_label,
                anchor_geno=anchor_geno,
                anchor_image_arr=anchor_image_arr,
                text_candidate_distances=text_candidate_distances,
                image_dim=image_dim,
                categories=categories,
                sut_call_fn=sut_call_fn,
                input_distance_fn=input_distance_fn,
                output_distance_fn=output_distance_fn,
            )
            results.append(sc)
            total_calls += 1

            if sc.flipped:
                n_flips += 1
                distinct_targets.add(sc.label)
                logger.debug(
                    "Stage1 flip found: strategy=%s  candidate_id=%d  "
                    "label=%s  d_i=%.2f  pdq=%.4f",
                    s_cfg.name, sc.candidate_id, sc.label, sc.d_i, sc.pdq_score,
                )

            # -- Early stopping -------------------------------------------
            if total_calls >= early_stop_cfg.min_calls_before_stop:
                if early_stop_cfg.on_flips_complete and n_flips >= max_flips:
                    logger.info("Stage1 early stop: max_flips=%d reached", max_flips)
                    return results
                if (
                    early_stop_cfg.on_targets_complete
                    and len(distinct_targets) >= max_distinct_targets
                ):
                    logger.info(
                        "Stage1 early stop: max_distinct_targets=%d reached",
                        max_distinct_targets,
                    )
                    return results

        logger.debug(
            "Stage1 strategy=%s  calls=%d  flips=%d",
            s_cfg.name, total_calls, n_flips,
        )

    return results
