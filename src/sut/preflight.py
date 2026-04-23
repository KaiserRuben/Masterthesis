"""Preflight cost estimation for boundary-test runs.

Measures per-SUT-call wall time on a handful of representative calls
before the main optimization loop starts, and prints a projection of
total runtime for the configured budget. Intended to be called from
experiment entry-point scripts (``run_boundary_test.py``,
``run_pdq_test.py``) behind a ``--preflight`` CLI flag so that any
run on new hardware surfaces its cost up front, not after several
hours of wall time.

Why this matters
----------------

Per-SUT-call cost scales approximately linearly with the number of
categories scored, because ``score_categories`` does one cached-prefix
forward pass and then one decoder pass per category. For a 4B-class
VLM on MPS with N=50 categories, observed cost is ~12 s per call
(~2 s image encode + 50 × ~0.2 s decodes); with N=2 it drops to
~0.5–1 s. That is a ~25× wall-time difference on identical budgets
and can turn a "4-hour run" into a "20-hour run" with no warning
unless a preflight check reports it.

The check is HARDWARE-DEPENDENT — running it on MPS, CUDA, or a new
GPU generation is exactly when you want this number. The flag is
permanent (not a one-off script) so every run on new hardware prints
its projection.

Usage
-----

From ``run_boundary_test.py``::

    if args.preflight:
        preflight_cost_check(
            sut=sut,
            manipulator=manipulator,
            seed=seeds[0],
            prompt_template=exp.prompt_template,
            answer_suffix=answer_suffix,
            categories=scored_categories,
            total_calls_projected=exp.generations * exp.pop_size * n_seeds,
            n_samples=20,
        )

From ``run_pdq_test.py``::

    if args.preflight:
        total_pdq = (
            cfg.stage1.budget_sut_calls
            + cfg.stage1.max_flips_per_seed * cfg.stage2.budget_sut_calls_per_flip
        ) * n_seeds
        preflight_cost_check(
            sut=sut,
            manipulator=manipulator,
            seed=seeds[0],
            prompt_template=cfg.prompt_template,
            answer_suffix=answer_suffix,
            categories=cfg.categories,
            total_calls_projected=total_pdq,
            n_samples=20,
        )

Behavior
--------

1. Prepares the manipulator on the given seed (once).
2. Runs ``n_warmup=2`` throw-away calls to flush first-use overhead
   (model lazy init, kernel compilation, cache warmup).
3. Runs ``n_samples`` measurement calls on fresh random genotypes;
   each one forces a cache miss (new image), so the measurement
   reflects realistic per-call cost under optimization, not the
   best-case cache-hit rate.
4. Prints a clear block with mean / median / min / max per-call
   timing, the projected total call count, and the projected total
   wall time in hours.
5. **Does not abort the run.** If the projection is unacceptable,
   the user can Ctrl-C the main process and reconfigure.

Caveats
-------

- N_samples should be 20–30 to get stable per-call statistics without
  burning a meaningful fraction of the cache. Fewer samples (< 10)
  give noisy projections; more samples (> 50) eat into the real run.
- The projection assumes constant per-call cost. In practice the
  optimizer's convergence behaviour, cache hit rate, and image
  encoder warm-up can make the first few hundred calls slower than
  steady state. The projection is therefore typically an
  *upper bound* for SMOO (cache hit rate grows over time) and
  accurate for PDQ (each call targets a fresh genotype).
- The preflight itself costs ``n_samples × per_call_time`` wall seconds.
  On a 12 s/call SMOO run with 20 samples, that is ~4 minutes of
  overhead before the real run begins. Consider that a fair price
  for knowing whether the run will finish in 4 h or 40 h.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from src.config import SeedTriple
    from src.manipulator.vlm_manipulator import VLMManipulator
    from src.sut.vlm_sut import VLMSUT

logger = logging.getLogger(__name__)


def preflight_cost_check(
    *,
    sut: VLMSUT,
    manipulator: VLMManipulator,
    seed: SeedTriple,
    prompt_template: str,
    answer_suffix: str,
    categories: tuple[str, ...] | list[str],
    total_calls_projected: int,
    n_samples: int = 20,
    n_warmup: int = 2,
    rng_seed: int = 0,
) -> dict[str, Any]:
    """Measure representative SUT calls and print a runtime projection.

    :param sut: Initialized VLMSUT. Must be ready to accept
        ``process_input`` calls.
    :param manipulator: VLMManipulator. Will be prepared on *seed*
        (the caller may already have prepared it; re-preparing is a
        no-op but does consume a tiny amount of wall time).
    :param seed: A seed triple to use as the anchor for measurement.
        Typically the first seed of the run.
    :param prompt_template: The experiment's question prompt.
    :param answer_suffix: The pair-specific answer-format suffix that
        will be appended to each mutated text before scoring.
    :param categories: The category list the SUT will score against
        during the real run. Passed through to ``process_input`` so
        the preflight matches production cost.
    :param total_calls_projected: Total SUT calls the full run will
        consume (computed per pipeline by the caller).
    :param n_samples: Number of measurement calls to make. Default 20.
    :param n_warmup: Number of warmup calls before measurement. These
        absorb first-call overhead (model lazy init, kernel compile,
        cache setup). Default 2.
    :param rng_seed: Seed for the random genotype RNG. Default 0.
    :returns: A dict with the raw timing statistics and the projection,
        so callers can log or assert on it programmatically::

            {
                "n_samples": int,
                "mean_s": float,
                "median_s": float,
                "min_s": float,
                "max_s": float,
                "std_s": float,
                "total_calls_projected": int,
                "projected_total_s": float,
                "projected_total_h": float,
                "n_categories_scored": int,
                "device": str,
                "model_id": str,
            }
    """
    categories = tuple(categories)
    logger.info(
        "Preflight: preparing manipulator on seed class_a=%s class_b=%s",
        seed.class_a, seed.class_b,
    )
    manipulator.prepare(seed.image, prompt_template)

    gene_bounds = manipulator.gene_bounds
    n_genes = len(gene_bounds)
    rng = np.random.default_rng(rng_seed)

    def _random_genotype() -> np.ndarray:
        """Draw a fresh random genotype uniformly within gene bounds."""
        g = np.empty(n_genes, dtype=np.int64)
        for i, ub in enumerate(gene_bounds):
            g[i] = rng.integers(0, int(ub) + 1)
        return g.reshape(1, -1)

    def _one_call() -> float:
        """Render a fresh genotype and time a single SUT call."""
        geno = _random_genotype()
        imgs, texts = manipulator.manipulate(
            candidates=None, weights=geno,
        )
        img, txt = imgs[0], texts[0]
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = sut.process_input(
                img, text=txt + answer_suffix, categories=categories,
            )
        return time.perf_counter() - t0

    # Warmup — not counted. Swallows first-call overhead.
    for _ in range(n_warmup):
        _one_call()

    # Measurement.
    times: list[float] = []
    for _ in range(n_samples):
        times.append(_one_call())

    arr = np.array(times, dtype=np.float64)
    mean_s = float(arr.mean())
    median_s = float(np.median(arr))
    min_s = float(arr.min())
    max_s = float(arr.max())
    std_s = float(arr.std())

    projected_total_s = mean_s * total_calls_projected
    projected_total_h = projected_total_s / 3600.0

    bar = "=" * 72
    lines = [
        "",
        bar,
        "PREFLIGHT COST CHECK",
        bar,
        f"  hardware:           device={sut._config.device}",
        f"  model:              {sut._config.sut.model_id}",
        f"  categories scored:  N={len(categories)}",
        f"  warmup calls:       {n_warmup}  (not counted)",
        f"  measurement calls:  {n_samples}",
        "",
        f"  per-call mean:      {mean_s:7.3f} s",
        f"  per-call median:    {median_s:7.3f} s",
        f"  per-call min:       {min_s:7.3f} s",
        f"  per-call max:       {max_s:7.3f} s",
        f"  per-call std:       {std_s:7.3f} s",
        "",
        f"  total calls (proj): {total_calls_projected:>10,}",
        f"  total wall  (proj): {projected_total_s:>10,.0f} s",
        f"                      ={projected_total_s / 60:>9,.1f} min",
        f"                      ={projected_total_h:>9,.2f} h",
        bar,
        "",
    ]
    for line in lines:
        logger.info(line)
    # Also print directly so it's visible under default WARNING root logger.
    print("\n".join(lines), flush=True)

    return {
        "n_samples": n_samples,
        "mean_s": mean_s,
        "median_s": median_s,
        "min_s": min_s,
        "max_s": max_s,
        "std_s": std_s,
        "total_calls_projected": total_calls_projected,
        "projected_total_s": projected_total_s,
        "projected_total_h": projected_total_h,
        "n_categories_scored": len(categories),
        "device": sut._config.device,
        "model_id": sut._config.sut.model_id,
    }


__all__ = ["preflight_cost_check"]
