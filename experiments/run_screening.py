#!/usr/bin/env python3
"""Run a SMOO adaptive-resolution screening experiment (EXP-08).

Supports five optimization methods via ``config.optimizer.method``:

- ``M0`` — baseline (delegates to the existing :class:`VLMBoundaryTester`).
- ``M1`` — Stage 1 only: fuzzy one-hot screen at a single depth.
- ``M2`` — Stage 1 two-depth (monotonicity diagnostic).
- ``M3`` — Stage 1 + Stage 2 (precise scan on awake genes).
- ``M4`` — Stage 1 + Stage 2 + Stage 3 (evolution with early stopping).

Pair resolution: ``--pair NAME`` (repeatable) looks up the pair in the
live seed pool via :mod:`src.pair_resolver` and writes the resolved
``filter_indices`` into the config at runtime. ``--replicates N`` picks
the first N pool indices per pair.

Usage
-----

    # Tier 1 — shark M1 single replicate
    python experiments/run_screening.py configs/EXP-08/M1.yaml \\
        --pair "great_white_shark->tiger_shark" --replicates 1

    # Tier 2 — shark M4 three replicates on MPS
    python experiments/run_screening.py configs/EXP-08/M4.yaml \\
        --pair shark_gws-ts --replicates 3 --device mps
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from time import time
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dacite
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from numpy.typing import NDArray
from PIL import Image
from tqdm import tqdm

from src.config import ExperimentConfig, SeedTriple, resolve_categories
from src.data import ImageNetCache
from src.manipulator.image.manipulator import ImageManipulator
from src.manipulator.image.types import CandidateStrategy, PatchStrategy
from src.manipulator.text.manipulator import TextManipulator
from src.manipulator.vlm_manipulator import VLMManipulator
from src.objectives import (
    CriterionCollection,
    MatrixDistance,
    TargetedBalance,
    TextReplacementDistance,
)
from src.optimizer.discrete_pymoo_optimizer import DiscretePymooOptimizer
from src.optimizer.early_stop import EarlyStopChecker, EarlyStopConfig
from src.optimizer.seed_matrix import (
    build_fuzzy_onehot,
    build_pareto_init,
    build_precise_scan,
)
from src.pair_resolver import resolve_pair
from src.pdq.runner import _apply_seed_filter
from src.sut import VLMSUT
from src.tester import VLMBoundaryTester, generate_seeds
from src.tester.vlm_boundary_tester import (
    SMOO_SCHEMA_VERSION,
    _build_context_meta,
    _build_stats,
    _build_trace_rows,
    _pil_to_tensor,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("src").setLevel(logging.INFO)
logging.getLogger(__name__).setLevel(logging.INFO)
logger = logging.getLogger(__name__)


_DACITE_CONFIG = dacite.Config(
    cast=[tuple, frozenset],
    type_hooks={
        Path: lambda v: Path(v).expanduser() if isinstance(v, str) else v,
        PatchStrategy: lambda v: PatchStrategy[v] if isinstance(v, str) else v,
        CandidateStrategy: (
            lambda v: CandidateStrategy[v] if isinstance(v, str) else v
        ),
    },
)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_config(cfg: dict) -> ExperimentConfig:
    """Hydrate a YAML dict into an :class:`ExperimentConfig`."""
    return dacite.from_dict(ExperimentConfig, cfg, config=_DACITE_CONFIG)


def _resolve_depths(
    depths: Sequence[int], gene_bounds: NDArray[np.int64]
) -> list[int]:
    """Map fuzzy depth sentinels (-1 → max_k - 1) to concrete values.

    ``max_k - 1`` is the largest value reachable under the exclusive
    upper-bound convention. Uses the minimum gene bound as the common
    max_k across the vector — depths are global, so genes with smaller
    bounds clamp naturally in the seed-matrix builder.
    """
    max_k = int(gene_bounds.min()) - 1
    return [max_k if d == -1 else int(d) for d in depths]


# ---------------------------------------------------------------------------
# Awake classification
# ---------------------------------------------------------------------------


def _classify_awake(
    seed_matrix: NDArray[np.int64],
    fitness: NDArray[np.float64],
    threshold: str,
    fixed_delta: float,
    baseline_fitness: NDArray[np.float64] | None,
) -> NDArray[np.bool_]:
    """Return a per-gene awake mask from Stage 1 output.

    The fuzzy one-hot seed matrix has exactly one non-zero gene per row
    (plus the all-zero baseline). This function walks each row, reads
    the fitness delta against baseline, and marks the touched gene as
    awake if the delta exceeds the threshold.

    :param seed_matrix: The Stage-1 sampling matrix, shape
        ``(n_individuals, n_genes)``.
    :param fitness: Fitness vector per individual, shape
        ``(n_individuals, n_objectives)``. Uses objective 0 (TgtBal
        family) by convention.
    :param threshold: ``"permutation_shuffle"`` or ``"fixed_delta"``.
    :param fixed_delta: Threshold for ``fixed_delta`` mode.
    :param baseline_fitness: Fitness of the all-zero individual, or
        ``None`` if baseline wasn't included; in that case uses
        median fitness across the matrix as the reference.
    :returns: Boolean mask of shape ``(n_genes,)``.
    """
    n_genes = seed_matrix.shape[1]
    awake = np.zeros(n_genes, dtype=bool)

    # Track max |Δfitness| seen per gene (Stage 1 only touches each
    # gene once at a single depth, but in M2 each gene may be probed
    # twice; take max to flag any wake event).
    gene_delta = np.zeros(n_genes)

    ref = (
        baseline_fitness
        if baseline_fitness is not None
        else np.median(fitness, axis=0)
    )
    # Objective 0 is TgtBal by convention in the existing pipeline
    # (after the Schema-v2 switch it's the log-prob-gap family).
    ref_scalar = float(ref[0]) if ref.ndim else float(ref)

    for i, row in enumerate(seed_matrix):
        touched = np.flatnonzero(row != 0)
        if len(touched) != 1:
            continue  # baseline or malformed; skip
        g = int(touched[0])
        delta = abs(float(fitness[i, 0]) - ref_scalar)
        if delta > gene_delta[g]:
            gene_delta[g] = delta

    if threshold == "fixed_delta":
        awake = gene_delta > fixed_delta
    elif threshold == "mad_outlier":
        # Median + 3 * MAD. Robust-outlier cutoff against the
        # *empirical distribution of all gene deltas* (which IS the
        # null in this setup — each gene probed in exactly one
        # individual, so per-gene deltas are i.i.d. samples under
        # "no signal" by construction). Every gene whose delta exceeds
        # the cutoff is flagged as awake.
        med = float(np.median(gene_delta))
        mad = float(np.median(np.abs(gene_delta - med)))
        # 1.4826 converts MAD to σ under Gaussian assumption; 3σ ≈ 99.7%
        cutoff = med + 3.0 * 1.4826 * mad
        awake = gene_delta > cutoff
    elif threshold == "top_quantile":
        # Flag genes whose delta is in the top 5% of the distribution.
        cutoff = float(np.quantile(gene_delta, 0.95))
        awake = gene_delta > cutoff
    else:
        raise ValueError(
            f"Unknown awake_threshold: {threshold!r}. "
            f"Valid: 'fixed_delta', 'mad_outlier', 'top_quantile'."
        )

    return awake


# ---------------------------------------------------------------------------
# Per-seed stage runner — low-level primitive
# ---------------------------------------------------------------------------


def _run_one_stage(
    *,
    tester: VLMBoundaryTester,
    seed_idx: int,
    seed: SeedTriple,
    sampling: NDArray[np.int64],
    max_generations: int,
    stage_tag: str,
    run_dir: Path,
    origin_tensor: torch.Tensor,
    scored_categories: tuple[str, ...],
    target_classes: tuple[int, int],
    answer_suffix: str,
    pair: tuple[str, str],
    parquet_metadata: dict[bytes, bytes],
    early_stop: EarlyStopChecker | None = None,
    tgtbal_idx: int = 2,
) -> tuple[list[list[Any]], NDArray[np.int64], NDArray[np.float64], int]:
    """Run a single screening stage with the given sampling matrix.

    Re-initializes the optimizer with ``sampling`` as the initial
    population, then runs up to ``max_generations`` evaluations. Writes
    a per-stage trace parquet and returns the Pareto state.

    :returns: Tuple ``(trace_rows, pareto_genotypes, pareto_fitness, n_gens_run)``.
    """
    tester._optimizer.set_initial_population(sampling)

    trace_path = run_dir / f"trace_{stage_tag}.parquet"
    trace_rows: list[dict[str, Any]] = []
    trace_writer: pq.ParquetWriter | None = None
    n_gens_run = 0

    try:
        # For single-pass stages (Stage 1 / Stage 2) we want tqdm on the
        # SUT-call loop itself. For evolution stages (Stage 3) we want an
        # outer per-generation bar and quieter inner progress.
        show_inner_tqdm = max_generations == 1
        outer_pbar: tqdm | None = None
        if max_generations > 1:
            outer_pbar = tqdm(
                total=max_generations,
                desc=f"{stage_tag} gens",
                unit="gen",
                leave=False,
            )

        for gen in range(max_generations):
            sut_desc = None
            if show_inner_tqdm:
                sut_desc = f"seed {seed_idx} · {stage_tag}"
            elif outer_pbar is not None:
                sut_desc = f"gen {gen}"
            gen_traces = tester._run_generation(
                seed_idx, gen, target_classes, origin_tensor,
                scored_categories, answer_suffix,
                sut_progress_desc=sut_desc,
            )
            n_gens_run = gen + 1

            if gen_traces:
                trace_rows.extend(gen_traces)
                df = pd.DataFrame(gen_traces)
                table = pa.Table.from_pandas(df, preserve_index=False)
                if trace_writer is None:
                    existing = dict(table.schema.metadata or {})
                    existing.update(parquet_metadata)
                    existing[b"stage_tag"] = stage_tag.encode()
                    schema = table.schema.with_metadata(existing)
                    trace_writer = pq.ParquetWriter(
                        trace_path, schema, compression="zstd",
                    )
                    table = table.replace_schema_metadata(existing)
                trace_writer.write_table(table)

            if outer_pbar is not None:
                pareto = tester._optimizer.best_candidates
                postfix = {"pareto": len(pareto)}
                if pareto:
                    pf = np.array([c.fitness for c in pareto])
                    postfix["tb"] = f"{pf[:, tgtbal_idx].min():.3e}"
                outer_pbar.set_postfix(postfix)
                outer_pbar.update(1)

            if early_stop is not None:
                pareto = tester._optimizer.best_candidates
                if pareto:
                    pf = np.array([c.fitness for c in pareto])
                    trig = early_stop.update(
                        generation=gen,
                        pareto_fitness=pf,
                        tgtbal_index=tgtbal_idx,
                    )
                    if trig is not None:
                        logger.info(
                            f"  [{stage_tag}] early stop: {trig.trigger} at gen {gen} {trig.details}"
                        )
                        break
    finally:
        if trace_writer is not None:
            trace_writer.close()
        if outer_pbar is not None:
            outer_pbar.close()

    pareto = tester._optimizer.best_candidates
    pareto_geno = np.array(
        [c.solution.astype(np.int64) for c in pareto], dtype=np.int64,
    )
    pareto_fit = np.array(
        [c.fitness for c in pareto], dtype=np.float64,
    )
    return trace_rows, pareto_geno, pareto_fit, n_gens_run


# ---------------------------------------------------------------------------
# Per-seed orchestrator — dispatches stages by method
# ---------------------------------------------------------------------------


def _run_seed_screening(
    tester: VLMBoundaryTester,
    seed_idx: int,
    seed: SeedTriple,
) -> None:
    """Run one seed under an M1/M2/M3/M4 screening method.

    Shares the seed-setup and output-management logic with the baseline
    tester's ``_run_seed`` but uses staged seed matrices + early stop
    instead of a single random-init evolution loop.
    """
    cfg = tester._config
    method = cfg.optimizer.method
    start = time()

    # -- Seed prep (mirrors _run_seed lines 392-456) ------------------------
    pair = (seed.class_a, seed.class_b)
    answer_suffix = cfg.answer_format.format(
        categories=", ".join(pair),
    )
    full_categories = tuple(cfg.categories)

    if cfg.score_full_categories:
        scored_categories = full_categories
        target_classes = (
            full_categories.index(seed.class_a),
            full_categories.index(seed.class_b),
        )
    else:
        scored_categories = pair
        target_classes = (0, 1)

    tester._manipulator.prepare(seed.image, cfg.prompt_template)
    tester._optimizer.update_gene_bounds(tester._manipulator.gene_bounds)

    zero_geno = tester._manipulator.zero_genotype().reshape(1, -1)
    baseline_imgs, _ = tester._manipulator.manipulate(
        candidates=None, weights=zero_geno,
    )
    origin_tensor = _pil_to_tensor(baseline_imgs[0])

    run_dir = (
        cfg.save_dir
        / f"{cfg.name}_seed_{seed_idx}_{int(start)}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    parquet_metadata: dict[bytes, bytes] = {
        b"schema_version": str(SMOO_SCHEMA_VERSION).encode(),
        b"pipeline": b"smoo",
        b"method": method.encode(),
        b"categories": json.dumps(list(scored_categories)).encode(),
        b"pair": json.dumps(list(pair)).encode(),
        b"target_classes": json.dumps(list(target_classes)).encode(),
    }

    gene_bounds = tester._manipulator.gene_bounds
    n_genes = len(gene_bounds)
    tgtbal_idx = 2  # MatrixDistance, TextReplacementDistance, TargetedBalance

    # -- Stage 1 — fuzzy screen --------------------------------------------
    fuzzy_depths = _resolve_depths(cfg.optimizer.fuzzy_depths, gene_bounds)
    stage1_matrix = build_fuzzy_onehot(
        n_genes=n_genes,
        gene_bounds=gene_bounds,
        depths=fuzzy_depths,
        include_zero=cfg.optimizer.include_zero_in_seed,
    )

    logger.info(
        f"  [{method} seed {seed_idx}] Stage 1: "
        f"{stage1_matrix.shape[0]} individuals over {n_genes} genes "
        f"at depths {fuzzy_depths}"
    )

    s1_rows, s1_pareto_geno, s1_pareto_fit, _ = _run_one_stage(
        tester=tester,
        seed_idx=seed_idx,
        seed=seed,
        sampling=stage1_matrix,
        max_generations=1,
        stage_tag="stage1",
        run_dir=run_dir,
        origin_tensor=origin_tensor,
        scored_categories=scored_categories,
        target_classes=target_classes,
        answer_suffix=answer_suffix,
        pair=pair,
        parquet_metadata=parquet_metadata,
        tgtbal_idx=tgtbal_idx,
    )

    # Extract fitness per row for awake classification.
    s1_fitness = np.array(
        [
            [row[f"fitness_TgtBal"]] if method in ("M1",) else
            [row["fitness_TgtBal"]]
            for row in s1_rows
        ]
    )
    # Full 3-objective matrix for downstream stages
    if s1_rows:
        obj_keys = [k for k in s1_rows[0] if k.startswith("fitness_")]
        s1_fit_full = np.array(
            [[row[k] for k in obj_keys] for row in s1_rows],
        )
    else:
        s1_fit_full = np.zeros((0, 3))

    baseline_fit = None
    if cfg.optimizer.include_zero_in_seed and len(s1_rows) > 0:
        # First individual in the stage-1 matrix is the all-zero baseline.
        baseline_fit = s1_fit_full[0]

    if method in ("M1", "M2"):
        _finalize_seed(
            tester, run_dir, seed_idx, seed, pair, scored_categories,
            target_classes, answer_suffix, start, s1_pareto_geno,
            s1_pareto_fit, method, stages_ran=["stage1"],
        )
        return

    # -- Stage 2 — precise characterization --------------------------------
    awake_mask = _classify_awake(
        seed_matrix=stage1_matrix,
        fitness=s1_fit_full,
        threshold=cfg.optimizer.awake_threshold,
        fixed_delta=cfg.optimizer.awake_fixed_delta,
        baseline_fitness=baseline_fit,
    )
    n_awake = int(awake_mask.sum())
    logger.info(
        f"  [{method} seed {seed_idx}] Stage 1 → Stage 2: "
        f"{n_awake} awake genes out of {n_genes} "
        f"({100*n_awake/n_genes:.1f}%)"
    )

    with open(run_dir / "awake_mask.json", "w") as f:
        json.dump({
            "awake_mask": awake_mask.tolist(),
            "n_awake": n_awake,
            "n_genes": n_genes,
            "threshold": cfg.optimizer.awake_threshold,
        }, f, indent=2)

    if n_awake == 0:
        logger.warning(
            f"  [{method} seed {seed_idx}] No awake genes detected — "
            f"'diffuse pair' regime. Stage 2/3 skipped."
        )
        _finalize_seed(
            tester, run_dir, seed_idx, seed, pair, scored_categories,
            target_classes, answer_suffix, start, s1_pareto_geno,
            s1_pareto_fit, method, stages_ran=["stage1"],
        )
        return

    stage2_matrix = build_precise_scan(
        awake_mask=awake_mask,
        gene_bounds=gene_bounds,
        depths=cfg.optimizer.precise_depths,
    )
    logger.info(
        f"  [{method} seed {seed_idx}] Stage 2: "
        f"{stage2_matrix.shape[0]} individuals"
    )

    s2_rows, s2_pareto_geno, s2_pareto_fit, _ = _run_one_stage(
        tester=tester,
        seed_idx=seed_idx,
        seed=seed,
        sampling=stage2_matrix,
        max_generations=1,
        stage_tag="stage2",
        run_dir=run_dir,
        origin_tensor=origin_tensor,
        scored_categories=scored_categories,
        target_classes=target_classes,
        answer_suffix=answer_suffix,
        pair=pair,
        parquet_metadata=parquet_metadata,
        tgtbal_idx=tgtbal_idx,
    )

    if method == "M3":
        _finalize_seed(
            tester, run_dir, seed_idx, seed, pair, scored_categories,
            target_classes, answer_suffix, start, s2_pareto_geno,
            s2_pareto_fit, method, stages_ran=["stage1", "stage2"],
        )
        return

    # -- Stage 3 — evolution seeded from Stage 2 Pareto --------------------
    pop_size = max(len(s2_pareto_geno), 10)
    # If Pareto is very small, inflate with random variants so AGE-MOEA-2
    # has enough diversity. build_pareto_init handles the expansion.
    stage3_sampling = build_pareto_init(
        pareto_genotypes=s2_pareto_geno,
        pop_size=pop_size,
        gene_bounds=gene_bounds,
        perturbation_prob=0.1,
        rng=np.random.default_rng(seed_idx),
    )

    early_stop_cfg_raw = cfg.optimizer.early_stop
    if early_stop_cfg_raw.enable:
        # HV reference: one unit past each objective's worst-case value.
        # Use conservative (3.0, 3.0, 5.0) as a safe reference given our
        # objective scales; downstream analysis can normalise.
        es_cfg = EarlyStopConfig(
            epsilon_margin=early_stop_cfg_raw.epsilon_margin,
            plateau_patience=early_stop_cfg_raw.plateau_patience,
            no_improvement_warmup=early_stop_cfg_raw.no_improvement_warmup,
            hypervolume_reference=(3.0, 3.0, 5.0),
            max_generations=cfg.optimizer.evolution_generations,
        )
        early_stop = EarlyStopChecker(es_cfg)
    else:
        early_stop = None

    logger.info(
        f"  [{method} seed {seed_idx}] Stage 3: "
        f"pop_size={pop_size}, max_gens={cfg.optimizer.evolution_generations}"
    )

    s3_rows, s3_pareto_geno, s3_pareto_fit, gens_run = _run_one_stage(
        tester=tester,
        seed_idx=seed_idx,
        seed=seed,
        sampling=stage3_sampling,
        max_generations=cfg.optimizer.evolution_generations,
        stage_tag="stage3",
        run_dir=run_dir,
        origin_tensor=origin_tensor,
        scored_categories=scored_categories,
        target_classes=target_classes,
        answer_suffix=answer_suffix,
        pair=pair,
        parquet_metadata=parquet_metadata,
        early_stop=early_stop,
        tgtbal_idx=tgtbal_idx,
    )

    _finalize_seed(
        tester, run_dir, seed_idx, seed, pair, scored_categories,
        target_classes, answer_suffix, start, s3_pareto_geno,
        s3_pareto_fit, method, stages_ran=["stage1", "stage2", "stage3"],
        gens_run=gens_run,
    )


# ---------------------------------------------------------------------------
# Finalisation — save pareto, stats, context
# ---------------------------------------------------------------------------


def _finalize_seed(
    tester: VLMBoundaryTester,
    run_dir: Path,
    seed_idx: int,
    seed: SeedTriple,
    pair: tuple[str, str],
    scored_categories: tuple[str, ...],
    target_classes: tuple[int, int],
    answer_suffix: str,
    start: float,
    pareto_geno: NDArray[np.int64],
    pareto_fit: NDArray[np.float64],
    method: str,
    stages_ran: list[str],
    gens_run: int | None = None,
) -> None:
    """Write pareto images/JSON + stats.json + context.json for a seed."""
    cfg = tester._config
    runtime = time() - start

    for i, (geno, fit) in enumerate(zip(pareto_geno, pareto_fit)):
        g = geno.astype(np.int64).reshape(1, -1)
        imgs, txts = tester._manipulator.manipulate(candidates=None, weights=g)
        imgs[0].save(run_dir / f"pareto_{i}.png")
        with open(run_dir / f"pareto_{i}.json", "w") as f:
            json.dump({
                "genotype": geno.tolist(),
                "fitness": fit.tolist(),
                "text": txts[0],
                "full_prompt": txts[0] + answer_suffix,
            }, f, indent=2)

    seed.image.save(run_dir / "origin.png")

    stats = _build_stats(
        seed_idx, seed, cfg, tester._manipulator,
        len(pareto_geno), runtime, scored_categories, pair, target_classes,
        cache_stats=tester._sut.cache_stats,
    )
    stats["method"] = method
    stats["stages_ran"] = stages_ran
    if gens_run is not None:
        stats["evolution_generations_run"] = gens_run
    with open(run_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    ctx_meta = _build_context_meta(tester._manipulator)
    with open(run_dir / "context.json", "w") as f:
        json.dump(ctx_meta, f, indent=2)

    logger.info(f"  Saved {method} results to {run_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_experiment(cfg: dict, pair_names: list[str], replicates: int) -> None:
    """Build components, resolve pairs, dispatch per-method per-seed."""
    exp = load_config(cfg)
    data_source = ImageNetCache(dirs=exp.cache_dirs)
    exp = resolve_categories(exp, data_source.labels())

    pool = ThreadPoolExecutor(max_workers=3)

    logger.info("Text manipulator starting...")
    text_fut: Future[TextManipulator] = pool.submit(
        TextManipulator.from_pretrained, config=exp.text,
    )
    logger.info(f"Image manipulator starting...  preset={exp.image.preset}")
    image_fut: Future[ImageManipulator] = pool.submit(
        ImageManipulator.from_preset, device=exp.device, config=exp.image,
    )
    logger.info(f"SUT starting...  {exp.sut.model_id} on {exp.device}")
    sut_fut: Future[VLMSUT] = pool.submit(VLMSUT, exp)

    objectives = CriterionCollection(
        MatrixDistance(),
        TextReplacementDistance(),
        TargetedBalance(),
    )
    optimizer = DiscretePymooOptimizer(
        gene_bounds=np.zeros(1, dtype=np.int64),
        num_objectives=3,
        pop_size=max(exp.pop_size, 1),
    )

    sut = sut_fut.result()
    logger.info("SUT loaded")

    # -- Resolve pairs to filter_indices -----------------------------------
    filter_indices: list[int] = []
    for pair_name in pair_names:
        indices = resolve_pair(
            pair=pair_name,
            config=exp,
            sut=sut,
            data_source=data_source,
            replicates=replicates,
        )
        logger.info(f"Resolved pair {pair_name!r} → filter_indices={indices}")
        filter_indices.extend(indices)

    exp = dataclasses.replace(
        exp,
        seeds=dataclasses.replace(exp.seeds, filter_indices=tuple(filter_indices)),
    )

    logger.info("Generating seed pool")
    seeds = generate_seeds(sut, exp, data_source)
    image_manip = image_fut.result()
    text_manip = text_fut.result()
    pool.shutdown(wait=False)

    manipulator = VLMManipulator(image_manip, text_manip)

    tester = VLMBoundaryTester(
        sut=sut,
        manipulator=manipulator,
        optimizer=optimizer,
        objectives=objectives,
        config=exp,
    )

    method = exp.optimizer.method
    logger.info(f"Method: {method}")

    # -- Method dispatch ---------------------------------------------------
    if method == "M0":
        tester.test(seeds)
        return

    if method not in ("M1", "M2", "M3", "M4"):
        raise ValueError(
            f"Unknown optimizer.method={method!r}. "
            f"Expected one of M0, M1, M2, M3, M4."
        )

    indexed_seeds = _apply_seed_filter(list(seeds), exp.seeds.filter_indices)
    if not indexed_seeds:
        logger.warning("No seeds after filter — nothing to run.")
        return

    logger.info(
        f"Running {method} on {len(indexed_seeds)} seed(s)"
    )
    for pos, (seed_idx, seed) in enumerate(indexed_seeds):
        logger.info(
            f"[{pos + 1}/{len(indexed_seeds)}] seed {seed_idx}: "
            f"{seed.class_a} vs {seed.class_b}"
        )
        _run_seed_screening(tester, seed_idx, seed)
        optimizer.reset()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a SMOO adaptive-resolution screening experiment.",
    )
    parser.add_argument("config", type=Path, help="Path to experiment YAML")
    parser.add_argument(
        "--pair", action="append", default=[],
        help=(
            "Pair to resolve (repeatable). Format: 'class_a->class_b' or "
            "'class_a-class_b'. Underscores replace spaces."
        ),
    )
    parser.add_argument(
        "--replicates", type=int, default=1,
        help="Number of seed replicates per pair (default 1).",
    )
    parser.add_argument("--device", help="Override device (cpu|cuda|mps)")
    parser.add_argument("--save-dir", type=str, help="Override save_dir")
    parser.add_argument("--name", type=str, help="Override experiment name")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}

    if args.device:
        cfg["device"] = args.device
    if args.save_dir:
        cfg["save_dir"] = args.save_dir
    if args.name:
        cfg["name"] = args.name

    if not args.pair and not cfg.get("seeds", {}).get("filter_indices"):
        raise SystemExit(
            "Error: must pass at least one --pair or set "
            "seeds.filter_indices in the YAML."
        )

    run_experiment(cfg, args.pair, args.replicates)
    os._exit(0)


if __name__ == "__main__":
    main()
