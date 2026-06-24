"""VLM boundary tester — the main orchestrator.

Wires together :class:`VLMManipulator`, :class:`VLMSUT`,
:class:`DiscretePymooOptimizer`, and 3 live objectives
(MatrixDistance, TextEmbeddingDistance, TargetedBalance) into a
multi-objective evolutionary search loop. Objectives are batched and
run through a :class:`CriterionCollection`.

Each seed triple ``(image, class_A, class_B)`` is tested independently:
the manipulator prepares image + text contexts, the optimizer evolves
integer genotypes, and every SUT interaction is logged to a Parquet
trace for offline analysis.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from time import time
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from PIL import Image
from smoo import SMOO
from torch import Tensor
from tqdm import tqdm

from src import distlock
from src.common import (
    apply_seed_filter,
    build_context_meta,
    seed_target_class,
)
from src.common.artifacts import EVOLUTIONARY_SCHEMA_VERSION, ParquetBuffer
from src.manipulator.vlm_manipulator import VLMManipulator
from src.objectives import CriterionCollection
from src.optimizer.discrete_pymoo_optimizer import DiscretePymooOptimizer
from src.optimizer.early_stop import EarlyStopChecker, EarlyStopConfig, EarlyStopTrigger
from src.optimizer.sparse_sampling import build_sampler_from_config

from src.config import ExperimentConfig, SeedTriple

logger = logging.getLogger(__name__)


def effective_prompt_template(seed: SeedTriple, config: ExperimentConfig) -> str:
    """Per-seed prompt override (grounding referent) → else the global template."""
    if seed.metadata and seed.metadata.get("prompt_template"):
        return seed.metadata["prompt_template"]
    return config.prompt_template


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def pil_to_tensor(img: Image.Image) -> Tensor:
    """Convert a PIL image to a ``(C, H, W)`` float tensor in ``[0, 1]``."""
    arr = np.array(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    return torch.from_numpy(arr).permute(2, 0, 1)


def _fmt_metric(v: float) -> str:
    """Compact, scale-aware formatter for pbar postfix values.

    Fixed 3-decimal for values >= 1e-3, scientific for smaller non-zero
    values so tiny-but-nonzero distances no longer display as ``0.000``.
    """
    if v is None or not np.isfinite(v):
        return "nan"
    if v == 0.0:
        return "0"
    if abs(v) >= 1e-3:
        return f"{v:.3f}"
    return f"{v:.2e}"


def build_trace_rows(
    seed_idx: int,
    gen: int,
    genotypes: NDArray,
    logits: Tensor,
    texts: list[str],
    cache_hits: list[bool],
    batched_results: dict[str, Any],
    batched_fitness: tuple,
    target_classes: tuple[int, int],
    categories: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Build per-individual trace rows (pure — no side effects).

    Three representations of the per-individual SUT state are stored:

    * ``logprobs`` — length-N length-normalized log-prob vector as
      returned by the scorer. The raw, canonical source; everything
      else is derivable from it. Full FP32 precision.
    * ``probs`` — length-N softmax of ``logprobs``, the full N-class
      probability distribution. Convenience for post-hoc analyses that
      want to work in probability space (entropy, KL, top-k, trees).
    * ``p_class_a`` / ``p_class_b`` — pair-conditional probabilities
      (2-class softmax over just the target pair). These sum to 1 per
      row and preserve the historical 2-class semantics that existing
      viz code in ``analysis/viz_*`` relies on. Prefer ``probs`` in new
      code.

    All three are redundant in information content but serve different
    ergonomic niches; storage is ~N floats per row, negligible.
    """
    idx_a, idx_b = target_classes

    # Full N-class distribution.
    probs = F.softmax(logits, dim=-1)

    # Pair-conditional 2-class softmax for backward-compatible
    # p_class_a / p_class_b columns. Computed from raw logits (not
    # re-normalised from `probs`) so the pair-conditional semantics is
    # exact regardless of how much N-class mass sits on other classes.
    pair_lp = logits[:, [idx_a, idx_b]]
    pair_probs = F.softmax(pair_lp, dim=-1)

    fitness_names = list(batched_results.keys())
    return [
        {
            "seed_id": seed_idx,
            "generation": gen,
            "individual": i,
            "genotype": genotypes[i].tolist(),
            "logprobs": logits[i].tolist(),
            "probs": probs[i].tolist(),
            "decoded_text": texts[i],
            "cache_hit": cache_hits[i],
            "predicted_class": categories[int(logits[i].argmax())],
            "p_class_a": float(pair_probs[i, 0]),
            "p_class_b": float(pair_probs[i, 1]),
            **{
                f"fitness_{name}": float(batched_fitness[j][i])
                for j, name in enumerate(fitness_names)
            },
        }
        for i in range(len(genotypes))
    ]


def build_stats(
    seed_idx: int,
    seed: SeedTriple,
    config: ExperimentConfig,
    manipulator: VLMManipulator,
    n_pareto: int,
    runtime: float,
    scored_categories: tuple[str, ...],
    pair: tuple[str, str],
    target_classes: tuple[int, int],
    cache_stats: dict[str, int],
) -> dict[str, Any]:
    """Build stats metadata dict (pure).

    The ``categories`` field records the category list the SUT actually
    scored against — length 2 in pair-only mode (default), length N in
    ``score_full_categories`` mode. The ``pair`` / ``target_classes``
    fields record which two entries are the target pair, so readers can
    recover ``class_a`` / ``class_b`` positions in the ``logprobs``
    column without any external lookup.

    :param scored_categories: The actually-scored category list
        (canonical index map for the ``logprobs`` column).
    :param pair: ``(class_a, class_b)`` labels.
    :param target_classes: Positions of ``pair`` inside
        ``scored_categories``. ``(0, 1)`` in pair-only mode; the pair
        positions in the full list in N-class mode.
    :param cache_stats: Aggregate ``{"hits", "misses"}`` counts at save time.
    """
    out: dict[str, Any] = {
        "schema_version": EVOLUTIONARY_SCHEMA_VERSION,
        "seed_idx": seed_idx,
        "class_a": seed.class_a,
        "class_b": seed.class_b,
        "prompt_template": config.prompt_template,
        "answer_format": config.answer_format,
        "runtime_seconds": runtime,
        "generations": config.generations,
        "pop_size": config.pop_size,
        "gene_bounds": manipulator.gene_bounds.tolist(),
        "image_dim": manipulator.image_dim,
        "text_dim": manipulator.text_dim,
        # Actually-scored category list. Length-2 when
        # score_full_categories is False (default), length-N when True.
        # Readers use this to map logprob-column indices back to class
        # names, regardless of scope.
        "categories": list(scored_categories),
        "pair": list(pair),
        "target_classes": list(target_classes),
        "n_categories_scored": len(scored_categories),
        # Full category list from the config (ignoring scoring scope) —
        # useful for downstream analyses that need to know the universe
        # of classes even in pair-only runs.
        "categories_universe": list(config.categories),
        "n_categories_universe": len(config.categories),
        "score_full_categories": config.score_full_categories,
        "n_pareto": n_pareto,
        "cache_hits": cache_stats["hits"],
        "cache_misses": cache_stats["misses"],
        # Reproducibility metadata
        "model_id": config.sut.model_id,
        "seed_selection_mode": config.seeds.mode,
        "n_categories_seed": config.n_categories,
        "image_preset": config.image.preset,
        "image_patch_ratio": config.image.patch_ratio,
        "image_patch_strategy": config.image.patch_strategy.name,
        "image_candidate_strategy": config.image.candidate_strategy.name,
        "image_n_candidates": config.image.n_candidates,
        "device": config.device,
    }
    # Mode-specific seed-config provenance.
    if config.seeds.mode == "gap_filter" and config.seeds.gap_filter is not None:
        out["max_logprob_gap"] = config.seeds.gap_filter.max_logprob_gap
        out["n_per_class"] = config.seeds.gap_filter.n_per_class
    elif config.seeds.mode == "roster" and config.seeds.roster is not None:
        out["roster_seeds_per_class"] = config.seeds.roster.seeds_per_class
        out["roster_min_anchor_confidence"] = (
            config.seeds.roster.min_anchor_confidence
        )
        out["roster_class_list"] = list(config.seeds.roster.class_list)
    elif config.seeds.mode == "refcocoplus" and config.seeds.refcocoplus is not None:
        out["refcoco_split"] = config.seeds.refcocoplus.split
        out["refcoco_n_items"] = config.seeds.refcocoplus.n_items
        out["coordinate_space"] = config.grounding.coordinate_space
        out["grounding_answer_format"] = config.grounding.answer_format
    # Per-seed metadata (Exp-100 roster pipeline emits taxonomy / abstraction
    # bookkeeping; gap_filter leaves it None). Stored under a namespaced key
    # so it can never shadow an existing stats field.
    if seed.metadata is not None:
        out["seed_metadata"] = dict(seed.metadata)
    return out


# ---------------------------------------------------------------------------
# Tester
# ---------------------------------------------------------------------------


class VLMBoundaryTester(SMOO):
    """Multi-objective boundary tester for Vision-Language Models.

    Orchestrates the full optimisation loop per seed::

        seed → prepare → [ask → manipulate → SUT → objectives → fitness
                          → update] × generations → save

    All 3 live objectives (MatrixDistance, TextEmbeddingDistance,
    TargetedBalance) are batched and passed via the
    :class:`CriterionCollection`.

    :param sut: The VLM system-under-test.
    :param manipulator: Multi-modal manipulator (image + text).
    :param optimizer: Discrete evolutionary optimizer.
    :param objectives: Criterion collection (3 batched objectives).
    :param config: Experiment-level settings.
    """

    _manipulator: VLMManipulator
    _optimizer: DiscretePymooOptimizer

    def __init__(
        self,
        *,
        sut,
        manipulator: VLMManipulator,
        optimizer: DiscretePymooOptimizer,
        objectives: CriterionCollection,
        config: ExperimentConfig,
    ) -> None:
        super().__init__(
            sut=sut,
            manipulator=manipulator,
            optimizer=optimizer,
            objectives=objectives,
            restrict_classes=None,
            use_wandb=False,
        )
        self._config = config

        # Modality-driven gates — skip per-modality work when the
        # corresponding criterion isn't in the live collection.
        # Result-key naming: MatrixDistance_<aggregator> (e.g. _fro),
        # TextEmbeddingDistance → "TextDist", TargetedBalance → "TgtBal".
        names = list(objectives.names)
        self._has_matrix_dist = any(
            n.startswith("MatrixDistance") for n in names
        )
        self._has_text_dist = "TextDist" in names

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def test(
        self,
        seeds: Sequence[SeedTriple],
        worker_id: int | None = None,
        worker_stride: int | None = None,
    ) -> None:
        """Run boundary testing for all seeds.

        Applies the optional ``config.seeds.filter_indices`` post-generation
        filter so that targeted re-runs (e.g. Exp-05 Phase A single-seed
        deep probes) use the same indexing as the unfiltered pool — a
        filtered seed at original position 0 still reports as
        ``seed_idx=0`` in output naming.

        :param seeds: Sequence of ``(image, class_a, class_b)`` triples,
            as emitted by :func:`generate_seeds`.
        :param worker_id: When set with *worker_stride*, evaluates only
            the round-robin slice ``indexed_seeds[worker_id::worker_stride]``.
            Slicing happens AFTER ``filter_indices`` so per-seed naming
            stays consistent across the worker pool.
        :param worker_stride: Total number of parallel workers.
        """
        indexed_seeds = apply_seed_filter(
            list(seeds), self._config.seeds.filter_indices,
        )

        if worker_id is not None and worker_stride is not None:
            before = len(indexed_seeds)
            indexed_seeds = indexed_seeds[worker_id::worker_stride]
            logger.info(
                "Worker %d/%d: %d/%d seeds assigned",
                worker_id, worker_stride, len(indexed_seeds), before,
            )

        if not indexed_seeds:
            logger.warning("No seeds after filter — nothing to test.")
            return

        n_gens = self._config.generations
        pbar = tqdm(total=n_gens, unit="gen", position=worker_id or 0)

        for pos, (seed_idx, seed) in enumerate(indexed_seeds):
            pbar.reset()
            pbar.set_description(
                f"[{pos + 1}/{len(indexed_seeds)}] "
                f"seed_{seed_idx} {seed.class_a} vs {seed.class_b}"
            )
            self._run_seed(seed_idx, seed, pbar)
            self._optimizer.reset()
            self._cleanup()

        pbar.close()

    # ------------------------------------------------------------------
    # Per-seed loop
    # ------------------------------------------------------------------

    def run_one_seed(
        self,
        seed_idx: int,
        seed: SeedTriple,
        run_dir: Path | None = None,
        reset_optimizer: bool = False,
    ) -> list:
        """Run optimisation for a single seed; return Pareto candidates.

        Combined-pipeline entry point: lets the main runner interleave
        the evolutionary stage with PDQ on a per-seed basis without
        spinning up a fresh tester each time.

        :param seed_idx: 0-based seed index.
        :param seed: Seed triple.
        :param run_dir: Output directory override. ``None`` falls back to
            the legacy ``save_dir/<name>_seed_<i>_<ts>`` layout.
        :param reset_optimizer: When True, resets optimiser state and
            cleans up before returning (matches legacy ``test()`` flow).
            Combined-pipeline callers leave this False so they can read
            the Pareto front and reset themselves after the handoff.
        :returns: List of Pareto candidates from
            :attr:`optimizer.best_candidates`.
        """
        pbar = tqdm(
            total=self._config.generations,
            unit="gen",
            position=0,
            desc=f"seed_{seed_idx} {seed.class_a} vs {seed.class_b}",
        )
        try:
            self._run_seed(seed_idx, seed, pbar, run_dir=run_dir)
        finally:
            pbar.close()
        pareto = list(self._optimizer.best_candidates)
        if reset_optimizer:
            self._optimizer.reset()
            self._cleanup()
        return pareto

    def _run_seed(
        self,
        seed_idx: int,
        seed: SeedTriple,
        pbar: tqdm,
        run_dir: Path | None = None,
    ) -> None:
        start_time = time()

        # Prompt suffix always pair-constrained — the VLM sees "Answer with
        # A or B" regardless of how many categories the SUT scores against.
        pair = (seed.class_a, seed.class_b)
        answer_suffix = self._config.answer_format.format(
            categories=", ".join(pair),
        )
        full_categories = tuple(self._config.categories)

        # ---- Scoring-scope: pair (default) vs. full N-class ----------------
        # Controlled by ExperimentConfig.score_full_categories (default False).
        # Pair-only is ~25× cheaper per call on N=50 and is the right choice
        # unless a downstream post-hoc analysis explicitly needs the full
        # N-dim log-prob vector at every individual (tree distances,
        # cross-class entropy, etc.). When True, ``target_classes`` becomes
        # the pair's position in the full list so TargetedBalance still
        # reads the right two entries.
        if self._config.score_full_categories:
            scored_categories = full_categories
            target_classes = (
                full_categories.index(seed.class_a),
                full_categories.index(seed.class_b),
            )
        else:
            scored_categories = pair
            target_classes = (0, 1)

        # 1. Prepare manipulator with just the question prompt. The
        #    seed's L0 target class is forwarded so cone-filter mode can
        #    route through the per-class modal target grid. Legacy KNN
        #    mode ignores the value but still records it on the context
        #    for trace metadata.
        target_class = seed_target_class(seed)
        prompt_template = effective_prompt_template(seed, self._config)
        self._manipulator.prepare(
            seed.image,
            prompt_template,
            target_class=target_class,
            origin_class=seed.class_a,
        )

        # 1b. Anchor sentence embedding for TextEmbeddingDistance. The
        #     manipulator at gene=0 reproduces the original prompt, so
        #     using ``prompt_template`` here keeps the distance for any
        #     all-zero individual at exactly 0.
        #
        #     Skipped entirely when TextEmbeddingDistance is not in the
        #     objective set (modality=image_only). ``text_embedder`` is
        #     also None on backends that can't encode_text (currently
        #     OpenVINO) — in that case TextEmbeddingDistance receives
        #     zeros and effectively drops out of the objective set.
        if self._has_text_dist and self._sut.text_embedder is not None:
            self._anchor_text_embedding = self._sut.text_embedder.embed(
                prompt_template,
            )
        else:
            self._anchor_text_embedding = None

        # 2. Install per-seed init sampler. Only relevant when the
        #    OptimizerConfig selected a non-uniform sampling mode; the
        #    default "uniform" path leaves algo_params untouched and
        #    inherits PyMoo's IntegerRandomSampling from the optimizer.
        #
        #    `sparse_multitier_fps` additionally requires the per-seed
        #    VQGAN codebook + KNN-ordered candidate lists to do FPS
        #    over codebook embeddings at active positions. These come
        #    from the image manipulator's prepared selection.
        sampling_cfg = self._config.optimizer.sampling
        codebook = None
        candidates_per_position = None
        if sampling_cfg.mode == "sparse_multitier_fps":
            img_ctx = self._manipulator.image_context
            if img_ctx is not None:
                codebook = self._manipulator.image_manipulator.codec.codebook
                candidates_per_position = img_ctx.selection.candidates

        sampler = build_sampler_from_config(
            sampling_cfg,
            text_dim=self._manipulator.text_dim,
            codebook=codebook,
            candidates_per_position=candidates_per_position,
        )
        if sampler is not None:
            self._optimizer.set_sampling(sampler)

        # 3. Re-configure optimizer for this seed's genotype dimensions.
        self._optimizer.update_gene_bounds(self._manipulator.gene_bounds)

        # 4. VQGAN-reconstructed baseline for MatrixDistance.
        #    Skipped when MatrixDistance isn't in the objective set
        #    (modality=text_only). _run_generation will short-circuit the
        #    per-individual tensor conversion in that case.
        if self._has_matrix_dist:
            # κ=0 reference for MatrixDistance. Delegated to the backend
            # so StyleGAN can reuse its precheck-cached origin image and
            # VQGAN can do its single decode — no batched manipulate
            # call, no SMOO batch>=2 quirk.
            origin_tensor = pil_to_tensor(self._manipulator.baseline_image())
        else:
            origin_tensor = None

        # 5. Output directory created up-front so incremental writers can
        #    target it. Previously created at end-of-seed by _save_seed_results.
        if run_dir is None:
            run_dir = (
                self._config.save_dir
                / f"{self._config.name}_seed_{seed_idx}_{int(start_time)}"
            )
        run_dir.mkdir(parents=True, exist_ok=True)

        # File-level metadata stamped onto every parquet this seed produces.
        parquet_metadata: dict[bytes, bytes] = {
            b"schema_version": str(EVOLUTIONARY_SCHEMA_VERSION).encode(),
            b"pipeline": b"smoo",
            b"categories": json.dumps(list(scored_categories)).encode(),
            b"pair": json.dumps(list(pair)).encode(),
            b"target_classes": json.dumps(list(target_classes)).encode(),
        }

        # 6. Early-stop checker. OR-combines four triggers (flip /
        #    plateau-HV / no-improvement-since-seed / hard-cap). Disabled
        #    when ``optimizer.early_stop.enable`` is False — falls back
        #    to the hard cap at ``config.generations``.
        early_cfg = self._config.optimizer.early_stop
        early_stop_checker: EarlyStopChecker | None = None
        if early_cfg.enable:
            early_stop_checker = EarlyStopChecker(
                EarlyStopConfig(
                    epsilon_margin=early_cfg.epsilon_margin,
                    plateau_patience=early_cfg.plateau_patience,
                    no_improvement_warmup=early_cfg.no_improvement_warmup,
                    hypervolume_reference=early_cfg.hypervolume_reference,
                    max_generations=self._config.generations,
                )
            )
        early_stop_trigger: EarlyStopTrigger | None = None

        # 7. Generation loop with incremental flushing. Trace rows flush
        #    every 100 rows (bounded memory), convergence flushes every
        #    row (one per gen — crash-safety matters more than row-group
        #    efficiency for this tiny file).
        trace_buf = ParquetBuffer(
            run_dir / "trace.parquet",
            flush_interval=100,
            file_metadata=parquet_metadata,
        )
        conv_buf = ParquetBuffer(
            run_dir / "convergence.parquet",
            flush_interval=1,
            file_metadata=parquet_metadata,
        )

        try:
            for gen in range(self._config.generations):
                gen_start = time()

                gen_traces = self._run_generation(
                    seed_idx, gen, target_classes, origin_tensor,
                    scored_categories, answer_suffix,
                )

                # -- Flush trace rows for this generation -----------------
                if gen_traces:
                    trace_buf.append_many(gen_traces)

                # -- Convergence row from Pareto front --------------------
                pareto = self._optimizer.best_candidates
                pareto_fitness = np.array([c.fitness for c in pareto])
                obj_names = list(self._objectives.results.keys())

                conv_row: dict[str, Any] = {
                    "generation": gen,
                    "n_pareto": len(pareto),
                    "wall_time": time() - gen_start,
                }
                for j, name in enumerate(obj_names):
                    conv_row[f"pareto_min_{name}"] = float(
                        pareto_fitness[:, j].min()
                    )
                    conv_row[f"pareto_mean_{name}"] = float(
                        pareto_fitness[:, j].mean()
                    )

                # Coordinates of the Pareto individual with minimum TgtBal —
                # answers "best boundary flip, and at what (img, text) cost?"
                # rather than three independent per-axis minima.
                if "TgtBal" in obj_names:
                    tgtbal_j = obj_names.index("TgtBal")
                    best_idx = int(np.argmin(pareto_fitness[:, tgtbal_j]))
                    for j, name in enumerate(obj_names):
                        conv_row[f"pareto_atbest_TgtBal_{name}"] = float(
                            pareto_fitness[best_idx, j]
                        )
                else:
                    best_idx = None

                # Population stats from this generation's traces.
                pop_fitness = {name: [] for name in obj_names}
                for row in gen_traces:
                    for name in obj_names:
                        pop_fitness[name].append(row[f"fitness_{name}"])
                for name in obj_names:
                    vals = np.array(pop_fitness[name])
                    conv_row[f"pop_min_{name}"] = float(vals.min())
                    conv_row[f"pop_mean_{name}"] = float(vals.mean())

                conv_buf.append(conv_row)

                # -- Progress bar with convergence info -------------------
                # Show the best-TgtBal Pareto individual with the (img, text)
                # distances at which that flip was achieved — the condition
                # under which the boundary was crossed, not three decoupled
                # per-axis minima.
                postfix: dict[str, Any] = {"pareto": len(pareto)}
                if best_idx is not None:
                    postfix["bal"] = _fmt_metric(
                        conv_row["pareto_atbest_TgtBal_TgtBal"]
                    )
                    for name in obj_names:
                        if name == "TgtBal":
                            continue
                        short = (
                            "img" if name.startswith("MatrixDistance_")
                            else "txt" if "Text" in name
                            else name
                        )
                        postfix[f"@{short}"] = _fmt_metric(
                            conv_row[f"pareto_atbest_TgtBal_{name}"]
                        )
                postfix["t"] = f"{time() - gen_start:.0f}s"
                pbar.set_postfix(postfix)
                pbar.update(1)

                # -- Early-stop check. Fires after the current gen is
                #    fully written, so the trace / convergence parquet
                #    always matches what the optimizer actually saw.
                if early_stop_checker is not None:
                    try:
                        tgtbal_col_idx = obj_names.index("TgtBal")
                    except ValueError:
                        tgtbal_col_idx = None
                    if tgtbal_col_idx is not None:
                        trig = early_stop_checker.update(
                            generation=gen,
                            pareto_fitness=pareto_fitness,
                            tgtbal_index=tgtbal_col_idx,
                        )
                        if trig is not None and trig.trigger != "hard_cap":
                            logger.info(
                                "[seed %d] early stop at gen %d: %s (%s)",
                                seed_idx, gen, trig.trigger, trig.details,
                            )
                            early_stop_trigger = trig
                            break
        finally:
            # Close writers even on exception so partial parquets are
            # still readable after a mid-run Ctrl-C or kill.
            trace_buf.close()
            conv_buf.close()

        # 8. Finalize — write non-streaming artefacts (stats, pareto
        #    images, context, origin image) now that trace / convergence
        #    are on disk.
        runtime = time() - start_time
        self._finalize_seed_output(
            seed_idx, seed, run_dir,
            scored_categories, pair, target_classes,
            answer_suffix, runtime,
            early_stop=early_stop_trigger,
        )

    # ------------------------------------------------------------------
    # Single generation
    # ------------------------------------------------------------------

    def _run_generation(
        self,
        seed_idx: int,
        gen: int,
        target_classes: tuple[int, int],
        origin_tensor: Tensor,
        categories: tuple[str, ...],
        answer_suffix: str,
        sut_progress_desc: str | None = None,
    ) -> list[dict[str, Any]]:
        """Evaluate one generation and advance the optimizer.

        :param sut_progress_desc: If not None, renders an inner tqdm bar
            around the SUT calls with this description. Used by the
            screening runner for stages with hundreds of individuals
            per "generation"; default behaviour (silent inner loop)
            is preserved for the baseline tester.

        Returns a list of trace-row dicts (one per individual).
        """
        genotypes = self._optimizer.get_x_current()

        # -- Manipulate ---------------------------------------------------
        images, texts = self._manipulator.manipulate(
            candidates=None, weights=genotypes,
        )

        # -- SUT evaluation -----------------------------------------------
        # Append answer options AFTER text mutation so categories are
        # never exposed to the text optimizer.
        logits_list: list[Tensor] = []
        cache_hits: list[bool] = []
        iterator = zip(images, texts)
        if sut_progress_desc is not None:
            iterator = tqdm(
                iterator,
                total=len(images),
                desc=sut_progress_desc,
                unit="call",
                leave=False,
            )
        with distlock.lock(self._sut.device_str):
            for img, txt in iterator:
                logits_list.append(
                    self._sut.process_input(
                        img, text=txt + answer_suffix, categories=categories,
                    )
                )
                cache_hits.append(self._sut.last_call_cached)

        logits = torch.stack(logits_list)

        # -- Tensor conversion for MatrixDistance -------------------------
        # Skipped when modality=text_only (no MatrixDistance criterion);
        # CriterionCollection ignores ``images`` if no image-criterion
        # consumes it, so passing None through is fine.
        if self._has_matrix_dist:
            perturbed_batch = torch.stack(
                [pil_to_tensor(img) for img in images]
            )
            origin_batch = origin_tensor.unsqueeze(0).expand_as(perturbed_batch)
            images_arg: list | None = [origin_batch, perturbed_batch]
        else:
            images_arg = None

        # -- Text-distance objective ---------------------------------------
        # Skipped when modality=image_only (no TextEmbeddingDistance
        # criterion). Backends without encode_text feed zeros so the
        # criterion passes through with no selection pressure.
        text_kwargs: dict = {}
        if self._has_text_dist:
            if self._sut.text_embedder is not None:
                text_distances = self._sut.text_embedder.cosine_distances_to(
                    self._anchor_text_embedding, texts,
                )
            else:
                import numpy as _np
                text_distances = _np.zeros(len(texts), dtype=_np.float32)
            text_kwargs["text_distances"] = text_distances

        # -- Evaluate batched objectives ---------------------------------
        self._objectives.evaluate_all(
            images=images_arg,
            logits=logits,
            target_classes=target_classes,
            batch_dim=0,
            **text_kwargs,
        )

        batched_results = self._objectives.results
        fitness = tuple(
            f.cpu().numpy() if isinstance(f, Tensor) else np.asarray(f)
            for f in batched_results.values()
        )

        # -- Assign fitness & update optimizer ----------------------------
        self._optimizer.assign_fitness(fitness)
        self._optimizer.update()

        # -- Build trace rows ---------------------------------------------
        return build_trace_rows(
            seed_idx, gen, genotypes, logits, texts, cache_hits,
            batched_results, fitness,
            target_classes, categories,
        )

    # ------------------------------------------------------------------
    # Result saving
    # ------------------------------------------------------------------

    def _finalize_seed_output(
        self,
        seed_idx: int,
        seed: SeedTriple,
        run_dir: Path,
        scored_categories: tuple[str, ...],
        pair: tuple[str, str],
        target_classes: tuple[int, int],
        answer_suffix: str,
        runtime: float,
        early_stop: EarlyStopTrigger | None = None,
    ) -> None:
        """Write non-streaming seed artefacts.

        ``trace.parquet`` and ``convergence.parquet`` are written
        incrementally during :meth:`_run_seed` via ``ParquetBuffer``; by
        the time this method is called, both files are already closed
        on disk. This method only handles the bits that need the whole
        seed's final state: the Pareto candidate images and JSONs, the
        origin image, ``stats.json``, and ``context.json``.

        :param seed_idx: 0-based seed index.
        :param seed: The seed triple.
        :param run_dir: Output directory created at the start of
            :meth:`_run_seed`.
        :param scored_categories: The category list the SUT actually
            scored against during this seed — either the 2-element pair
            (pair-only scoring, default) or the full N-category list
            (``config.score_full_categories = True``). This is the
            canonical index→class mapping for the ``logprobs`` column.
        :param pair: ``(class_a, class_b)`` labels. Equals
            *scored_categories* in pair-only mode.
        :param target_classes: Positions of ``pair`` inside
            *scored_categories*. ``(0, 1)`` in pair-only mode, or the
            pair indices in the full list in N-class mode.
        :param answer_suffix: Pair-constrained answer format suffix.
        :param runtime: Wall-clock seconds spent on this seed.
        """
        # -- Pareto-optimal candidates ------------------------------------
        # Batched manipulate over all Pareto candidates — one synthesis
        # forward instead of N batch=1 calls. Edge case: if the front
        # degenerates to a single candidate, pad to two and discard the
        # duplicate (SMOO's manipulate asserts `cond[1]` and IndexErrors
        # at batch=1; we accept one wasted render in this rare path
        # rather than skipping the save).
        candidates_list = list(self._optimizer.best_candidates)
        if candidates_list:
            genotypes = np.stack(
                [c.solution.astype(np.int64) for c in candidates_list]
            )
            n_keep = len(candidates_list)
            if n_keep == 1:
                genotypes = np.tile(genotypes, (2, 1))
            imgs, txts = self._manipulator.manipulate(
                candidates=None, weights=genotypes,
            )
            for i, (cand, img, txt) in enumerate(
                zip(candidates_list, imgs[:n_keep], txts[:n_keep])
            ):
                img.save(run_dir / f"pareto_{i}.png")
                with open(run_dir / f"pareto_{i}.json", "w") as f:
                    json.dump(
                        {
                            "genotype": cand.solution.astype(np.int64).tolist(),
                            "fitness": cand.fitness.tolist()
                            if isinstance(cand.fitness, np.ndarray)
                            else list(cand.fitness),
                            "text": txt,
                            "full_prompt": txt + answer_suffix,
                        },
                        f,
                        indent=2,
                    )

        # -- Origin image -------------------------------------------------
        seed.image.save(run_dir / "origin.png")

        # -- Stats JSON ---------------------------------------------------
        stats = build_stats(
            seed_idx, seed, self._config, self._manipulator,
            len(self._optimizer.best_candidates),
            runtime, scored_categories, pair, target_classes,
            cache_stats=self._sut.cache_stats,
        )
        if early_stop is not None:
            stats["early_stop"] = {
                "trigger": early_stop.trigger,
                "generation": early_stop.generation,
                "details": early_stop.details,
            }
        with open(run_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        # -- Context metadata (for offline reconstruction) ----------------
        ctx_meta = build_context_meta(self._manipulator)
        with open(run_dir / "context.json", "w") as f:
            json.dump(ctx_meta, f, indent=2)

        logger.info(f"  Saved results to {run_dir}")
