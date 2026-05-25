"""Boundary-pair runner — orchestrates evolutionary → PDQ per seed.

Per seed::

    1. Evolutionary stage runs AGE-MOEA-II to find a Pareto front of
       near-boundary individuals (writes trace / convergence / pareto_*
       under ``<seed_dir>/evolutionary/``).
    2. Each Pareto member is selected as a PDQ anchor.  The anchor image
       is rendered, the SUT yields the anchor label via pair-restricted
       softmax, and :func:`~src.pdq.runner.run_pdq_core` runs Stage 1 +
       Stage 2 with that anchor.  All anchors for one seed share a
       single :class:`SeedLogger` at ``<seed_dir>/pdq/``; the
       archive.parquet rows are tagged by ``pareto_idx``.
    3. Optimiser is reset between seeds.

Shared models (SUT, VQGAN/StyleGAN, text composite) are loaded exactly
once at the start of the run.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from time import time
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.common import (
    apply_seed_filter,
    build_context_meta,
    dispatch_workers,
    init_shared_components,
    precompute_image_backend,
    prepare_pipeline_seeds,
)
from src.config import (
    ExperimentConfig,
    SeedTriple,
    resolve_categories as resolve_evolutionary_categories,
)
from src.data import ImageNetCache
from src.evolutionary import VLMBoundaryTester
from src.manipulator.image_backend import ImageBackend
from src.manipulator.text.composite import CompositeTextManipulator
from src.manipulator.vlm_manipulator import VLMManipulator
from src.objectives import (
    CriterionCollection,
    MatrixDistance,
    TargetedBalance,
    TextEmbeddingDistance,
)
from src.optimizer.discrete_pymoo_optimizer import DiscretePymooOptimizer
from src.pdq.artifacts import PDQ_SCHEMA_VERSION, SeedLogger
from src.pdq.config import (
    PDQExperimentConfig,
    config_to_dict as pdq_config_to_dict,
)
from src.pdq.runner import run_pdq_core
from src.pdq.sut_adapter import SUTAdapter
from src.sut import VLMSUT

from .config import (
    AnchorSelectionConfig,
    BoundaryPairExperimentConfig,
    boundary_pair_config_to_dict,
    to_evolutionary_config,
    to_pdq_config,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-worker thin bundle (one set of per-seed mutable state).
# ---------------------------------------------------------------------------


@dataclass
class _WorkerBundle:
    """Per-worker state for the boundary-pair pipeline.

    Threads share the underlying model objects (``VQGAN``, ``VLMSUT.scorer``,
    text composite) — these fields wrap them with per-worker counters,
    contexts, and RNG streams.
    """

    worker_id: int
    sut: VLMSUT
    manipulator: VLMManipulator
    optimizer: DiscretePymooOptimizer
    tester: VLMBoundaryTester
    adapter: SUTAdapter
    rng: np.random.Generator


# ---------------------------------------------------------------------------
# Anchor selection helpers
# ---------------------------------------------------------------------------


def _select_anchors(
    pareto: list, selection: AnchorSelectionConfig,
) -> list[tuple[int, Any]]:
    """Pick anchors from the Pareto front per selection config.

    Returns ``(pareto_idx, candidate)`` tuples in evaluation order.
    ``pareto_idx`` is the original index in the unsorted Pareto front
    (preserved across selection modes for traceability).
    """
    if selection.source != "pareto_front":
        raise ValueError(
            f"Unsupported anchor_selection.source {selection.source!r}; "
            "only 'pareto_front' is implemented."
        )

    indexed = list(enumerate(pareto))
    if selection.k is None or selection.k >= len(indexed):
        return indexed

    # Top-K by TgtBal (the last fitness axis across all modality configs —
    # MatrixDistance? + TextEmbeddingDistance? + TargetedBalance).
    fitness_sorted = sorted(
        indexed, key=lambda pair: float(pair[1].fitness[-1]),
    )
    return fitness_sorted[: selection.k]


def _pair_softmax_argmax(
    logprobs: list[float],
    categories: tuple[str, ...],
    class_a: str,
    class_b: str,
) -> tuple[str, float, float]:
    """Pair-restricted softmax over ``(class_a, class_b)`` → argmax label.

    Returns ``(anchor_label, p_a, p_b)``.  ``anchor_label`` is
    ``class_a`` when ``p_a >= p_b`` else ``class_b``.
    """
    idx_a = categories.index(class_a)
    idx_b = categories.index(class_b)
    lp = np.asarray([logprobs[idx_a], logprobs[idx_b]], dtype=np.float64)
    lp -= lp.max()  # numerical stability
    probs = np.exp(lp)
    probs /= probs.sum()
    p_a, p_b = float(probs[0]), float(probs[1])
    label = class_a if p_a >= p_b else class_b
    return label, p_a, p_b


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class BoundaryPairRunner:
    """Boundary-pair (evolutionary → PDQ) pipeline runner.

    With ``parallel.workers > 1`` the per-seed loop fans out across N
    worker threads sharing one model set: VQGAN/StyleGAN, the VLM
    scorer, and the text embedder are loaded once; each thread owns
    thin per-seed state (VLMSUT wrapper, VLMManipulator, optimiser,
    objectives, tester, SUTAdapter, RNG, per-seed SeedLogger).  GPU
    access is serialised by the process-local device mutex
    (:mod:`src.distlock`).  When SUT and VQGAN run on different device
    strings (e.g. OpenVINO Arc + torch CPU), threads run truly
    concurrently on the GPU step; otherwise the gain is CPU-prep
    overlap only.

    :param config: Boundary-pair experiment config.  Categories may be
        unresolved; :meth:`run` resolves them via ImageNetCache.
    """

    def __init__(self, config: BoundaryPairExperimentConfig) -> None:
        self._cfg = config

    def run(self) -> None:
        """Initialise components and run all seeds."""
        cfg = self._cfg

        # Resolve categories on the evolutionary projection, then mirror
        # back onto the boundary-pair config so PDQ/SeedLogger see the
        # same canonical list.
        data_source = ImageNetCache(dirs=list(cfg.cache_dirs))
        evo_cfg = to_evolutionary_config(cfg)
        evo_cfg = resolve_evolutionary_categories(evo_cfg, data_source.labels())
        cfg = self._with_resolved_categories(cfg, evo_cfg.categories)
        self._cfg = cfg
        evo_cfg = to_evolutionary_config(cfg)
        pdq_cfg = to_pdq_config(cfg)

        # -- Shared component init + seed gen + backend precompute -----
        components = init_shared_components(evo_cfg, data_source)
        seeds = prepare_pipeline_seeds(components, evo_cfg)
        if not seeds:
            logger.warning("No seeds — nothing to test.")
            return
        precompute_image_backend(components, seeds, evo_cfg)

        # -- Objective spec (per-worker collections built below) --------
        # ``crit_types`` is the ordered list of criterion classes; each
        # worker instantiates its own collection because
        # ``CriterionCollection.results`` is mutated per call.
        crit_types: list = []
        if evo_cfg.modality != "text_only":
            crit_types.append(MatrixDistance)
        if evo_cfg.modality != "image_only":
            crit_types.append(TextEmbeddingDistance)
        crit_types.append(TargetedBalance)
        logger.info(
            "modality=%s → %d objectives: %s",
            evo_cfg.modality,
            len(crit_types),
            ", ".join(c.__name__ for c in crit_types),
        )

        # -- Seed dispatch ----------------------------------------------
        indexed_seeds = apply_seed_filter(
            list(seeds), cfg.seeds.filter_indices,
        )
        workers = max(1, cfg.parallel.workers)
        logger.info(
            "%d seed(s) — boundary-pair pipeline (evolutionary → PDQ), workers=%d",
            len(indexed_seeds), workers,
        )

        dispatch_workers(
            workers=workers,
            indexed_seeds=indexed_seeds,
            build_bundle=lambda wid: self._build_worker_bundle(
                worker_id=wid,
                evo_cfg=evo_cfg,
                pdq_cfg=pdq_cfg,
                shared_sut=components.sut,
                image_manip=components.image_manip,
                text_manip=components.text_manip,
                crit_types=crit_types,
            ),
            run_slice=lambda slice_, bundle: self._run_seed_slice(
                indexed_seeds=slice_,
                bundle=bundle,
                pdq_cfg=pdq_cfg,
            ),
            thread_name_prefix="bp-worker",
        )

        logger.info("Boundary-pair pipeline complete.")

    # ------------------------------------------------------------------
    # Per-worker bundle construction + seed loop
    # ------------------------------------------------------------------

    def _build_worker_bundle(
        self,
        *,
        worker_id: int,
        evo_cfg: ExperimentConfig,
        pdq_cfg: PDQExperimentConfig,
        shared_sut: VLMSUT,
        image_manip: ImageBackend,
        text_manip: CompositeTextManipulator,
        crit_types: list,
    ) -> "_WorkerBundle":
        """Create a per-worker thin bundle over the shared models.

        Each worker owns its own VLMSUT wrapper (counters,
        last_call_cached), VLMManipulator (per-seed contexts), optimiser
        (per-seed reset state), objective collection (per-call results
        buffer), tester, SUTAdapter, and RNG.  The shared models live on
        *shared_sut.scorer*, *image_manip*, *text_manip* — no per-worker
        copy of those.
        """
        sut_thread = VLMSUT(
            evo_cfg,
            scorer=shared_sut.scorer,
            text_embedder=shared_sut.text_embedder,
            redis_client=shared_sut.redis_client,
        )
        objectives = CriterionCollection(*[c() for c in crit_types])
        optimizer = DiscretePymooOptimizer(
            gene_bounds=np.zeros(1, dtype=np.int64),
            num_objectives=len(crit_types),
            pop_size=evo_cfg.pop_size,
        )
        manipulator = VLMManipulator(image_manip, text_manip)
        tester = VLMBoundaryTester(
            sut=sut_thread,
            manipulator=manipulator,
            optimizer=optimizer,
            objectives=objectives,
            config=evo_cfg,
        )
        adapter = SUTAdapter(sut_thread)
        rng = np.random.default_rng(pdq_cfg.reproducibility.seed_int + worker_id)
        return _WorkerBundle(
            worker_id=worker_id,
            sut=sut_thread,
            manipulator=manipulator,
            optimizer=optimizer,
            tester=tester,
            adapter=adapter,
            rng=rng,
        )

    def _run_seed_slice(
        self,
        *,
        indexed_seeds: list[tuple[int, SeedTriple]],
        bundle: "_WorkerBundle",
        pdq_cfg: PDQExperimentConfig,
    ) -> None:
        """Run the per-seed boundary-pair flow for one worker's slice."""
        for seed_idx, seed in indexed_seeds:
            self._run_seed(
                seed_idx=seed_idx,
                seed=seed,
                tester=bundle.tester,
                manipulator=bundle.manipulator,
                adapter=bundle.adapter,
                rng=bundle.rng,
                pdq_cfg=pdq_cfg,
            )
            bundle.optimizer.reset()
            bundle.tester._cleanup()  # noqa: SLF001 — mirror tester.test() reset chain
        logger.info(
            "Worker %d done: %d seed(s), %d SUT calls (%d cache misses)",
            bundle.worker_id, len(indexed_seeds),
            bundle.adapter.call_count, bundle.adapter.miss_count,
        )

    def _run_seed(
        self,
        *,
        seed_idx: int,
        seed: SeedTriple,
        tester: VLMBoundaryTester,
        manipulator: VLMManipulator,
        adapter: SUTAdapter,
        rng: np.random.Generator,
        pdq_cfg,
    ) -> None:
        cfg = self._cfg
        ts = int(time())
        seed_dir = cfg.save_dir / cfg.name / f"seed_{seed_idx:04d}_{ts}"
        evo_dir = seed_dir / "evolutionary"
        pdq_dir = seed_dir / "pdq"
        seed_dir.mkdir(parents=True, exist_ok=True)
        evo_dir.mkdir(exist_ok=True)
        pdq_dir.mkdir(exist_ok=True)

        # -- Top-level run config ---------------------------------------
        with open(seed_dir / "config.json", "w") as f:
            cfg_dict = boundary_pair_config_to_dict(cfg)
            cfg_dict["schema_version"] = PDQ_SCHEMA_VERSION
            cfg_dict["pipeline"] = "boundary_pair"
            json.dump(cfg_dict, f, indent=2)

        # -- Stage 1: evolutionary --------------------------------------
        logger.info(
            "[seed %d] Stage 1: evolutionary (%d gen × %d pop)",
            seed_idx, cfg.evolutionary.generations, cfg.evolutionary.pop_size,
        )
        pareto = tester.run_one_seed(
            seed_idx=seed_idx,
            seed=seed,
            run_dir=evo_dir,
            reset_optimizer=False,
        )
        if not pareto:
            logger.warning(
                "[seed %d] No Pareto candidates — skipping PDQ stage.",
                seed_idx,
            )
            return

        # -- Anchor selection -------------------------------------------
        selected = _select_anchors(pareto, cfg.anchor_selection)
        logger.info(
            "[seed %d] Anchor selection: %d/%d Pareto member(s) → PDQ anchors",
            seed_idx, len(selected), len(pareto),
        )

        # -- Stage 2: PDQ on each anchor --------------------------------
        # tester.run_one_seed leaves manipulator already prepared for
        # this seed; we reuse that preparation across all anchors.
        answer_suffix = cfg.answer_format.format(
            categories=", ".join(cfg.categories),
        )

        # Pair-softmax labels for PDQ must match the SUT's concrete
        # category vocabulary. In roster mode with taxonomy abstraction
        # (level > 0 or even level 0, which already maps to a
        # taxonomy-canonical L0 string), ``seed.class_a`` / ``class_b``
        # can be abstracted strings ("shark", "salamander") that are not
        # in ``cfg.categories``.  Fall back to the concrete metadata
        # when present.
        if seed.metadata is not None:
            pair_class_a = seed.metadata.get(
                "anchor_class_concrete", seed.class_a,
            )
            pair_class_b = seed.metadata.get(
                "target_class_concrete", seed.class_b,
            )
        else:
            pair_class_a = seed.class_a
            pair_class_b = seed.class_b

        anchors_dir = pdq_dir / "anchors"
        anchors_dir.mkdir(exist_ok=True)

        anchor_summaries: list[dict[str, Any]] = []

        with SeedLogger(
            pdq_dir,
            compression=cfg.pdq.logging.parquet_compression,
            flush_interval=cfg.pdq.logging.flush_interval_calls,
        ) as sl:
            pdq_cfg_dict = pdq_config_to_dict(pdq_cfg)
            pdq_cfg_dict["schema_version"] = PDQ_SCHEMA_VERSION
            pdq_cfg_dict["pipeline"] = "boundary_pair"
            pdq_cfg_dict["boundary_pair_name"] = cfg.name
            sl.write_config_json(pdq_cfg_dict)
            sl.write_context_json(build_context_meta(manipulator))

            pbar = tqdm(
                total=len(selected),
                desc=f"[seed {seed_idx}] PDQ anchors",
                unit="anchor",
                position=0,
            )

            flip_id_cursor = 0
            for pareto_idx, cand in selected:
                anchor_geno = np.asarray(cand.solution, dtype=np.int64)
                fitness_vec = np.asarray(cand.fitness)
                evolutionary_gen = _maybe_get_gen(cand)

                # 1. Render anchor + call SUT for its label.
                imgs, texts = manipulator.manipulate(
                    candidates=None,
                    weights=anchor_geno.reshape(1, -1),
                )
                anchor_image: Image.Image = imgs[0]
                anchor_text: str = texts[0]
                anchor_prompt = anchor_text + answer_suffix

                anchor_logprobs_t, anchor_call_id = adapter.call(
                    anchor_image,
                    text=anchor_prompt,
                    categories=cfg.categories,
                    stage="anchor",
                    candidate_id=-1,
                )
                anchor_logprobs = anchor_logprobs_t.tolist()
                anchor_label, p_a, p_b = _pair_softmax_argmax(
                    anchor_logprobs, cfg.categories, pair_class_a, pair_class_b,
                )

                # 2. Save anchor image + summary.
                anchor_image.save(anchors_dir / f"anchor_{pareto_idx:03d}.png")
                with open(
                    anchors_dir / f"anchor_{pareto_idx:03d}.json", "w",
                ) as f:
                    json.dump(
                        {
                            "pareto_idx": pareto_idx,
                            "genotype": anchor_geno.tolist(),
                            "fitness": fitness_vec.tolist(),
                            "anchor_label": anchor_label,
                            "p_class_a": p_a,
                            "p_class_b": p_b,
                            "evolutionary_gen": evolutionary_gen,
                            "anchor_call_id": anchor_call_id,
                        },
                        f,
                        indent=2,
                    )

                sl.append_sut_calls(adapter.pop_records())
                sl.flush_all()

                # 3. Run PDQ core for this anchor.
                pbar.set_postfix({
                    "anchor": anchor_label,
                    "par": pareto_idx,
                    "p_a": f"{p_a:.2f}",
                })
                counts = run_pdq_core(
                    cfg=pdq_cfg,
                    seed_idx=seed_idx,
                    seed_id=f"seed_{seed_idx:04d}",
                    run_id=cfg.name,
                    manipulator=manipulator,
                    adapter=adapter,
                    rng=rng,
                    sl=sl,
                    pbar=pbar,
                    answer_suffix=answer_suffix,
                    anchor_genotype=anchor_geno,
                    anchor_label=anchor_label,
                    anchor_image=anchor_image,
                    anchor_logprobs=anchor_logprobs,
                    flip_id_start=flip_id_cursor,
                    pareto_idx=pareto_idx,
                    evolutionary_gen=evolutionary_gen,
                    anchor_source="evolutionary",
                )
                flip_id_cursor = counts["next_flip_id"]
                anchor_summaries.append({
                    "pareto_idx": pareto_idx,
                    "anchor_label": anchor_label,
                    "p_a": p_a,
                    "p_b": p_b,
                    **{
                        k: counts[k]
                        for k in (
                            "n_stage1_candidates",
                            "n_stage1_flips",
                            "n_distinct_targets",
                            "n_stage2_flips",
                            "n_stage2_sut_calls",
                        )
                    },
                })
                pbar.update(1)

            pbar.close()

        # -- Top-level manifest -----------------------------------------
        manifest = {
            "schema_version": PDQ_SCHEMA_VERSION,
            "pipeline": "boundary_pair",
            "seed_idx": seed_idx,
            "class_a": seed.class_a,
            "class_b": seed.class_b,
            "n_pareto": len(pareto),
            "n_anchors_evaluated": len(selected),
            "anchor_selection": {
                "source": cfg.anchor_selection.source,
                "k": cfg.anchor_selection.k,
                "label_assignment": cfg.anchor_selection.label_assignment,
            },
            "anchors": anchor_summaries,
            "evolutionary_dir": str(evo_dir.relative_to(seed_dir)),
            "pdq_dir": str(pdq_dir.relative_to(seed_dir)),
        }
        with open(seed_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(
            "[seed %d] done: %d anchor(s), total %d Stage-1 flips, "
            "%d Stage-2 minimised pairs",
            seed_idx,
            len(selected),
            sum(a["n_stage1_flips"] for a in anchor_summaries),
            sum(a["n_stage2_flips"] for a in anchor_summaries),
        )

    @staticmethod
    def _with_resolved_categories(
        cfg: BoundaryPairExperimentConfig, categories: tuple[str, ...],
    ) -> BoundaryPairExperimentConfig:
        if cfg.categories == categories:
            return cfg
        import dataclasses
        return dataclasses.replace(cfg, categories=categories)


def _maybe_get_gen(cand: Any) -> int | None:
    """Return the generation the Pareto member appeared at, or None."""
    for attr in ("gen", "generation", "iteration"):
        if hasattr(cand, attr):
            value = getattr(cand, attr)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return None
    return None


__all__ = ["BoundaryPairRunner"]
