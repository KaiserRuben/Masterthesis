"""PDQ runner — Phase 3.

Orchestrates the full PDQ pipeline for all seeds.  Phase 3 adds Stage 2
minimisation: for each Stage-1 flip, the smallest perturbation that still
crosses the class boundary is found via three greedy passes (zeroing,
rank-reduction, random-subset), and the archive row is written with the
minimised genotype and ``validity="VV"``.

Architecture mirrors ``experiments/runners/run_boundary_test.py``:
- Parallel component init (SUT, image manipulator, text manipulator)
- Shared ``generate_seeds()`` from ``src.evolutionary``
- Per-seed: prepare → anchor → Stage 1 search → Stage 2 minimisation
  → write artifacts
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from time import time
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.config import ExperimentConfig, SeedTriple
from src.data import ImageNetCache
from src.manipulator.image.manipulator import ImageManipulator
from src.manipulator.text.manipulator import TextManipulator
from src.common import apply_seed_filter, build_context_meta
from src.manipulator.vlm_manipulator import VLMManipulator
from src.sut import VLMSUT
from src.common import generate_seeds

from .archive import append_archive_row_stage2
from .artifacts import PDQ_SCHEMA_VERSION, SeedLogger
from .config import (
    PDQExperimentConfig,
    config_to_dict,
    resolve_categories as pdq_resolve_categories,
)
from .metric import INPUT_DISTANCES, OUTPUT_DISTANCES
from .search.base import ScoredCandidate
from .search.stage1 import run_stage1
from .search.stage2 import CheckResult, minimise_flip
from .sut_adapter import SUTAdapter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage-1 flip record (held in memory until Stage-2 writes archive row)
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class _FlipRecord:
    """All data needed to build an archive row after Stage-2 completes.

    Stage-1 archive writes are deferred so Stage-2 can write the final
    ``VV`` row directly, avoiding a read-modify-write on the parquet file.

    :param sc: The Stage-1 :class:`~src.pdq.search.base.ScoredCandidate`.
    :param flip_id: 0-based flip index within this seed.
    :param anchor_geno_list: Anchor genotype as plain list.
    :param anchor_logprobs: Anchor SUT logprobs.
    :param anchor_label: VLM label on the anchor.
    :param stage1_sut_calls: Total SUT calls at end of Stage 1.
    """

    sc: ScoredCandidate
    flip_id: int
    anchor_geno_list: list[int]
    anchor_logprobs: list[float]
    anchor_label: str
    stage1_sut_calls: int


# ---------------------------------------------------------------------------
# GPU allocator-pool maintenance
# ---------------------------------------------------------------------------


def _release_gpu_cache(device: str) -> None:
    """Return cached but unused GPU memory to the OS.

    PyTorch's CUDA and MPS allocators keep freed tensors in an internal
    pool for reuse, which is great for throughput but problematic for
    long-running multi-stage runs: per-call KV-cache tensors (hundreds
    of MB each on a 4B model) accumulate in the pool over thousands of
    SUT calls and can push process RSS into the tens of GB even though
    Python no longer references the underlying tensors.

    This helper calls the device-specific ``empty_cache`` to flush the
    pool. It is cheap (a no-op when there is nothing to release) and
    safe to call between stages or per-flip without affecting numerical
    behaviour. CPU runs are no-ops.

    :param device: Torch device string from the experiment config
        (``"cpu"``, ``"cuda"``, ``"mps"``).
    """
    if device.startswith("cuda"):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elif device == "mps":
        if torch.backends.mps.is_available():
            # mps.empty_cache exists on PyTorch >= 2.0; guard for older.
            empty = getattr(torch.mps, "empty_cache", None)
            if callable(empty):
                empty()


# ---------------------------------------------------------------------------
# Config shim: PDQExperimentConfig → ExperimentConfig
# ---------------------------------------------------------------------------


def _make_exp_config(cfg: PDQExperimentConfig) -> ExperimentConfig:
    """Build a minimal ExperimentConfig for shared utilities (VLMSUT, seeds).

    Only the fields that VLMSUT and ``generate_seeds`` actually read are
    forwarded — no duplication of PDQ-specific knobs.

    :param cfg: PDQ config (resolved categories required).
    :returns: Matching ExperimentConfig.
    """
    return ExperimentConfig(
        device=cfg.device,
        categories=cfg.categories,
        prompt_template=cfg.prompt_template,
        answer_format=cfg.answer_format,
        cache_dirs=cfg.cache_dirs,
        sut=cfg.sut,
        image=cfg.image,
        text=cfg.text,
        seeds=cfg.seeds,
    )


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------


def _rng_state_dict(rng: np.random.Generator) -> dict[str, Any]:
    """Serialise numpy RNG state to a JSON-compatible dict."""
    raw = rng.bit_generator.state
    # Convert numpy arrays to lists so json.dump works.
    def _fix(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _fix(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_fix(x) for x in obj]
        return obj
    return _fix(raw)


def _git_hash() -> str | None:
    """Return HEAD commit hash, or None if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return None


def _env_dict() -> dict[str, str]:
    """Capture minimal Python / OS environment for provenance."""
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": _safe_version("torch"),
        "transformers": _safe_version("transformers"),
        "numpy": _safe_version("numpy"),
    }


def _safe_version(pkg: str) -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version(pkg)
    except Exception:  # noqa: BLE001
        return "unknown"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class PDQRunner:
    """PDQ pipeline runner.

    Handles component initialisation, seed generation, and the per-seed
    two-stage search loop.

    :param config: PDQ experiment config (categories need not be resolved
        yet — :meth:`run` resolves them against the data source).
    """

    def __init__(self, config: PDQExperimentConfig) -> None:
        self._cfg = config

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, preflight: bool = False) -> None:
        """Initialise all components and run all seeds.

        :param preflight: If True, run a representative-cost measurement
            on the first filtered seed before the main loop starts.
            Prints per-call wall time and a projected total runtime
            based on the configured budget (stage1 + max_flips ×
            stage2). Does NOT abort; the user should Ctrl-C if the
            projection is unacceptable. The preflight reuses the
            already-loaded SUT and manipulator, so it adds only the
            measurement calls' wall time (~20 × per-call seconds) to
            the run.
        """
        cfg = self._cfg

        # -- Data source (always needed first for category resolution) ----
        data_source = ImageNetCache(dirs=list(cfg.cache_dirs))
        cfg = pdq_resolve_categories(cfg, data_source.labels())
        self._cfg = cfg

        exp_cfg = _make_exp_config(cfg)

        # -- Parallel component init (same pattern as SMOO runner) --------
        pool = ThreadPoolExecutor(max_workers=3)

        logger.info("Text manipulator starting...")
        text_fut: Future[TextManipulator] = pool.submit(
            TextManipulator.from_pretrained, config=cfg.text,
        )

        logger.info("Image manipulator starting...  preset=%s", cfg.image.preset)
        image_fut: Future[ImageManipulator] = pool.submit(
            ImageManipulator.from_preset, device=cfg.device, config=cfg.image,
        )

        logger.info("SUT starting...  %s on %s", cfg.sut.model_id, cfg.device)
        sut_fut: Future[VLMSUT] = pool.submit(VLMSUT, exp_cfg)

        # SUT needed first for seed generation.
        sut: VLMSUT = sut_fut.result()
        logger.info("SUT loaded")

        logger.info("Generating seeds...")
        seeds = generate_seeds(sut, exp_cfg, data_source)

        image_manip: ImageManipulator = image_fut.result()
        logger.info("Image manipulator loaded")

        text_manip: TextManipulator = text_fut.result()
        logger.info("Text manipulator loaded")

        pool.shutdown(wait=False)

        if not seeds:
            logger.warning("No seeds passed filters — nothing to test.")
            return

        # Apply post-generation seed filter (preserves original indices so
        # `seed_0032` in a filtered run matches `seed_0032` in the full run).
        indexed_seeds = apply_seed_filter(seeds, cfg.seeds.filter_indices)

        logger.info(
            "%d seed(s) to test (PDQ pipeline, %d strategies, "
            "stage1_budget=%d)",
            len(indexed_seeds),
            len(cfg.stage1.strategies),
            cfg.stage1.budget_sut_calls,
        )

        manipulator = VLMManipulator(image_manip, text_manip)
        adapter = SUTAdapter(sut)

        # -- Deterministic RNG -------------------------------------------
        rng = np.random.default_rng(cfg.reproducibility.seed_int)

        # -- Preflight cost check (optional) -----------------------------
        if preflight:
            from src.sut import preflight_cost_check

            first_seed = indexed_seeds[0][1]
            answer_suffix = cfg.answer_format.format(
                categories=", ".join(cfg.categories),
            )
            # Total PDQ SUT calls per seed: stage1 budget (at most
            # budget_sut_calls, reached if no early stop) + up to
            # max_flips_per_seed × budget_sut_calls_per_flip for stage2.
            # Multiplied across all filtered seeds.
            per_seed_calls = (
                cfg.stage1.budget_sut_calls
                + cfg.stage1.max_flips_per_seed
                * cfg.stage2.budget_sut_calls_per_flip
            )
            total_calls = per_seed_calls * len(indexed_seeds)
            preflight_cost_check(
                sut=sut,
                manipulator=manipulator,
                seed=first_seed,
                prompt_template=cfg.prompt_template,
                answer_suffix=answer_suffix,
                categories=cfg.categories,
                total_calls_projected=total_calls,
                n_samples=20,
            )

        pbar = tqdm(indexed_seeds, unit="seed", desc="PDQ")
        for pos, (seed_idx, seed) in enumerate(pbar):
            pbar.set_description(
                f"[{pos + 1}/{len(indexed_seeds)}] "
                f"seed_{seed_idx:04d} {seed.class_a}"
            )
            self._run_seed(
                seed_idx=seed_idx,
                seed=seed,
                manipulator=manipulator,
                adapter=adapter,
                rng=rng,
                pbar=pbar,
            )

        pbar.close()
        logger.info(
            "PDQ run complete. Total SUT calls: %d (%d cache misses)",
            adapter.call_count,
            adapter.miss_count,
        )

    # ------------------------------------------------------------------
    # Per-seed orchestration
    # ------------------------------------------------------------------

    def _run_seed(
        self,
        seed_idx: int,
        seed: SeedTriple,
        manipulator: VLMManipulator,
        adapter: SUTAdapter,
        rng: np.random.Generator,
        pbar: tqdm,
    ) -> None:
        """Run one seed: anchor + Phase-1 skeleton artifacts.

        :param seed_idx: 0-based seed index.
        :param seed: Seed triple (image, class_a, class_b).
        :param manipulator: Prepared VLMManipulator (will be re-prepared here).
        :param adapter: SUT adapter (shared across seeds, state accumulates).
        :param rng: Seeded random number generator.
        :param pbar: Parent tqdm bar.
        """
        cfg = self._cfg
        t_seed_start = time()

        # Seed dir: runs/{name}/seed_{idx}_{ts}/
        ts = int(t_seed_start)
        seed_dir = cfg.save_dir / cfg.name / f"seed_{seed_idx:04d}_{ts}"

        with SeedLogger(
            seed_dir,
            compression=cfg.logging.parquet_compression,
            flush_interval=cfg.logging.flush_interval_calls,
        ) as sl:
            self._run_seed_inner(
                seed_idx=seed_idx,
                seed=seed,
                manipulator=manipulator,
                adapter=adapter,
                rng=rng,
                sl=sl,
                t_start=t_seed_start,
                pbar=pbar,
            )

        # Clear postfix for next seed
        pbar.set_postfix({})

    def _run_seed_inner(
        self,
        seed_idx: int,
        seed: SeedTriple,
        manipulator: VLMManipulator,
        adapter: SUTAdapter,
        rng: np.random.Generator,
        sl: SeedLogger,
        t_start: float,
        pbar: tqdm,
    ) -> None:
        cfg = self._cfg
        categories = cfg.categories
        answer_suffix = cfg.answer_format.format(
            categories=", ".join(categories),
        )

        pbar.set_postfix({"stage": "prep"})

        # 1. Prepare manipulator for this seed.
        manipulator.prepare(seed.image, cfg.prompt_template)

        # 2. Compute anchor (zero genotype → VQGAN-reconstructed image +
        #    original prompt text).
        zero_geno = manipulator.zero_genotype().reshape(1, -1)
        anchor_imgs, anchor_texts = manipulator.manipulate(
            candidates=None, weights=zero_geno,
        )
        anchor_image: Image.Image = anchor_imgs[0]
        anchor_text: str = anchor_texts[0]
        full_anchor_prompt = anchor_text + answer_suffix

        # 3. Call SUT on anchor.
        # candidate_id=-1 sentinel: anchor is not a candidate entry; -1 keeps
        # the column typed as int64 across all sut_calls.parquet row groups.
        anchor_logprobs, anchor_call_id = adapter.call(
            anchor_image,
            text=full_anchor_prompt,
            categories=categories,
            stage="anchor",
            candidate_id=-1,
        )
        label_anchor = categories[int(anchor_logprobs.argmax().item())]

        pbar.set_postfix({"stage": "S1", "anchor": label_anchor})

        # 4. Flush anchor SUT call record.
        sl.append_sut_calls(adapter.pop_records())
        sl.flush_all()

        # 5. Write image / text artifacts.
        if cfg.logging.save_anchor_images:
            sl.save_anchor_original(seed.image)
            sl.save_anchor_baseline(anchor_image)
        sl.save_anchor_prompt(full_anchor_prompt)

        # 6. Write JSON metadata.
        # Stamp the schema version on config.json for redundancy with
        # the per-parquet file-level marker — a reader that only sees
        # config.json can still determine the layout.
        cfg_dict = config_to_dict(cfg)
        cfg_dict["schema_version"] = PDQ_SCHEMA_VERSION
        sl.write_config_json(cfg_dict)
        sl.write_context_json(build_context_meta(manipulator))

        if cfg.reproducibility.dump_rng_state:
            sl.write_rng_state_json(_rng_state_dict(rng))

        # 7. Stage 1 — flip discovery.
        anchor_logprobs_list = anchor_logprobs.tolist()
        anchor_geno = manipulator.zero_genotype()
        anchor_image_arr = np.array(anchor_image)

        # Resolve distance functions from config.
        input_dist_fn = INPUT_DISTANCES.get(cfg.distances.d_i_primary)
        if input_dist_fn is None:
            raise ValueError(
                f"d_i_primary={cfg.distances.d_i_primary!r} is not in "
                "INPUT_DISTANCES registry"
            )
        output_dist_fn = OUTPUT_DISTANCES[cfg.distances.d_o_primary]

        # SUT call closure for Stage 1 — captures manipulator, adapter, prompt.
        def _sut_call(genotype: np.ndarray) -> tuple[
            list[float], int, Image.Image, str, float
        ]:
            candidate_id = adapter.call_count  # equals upcoming call_id
            imgs, texts = manipulator.manipulate(
                candidates=None, weights=genotype.reshape(1, -1),
            )
            img, text = imgs[0], texts[0]
            logprobs_t, call_id = adapter.call(
                img,
                text=text + answer_suffix,
                categories=categories,
                stage="stage1",
                candidate_id=candidate_id,
            )
            return logprobs_t.tolist(), call_id, img, text, adapter.wall_time_cumulative

        scored = run_stage1(
            anchor_geno=anchor_geno,
            anchor_label=label_anchor,
            anchor_image_arr=anchor_image_arr,
            gene_bounds=manipulator.gene_bounds,
            image_dim=manipulator.image_dim,
            text_candidate_distances=manipulator.text_candidate_distances,
            seed_idx=seed_idx,
            strategies=cfg.stage1.strategies,
            budget=cfg.stage1.budget_sut_calls,
            rng=rng,
            sut_call_fn=_sut_call,
            input_distance_fn=input_dist_fn,
            output_distance_fn=output_dist_fn,
            categories=categories,
            early_stop_cfg=cfg.stage1.early_stop,
            max_flips=cfg.stage1.max_flips_per_seed,
            max_distinct_targets=cfg.stage1.max_distinct_targets,
        )

        # -- Write Stage-1 results to parquet --------------------------------
        # Archive rows are NOT written here; they are deferred to after Stage 2
        # so the archive always contains the minimised (VV) genotype.
        seed_id = f"seed_{seed_idx:04d}"
        run_id = cfg.name
        flip_id = 0
        seen_targets: set[str] = set()
        flip_records: list[_FlipRecord] = []
        stage1_total_calls_at_end = adapter.call_count  # updated below

        for sc in scored:
            # candidates.parquet — every evaluated candidate
            sl.append_candidate(_to_candidate_row(sc))

            if sc.flipped:
                is_first = sc.label not in seen_targets
                seen_targets.add(sc.label)

                # stage1_flips.parquet
                sl.append_stage1_flip(
                    _to_stage1_flip_row(sc, flip_id, label_anchor, is_first)
                )

                # Defer archive write — collect for Stage 2.
                flip_records.append(_FlipRecord(
                    sc=sc,
                    flip_id=flip_id,
                    anchor_geno_list=anchor_geno.tolist(),
                    anchor_logprobs=anchor_logprobs_list,
                    anchor_label=label_anchor,
                    stage1_sut_calls=adapter.call_count,
                ))

                # Save flip image if configured.
                if cfg.logging.save_flip_images:
                    sl.save_flip_image(
                        flip_id=flip_id,
                        stage="stage1",
                        image=sc.rendered_image,
                        meta={
                            "candidate_id": sc.candidate_id,
                            "label": sc.label,
                            "d_i": sc.d_i,
                            "d_o": sc.d_o,
                            "pdq": sc.pdq_score,
                            "strategy": sc.candidate.strategy,
                        },
                    )

                flip_id += 1

        stage1_total_calls_at_end = adapter.call_count
        n_flips = flip_id

        # Drain Stage-1 SUT call records and release accumulated GPU
        # allocator-pool memory before Stage 2 begins. After 150 Stage-1
        # calls on a 4B model, the MPS / CUDA allocator pool can hold
        # several GB of cached KV-cache tensors that are no longer
        # referenced — force them back to the OS so Stage 2 starts with
        # a clean memory footprint.
        sl.append_sut_calls(adapter.pop_records())
        sl.flush_all()
        _release_gpu_cache(cfg.device)

        pbar.set_postfix({
            "stage": "S2",
            "anchor": label_anchor,
            "S1_flips": n_flips,
            "S1_targets": len(seen_targets),
        })

        # -- Stage 2 — minimisation ------------------------------------------
        # SUT check closure for Stage 2. Each call also bumps a counter
        # that triggers periodic GPU allocator-pool flushes mid-flip:
        # without them the MPS / CUDA allocator pool grows by roughly the
        # KV-cache size (~15-20 MB on Qwen3.5-4B at N=50 categories) per
        # call, accumulating tens of GB inside a single 2000-call flip
        # before the per-flip _release_gpu_cache below has a chance to
        # reclaim it. With release_every=200, mid-flip RSS growth is
        # bounded at ~3-4 GB above the post-release baseline.
        _stage2_call_counter = [0]
        _STAGE2_RELEASE_EVERY = 200

        def _stage2_check(geno: np.ndarray) -> CheckResult:
            imgs, texts = manipulator.manipulate(
                candidates=None, weights=geno.reshape(1, -1),
            )
            img, text = imgs[0], texts[0]
            logprobs_t, call_id = adapter.call(
                img,
                text=text + answer_suffix,
                categories=categories,
                stage="stage2",
                candidate_id=-1,
            )
            lbl = categories[int(logprobs_t.argmax().item())]

            _stage2_call_counter[0] += 1
            if _stage2_call_counter[0] >= _STAGE2_RELEASE_EVERY:
                _release_gpu_cache(cfg.device)
                _stage2_call_counter[0] = 0

            return CheckResult(lbl != label_anchor, lbl, call_id, adapter.wall_time_cumulative)

        n_stage2_calls = 0
        n_vv = 0

        for fr in flip_records:
            result = minimise_flip(
                flipped_geno=fr.sc.candidate.genotype,
                sut_check_fn=_stage2_check,
                anchor_label=label_anchor,
                budget_total=cfg.stage2.budget_sut_calls_per_flip,
                rng=rng,
                cfg=cfg.stage2,
                input_distance_fn=input_dist_fn,
                stage1_flip_label=fr.sc.label,
                seed_idx=seed_idx,
                flip_id=fr.flip_id,
            )
            n_stage2_calls += result.sut_calls_used
            n_vv += 1

            # stage2_trajectories.parquet — one row per SUT call in Stage 2.
            label_before = fr.sc.label
            rs_before = fr.sc.total_rank_sum
            sp_before = fr.sc.total_sparsity
            for pass_traj in result.trajectory_per_pass.values():
                for step in pass_traj:
                    sl.append_stage2_step(_to_stage2_traj_row(
                        flip_id=fr.flip_id,
                        step=step,
                        label_before=label_before,
                        rank_sum_before_first=rs_before,
                        sparsity_before_first=sp_before,
                    ))
                    if step.get("accepted"):
                        label_before = step.get("label_after", label_before)

            # archive.parquet — VV row with minimised genotype.
            append_archive_row_stage2(
                buffer=sl._bufs["archive.parquet"],  # noqa: SLF001
                sc=fr.sc,
                flip_id=fr.flip_id,
                seed_id=seed_id,
                run_id=run_id,
                anchor_geno_list=fr.anchor_geno_list,
                anchor_logprobs=fr.anchor_logprobs,
                anchor_label=fr.anchor_label,
                stage1_sut_calls=fr.stage1_sut_calls,
                stage2_result=result,
            )

            # Per-flip housekeeping. Drain SUT call records to disk so
            # the adapter's in-memory buffer stays bounded at one flip's
            # worth (~2000 records × ~3 KB ≈ 7 MB) instead of growing to
            # all flips' worth (~70 MB). Then release the GPU allocator
            # pool — each flip generates ``budget_sut_calls_per_flip``
            # new KV-cache tensors that PyTorch's MPS/CUDA allocator
            # would otherwise retain across the whole seed.
            sl.append_sut_calls(adapter.pop_records())
            sl.flush_all()
            _release_gpu_cache(cfg.device)

        # Final drain (no-op if all flips already drained, but keeps the
        # invariant that the seed ends with an empty adapter buffer).
        sl.append_sut_calls(adapter.pop_records())
        sl.flush_all()
        _release_gpu_cache(cfg.device)

        pbar.set_postfix({
            "anchor": label_anchor,
            "S1_flips": n_flips,
            "S2_flips": n_vv,
            "time": f"{time() - t_start:.1f}s",
        })

        # -- Stats JSON (final with Stage-2 summary) -------------------------
        stats = _build_stats(
            seed_idx=seed_idx,
            seed=seed,
            cfg=cfg,
            manipulator=manipulator,
            label_anchor=label_anchor,
            anchor_call_id=anchor_call_id,
            anchor_logprobs=anchor_logprobs_list,
            n_stage1_candidates=len(scored),
            n_stage1_flips=n_flips,
            n_distinct_targets=len(seen_targets),
            n_stage2_flips=n_vv,
            n_stage2_sut_calls=n_stage2_calls,
            wall_time_s=time() - t_start,
            git_hash=_git_hash() if cfg.reproducibility.dump_git_hash else None,
            env=_env_dict() if cfg.reproducibility.dump_env else None,
            cache_stats=adapter.cache_stats,
        )
        sl.write_stats_json(stats)


# ---------------------------------------------------------------------------
# Parquet row builders (pure functions — all fields explicit)
# ---------------------------------------------------------------------------


def _to_candidate_row(sc: Any) -> dict[str, Any]:
    """Convert a ScoredCandidate to a ``candidates.parquet`` row dict."""
    return {
        "candidate_id": sc.candidate_id,
        "stage": "stage1",
        "parent_candidate_id": None,
        "operation": sc.candidate.strategy,
        "target_gene": None,
        "old_value": None,
        "new_value": None,
        "genotype": sc.candidate.genotype.tolist(),
        "img_sparsity": sc.img_sparsity,
        "txt_sparsity": sc.txt_sparsity,
        "total_sparsity": sc.total_sparsity,
        "img_rank_sum": sc.img_rank_sum,
        "txt_rank_sum": sc.txt_rank_sum,
        "total_rank_sum": sc.total_rank_sum,
        "hamming_to_anchor": sc.hamming_to_anchor,
        "rendered_text": sc.rendered_text,
        "image_pixel_L2": sc.image_pixel_L2,
        "text_cosine_sum": sc.text_cosine_sum,
        "image_path": None,
        "sut_call_id": sc.sut_call_id,
        "label": sc.label,
        "flipped_vs_anchor": sc.flipped,
        "accepted": True,
    }


def _to_stage1_flip_row(
    sc: Any,
    flip_id: int,
    anchor_label: str,
    is_first_for_target: bool,
) -> dict[str, Any]:
    """Convert a flipped ScoredCandidate to a ``stage1_flips.parquet`` row."""
    return {
        "flip_id": flip_id,
        "candidate_id": sc.candidate_id,
        "discovery_sut_call": sc.sut_call_id,
        "discovery_wall_time_s": sc.discovery_wall_time_cum,
        "operation": sc.candidate.strategy,
        "L_anchor": anchor_label,
        "L_target": sc.label,
        "genotype_flipped": sc.candidate.genotype.tolist(),
        "total_sparsity_at_discovery": sc.total_sparsity,
        "total_rank_sum_at_discovery": sc.total_rank_sum,
        "is_first_for_target": is_first_for_target,
        "stage2_refined_candidate_id": None,
    }


def _to_stage2_traj_row(
    flip_id: int,
    step: dict[str, Any],
    label_before: str,
    rank_sum_before_first: int,
    sparsity_before_first: int,
) -> dict[str, Any]:
    """Convert a Stage-2 trajectory step dict to a ``stage2_trajectories.parquet`` row.

    The step dict is produced by :func:`~src.pdq.search.stage2.greedy_zeroing`,
    :func:`~src.pdq.search.stage2.rank_reduction`, or
    :func:`~src.pdq.search.stage2.random_subset` and already contains
    ``rank_sum_before/after``, ``sparsity_before/after``, and SUT metadata.
    This function adds ``flip_id``, ``candidate_id_before``, and
    ``label_before``.

    :param flip_id: Flip index within the seed.
    :param step: Per-step dict from a Stage-2 pass trajectory.
    :param label_before: VLM label before this step (updated by caller
        after each accepted step).
    :param rank_sum_before_first: rank_sum of the Stage-1 flip genotype
        (used only if the step dict lacks ``rank_sum_before``).
    :param sparsity_before_first: sparsity of the Stage-1 flip genotype.
    :returns: Row dict matching ``STAGE2_TRAJECTORIES_COLUMNS``.
    """
    return {
        "flip_id": flip_id,
        "step": step.get("step"),
        "pass_name": step.get("pass_name"),
        "target_gene": step.get("target_gene"),
        "old_value": step.get("old_value"),
        "new_value": step.get("new_value"),
        "candidate_id_before": None,
        "candidate_id_after": step.get("sut_call_id"),
        "label_before": label_before,
        "label_after": step.get("label_after", ""),
        "still_flipped": step.get("still_flipped"),
        "accepted": step.get("accepted"),
        "sparsity_before": step.get("sparsity_before", sparsity_before_first),
        "sparsity_after": step.get("sparsity_after"),
        "rank_sum_before": step.get("rank_sum_before", rank_sum_before_first),
        "rank_sum_after": step.get("rank_sum_after"),
        "sut_call_id": step.get("sut_call_id"),
        "wall_time_cumulative_s": step.get("wall_time_cumulative_s"),
    }


# ---------------------------------------------------------------------------
# Stats builder
# ---------------------------------------------------------------------------


def _build_stats(
    seed_idx: int,
    seed: SeedTriple,
    cfg: PDQExperimentConfig,
    manipulator: VLMManipulator,
    label_anchor: str,
    anchor_call_id: int,
    anchor_logprobs: list[float],
    n_stage1_candidates: int,
    n_stage1_flips: int,
    n_distinct_targets: int,
    n_stage2_flips: int,
    n_stage2_sut_calls: int,
    wall_time_s: float,
    git_hash: str | None,
    env: dict[str, str] | None,
    cache_stats: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build the per-seed stats dict for ``stats.json``.

    Includes the ``schema_version`` marker (see
    :data:`~src.pdq.artifacts.PDQ_SCHEMA_VERSION`) plus the full
    category list so that downstream readers can recover the
    index→class_name mapping for every N-dim ``logprobs`` column in
    ``sut_calls.parquet`` / ``archive.parquet`` from a single file.

    :param cache_stats: Optional ``{"hits", "misses"}`` aggregate
        (forwarded from the underlying ``VLMSUT.cache_stats``).
    """
    return {
        "schema_version": PDQ_SCHEMA_VERSION,
        "pipeline": "pdq",
        "seed_idx": seed_idx,
        "class_a": seed.class_a,
        "class_b": seed.class_b,
        "label_anchor": label_anchor,
        "anchor_call_id": anchor_call_id,
        "anchor_logprobs": anchor_logprobs,
        "categories": list(cfg.categories),
        "n_categories_total": len(cfg.categories),
        "n_img_genes": manipulator.image_dim,
        "n_txt_genes": manipulator.text_dim,
        "genotype_dim": manipulator.genotype_dim,
        "gene_bounds": manipulator.gene_bounds.tolist(),
        "n_stage1_candidates": n_stage1_candidates,
        "n_stage1_flips": n_stage1_flips,
        "n_distinct_targets": n_distinct_targets,
        "n_stage2_flips": n_stage2_flips,
        "n_stage2_sut_calls": n_stage2_sut_calls,
        "wall_time_s": wall_time_s,
        "phase": "stage2",
        "cache_hits": (cache_stats or {}).get("hits"),
        "cache_misses": (cache_stats or {}).get("misses"),
        "git_hash": git_hash,
        "env": env,
    }
