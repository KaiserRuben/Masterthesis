"""Shared component-init + seed-gen + backend-precompute helpers.

Three pipelines (``run_boundary_test.py``, ``run_pdq_test.py``,
``run_boundary_pair_test.py``) all need to:

1. Load text manipulator, image manipulator, and SUT in parallel
   (with the StyleGAN-XL deferred branch that waits on SUT).
2. Generate the seed pool (``gap_filter`` or ``roster`` + combinatorial
   abstraction expansion).
3. Pre-populate backend-specific caches (cone modal grids for
   VQGAN+cone, modal-w + accepted-origin for StyleGAN).

Centralising the three concerns here keeps the per-pipeline runners
focused on stage logic + worker dispatch.

All helpers take the canonical :class:`~src.config.ExperimentConfig`.
Pipelines that use their own config dataclass (PDQ, boundary-pair)
project to ``ExperimentConfig`` before calling.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Sequence

from .redis_cache import BytesRedisCache
from .roster_seed_generator import roster_seeds
from .seed_context import apply_seed_filter, collect_target_classes
from .seed_generator import generate_seeds
from .combinatorial_pair_generator import combinatorial_pairs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared component bundle
# ---------------------------------------------------------------------------


@dataclass
class SharedComponents:
    """All process-wide singletons needed by the per-seed loop.

    Threads share these objects by reference; per-worker wrappers
    (VLMSUT thread wrapper, VLMManipulator, …) hold thin per-seed
    state on top.
    """

    sut: "VLMSUT"  # noqa: F821 — forward, resolved at runtime
    image_manip: "ImageBackend"  # noqa: F821
    text_manip: "CompositeTextManipulator"  # noqa: F821
    data_source: "ImageNetCache"  # noqa: F821


# ---------------------------------------------------------------------------
# Component init
# ---------------------------------------------------------------------------


def init_shared_components(
    exp_cfg: "ExperimentConfig",  # noqa: F821
    data_source: "ImageNetCache",  # noqa: F821
) -> SharedComponents:
    """Parallel load text manipulator, image manipulator, and SUT.

    Three load futures run on a ``ThreadPoolExecutor`` so the heavy
    HuggingFace weight loads overlap.  The SUT future is resolved
    before the image manipulator if the image backend is StyleGAN-XL
    (StyleGAN's origin-precheck needs the SUT).

    Categories are expected to be already resolved on *exp_cfg* — the
    caller is responsible for invoking
    :func:`src.config.resolve_categories` against the data source's
    label set before this point.

    :param exp_cfg: Canonical experiment config with resolved categories.
    :param data_source: Open :class:`ImageNetCache`.
    :returns: :class:`SharedComponents` with all three model bundles loaded.
    """
    # Local imports to avoid circular deps at module-import time:
    # src.manipulator.* and src.sut import from src.common in some
    # paths; deferring these breaks the cycle.
    from src.manipulator.image.manipulator import ImageManipulator
    from src.manipulator.image_backend import ImageBackend
    from src.manipulator.image_factory import build_image_backend
    from src.manipulator.text.composite import CompositeTextManipulator
    from src.sut import VLMSUT

    pool = ThreadPoolExecutor(max_workers=3)

    logger.info(
        "Composite text manipulator starting (profile=%s)...",
        exp_cfg.text.composite.profile,
    )
    text_fut: Future[CompositeTextManipulator] = pool.submit(
        CompositeTextManipulator.from_config,
        text_config=exp_cfg.text,
        device=exp_cfg.device,
        redis_url=exp_cfg.sut.redis_url,
    )

    # VQGAN can load in parallel with the SUT.  StyleGAN needs the SUT
    # for origin-precheck, so we defer the image-backend build.
    image_fut: Future[ImageBackend] | None
    if exp_cfg.image.backend == "vqgan_codebook":
        logger.info(
            "Image manipulator starting (vqgan_codebook)...  preset=%s",
            exp_cfg.image.preset,
        )
        image_fut = pool.submit(
            ImageManipulator.from_preset,
            device=exp_cfg.device,
            config=exp_cfg.image,
        )
    else:
        logger.info(
            "Image manipulator deferred (stylegan_xl): builds after SUT."
        )
        image_fut = None

    sut_device = (
        exp_cfg.sut.ov_device
        if exp_cfg.sut.backend == "openvino"
        else exp_cfg.device
    )
    logger.info(
        "SUT starting...  %s on %s", exp_cfg.sut.model_id, sut_device,
    )
    sut_fut: Future[VLMSUT] = pool.submit(VLMSUT, exp_cfg)

    sut: VLMSUT = sut_fut.result()
    logger.info("SUT loaded")

    text_manip: CompositeTextManipulator = text_fut.result()
    logger.info("Text manipulator loaded")

    if image_fut is not None:
        image_manip: ImageBackend = image_fut.result()
    else:
        full_prompt = exp_cfg.prompt_template + exp_cfg.answer_format.format(
            categories=", ".join(exp_cfg.categories),
        )
        redis_for_stylegan = BytesRedisCache.from_url(exp_cfg.sut.redis_url)
        logger.info(
            "Image manipulator starting (stylegan_xl): "
            "checkpoint=%s, kappa_levels=%d, target_m=%d",
            exp_cfg.image.stylegan.checkpoint_path,
            exp_cfg.image.stylegan.kappa_quant_levels,
            exp_cfg.image.stylegan.target_m,
        )
        image_manip = build_image_backend(
            image_config=exp_cfg.image,
            device=exp_cfg.device,
            categories=exp_cfg.categories,
            sut=sut,
            redis_cache=redis_for_stylegan,
            prompt=full_prompt,
        )
    logger.info("Image manipulator loaded")

    pool.shutdown(wait=False)

    return SharedComponents(
        sut=sut,
        image_manip=image_manip,
        text_manip=text_manip,
        data_source=data_source,
    )


# ---------------------------------------------------------------------------
# Seed generation
# ---------------------------------------------------------------------------


def prepare_pipeline_seeds(
    components: SharedComponents,
    exp_cfg: "ExperimentConfig",  # noqa: F821
) -> Sequence:
    """Generate seeds per the config's seed mode.

    * ``gap_filter`` — :func:`src.common.generate_seeds`: scans
      ImageNet validation, returns seeds whose VLM logprob-gap meets
      the threshold.
    * ``roster`` — :func:`src.common.roster_seeds` collects anchors
      per explicit class, then :func:`src.common.combinatorial_pairs`
      expands across abstraction-level cells.

    :param components: Shared components (uses ``sut`` + ``data_source``).
    :param exp_cfg: Canonical experiment config.
    :returns: List of :class:`SeedTriple` objects (possibly empty).
    """
    if exp_cfg.seeds.mode == "roster":
        if exp_cfg.seeds.roster is None:
            raise ValueError(
                "seeds.mode='roster' requires a seeds.roster config block."
            )
        logger.info(
            "Generating seeds (roster: %d classes × %d seeds, "
            "combinatorial abstraction expansion)",
            len(exp_cfg.seeds.roster.class_list),
            exp_cfg.seeds.roster.seeds_per_class,
        )
        seed_images = roster_seeds(components.sut, exp_cfg, components.data_source)
        return combinatorial_pairs(
            seed_images,
            exp_cfg.seeds.roster.class_list,
            exp_cfg.seeds.roster.abstraction,
        )
    if exp_cfg.seeds.mode == "gap_filter":
        logger.info("Generating seeds (gap_filter)")
        return generate_seeds(components.sut, exp_cfg, components.data_source)
    raise ValueError(  # pragma: no cover — guarded by SeedConfig.__post_init__
        f"Unknown seeds.mode={exp_cfg.seeds.mode!r}"
    )


# ---------------------------------------------------------------------------
# Backend-specific precompute
# ---------------------------------------------------------------------------


def precompute_image_backend(
    components: SharedComponents,
    seeds: Sequence,
    exp_cfg: "ExperimentConfig",  # noqa: F821
) -> None:
    """Pre-populate backend-specific caches for the actually-evaluated seeds.

    * VQGAN + cone_filter.enabled → modal-grid for every target class
      that will be tested.  Built via
      :class:`~src.manipulator.image.class_target.ModalTargetBuilder`,
      attached to the image manipulator, then ``precompute_targets()``.
    * StyleGAN-XL → modal-w + accepted-origin precheck for every class
      that appears as either origin (``seed.class_a``) or target.

    Seeds are first scoped via :func:`apply_seed_filter` so unused
    classes (those filtered out by ``filter_indices``) are not
    precomputed.

    No-op for VQGAN without cone or when *seeds* is empty.

    :param components: Shared components (uses ``image_manip`` + ``data_source``).
    :param seeds: Full seed pool from :func:`prepare_pipeline_seeds`.
    :param exp_cfg: Canonical experiment config.
    """
    from src.manipulator.image.class_target import ModalTargetBuilder

    precompute_seeds = [
        s for _, s in apply_seed_filter(list(seeds), exp_cfg.seeds.filter_indices)
    ]
    if not precompute_seeds:
        return

    if (
        exp_cfg.image.backend == "vqgan_codebook"
        and exp_cfg.image.cone_filter.enabled
    ):
        target_classes = collect_target_classes(precompute_seeds)
        redis_cache = BytesRedisCache.from_url(exp_cfg.sut.redis_url)
        builder = ModalTargetBuilder(
            codec=components.image_manip.codec,  # type: ignore[attr-defined]
            data_source=components.data_source,
            preset=exp_cfg.image.preset,
            target_m=exp_cfg.image.cone_filter.target_m,
            redis_cache=redis_cache,
        )
        components.image_manip.attach_modal_builder(builder)
        logger.info(
            "Cone filter enabled (alpha=%.1f°, m=%d) — "
            "pre-populating modal targets for %d class(es)",
            exp_cfg.image.cone_filter.alpha_deg,
            exp_cfg.image.cone_filter.target_m,
            len(target_classes),
        )
        components.image_manip.precompute_targets(target_classes)
        return

    if exp_cfg.image.backend == "stylegan_xl":
        from src.common.seed_context import seed_target_class
        from src.manipulator.image_stylegan import StyleGANImageManipulator

        target_classes = collect_target_classes(precompute_seeds)
        # Pairwise origin precheck: SUT-accept depends on
        # P(origin) > P(target) under the actual (origin, target)
        # contrast, not a global top-K — so we precompute one accepted
        # origin per pair, not one per class.
        origin_pairs = tuple(sorted({
            (s.class_a, seed_target_class(s)) for s in precompute_seeds
        }))
        logger.info(
            "StyleGAN backend: pre-populating modal-w for %d target "
            "class(es), pairwise origin precheck for %d pair(s)",
            len(target_classes), len(origin_pairs),
        )
        assert isinstance(components.image_manip, StyleGANImageManipulator)
        components.image_manip.precompute_targets(target_classes, origin_pairs)


__all__ = [
    "SharedComponents",
    "init_shared_components",
    "prepare_pipeline_seeds",
    "precompute_image_backend",
]
