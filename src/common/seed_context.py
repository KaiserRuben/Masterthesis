"""Pure helpers shared by the evolutionary and PDQ pipelines.

These used to live as ``_`` -prefixed helpers inside the two packages, with
each package reaching into the other's privates at import time. Promoting them
here breaks that cycle and documents them as the shared seam.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Sequence

from src.config import SeedTriple
from src.manipulator.vlm_manipulator import VLMManipulator

logger = logging.getLogger(__name__)


def seed_target_class(seed: SeedTriple) -> str:
    """Resolve the seed's concrete (L0) target class name.

    The cone-filter modal-grid builder needs an L0 ImageNet label
    (``ImageNetCache.load_samples`` only accepts L0). Two cases:

    * Roster mode: ``seed.metadata["target_class_concrete"]`` is the L0
      name (``class_b`` may be an abstracted label like ``"bird"``).
    * Gap-filter mode: ``class_b`` is already an L0 label.

    :param seed: A :class:`SeedTriple` from either generator.
    :returns: The L0 ImageNet class name.
    """
    meta = seed.metadata or {}
    concrete = meta.get("target_class_concrete")
    if concrete:
        return str(concrete)
    return seed.class_b


def collect_target_classes(seeds: Iterable[SeedTriple]) -> tuple[str, ...]:
    """Return the deduplicated, sorted L0 target classes for ``seeds``.

    Used by runners to pre-populate the modal-target cache for all classes
    that will appear during the run, before any worker starts iterating.
    """
    return tuple(sorted({seed_target_class(s) for s in seeds}))


def build_context_meta(manipulator: VLMManipulator) -> dict[str, Any]:
    """Snapshot the manipulator's prepared context for offline reconstruction.

    Requires ``manipulator.prepare()`` to have been called.

    Image side: dispatches on the backend. For the VQGAN backend the
    flat ``selection`` with ``positions`` / ``original_codes`` /
    ``candidates`` is recorded along with ``image_candidate_strategy``
    (``"knn"`` or ``"cone_filter"``) and ``image_target_class``. For the
    StyleGAN backend the genome is per-layer κ levels, so the meta
    records ``image_num_ws`` and ``image_kappa_quant_levels`` plus the
    origin / target class labels. The ``image_backend`` field marks
    which case applies; see ``EVOLUTIONARY_SCHEMA_VERSION`` v4.

    Text side (post 2026-04-28 composite cleanup): the text context is a
    :class:`CompositeManipulationContext` carrying the original tokenised
    prompt (with PoS tags) and the per-operator gene-block layout.
    """
    img_ctx = manipulator.image_context
    txt_ctx = manipulator.text_context

    text_meta = {
        "text_original_tokens": list(txt_ctx.original_tokens.tokens),
        "text_pos_tags": list(txt_ctx.original_tokens.pos_tags),
        "text_op_order": list(txt_ctx.op_order),
        "text_op_gene_dims": list(txt_ctx.op_gene_dims),
        "text_gene_bounds": txt_ctx.gene_bounds.tolist(),
    }

    if hasattr(img_ctx, "selection"):
        # VQGAN backend.
        img_sel = img_ctx.selection
        return {
            "image_backend": "vqgan_codebook",
            "image_patch_positions": img_sel.positions.tolist(),
            "image_original_codes": img_sel.original_codes.tolist(),
            "image_candidates": [c.tolist() for c in img_sel.candidates],
            "image_candidate_strategy": img_ctx.candidate_strategy,
            "image_target_class": img_ctx.target_class,
            **text_meta,
        }

    # StyleGAN backend.
    return {
        "image_backend": "stylegan_xl",
        "image_num_ws": int(getattr(img_ctx, "num_ws", img_ctx.genotype_dim)),
        "image_kappa_quant_levels": int(
            getattr(img_ctx, "kappa_quant_levels", img_ctx.gene_bounds[0] - 1)
        ),
        "image_origin_class": getattr(img_ctx, "origin_class", None),
        "image_target_class": img_ctx.target_class,
        "image_candidate_strategy": img_ctx.candidate_strategy,
        **text_meta,
    }


def apply_seed_filter(
    seeds: Sequence[SeedTriple],
    filter_indices: tuple[int, ...],
) -> list[tuple[int, SeedTriple]]:
    """Keep only the requested indices from a generated seed pool.

    Preserves original 0-based indices so output directories and
    ``seed_idx`` metadata stay consistent with an unfiltered run — a
    filtered seed at original position 32 is still reported as
    ``seed_0032``.

    :param seeds: Full generated seed pool (``generate_seeds`` output).
    :param filter_indices: Indices to keep. Empty tuple → keep all.
    :returns: List of ``(original_index, seed)`` pairs in index order.
    :raises ValueError: If any filter index is out of range for the pool.
    """
    if not filter_indices:
        return list(enumerate(seeds))
    filter_set = set(filter_indices)
    out_of_range = {i for i in filter_set if i < 0 or i >= len(seeds)}
    if out_of_range:
        raise ValueError(
            f"seeds.filter_indices references out-of-range indices "
            f"{sorted(out_of_range)} for a pool of size {len(seeds)}. "
            "Either adjust filter_indices or regenerate the seed pool "
            "with the same n_per_class / max_logprob_gap / n_categories."
        )
    kept = [(i, s) for i, s in enumerate(seeds) if i in filter_set]
    logger.info(
        "Seed filter active: %d of %d seeds retained (indices %s)",
        len(kept), len(seeds), sorted(filter_set),
    )
    return kept
