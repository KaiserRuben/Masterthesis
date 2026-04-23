"""Pure helpers shared by the evolutionary and PDQ pipelines.

These used to live as ``_`` -prefixed helpers inside the two packages, with
each package reaching into the other's privates at import time. Promoting them
here breaks that cycle and documents them as the shared seam.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from src.config import SeedTriple
from src.manipulator.vlm_manipulator import VLMManipulator

logger = logging.getLogger(__name__)


def build_context_meta(manipulator: VLMManipulator) -> dict[str, Any]:
    """Snapshot the manipulator's prepared context for offline reconstruction.

    Requires ``manipulator.prepare()`` to have been called.
    """
    img_sel = manipulator.image_context.selection
    txt_sel = manipulator.text_context.selection
    return {
        "image_patch_positions": img_sel.positions.tolist(),
        "image_original_codes": img_sel.original_codes.tolist(),
        "image_candidates": [c.tolist() for c in img_sel.candidates],
        "text_word_positions": txt_sel.positions.tolist(),
        "text_original_words": list(txt_sel.original_words),
        "text_candidates": [list(c) for c in txt_sel.candidates],
        "text_candidate_distances": [
            d.tolist() for d in manipulator.text_candidate_distances
        ],
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
