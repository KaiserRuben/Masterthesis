"""Combinatorial pair expansion for the Exp-100 roster pipeline.

Takes a roster anchor pool (list of :class:`SeedImage`) and expands it
into the SMOO-tester-ready :class:`SeedTriple` list, enumerating all
ordered (anchor, target) class pairs in ``class_list`` and all valid
(L_anchor, L_target) abstraction-level cells per pair.

Two filters apply per cell:

1. **Direction filter** — ``directions in {"forward", "reverse", "both"}``.
   Forward = ``class_list[i]`` is anchor against ``class_list[j]`` for
   ``i < j``; reverse = ``i > j``. ``both`` covers both halves and
   yields the symmetric anchor/target swap that the experiment design
   relies on.

2. **Disjointness filter** — when ``apply_disjointness=True`` (default),
   only cells satisfying ``max(L_a, L_b) < common_ancestor_level`` are
   emitted. This drops semantically nested prompt pairs like
   "Is this a Junco or a bird?" (Junco ⊂ bird → ill-defined).

Each emitted SeedTriple carries the Exp-100 metadata dict so that the
tester can write it into ``stats.json`` for post-hoc aggregation along
the (Anchor-class × Target-class × L_anchor × L_target × Seed) axes.
"""

from __future__ import annotations

import logging
from typing import Sequence

from src.common.abstraction import resolve_label
from src.common.roster_seed_generator import SeedImage
from src.config import AbstractionConfig, SeedTriple
from src.data.taxonomy import common_ancestor_level

logger = logging.getLogger(__name__)


def combinatorial_pairs(
    seed_images: Sequence[SeedImage],
    class_list: Sequence[str],
    abstraction: AbstractionConfig,
) -> list[SeedTriple]:
    """Expand a roster anchor pool into ordered + abstracted SeedTriples.

    :param seed_images: Anchor pool from :func:`roster_seeds`. Must be
        grouped by class in ``class_list`` order, with ``seed_idx_in_class``
        running 0..N-1 within each group.
    :param class_list: Concrete (L0) class names. Position in this list
        determines forward/reverse direction semantics.
    :param abstraction: Levels, disjointness flag, direction filter.
    :returns: SeedTriples in deterministic emission order — outer loop
        over ordered pairs ``(i, j)`` (i != j), inner loop over
        ``(L_anchor, L_target)`` cells in
        ``levels_anchor × levels_target`` order, innermost over
        ``seed_idx_in_class`` of the anchor class. Each carries a full
        ``metadata`` dict.
    """
    class_list = tuple(class_list)
    if not class_list:
        raise ValueError("class_list must be non-empty.")

    # Group seed images by class for O(1) per-anchor lookup.
    by_class: dict[str, list[SeedImage]] = {c: [] for c in class_list}
    for s in seed_images:
        if s.class_concrete not in by_class:
            raise ValueError(
                f"SeedImage for class {s.class_concrete!r} has no slot in "
                f"class_list={list(class_list)!r}."
            )
        by_class[s.class_concrete].append(s)

    out: list[SeedTriple] = []
    n_pairs_total = 0
    n_pairs_filtered_disjoint = 0
    n_cells_total = 0
    n_cells_filtered = 0

    for i, anchor_cls in enumerate(class_list):
        for j, target_cls in enumerate(class_list):
            if i == j:
                continue
            # Direction filter: forward = i<j, reverse = i>j, both = always.
            if abstraction.directions == "forward" and not (i < j):
                continue
            if abstraction.directions == "reverse" and not (i > j):
                continue

            n_pairs_total += 1
            cancestor = common_ancestor_level(anchor_cls, target_cls)

            pair_cell_count = 0
            for la in abstraction.levels_anchor:
                for lt in abstraction.levels_target:
                    n_cells_total += 1
                    if abstraction.apply_disjointness:
                        # Cell valid iff max(la, lt) < common_ancestor_level.
                        # When cancestor is None (different super-cats), ANY
                        # finite max satisfies the rule, so all cells pass.
                        if cancestor is not None and max(la, lt) >= cancestor:
                            n_cells_filtered += 1
                            continue

                    anchor_label = resolve_label(anchor_cls, la)
                    target_label = resolve_label(target_cls, lt)

                    pair_cell_count += 1
                    for seed in by_class.get(anchor_cls, []):
                        meta = {
                            "level_anchor": la,
                            "level_target": lt,
                            "anchor_class_concrete": anchor_cls,
                            "target_class_concrete": target_cls,
                            "anchor_label_in_prompt": anchor_label,
                            "target_label_in_prompt": target_label,
                            "common_ancestor_level": cancestor,
                            "seed_idx_in_class": seed.seed_idx_in_class,
                            "anchor_position": i,
                            "target_position": j,
                        }
                        out.append(SeedTriple(
                            image=seed.image,
                            class_a=anchor_label,
                            class_b=target_label,
                            metadata=meta,
                        ))

            if pair_cell_count == 0:
                # Whole pair was filtered out. c=0 (same fine cluster) is
                # the typical reason — those classes are taxonomy-siblings
                # at L0 and have no semantically disjoint cells.
                n_pairs_filtered_disjoint += 1

    logger.info(
        "Combinatorial expansion: %d ordered pairs (%d fully filtered), "
        "%d cells total (%d filtered by disjointness), %d SeedTriples emitted.",
        n_pairs_total, n_pairs_filtered_disjoint,
        n_cells_total, n_cells_filtered, len(out),
    )
    return out


__all__ = ["combinatorial_pairs"]
