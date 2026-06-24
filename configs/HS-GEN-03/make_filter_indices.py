#!/usr/bin/env python3
"""Generate ``seeds.filter_indices`` for HS-GEN-03 — fresh joint-diversity pool.

PURPOSE
-------
HS-01's joint-modality strata (pair balanced/text_heavy/image_heavy,
image/boundary_joint) still trail by ~30 items after HS-GEN-02. HS-GEN-02
exhausted its 6-class roster (junco, ostrich, green iguana, boa constrictor,
cello, marimba) at the ``seeds_per_class=12`` ceiling — all 27 crossers were
consumed, 0 spare. HS-GEN-03 reuses the *exact* HS-GEN-02 / Exp-101 pipeline
(joint modality, LLaVA-OV-INT8, cone alpha=20, full_stack text,
sparse_multitier init, 30 gen x 30 pop, early_stop disabled) but with a NEW,
DISJOINT 6-class roster so the campaign produces FRESH distinct anchor photos.

The new roster is three within-L2 "buckets" of two visually-distinct,
lay-distinguishable classes each — mirroring HS-GEN-02's structure (bird,
reptile, instrument) with brand-new classes:

    bird     : peacock          -> flamingo
    mammal   : zebra            -> Golden Retriever
    vehicle  : school bus       -> airliner

Each within-bucket forward (0,0) pair is the highest-yield cell family (the
within-bucket forward (0,0) cells crossed ~80% in the Exp-101 analysis).

PHOTO ACCOUNTING (the load-bearing fact)
----------------------------------------
A distinct anchor PHOTO is a ``(anchor_class_concrete, seed_idx_in_class)``
pair. The combinatorial enumeration emits one SeedTriple per
``(anchor, target, level_anchor, level_target, seed_idx)``. Every selected
SeedTriple is one RUN and exercises exactly ONE photo (its anchor). If the
SAME photo (a,k) is selected for several target cells, those runs all reuse
that one photo -> poor distinct-photo yield. To MAXIMISE distinct photos we
therefore assign each photo a SMALL, ROTATED set of forward (0,0) targets
(``TARGETS_PER_PHOTO``) instead of all partners. With one target per photo,
runs == distinct photos; with two, runs == 2 x distinct photos but the
corpus spans twice as many label-pairs.

CELL FAMILY (why forward (0,0))
-------------------------------
Empirically (46-run Exp-101 scan, configs/HS-GEN-02/_analyze_exp101_yield.py)
the HIGH-YIELD family is forward, level-(0,0):
    within-bucket forward (0,0): crossed 4/5 (80%)
    cross-bucket  forward (0,0): crossed 2/6 (33%)
    forward beats reverse everywhere (within 67% vs 11%; cross 36% vs 27%).
So HS-GEN-03 selects forward (i<j) cells at (la,lt)=(0,0) only. We INCLUDE
the within-bucket forward (0,0) pairs (peacock->flamingo, zebra->Golden
Retriever, school bus->airliner) because they are the single highest-yield
cells. ``CROSS_ONLY`` can flip this off if a pure cross-bucket corpus is
wanted.

LAY-DISTINGUISHABILITY
----------------------
All forward pairs in this roster are lay-distinguishable. Within-bucket pairs
are visually unmistakable (an iridescent fanned peacock vs a pink long-legged
flamingo; a black-and-white striped zebra vs a golden dog; a yellow school
bus vs a jet airliner). Cross-bucket pairs span different L2 super-categories
(bird / mammal / vehicle) and are trivially distinct. So every (anchor,
target) stimulus this campaign can promote is clearly tellable apart by a
layperson.

SEEDS_PER_CLASS CEILING
-----------------------
RosterSeedGenerator over-fetches ``seeds_per_class * 4`` candidates/class
(``_OVERSAMPLE_FACTOR``). ImageNet-1k validation has exactly 50 images/class,
so ``seeds_per_class * 4 <= 50`` -> ``seeds_per_class <= 12`` is the hard
no-rejection ceiling; beyond it the loader streams past the val pool and a
strict class risks RuntimeError pool-exhaustion. Realistic rejection
(misclass + below-threshold) lowers the achievable count further, so 12 is an
upper bound the workstation must confirm (preflight / roster dry-run). If a
class cannot fill 12, lower SEEDS_PER_CLASS here, regenerate, and re-paste.

Run (env ``uni``):
    conda run -n uni python configs/HS-GEN-03/make_filter_indices.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

# Repo root is two levels up (configs/HS-GEN-03/ -> repo root).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.common.combinatorial_pair_generator import combinatorial_pairs
from src.common.roster_seed_generator import SeedImage
from src.config import AbstractionConfig
from src.data.taxonomy import common_ancestor_level

# Must match the YAML exactly (order is load-bearing for direction semantics
# AND for the enumeration index space). Three within-L2 buckets, anchor first:
#   bird     : peacock          -> flamingo
#   mammal   : zebra            -> Golden Retriever
#   vehicle  : school bus       -> airliner
CLASS_LIST = (
    "peacock",
    "flamingo",
    "zebra",
    "Golden Retriever",
    "school bus",
    "airliner",
)

# YAML seeds.roster.seeds_per_class. 12 = hard oversample ceiling (12*4=48<=50
# val pool). Lower this (and re-run) if a class cannot fill on the workstation.
SEEDS_PER_CLASS = 12

# Forward (0,0) targets to assign each anchor photo. 2 -> runs == 2 x distinct
# photos, doubling label-pair coverage while keeping the distinct-photo count
# high. Set to 1 for runs == distinct photos (max photos / min runs).
TARGETS_PER_PHOTO = 2

# Include the 3 within-bucket forward (0,0) pairs (highest-yield cells). With
# CROSS_ONLY=False every forward (0,0) cell (within + cross) is selected.
CROSS_ONLY = False

# Full abstraction grid — must match the YAML so the enumeration (and thus the
# index space) is identical to the runner's.
ABSTRACTION = AbstractionConfig(
    levels_anchor=(0, 1, 2),
    levels_target=(0, 1, 2),
    apply_disjointness=True,
    directions="both",
)


def build_enumeration() -> list:
    """Re-run the real combinatorial enumeration with mock SeedImages."""
    mock_pool: list[SeedImage] = [
        SeedImage(image=None, class_concrete=cls, seed_idx_in_class=k)
        for cls in CLASS_LIST
        for k in range(SEEDS_PER_CLASS)
    ]
    return combinatorial_pairs(mock_pool, CLASS_LIST, ABSTRACTION)


def forward_00_partners(anchor: str) -> list[str]:
    """Forward (i<j) partners for *anchor* at the concrete (L0) level."""
    i = CLASS_LIST.index(anchor)
    parts = [t for j, t in enumerate(CLASS_LIST) if i < j]
    if CROSS_ONLY:
        parts = [t for t in parts if common_ancestor_level(anchor, t) is None]
    return parts


def main() -> None:
    triples = build_enumeration()

    # (anchor, target, seed_idx) -> enumeration idx, restricted to (la,lt)=(0,0).
    idx_of: dict[tuple[str, str, int], int] = {}
    for idx, tr in enumerate(triples):
        m = tr.metadata
        if m["level_anchor"] == 0 and m["level_target"] == 0:
            idx_of[(
                m["anchor_class_concrete"],
                m["target_class_concrete"],
                m["seed_idx_in_class"],
            )] = idx

    selected: list[int] = []
    rows: list[tuple] = []
    pair_counts: Counter[tuple[str, str]] = Counter()
    photos: set[tuple[str, int]] = set()

    for anchor in CLASS_LIST:
        partners = forward_00_partners(anchor)
        if not partners:
            continue  # airliner (last) anchors nothing forward
        for k in range(SEEDS_PER_CLASS):
            # Rotate a contiguous window of TARGETS_PER_PHOTO partners so the
            # photo's targets vary seed-to-seed and the pair distribution stays
            # balanced.
            for off in range(TARGETS_PER_PHOTO):
                t = partners[(k * TARGETS_PER_PHOTO + off) % len(partners)]
                key = (anchor, t, k)
                idx = idx_of.get(key)
                if idx is None:
                    raise SystemExit(
                        f"ERROR: cell {key} absent from enumeration "
                        f"(disjointness-filtered or bad class/level)."
                    )
                if idx in selected:
                    # Same (anchor,target,seed) chosen twice by the rotation
                    # (happens when TARGETS_PER_PHOTO >= len(partners)); skip
                    # the duplicate so every run is a unique cell.
                    continue
                selected.append(idx)
                photos.add((anchor, k))
                pair_counts[(anchor, t)] += 1
                ca = common_ancestor_level(anchor, t)
                rows.append((idx, anchor, t, k,
                             "within" if ca is not None else "cross"))

    selected.sort()

    print(f"# Enumeration size (seeds_per_class={SEEDS_PER_CLASS}): "
          f"{len(triples)} SeedTriples")
    print(f"# CROSS_ONLY={CROSS_ONLY}  TARGETS_PER_PHOTO={TARGETS_PER_PHOTO}")
    print(f"# runs (selected cells): {len(selected)}")
    print(f"# DISTINCT anchor photos: {len(photos)}  "
          f"(classes: {sorted({a for a, _ in photos})})")
    print(f"# label-pairs covered: {len(pair_counts)}")
    print()
    hdr = f"{'idx':>5}  {'anchor':<16} {'target':<16} seed  kind"
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows):
        print(f"{r[0]:>5}  {r[1]:<16} {r[2]:<16} {r[3]:>4}  {r[4]}")

    print()
    print("# pair distribution:")
    for (a, t), n in pair_counts.most_common():
        print(f"#   {a:16} -> {t:16}: {n}")

    print()
    print("filter_indices:")
    for i in range(0, len(selected), 10):
        chunk = ", ".join(str(x) for x in selected[i:i + 10])
        suffix = "," if i + 10 < len(selected) else ""
        print(f"    # {chunk}{suffix}")
    print("  FLAT:", selected)


if __name__ == "__main__":
    main()
