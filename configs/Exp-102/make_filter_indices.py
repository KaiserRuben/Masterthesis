#!/usr/bin/env python3
"""Generate the ``seeds.filter_indices`` list for Exp-102-Basin-Generality.

Exp-102 tests whether the Exp-100 PDQ-layer findings (boa-constrictor as
global attractor sink, sparse text-dominated minimal pair-flips) and the
deep-search label walls generalise beyond the junco-anchored / forward-only
slice that produced them. All 5 non-junco roster classes get PDQ
neighbourhoods; a small junco subset bridges back to the Exp-100 slice.

The roster combinatorial enumeration is deterministic given
``(class_list, AbstractionConfig, seeds_per_class)`` — see
``src/common/combinatorial_pair_generator.combinatorial_pairs``. This script
re-runs that exact enumeration with *mock* SeedImages (image=None — the
generator only reads ``class_concrete`` / ``seed_idx_in_class``), then selects
the 0-based positions whose ``(anchor, target, level_anchor, level_target,
seed_idx)`` match the pre-registered cell design below. Because it calls the
real generator, the emitted index list is provably aligned with what
``run_boundary_pair_test.py`` will produce at run time.

CLASS_LIST, SEEDS_PER_CLASS and ABSTRACTION are IDENTICAL to Exp-101's
``make_filter_indices.py`` — the enumeration index space is therefore shared
across Exp-101/Exp-102 (e.g. index 28 = junco->boa (0,1) "snake" in both),
which keeps cross-experiment joins on ``seed_idx`` trivial.

The resulting ``FILTER_INDICES`` list is embedded verbatim into
``exp102_basin_generality.yaml``. Re-run this script (env ``uni``) and
re-paste if the cell design or class order ever changes:

    conda run -n uni python configs/Exp-102/make_filter_indices.py

Pre-registered cell design (see YAML header for hypotheses P1-P4):

  Per anchor class: >=1 within-bucket pair + >=1 cross-bucket pair, mostly
  level-(0,0) cells (PDQ operates on concrete classes; abstraction depth is
  spent only on the wall cells). 23 cells, 4 upgraded to n=2 -> 27 runs.

  Wall cells (level-(0,1), evolutionary depth matters):
    junco->boa   (0,1) "snake"     — Exp-100 wall, BRIDGE replication, n=2
    ostrich->boa (0,1) "snake"     — same wall from the second bird anchor, n=2
    iguana->boa  (0,1) "snake"     — within-bucket variant, n=1 (secondary)
    boa->junco   (0,1) "songbird"  — reverse-direction wall, n=2
    cello->junco (0,1) "songbird"  — reverse wall from instrument anchor, n=2

  n=2 rationale: the four cross-bucket wall cells carry the depth/direction
  claims (P3) whose noise term is between-seed floor variance; one extra seed
  each gives a minimal within-cell replicate. All diagonal (0,0) cells are
  breadth (basin + flip geometry) and stay n=1 (wide > deep).

Total: 23 distinct cells + 4 extra seed rows = 27 runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repo root is two levels up (configs/Exp-102/ -> repo root); make `src`
# importable regardless of the caller's CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.common.combinatorial_pair_generator import combinatorial_pairs
from src.common.roster_seed_generator import SeedImage
from src.config import AbstractionConfig

# Must match the YAML exactly (order is load-bearing for direction semantics
# AND the enumeration index space). Identical to Exp-101.
CLASS_LIST = (
    "junco",
    "ostrich",
    "green iguana",
    "boa constrictor",
    "cello",
    "marimba",
)
SEEDS_PER_CLASS = 2  # YAML seeds.roster.seeds_per_class

# Full abstraction grid — must match the YAML so the enumeration (and thus the
# index space) is identical to the runner's. Identical to Exp-101.
ABSTRACTION = AbstractionConfig(
    levels_anchor=(0, 1, 2),
    levels_target=(0, 1, 2),
    apply_disjointness=True,
    directions="both",
)

# Pre-registered cells: (anchor, target, level_anchor, level_target, n_seeds).
# n_seeds=1 -> seed_idx 0 only; n_seeds=2 -> seed_idx 0 and 1.
J, O, I, B, C, M = (
    "junco",
    "ostrich",
    "green iguana",
    "boa constrictor",
    "cello",
    "marimba",
)

TARGET_CELLS: list[tuple[str, str, int, int, int]] = [
    # --- Junco (bridge to the Exp-100 slice, 3 cells / 4 runs) --------------
    (J, O, 0, 0, 1),   # within-bucket diagonal
    (J, B, 0, 0, 1),   # cross-bucket diagonal (basin + flip-geometry baseline)
    (J, B, 0, 1, 2),   # "snake" WALL bridge @ depth

    # --- Ostrich (4 cells / 5 runs) -----------------------------------------
    (O, J, 0, 0, 1),   # within-bucket diagonal
    (O, I, 0, 0, 1),   # cross-bucket, reptile target that is NOT boa (P1 probe)
    (O, B, 0, 0, 1),   # cross-bucket diagonal, boa target
    (O, B, 0, 1, 2),   # "snake" WALL from the second bird anchor

    # --- Green iguana (4 cells / 4 runs) ------------------------------------
    (I, B, 0, 0, 1),   # within-bucket diagonal
    (I, B, 0, 1, 1),   # within-bucket "snake" wall variant (secondary, n=1)
    (I, C, 0, 0, 1),   # cross-bucket, boa-free pair (critical P1 probe)
    (I, J, 0, 0, 1),   # cross-bucket, second boa-free pair

    # --- Boa constrictor (4 cells / 5 runs; P1b sink-stability block) -------
    (B, I, 0, 0, 1),   # within-bucket diagonal
    (B, J, 0, 0, 1),   # cross-bucket diagonal (reverse of Exp-100 main pair)
    (B, J, 0, 1, 2),   # "songbird" reverse WALL @ depth
    (B, M, 0, 0, 1),   # cross-bucket instrument target

    # --- Cello (4 cells / 5 runs) -------------------------------------------
    (C, M, 0, 0, 1),   # within-bucket diagonal
    (C, J, 0, 0, 1),   # cross-bucket diagonal
    (C, J, 0, 1, 2),   # "songbird" reverse WALL from instrument anchor
    (C, B, 0, 0, 1),   # cross-bucket, boa target from instrument anchor (P1)

    # --- Marimba (4 cells / 4 runs) -----------------------------------------
    (M, C, 0, 0, 1),   # within-bucket diagonal
    (M, B, 0, 0, 1),   # cross-bucket diagonal, boa target
    (M, J, 0, 0, 1),   # cross-bucket, boa-free pair (P1 probe)
    (M, I, 0, 0, 1),   # cross-bucket, second boa-free pair
]


def build_enumeration() -> list:
    """Re-run the real combinatorial enumeration with mock SeedImages."""
    mock_pool: list[SeedImage] = []
    for cls in CLASS_LIST:
        for k in range(SEEDS_PER_CLASS):
            mock_pool.append(
                SeedImage(image=None, class_concrete=cls, seed_idx_in_class=k)
            )
    return combinatorial_pairs(mock_pool, CLASS_LIST, ABSTRACTION)


def main() -> None:
    triples = build_enumeration()

    # Map (anchor, target, la, lt) -> {max n_seeds requested}.
    want: dict[tuple[str, str, int, int], int] = {}
    for a, t, la, lt, n in TARGET_CELLS:
        key = (a, t, la, lt)
        if key in want:
            raise ValueError(f"duplicate cell in TARGET_CELLS: {key}")
        want[key] = n

    selected: list[int] = []
    rows: list[tuple] = []
    matched_cells: set[tuple[str, str, int, int]] = set()
    for idx, tr in enumerate(triples):
        m = tr.metadata
        key = (
            m["anchor_class_concrete"],
            m["target_class_concrete"],
            m["level_anchor"],
            m["level_target"],
        )
        if key not in want:
            continue
        if m["seed_idx_in_class"] < want[key]:
            selected.append(idx)
            matched_cells.add(key)
            rows.append((
                idx, m["anchor_class_concrete"], m["target_class_concrete"],
                m["level_anchor"], m["level_target"], m["seed_idx_in_class"],
                m["anchor_label_in_prompt"], m["target_label_in_prompt"],
                m["common_ancestor_level"],
            ))

    # Fail loud if any requested cell never appeared (e.g. filtered by
    # disjointness, or a typo'd class/level).
    missing = set(want) - matched_cells
    if missing:
        raise SystemExit(
            f"ERROR: {len(missing)} requested cell(s) not found in enumeration "
            f"(disjointness-filtered or mis-specified): {sorted(missing)}"
        )

    selected.sort()
    print(f"# Enumeration size (seeds_per_class={SEEDS_PER_CLASS}): {len(triples)} SeedTriples")
    print(f"# Distinct cells requested: {len(want)}   selected rows: {len(selected)}")
    print()
    hdr = f"{'idx':>4}  {'anchor':<15} {'target':<15} la lt seed  {'prompt_anchor':<14} {'prompt_target':<14} c"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r[0]:>4}  {r[1]:<15} {r[2]:<15} {r[3]:>2} {r[4]:>2} {r[5]:>4}  "
            f"{r[6]:<14} {r[7]:<14} {r[8]}"
        )

    print()
    print("filter_indices (YAML, 10 per line):")
    for i in range(0, len(selected), 10):
        chunk = ", ".join(str(x) for x in selected[i:i + 10])
        print(f"    {chunk}")
    print("  FLAT:", selected)


if __name__ == "__main__":
    main()
