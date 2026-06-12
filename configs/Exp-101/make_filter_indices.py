#!/usr/bin/env python3
"""Generate the ``seeds.filter_indices`` list for Exp-101-Margin-Predictor.

Exp-101 tests whether the *gen-0 median TgtBal margin* predicts evolutionary
convergence SPEED across anchor classes, directions, and label-pair cells.
Full enumeration of the 6-class roster (both directions, all valid abstraction
cells) is 240 directed cells; we sub-sample a stratified ~40-cell subset and
upgrade a handful to n=2 seeds for the within-cell null test (P3).

The roster combinatorial enumeration is deterministic given
``(class_list, AbstractionConfig, seeds_per_class)`` — see
``src/common/combinatorial_pair_generator.combinatorial_pairs``. This script
re-runs that exact enumeration with *mock* SeedImages (image=None — the
generator only reads ``class_concrete`` / ``seed_idx_in_class``), then selects
the 0-based positions whose ``(anchor, target, level_anchor, level_target,
seed_idx)`` match the pre-registered cell design below. Because it calls the
real generator, the emitted index list is provably aligned with what
``run_boundary_test.py`` will produce at run time.

The resulting ``FILTER_INDICES`` list is embedded verbatim into
``exp101_margin_predictor.yaml``. Re-run this script (env ``uni``) and re-paste
if the cell design or class order ever changes:

    conda run -n uni python configs/Exp-101/make_filter_indices.py

Pre-registered cell design (see YAML header for hypotheses P1-P3):

  Stratum 1 — within-bucket (c=2; valid cells max(la,lt)<2), cells
              {(0,0),(0,1),(1,1)} for BOTH directions of all 3 same-L2 pairs.
              Diagonal (0,0)/(1,1) + one off-diagonal (0,1). 18 cells.
  Stratum 2 — cross-bucket REQUIRED junco label-wall replications + super-level
              controls + reverse directions. 12 cells.
                junco->boa:  lt-sweep @ la=0  -> (0,0),(0,1 "snake" WALL),(0,2 "reptile" ctrl)
                boa->junco:  lt-sweep @ la=0  -> (0,0),(0,1 "songbird"),(0,2 "bird")
                junco->cello: la-sweep @ lt=0 -> (0,0),(1,0 "songbird" WALL),(2,0 "bird" ctrl)
                cello->junco: lt-sweep @ la=0 -> (0,0),(0,1 "songbird"),(0,2 "bird")
  Stratum 3 — cross-bucket breadth on NON-junco anchors (powers the P1
              falsification test on the non-junco subset). Diagonal cells.
              10 cells.

  n=2 upgrade (P3 within-cell null): 6 cells spanning cell-kinds and the
  stuck/converged regimes get seed_idx 0 AND 1; everything else is seed_idx 0.

Total: 40 distinct cells + 6 extra seed rows = 46 runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repo root is two levels up (configs/Exp-101/ -> repo root); make `src`
# importable regardless of the caller's CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.common.combinatorial_pair_generator import combinatorial_pairs
from src.common.roster_seed_generator import SeedImage
from src.config import AbstractionConfig

# Must match the YAML exactly (order is load-bearing for direction semantics).
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
# index space) is identical to the runner's.
ABSTRACTION = AbstractionConfig(
    levels_anchor=(0, 1, 2),
    levels_target=(0, 1, 2),
    apply_disjointness=True,
    directions="both",
)

# Pre-registered cells: (anchor, target, level_anchor, level_target, n_seeds).
# n_seeds=1 -> seed_idx 0 only; n_seeds=2 -> seed_idx 0 and 1 (P3 null test).
J, O, I, B, C, M = (
    "junco",
    "ostrich",
    "green iguana",
    "boa constrictor",
    "cello",
    "marimba",
)

TARGET_CELLS: list[tuple[str, str, int, int, int]] = [
    # --- Stratum 1: within-bucket (c=2), cells (0,0),(0,1),(1,1) -------------
    (J, O, 0, 0, 2), (J, O, 0, 1, 1), (J, O, 1, 1, 1),   # junco -> ostrich
    (O, J, 0, 0, 1), (O, J, 0, 1, 1), (O, J, 1, 1, 1),   # ostrich -> junco
    (I, B, 0, 0, 1), (I, B, 0, 1, 1), (I, B, 1, 1, 2),   # iguana -> boa
    (B, I, 0, 0, 1), (B, I, 0, 1, 1), (B, I, 1, 1, 1),   # boa -> iguana
    (C, M, 0, 0, 2), (C, M, 0, 1, 1), (C, M, 1, 1, 1),   # cello -> marimba
    (M, C, 0, 0, 1), (M, C, 0, 1, 1), (M, C, 1, 1, 1),   # marimba -> cello

    # --- Stratum 2: cross-bucket junco label walls + controls + reverse -----
    (J, B, 0, 0, 1), (J, B, 0, 1, 2), (J, B, 0, 2, 1),   # junco->boa: snake WALL @ (0,1)
    (B, J, 0, 0, 1), (B, J, 0, 1, 2), (B, J, 0, 2, 1),   # boa->junco reverse
    (J, C, 0, 0, 1), (J, C, 1, 0, 1), (J, C, 2, 0, 1),   # junco->cello: songbird WALL @ (1,0)
    (C, J, 0, 0, 1), (C, J, 0, 1, 1), (C, J, 0, 2, 1),   # cello->junco reverse

    # --- Stratum 3: cross-bucket, non-junco anchors (breadth for P1) --------
    (O, I, 0, 0, 1), (O, I, 1, 1, 1),                    # ostrich -> iguana
    (I, C, 0, 0, 2), (I, C, 1, 1, 1),                    # iguana -> cello
    (M, B, 0, 0, 1), (M, B, 1, 1, 1),                    # marimba -> boa
    (C, O, 0, 0, 1), (C, O, 2, 2, 1),                    # cello -> ostrich
    (B, M, 0, 0, 1), (B, M, 1, 1, 1),                    # boa -> marimba
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
    print("filter_indices:")
    # 10 per line for readable YAML embedding.
    for i in range(0, len(selected), 10):
        chunk = ", ".join(str(x) for x in selected[i:i + 10])
        suffix = "," if i + 10 < len(selected) else ""
        print(f"  [{chunk}]" if False else f"    # {chunk}{suffix}")
    print("  FLAT:", selected)


if __name__ == "__main__":
    main()
