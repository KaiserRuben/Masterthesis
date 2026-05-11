"""Tests for src.common.combinatorial_pair_generator.

Covers:
- Disjointness rule for c=1 / c=2 / c=None pairs
- Direction filter: forward / reverse / both
- Metadata completeness on every emitted SeedTriple
- Apply_disjointness=False emits all 9 cells
- Empty class_list rejected
"""

from __future__ import annotations

import pytest
from PIL import Image

from src.common.combinatorial_pair_generator import combinatorial_pairs
from src.common.roster_seed_generator import SeedImage
from src.config import AbstractionConfig

# --- Reference classes selected by their known taxonomy structure --------
# (paths verified via src.data.taxonomy.path_of):
#   'junco'                  → sparrow / songbird / bird
#   'brambling'              → finch / songbird / bird     (c=1 with junco — same L1)
#   'cock'                   → chicken / poultry / bird    (c=2 with junco — same L2)
#   'Beagle'                 → scent hound / hound / dog   (c=None with junco — diff L2)


def _img() -> Image.Image:
    return Image.new("RGB", (4, 4), color=(0, 0, 0))


def _seeds_for(class_name: str, n: int = 1) -> list[SeedImage]:
    return [
        SeedImage(image=_img(), class_concrete=class_name, seed_idx_in_class=i)
        for i in range(n)
    ]


class TestDisjointness:
    def test_c1_pair_yields_only_l0_l0_cell(self) -> None:
        # junco vs brambling: common ancestor at level 1 (both 'songbird').
        # Only cell satisfying max(la, lt) < 1 is (0, 0).
        seeds = _seeds_for("junco", 1) + _seeds_for("brambling", 1)
        out = combinatorial_pairs(
            seeds, ["junco", "brambling"],
            AbstractionConfig(directions="forward"),
        )
        # 1 pair × 1 cell × 1 seed = 1 SeedTriple
        assert len(out) == 1
        assert out[0].metadata["level_anchor"] == 0
        assert out[0].metadata["level_target"] == 0
        assert out[0].metadata["common_ancestor_level"] == 1

    def test_c2_pair_yields_4_cells(self) -> None:
        # junco vs cock: common ancestor at level 2 (both 'bird').
        # Valid cells: (0,0), (0,1), (1,0), (1,1) — all where max(la,lt) < 2.
        seeds = _seeds_for("junco", 1) + _seeds_for("cock", 1)
        out = combinatorial_pairs(
            seeds, ["junco", "cock"],
            AbstractionConfig(directions="forward"),
        )
        # 1 pair × 4 cells × 1 seed = 4 SeedTriples
        assert len(out) == 4
        cells = {(t.metadata["level_anchor"], t.metadata["level_target"])
                 for t in out}
        assert cells == {(0, 0), (0, 1), (1, 0), (1, 1)}
        for t in out:
            assert t.metadata["common_ancestor_level"] == 2

    def test_c_none_pair_yields_all_9_cells(self) -> None:
        # junco vs Beagle: different super-cats (bird vs dog).
        # All 9 cells valid.
        seeds = _seeds_for("junco", 1) + _seeds_for("Beagle", 1)
        out = combinatorial_pairs(
            seeds, ["junco", "Beagle"],
            AbstractionConfig(directions="forward"),
        )
        # 1 pair × 9 cells × 1 seed = 9 SeedTriples
        assert len(out) == 9
        cells = {(t.metadata["level_anchor"], t.metadata["level_target"])
                 for t in out}
        expected = {(la, lt) for la in (0, 1, 2) for lt in (0, 1, 2)}
        assert cells == expected
        for t in out:
            assert t.metadata["common_ancestor_level"] is None

    def test_apply_disjointness_false_emits_all_9_for_c2(self) -> None:
        seeds = _seeds_for("junco", 1) + _seeds_for("cock", 1)
        out = combinatorial_pairs(
            seeds, ["junco", "cock"],
            AbstractionConfig(directions="forward", apply_disjointness=False),
        )
        # 1 pair × 9 cells × 1 seed = 9 SeedTriples
        assert len(out) == 9


class TestDirections:
    def test_both_emits_two_directions(self) -> None:
        # junco vs Beagle (c=None, 9 cells per direction).
        seeds = _seeds_for("junco", 1) + _seeds_for("Beagle", 1)
        out = combinatorial_pairs(
            seeds, ["junco", "Beagle"],
            AbstractionConfig(directions="both"),
        )
        # 2 directions × 9 cells × 1 seed = 18 SeedTriples
        assert len(out) == 18
        directions_seen = {
            (t.metadata["anchor_position"], t.metadata["target_position"])
            for t in out
        }
        assert directions_seen == {(0, 1), (1, 0)}

    def test_forward_only(self) -> None:
        seeds = _seeds_for("junco", 1) + _seeds_for("Beagle", 1)
        out = combinatorial_pairs(
            seeds, ["junco", "Beagle"],
            AbstractionConfig(directions="forward"),
        )
        assert len(out) == 9
        for t in out:
            i = t.metadata["anchor_position"]
            j = t.metadata["target_position"]
            assert i < j

    def test_reverse_only(self) -> None:
        seeds = _seeds_for("junco", 1) + _seeds_for("Beagle", 1)
        out = combinatorial_pairs(
            seeds, ["junco", "Beagle"],
            AbstractionConfig(directions="reverse"),
        )
        assert len(out) == 9
        for t in out:
            i = t.metadata["anchor_position"]
            j = t.metadata["target_position"]
            assert i > j


class TestSeedReplication:
    def test_seeds_per_class_replicated_per_cell(self) -> None:
        # 3 seeds for class A, 3 for class B, c=None pair = 9 cells × 2 dirs.
        # Each direction's 9 cells get the anchor's 3 seeds → 27 per dir.
        seeds = _seeds_for("junco", 3) + _seeds_for("Beagle", 3)
        out = combinatorial_pairs(
            seeds, ["junco", "Beagle"],
            AbstractionConfig(directions="both"),
        )
        # 2 dirs × 9 cells × 3 seeds = 54 (per Exp-100 spec).
        assert len(out) == 54
        # Check seed_idx_in_class spans 0..2 for each (direction, cell) tuple.
        from collections import defaultdict
        groups: defaultdict[tuple, list[int]] = defaultdict(list)
        for t in out:
            key = (t.metadata["anchor_position"],
                   t.metadata["level_anchor"],
                   t.metadata["level_target"])
            groups[key].append(t.metadata["seed_idx_in_class"])
        assert len(groups) == 18  # 2 dirs × 9 cells
        for key, idxs in groups.items():
            assert sorted(idxs) == [0, 1, 2]


class TestMetadata:
    def test_metadata_keys_complete(self) -> None:
        seeds = _seeds_for("junco", 1) + _seeds_for("Beagle", 1)
        out = combinatorial_pairs(
            seeds, ["junco", "Beagle"],
            AbstractionConfig(directions="forward",
                              levels_anchor=(0,), levels_target=(2,)),
        )
        assert len(out) == 1
        meta = out[0].metadata
        expected_keys = {
            "level_anchor", "level_target",
            "anchor_class_concrete", "target_class_concrete",
            "anchor_label_in_prompt", "target_label_in_prompt",
            "common_ancestor_level",
            "seed_idx_in_class",
            "anchor_position", "target_position",
        }
        assert set(meta.keys()) == expected_keys

    def test_labels_are_actually_substituted(self) -> None:
        # junco @ L2 = "bird"; Beagle @ L0 = "scent hound".
        seeds = _seeds_for("junco", 1) + _seeds_for("Beagle", 1)
        out = combinatorial_pairs(
            seeds, ["junco", "Beagle"],
            AbstractionConfig(directions="forward",
                              levels_anchor=(2,), levels_target=(0,)),
        )
        assert len(out) == 1
        t = out[0]
        assert t.class_a == "bird"
        assert t.class_b == "scent hound"
        assert t.metadata["anchor_label_in_prompt"] == "bird"
        assert t.metadata["target_label_in_prompt"] == "scent hound"


class TestFailureModes:
    def test_empty_class_list_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            combinatorial_pairs([], [], AbstractionConfig())

    def test_seed_for_class_outside_list_rejected(self) -> None:
        seeds = _seeds_for("junco", 1)
        with pytest.raises(ValueError, match="no slot in class_list"):
            combinatorial_pairs(seeds, ["brambling"], AbstractionConfig())
