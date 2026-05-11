"""Tests for src.common.abstraction — taxonomy-resolver helpers.

Uses real ImageNet taxonomy (no monkey-patching) since
``src.data.taxonomy`` is a pure-data lookup module with no I/O.
"""

from __future__ import annotations

import pytest

from src.common.abstraction import resolve_label, validate_class_list


class TestResolveLabel:
    def test_l0_l1_l2_for_known_class(self) -> None:
        # 'junco' has a complete L0/L1/L2 path: sparrow / songbird / bird
        assert resolve_label("junco", 0) == "sparrow"
        assert resolve_label("junco", 1) == "songbird"
        assert resolve_label("junco", 2) == "bird"

    def test_invalid_level_raises(self) -> None:
        with pytest.raises(ValueError, match="level must be in"):
            resolve_label("junco", 3)
        with pytest.raises(ValueError, match="level must be in"):
            resolve_label("junco", -1)

    def test_unknown_class_raises(self) -> None:
        with pytest.raises(KeyError, match="unknown class"):
            resolve_label("definitely_not_a_real_class", 0)


class TestValidateClassList:
    def test_accepts_classes_with_complete_paths(self) -> None:
        # All three are known to have full L0/L1/L2 paths.
        validate_class_list(["junco", "great white shark", "Beagle"])

    def test_rejects_unknown_class(self) -> None:
        with pytest.raises(ValueError, match="unknown to taxonomy"):
            validate_class_list(["junco", "totally_made_up_xyz"])

    def test_aggregates_failures(self) -> None:
        # Two unknown names — message should mention both.
        with pytest.raises(ValueError) as exc_info:
            validate_class_list(["unknown_a_xyz", "unknown_b_xyz"])
        assert "unknown_a_xyz" in str(exc_info.value)
        assert "unknown_b_xyz" in str(exc_info.value)

    def test_empty_list_is_noop(self) -> None:
        # No classes → nothing to validate, no exception.
        validate_class_list([])
