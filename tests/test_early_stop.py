"""Tests for :mod:`src.optimizer.early_stop`.

All tests run the real :class:`EarlyStopChecker` against synthetic
Pareto fronts — no mocks. Pymoo's HV indicator is used for the
plateau trigger; if pymoo is unavailable the plateau test is skipped.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.optimizer.early_stop import (
    EarlyStopChecker,
    EarlyStopConfig,
    EarlyStopTrigger,
)

try:  # pragma: no cover
    from pymoo.indicators.hv import HV  # noqa: F401

    _HAS_PYMOO_HV = True
except ImportError:  # pragma: no cover
    _HAS_PYMOO_HV = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# TgtBal is column 0 in every test below.
TGTBAL_IDX = 0


def _front(
    tgtbal_values: list[float],
    other_values: list[float] | None = None,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Build a 2-objective Pareto front array of shape (n, 2).

    :param tgtbal_values: TgtBal values (column 0).
    :param other_values: Second-objective values; defaults to ``0.5``
        for every row so the HV contribution is stable across tests.
    :param dtype: Numpy dtype for the returned array.
    """
    n = len(tgtbal_values)
    if other_values is None:
        other_values = [0.5] * n
    assert len(other_values) == n
    return np.asarray(
        np.column_stack([tgtbal_values, other_values]), dtype=dtype,
    )


def _default_config(**overrides) -> EarlyStopConfig:
    """Build a config with safe defaults for focused testing."""
    base = dict(
        epsilon_margin=1e-30,
        plateau_patience=30,
        no_improvement_warmup=20,
        hypervolume_reference=None,
        max_generations=200,
    )
    base.update(overrides)
    return EarlyStopConfig(**base)


# ===========================================================================
# Flip trigger
# ===========================================================================


class TestFlipTrigger:
    """The flip trigger fires when pareto_min_TgtBal <= dtype.tiny + margin."""

    def test_flip_fires_below_epsilon(self) -> None:
        checker = EarlyStopChecker(_default_config())
        # FP32 tiny is ~1.17e-38; with margin 1e-30, epsilon is ~1e-30.
        # A value well below it must trigger.
        front = _front([1e-40], dtype=np.float32)
        result = checker.update(0, front, TGTBAL_IDX)
        assert isinstance(result, EarlyStopTrigger)
        assert result.trigger == "flip"
        assert result.generation == 0
        assert "pareto_min_TgtBal" in result.details
        assert "epsilon" in result.details
        assert result.details["epsilon"] == pytest.approx(
            np.finfo(np.float32).tiny + 1e-30,
        )

    def test_flip_at_boundary(self) -> None:
        """Exactly at epsilon is considered a flip (<= semantics)."""
        checker = EarlyStopChecker(_default_config())
        eps = float(np.finfo(np.float32).tiny) + 1e-30
        front = _front([eps], dtype=np.float32)
        result = checker.update(0, front, TGTBAL_IDX)
        assert result is not None and result.trigger == "flip"

    def test_flip_does_not_fire_above_epsilon(self) -> None:
        """Realistic TgtBal values (e.g. 0.2) must not trigger the flip."""
        checker = EarlyStopChecker(_default_config())
        front = _front([0.2, 0.3], dtype=np.float32)
        assert checker.update(0, front, TGTBAL_IDX) is None

    def test_flip_epsilon_tied_to_dtype(self) -> None:
        """FP64 tiny is smaller than FP32 tiny — epsilon follows dtype."""
        # Isolate dtype.tiny by zeroing the margin, then feed a value
        # exactly at FP64 tiny. FP32 tiny (~1.17e-38) >> FP64 tiny
        # (~2.2e-308), so this also verifies the epsilon is pulled
        # from the input dtype rather than a hard-coded FP32 default.
        checker = EarlyStopChecker(_default_config(epsilon_margin=0.0))
        tiny64 = float(np.finfo(np.float64).tiny)
        front64 = _front([tiny64], dtype=np.float64)
        result = checker.update(0, front64, TGTBAL_IDX)
        assert result is not None and result.trigger == "flip"
        assert result.details["epsilon"] == pytest.approx(tiny64)
        assert result.details["pareto_min_TgtBal"] == pytest.approx(tiny64)

    def test_flip_epsilon_respects_fp32_over_fp64(self) -> None:
        """A value just above FP32 tiny must NOT trigger on an FP32 front."""
        checker = EarlyStopChecker(_default_config(epsilon_margin=0.0))
        # Slightly above FP32 tiny — representable, and above the
        # dtype-tied threshold.
        v = float(np.finfo(np.float32).tiny) * 2.0
        front = _front([v], dtype=np.float32)
        assert checker.update(0, front, TGTBAL_IDX) is None


# ===========================================================================
# Plateau trigger
# ===========================================================================


@pytest.mark.skipif(not _HAS_PYMOO_HV, reason="pymoo.indicators.hv unavailable")
class TestPlateauTrigger:
    """Plateau fires after K gens of no HV improvement.

    The HV reference point is fixed at (1.0, 1.0) for a 2D
    minimization problem, so any Pareto point that strictly dominates
    (1,1) contributes positive volume.
    """

    def test_plateau_fires_after_patience(self) -> None:
        config = _default_config(
            plateau_patience=3,
            hypervolume_reference=(1.0, 1.0),
            no_improvement_warmup=10_000,  # disable no-improvement
            max_generations=10_000,
        )
        checker = EarlyStopChecker(config)

        # Generation 0: establishes a positive HV (sets the record).
        assert (
            checker.update(0, _front([0.2], [0.2]), TGTBAL_IDX) is None
        )
        # Generations 1..3: identical fronts → no HV improvement.
        assert (
            checker.update(1, _front([0.2], [0.2]), TGTBAL_IDX) is None
        )
        assert (
            checker.update(2, _front([0.2], [0.2]), TGTBAL_IDX) is None
        )
        # Third stall generation → plateau_counter reaches 3.
        result = checker.update(3, _front([0.2], [0.2]), TGTBAL_IDX)
        assert result is not None
        assert result.trigger == "plateau"
        assert result.details["plateau_counter"] == 3.0

    def test_plateau_resets_on_hv_improvement(self) -> None:
        config = _default_config(
            plateau_patience=2,
            hypervolume_reference=(1.0, 1.0),
            no_improvement_warmup=10_000,
            max_generations=10_000,
        )
        checker = EarlyStopChecker(config)

        assert checker.update(0, _front([0.5], [0.5]), TGTBAL_IDX) is None
        assert checker.update(1, _front([0.5], [0.5]), TGTBAL_IDX) is None
        # Improve HV — counter must reset.
        assert checker.update(2, _front([0.1], [0.1]), TGTBAL_IDX) is None
        # Now stall again: need 2 more before firing.
        assert checker.update(3, _front([0.1], [0.1]), TGTBAL_IDX) is None
        result = checker.update(4, _front([0.1], [0.1]), TGTBAL_IDX)
        assert result is not None and result.trigger == "plateau"

    def test_plateau_disabled_without_reference(self) -> None:
        """No reference point → trigger is skipped entirely."""
        config = _default_config(
            plateau_patience=2,
            hypervolume_reference=None,
            no_improvement_warmup=10_000,
            max_generations=10_000,
        )
        checker = EarlyStopChecker(config)

        for g in range(20):
            assert (
                checker.update(g, _front([0.2], [0.2]), TGTBAL_IDX) is None
            )


# ===========================================================================
# No-improvement trigger
# ===========================================================================


class TestNoImprovementTrigger:
    """Trigger fires when TgtBal at gen g fails to beat gen 0 after warmup."""

    def test_warmup_suppresses_trigger(self) -> None:
        config = _default_config(
            no_improvement_warmup=10,
            max_generations=10_000,
        )
        checker = EarlyStopChecker(config)

        # Gen 0 establishes the baseline (0.3).
        assert checker.update(0, _front([0.3]), TGTBAL_IDX) is None
        # During warmup: no-improvement must never fire, even with a
        # stagnant or worse TgtBal.
        for g in range(1, 10):
            assert checker.update(g, _front([0.3]), TGTBAL_IDX) is None

    def test_fires_after_warmup_when_no_improvement(self) -> None:
        config = _default_config(
            no_improvement_warmup=5,
            max_generations=10_000,
        )
        checker = EarlyStopChecker(config)

        assert checker.update(0, _front([0.3]), TGTBAL_IDX) is None
        for g in range(1, 5):
            assert checker.update(g, _front([0.3]), TGTBAL_IDX) is None
        # Gen 5 is the first post-warmup generation.
        result = checker.update(5, _front([0.3]), TGTBAL_IDX)
        assert result is not None
        assert result.trigger == "no_improvement"
        assert result.details["pareto_min_TgtBal_gen0"] == pytest.approx(0.3)
        assert result.details["pareto_min_TgtBal_current"] == pytest.approx(
            0.3,
        )

    def test_does_not_fire_when_improved(self) -> None:
        config = _default_config(
            no_improvement_warmup=3,
            max_generations=10_000,
        )
        checker = EarlyStopChecker(config)

        assert checker.update(0, _front([0.5]), TGTBAL_IDX) is None
        for g in range(1, 3):
            assert checker.update(g, _front([0.5]), TGTBAL_IDX) is None
        # Strict improvement — trigger must NOT fire.
        assert checker.update(3, _front([0.4]), TGTBAL_IDX) is None
        assert checker.update(4, _front([0.35]), TGTBAL_IDX) is None


# ===========================================================================
# Hard-cap trigger
# ===========================================================================


class TestHardCap:
    """Safety-net trigger that always fires at max_generations."""

    def test_hard_cap_fires_at_max(self) -> None:
        config = _default_config(
            no_improvement_warmup=10_000,  # disable other triggers
            max_generations=5,
            hypervolume_reference=None,
        )
        checker = EarlyStopChecker(config)

        for g in range(4):
            assert checker.update(g, _front([0.3]), TGTBAL_IDX) is None
        # generation index 4 equals max_generations - 1 — the last
        # allowed generation must report the cap.
        result = checker.update(4, _front([0.3]), TGTBAL_IDX)
        assert result is not None
        assert result.trigger == "hard_cap"
        assert result.details["generation"] == 4.0
        assert result.details["max_generations"] == 5.0

    def test_hard_cap_fires_on_empty_front(self) -> None:
        """Safety net works even if every generation had an empty front."""
        config = _default_config(
            max_generations=3,
            no_improvement_warmup=10_000,
        )
        checker = EarlyStopChecker(config)
        empty = np.empty((0, 2), dtype=np.float32)
        assert checker.update(0, empty, TGTBAL_IDX) is None
        assert checker.update(1, empty, TGTBAL_IDX) is None
        result = checker.update(2, empty, TGTBAL_IDX)
        assert result is not None and result.trigger == "hard_cap"


# ===========================================================================
# Robustness
# ===========================================================================


class TestEmptyFront:
    """Empty Pareto fronts must not crash and must not fire triggers 1-3."""

    def test_empty_front_returns_none(self) -> None:
        config = _default_config(
            no_improvement_warmup=0,
            hypervolume_reference=(1.0, 1.0),
            max_generations=10_000,
        )
        checker = EarlyStopChecker(config)
        empty = np.empty((0, 2), dtype=np.float32)
        assert checker.update(0, empty, TGTBAL_IDX) is None

    def test_empty_front_recorded_in_history(self) -> None:
        config = _default_config(max_generations=10_000)
        checker = EarlyStopChecker(config)
        empty = np.empty((0, 2), dtype=np.float32)
        checker.update(0, empty, TGTBAL_IDX)
        checker.update(1, _front([0.3]), TGTBAL_IDX)

        hist = checker.history
        assert len(hist["gen"]) == 2
        assert np.isnan(hist["pareto_min_tgtbal"][0])
        assert hist["pareto_min_tgtbal"][1] == pytest.approx(0.3)

    def test_rejects_non_2d_input(self) -> None:
        checker = EarlyStopChecker(_default_config())
        with pytest.raises(ValueError, match="2D"):
            checker.update(0, np.zeros((3,)), TGTBAL_IDX)

    def test_rejects_out_of_range_tgtbal_index(self) -> None:
        checker = EarlyStopChecker(_default_config())
        with pytest.raises(ValueError, match="out of range"):
            checker.update(0, _front([0.3]), tgtbal_index=5)


# ===========================================================================
# History property
# ===========================================================================


class TestHistory:
    """Parallel, consistent-length history arrays."""

    def test_history_lengths_consistent(self) -> None:
        config = _default_config(
            hypervolume_reference=(1.0, 1.0),
            no_improvement_warmup=10_000,
            max_generations=10_000,
        )
        checker = EarlyStopChecker(config)

        for g in range(7):
            checker.update(g, _front([0.3 - 0.01 * g], [0.3]), TGTBAL_IDX)

        hist = checker.history
        assert set(hist.keys()) == {"gen", "pareto_min_tgtbal", "hypervolume"}
        lengths = {len(v) for v in hist.values()}
        assert lengths == {7}

    def test_history_returns_new_lists(self) -> None:
        """Mutating the returned dict must not corrupt internal state."""
        checker = EarlyStopChecker(_default_config(max_generations=10_000))
        checker.update(0, _front([0.3]), TGTBAL_IDX)

        hist = checker.history
        hist["gen"].append(9999)
        hist["pareto_min_tgtbal"].append(9999.0)

        # A fresh read must still match the internal state.
        hist2 = checker.history
        assert hist2["gen"] == [0.0]
        assert len(hist2["pareto_min_tgtbal"]) == 1


# ===========================================================================
# Trigger-precedence smoke check
# ===========================================================================


class TestTriggerPrecedence:
    """Flip wins over plateau/no-improvement when multiple conditions hold."""

    def test_flip_wins(self) -> None:
        config = _default_config(
            no_improvement_warmup=0,   # would fire immediately
            plateau_patience=0,        # would fire immediately
            hypervolume_reference=(1.0, 1.0),
            max_generations=10_000,
        )
        checker = EarlyStopChecker(config)

        # Values at/below epsilon on the first call — flip must win
        # even though no-improvement/plateau could also qualify.
        result = checker.update(
            0,
            _front([0.0], [0.5], dtype=np.float32),
            TGTBAL_IDX,
        )
        assert result is not None
        assert result.trigger == "flip"
