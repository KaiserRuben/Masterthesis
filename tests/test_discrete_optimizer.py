"""Tests for DiscretePymooOptimizer.

All tests use real PyMoo optimization -- no mocks.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.optimizer.discrete_pymoo_optimizer import DiscretePymooOptimizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_optimizer(
    gene_bounds: NDArray | list[int],
    num_objectives: int = 2,
    pop_size: int = 20,
) -> DiscretePymooOptimizer:
    """Shorthand factory for tests."""
    return DiscretePymooOptimizer(
        gene_bounds=np.asarray(gene_bounds, dtype=np.int64),
        num_objectives=num_objectives,
        pop_size=pop_size,
    )


def _random_fitness(pop_size: int, num_objectives: int) -> tuple[NDArray, ...]:
    """Generate random fitness arrays (one per objective)."""
    return tuple(np.random.rand(pop_size) for _ in range(num_objectives))


def _assign_and_update(opt: DiscretePymooOptimizer, num_objectives: int) -> None:
    """Run one assign_fitness + update cycle."""
    pop = opt.get_x_current()
    fitness = _random_fitness(pop.shape[0], num_objectives)
    opt.assign_fitness(fitness)
    opt.update()


# ===================================================================
# TestConstruction
# ===================================================================


class TestConstruction:
    """Verify that initial construction produces valid state."""

    def test_gene_bounds_stored(self) -> None:
        bounds = np.array([3, 10, 26, 5], dtype=np.int64)
        opt = _make_optimizer(bounds)
        np.testing.assert_array_equal(opt.gene_bounds, bounds)

    def test_n_var_matches(self) -> None:
        bounds = np.array([3, 10, 26, 5], dtype=np.int64)
        opt = _make_optimizer(bounds)
        assert opt.n_var == len(bounds)

    def test_initial_population_shape(self) -> None:
        bounds = np.array([3, 10, 26, 5], dtype=np.int64)
        pop_size = 20
        opt = _make_optimizer(bounds, pop_size=pop_size)
        pop = opt.get_x_current()
        assert pop.shape == (pop_size, len(bounds))

    def test_initial_population_integer(self) -> None:
        bounds = np.array([3, 10, 26, 5], dtype=np.int64)
        opt = _make_optimizer(bounds)
        pop = opt.get_x_current()
        np.testing.assert_array_equal(pop, pop.astype(np.int64))

    def test_initial_population_within_bounds(self) -> None:
        bounds = np.array([3, 10, 26, 5], dtype=np.int64)
        opt = _make_optimizer(bounds)
        pop = opt.get_x_current()
        assert (pop >= 0).all(), "Some genes are negative"
        for i, ub in enumerate(bounds):
            assert (pop[:, i] < ub).all(), (
                f"Gene {i} exceeds bound {ub}: max={pop[:, i].max()}"
            )

    def test_uniform_bounds(self) -> None:
        bounds = np.full(8, 10, dtype=np.int64)
        opt = _make_optimizer(bounds)
        pop = opt.get_x_current()
        assert (pop >= 0).all()
        assert (pop < 10).all()

    def test_variable_bounds(self) -> None:
        bounds = np.array([3, 10, 26, 5], dtype=np.int64)
        opt = _make_optimizer(bounds, pop_size=50)
        pop = opt.get_x_current()
        for i, ub in enumerate(bounds):
            assert pop[:, i].min() >= 0
            assert pop[:, i].max() < ub


# ===================================================================
# TestOptimizationLoop
# ===================================================================


class TestOptimizationLoop:
    """Verify that the optimization loop maintains integer constraints."""

    def test_update_produces_new_population(self) -> None:
        bounds = np.array([10, 20, 30], dtype=np.int64)
        opt = _make_optimizer(bounds, pop_size=20)
        pop_before = opt.get_x_current().copy()
        _assign_and_update(opt, num_objectives=2)
        pop_after = opt.get_x_current()
        # Populations should differ (extremely unlikely to be identical)
        assert not np.array_equal(pop_before, pop_after)

    def test_population_stays_integer_after_update(self) -> None:
        bounds = np.array([5, 15, 25, 8], dtype=np.int64)
        opt = _make_optimizer(bounds)
        _assign_and_update(opt, num_objectives=2)
        pop = opt.get_x_current()
        np.testing.assert_array_equal(pop, pop.astype(np.int64))

    def test_population_stays_within_bounds_after_update(self) -> None:
        bounds = np.array([5, 15, 25, 8], dtype=np.int64)
        opt = _make_optimizer(bounds)
        _assign_and_update(opt, num_objectives=2)
        pop = opt.get_x_current()
        assert (pop >= 0).all()
        for i, ub in enumerate(bounds):
            assert (pop[:, i] < ub).all(), (
                f"Gene {i} exceeds bound {ub} after update: max={pop[:, i].max()}"
            )

    def test_multiple_generations(self) -> None:
        bounds = np.array([3, 10, 26, 5, 12], dtype=np.int64)
        opt = _make_optimizer(bounds, pop_size=30)
        for gen in range(10):
            pop = opt.get_x_current()
            # Integer check
            np.testing.assert_array_equal(
                pop, pop.astype(np.int64), err_msg=f"Non-integer at gen {gen}"
            )
            # Bounds check
            assert (pop >= 0).all(), f"Negative value at gen {gen}"
            for i, ub in enumerate(bounds):
                assert (pop[:, i] < ub).all(), (
                    f"Gene {i} exceeds bound {ub} at gen {gen}"
                )
            _assign_and_update(opt, num_objectives=2)

    def test_fitness_assignment(self) -> None:
        bounds = np.array([10, 20], dtype=np.int64)
        opt = _make_optimizer(bounds, num_objectives=2, pop_size=20)
        pop = opt.get_x_current()
        fitness = (
            np.random.rand(pop.shape[0]),
            np.random.rand(pop.shape[0]),
        )
        opt.assign_fitness(fitness)
        # After assignment, best_candidates should be updated.
        candidates = opt.best_candidates
        assert len(candidates) >= 1
        for c in candidates:
            assert c.solution is not None
            assert len(c.fitness) == 2


# ===================================================================
# TestMinimization
# ===================================================================


class TestMinimization:
    """Verify that the optimizer actually improves over random init."""

    def test_simple_minimization(self) -> None:
        """Two objectives: minimize gene-sum and deviation from a target.

        After ~50 generations the Pareto front should dominate the
        initial random fitness values.
        """
        rng = np.random.default_rng(42)
        bounds = np.array([20, 20, 20, 20], dtype=np.int64)
        target = np.array([10, 10, 10, 10], dtype=np.int64)
        num_obj = 2
        opt = _make_optimizer(bounds, num_objectives=num_obj, pop_size=40)

        def evaluate(pop: NDArray) -> tuple[NDArray, NDArray]:
            obj1 = pop.sum(axis=1).astype(np.float64)  # minimize sum
            obj2 = np.abs(pop - target).sum(axis=1).astype(np.float64)  # minimize deviation
            return obj1, obj2

        # Record initial fitness.
        pop_init = opt.get_x_current()
        f1_init, f2_init = evaluate(pop_init)
        initial_min_f1 = f1_init.min()
        initial_min_f2 = f2_init.min()

        # Run optimization.
        for _ in range(50):
            pop = opt.get_x_current()
            f1, f2 = evaluate(pop)
            opt.assign_fitness((f1, f2))
            opt.update()

        # The Pareto front should show improvement on at least one objective.
        best = opt.best_candidates
        best_f1 = min(c.fitness[0] for c in best)
        best_f2 = min(c.fitness[1] for c in best)

        # At least one objective should be better than initial random.
        improved = (best_f1 < initial_min_f1) or (best_f2 < initial_min_f2)
        assert improved, (
            f"No improvement: initial ({initial_min_f1:.1f}, {initial_min_f2:.1f}) "
            f"vs best ({best_f1:.1f}, {best_f2:.1f})"
        )


# ===================================================================
# TestUpdateBounds
# ===================================================================


class TestUpdateBounds:
    """Verify dynamic bound updates."""

    def test_update_gene_bounds(self) -> None:
        bounds_old = np.array([10, 20, 30], dtype=np.int64)
        bounds_new = np.array([5, 8, 12], dtype=np.int64)
        opt = _make_optimizer(bounds_old)
        opt.update_gene_bounds(bounds_new)

        np.testing.assert_array_equal(opt.gene_bounds, bounds_new)
        pop = opt.get_x_current()
        assert (pop >= 0).all()
        for i, ub in enumerate(bounds_new):
            assert (pop[:, i] < ub).all()

    def test_update_gene_bounds_changes_dimension(self) -> None:
        bounds_old = np.array([10, 20, 30], dtype=np.int64)
        bounds_new = np.array([5, 8], dtype=np.int64)
        opt = _make_optimizer(bounds_old)
        opt.update_gene_bounds(bounds_new)

        assert opt.n_var == 2
        pop = opt.get_x_current()
        assert pop.shape[1] == 2
        assert (pop >= 0).all()
        for i, ub in enumerate(bounds_new):
            assert (pop[:, i] < ub).all()


# ===================================================================
# TestReset
# ===================================================================


class TestReset:
    """Verify reset behaviour."""

    def test_reset_produces_new_population(self) -> None:
        bounds = np.array([10, 20, 30], dtype=np.int64)
        opt = _make_optimizer(bounds, pop_size=20)
        pop_before = opt.get_x_current().copy()
        opt.reset()
        pop_after = opt.get_x_current()
        # Shape and bounds must still hold.
        assert pop_after.shape == pop_before.shape
        assert (pop_after >= 0).all()
        for i, ub in enumerate(bounds):
            assert (pop_after[:, i] < ub).all()
        # Content should differ (stochastic -- extremely unlikely to match).
        assert not np.array_equal(pop_before, pop_after)
