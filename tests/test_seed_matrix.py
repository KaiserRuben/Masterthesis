"""Tests for :mod:`src.optimizer.seed_matrix` builders.

Pure NumPy behaviour only -- no PyMoo, no optimizer round-trip needed.
Every assertion is against exact array content or shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.common.seed_matrix import (
    build_fuzzy_onehot,
    build_pareto_init,
    build_precise_scan,
)


# ===========================================================================
# build_fuzzy_onehot
# ===========================================================================


class TestFuzzyOneHot:
    """Verify fuzzy one-hot seed-matrix construction."""

    # --- shape & content ---------------------------------------------------

    def test_single_depth_with_zero(self) -> None:
        n_genes = 4
        bounds = np.full(n_genes, 26, dtype=np.int64)
        depths = [25]

        matrix = build_fuzzy_onehot(n_genes, bounds, depths, include_zero=True)

        # 1 (zero) + 4 genes * 1 depth = 5 rows.
        assert matrix.shape == (5, 4)
        assert matrix.dtype == np.int64
        # First row is zero.
        np.testing.assert_array_equal(matrix[0], np.zeros(4, dtype=np.int64))
        # Each subsequent row is one-hot at depth 25.
        expected = np.eye(4, dtype=np.int64) * 25
        np.testing.assert_array_equal(matrix[1:], expected)

    def test_single_depth_without_zero(self) -> None:
        n_genes = 3
        bounds = np.full(n_genes, 10, dtype=np.int64)

        matrix = build_fuzzy_onehot(n_genes, bounds, [9], include_zero=False)

        assert matrix.shape == (3, 3)
        expected = np.eye(3, dtype=np.int64) * 9
        np.testing.assert_array_equal(matrix, expected)

    def test_multi_depth(self) -> None:
        n_genes = 3
        bounds = np.full(n_genes, 26, dtype=np.int64)
        depths = [12, 25]

        matrix = build_fuzzy_onehot(n_genes, bounds, depths, include_zero=True)

        # 1 + 3 genes * 2 depths = 7 rows.
        assert matrix.shape == (7, 3)
        # Row 0: zero.
        np.testing.assert_array_equal(matrix[0], np.zeros(3, dtype=np.int64))
        # Rows 1..3: depth 12, one-hot per gene.
        np.testing.assert_array_equal(matrix[1], [12, 0, 0])
        np.testing.assert_array_equal(matrix[2], [0, 12, 0])
        np.testing.assert_array_equal(matrix[3], [0, 0, 12])
        # Rows 4..6: depth 25, one-hot per gene.
        np.testing.assert_array_equal(matrix[4], [25, 0, 0])
        np.testing.assert_array_equal(matrix[5], [0, 25, 0])
        np.testing.assert_array_equal(matrix[6], [0, 0, 25])

    def test_multi_depth_without_zero(self) -> None:
        n_genes = 2
        bounds = np.full(n_genes, 5, dtype=np.int64)
        depths = [1, 4]

        matrix = build_fuzzy_onehot(n_genes, bounds, depths, include_zero=False)

        assert matrix.shape == (4, 2)
        np.testing.assert_array_equal(matrix[0], [1, 0])
        np.testing.assert_array_equal(matrix[1], [0, 1])
        np.testing.assert_array_equal(matrix[2], [4, 0])
        np.testing.assert_array_equal(matrix[3], [0, 4])

    # --- gene_mask ---------------------------------------------------------

    def test_gene_mask_restricts_genes(self) -> None:
        n_genes = 5
        bounds = np.full(n_genes, 10, dtype=np.int64)
        mask = np.array([True, False, True, False, True], dtype=np.bool_)

        matrix = build_fuzzy_onehot(
            n_genes, bounds, [9], include_zero=True, gene_mask=mask,
        )

        # 1 + 3 active * 1 depth = 4 rows.
        assert matrix.shape == (4, 5)
        np.testing.assert_array_equal(matrix[0], np.zeros(5, dtype=np.int64))
        np.testing.assert_array_equal(matrix[1], [9, 0, 0, 0, 0])
        np.testing.assert_array_equal(matrix[2], [0, 0, 9, 0, 0])
        np.testing.assert_array_equal(matrix[3], [0, 0, 0, 0, 9])

    def test_gene_mask_all_false(self) -> None:
        """No active genes -> only the zero row (or nothing)."""
        n_genes = 4
        bounds = np.full(n_genes, 10, dtype=np.int64)
        mask = np.zeros(n_genes, dtype=np.bool_)

        with_zero = build_fuzzy_onehot(
            n_genes, bounds, [9], include_zero=True, gene_mask=mask,
        )
        assert with_zero.shape == (1, n_genes)
        np.testing.assert_array_equal(with_zero[0], np.zeros(n_genes, dtype=np.int64))

        no_zero = build_fuzzy_onehot(
            n_genes, bounds, [9], include_zero=False, gene_mask=mask,
        )
        assert no_zero.shape == (0, n_genes)

    # --- clamping ----------------------------------------------------------

    def test_clamp_variable_bounds(self) -> None:
        """Depth exceeding gene bound -> row uses gene's max - 1, not silently dropped."""
        # Genes 0 & 1 have bound 5 (max=4); genes 2 & 3 have bound 26 (max=25).
        bounds = np.array([5, 5, 26, 26], dtype=np.int64)
        n_genes = 4
        depths = [25]  # Exceeds first two gene bounds.

        matrix = build_fuzzy_onehot(n_genes, bounds, depths, include_zero=False)

        assert matrix.shape == (4, 4)
        # Gene 0: clamped to 4.
        np.testing.assert_array_equal(matrix[0], [4, 0, 0, 0])
        # Gene 1: clamped to 4.
        np.testing.assert_array_equal(matrix[1], [0, 4, 0, 0])
        # Genes 2 & 3: unclamped at 25.
        np.testing.assert_array_equal(matrix[2], [0, 0, 25, 0])
        np.testing.assert_array_equal(matrix[3], [0, 0, 0, 25])

    def test_clamp_multi_depth_variable_bounds(self) -> None:
        """Multiple depths, some exceeding some bounds."""
        bounds = np.array([3, 10], dtype=np.int64)  # max = 2, 9
        depths = [5, 9]

        matrix = build_fuzzy_onehot(2, bounds, depths, include_zero=False)

        assert matrix.shape == (4, 2)
        # Depth=5: gene 0 clamped to 2, gene 1 unclamped at 5.
        np.testing.assert_array_equal(matrix[0], [2, 0])
        np.testing.assert_array_equal(matrix[1], [0, 5])
        # Depth=9: gene 0 clamped to 2, gene 1 at 9.
        np.testing.assert_array_equal(matrix[2], [2, 0])
        np.testing.assert_array_equal(matrix[3], [0, 9])

    def test_depth_zero_is_valid(self) -> None:
        """Depth=0 is legal even though it duplicates the all-zero row."""
        bounds = np.full(3, 10, dtype=np.int64)
        matrix = build_fuzzy_onehot(3, bounds, [0], include_zero=True)
        assert matrix.shape == (4, 3)
        np.testing.assert_array_equal(matrix, np.zeros((4, 3), dtype=np.int64))

    # --- output guarantees -------------------------------------------------

    def test_output_is_contiguous_int64(self) -> None:
        bounds = np.full(4, 10, dtype=np.int64)
        matrix = build_fuzzy_onehot(4, bounds, [9])
        assert matrix.dtype == np.int64
        assert matrix.flags["C_CONTIGUOUS"]

    def test_bounds_respected_for_all_rows(self) -> None:
        """No entry should ever equal or exceed its gene's bound."""
        bounds = np.array([3, 5, 8, 26], dtype=np.int64)
        matrix = build_fuzzy_onehot(4, bounds, [1, 4, 7, 25])
        # matrix[:, i] must be < bounds[i] for every i.
        for i, ub in enumerate(bounds):
            assert matrix[:, i].max() < ub

    # --- invalid input -----------------------------------------------------

    def test_rejects_wrong_n_genes(self) -> None:
        bounds = np.array([10, 10, 10], dtype=np.int64)
        with pytest.raises(ValueError, match="does not match"):
            build_fuzzy_onehot(4, bounds, [5])

    def test_rejects_non_integer_bounds(self) -> None:
        bounds = np.array([10.0, 10.0], dtype=np.float64)
        with pytest.raises(ValueError, match="integer dtype"):
            build_fuzzy_onehot(2, bounds, [5])

    def test_rejects_non_ndarray_bounds(self) -> None:
        with pytest.raises(ValueError, match="np.ndarray"):
            build_fuzzy_onehot(2, [10, 10], [5])  # type: ignore[arg-type]

    def test_rejects_2d_bounds(self) -> None:
        bounds = np.array([[10, 10], [10, 10]], dtype=np.int64)
        with pytest.raises(ValueError, match="1-D"):
            build_fuzzy_onehot(2, bounds, [5])

    def test_rejects_negative_depth(self) -> None:
        bounds = np.array([10, 10], dtype=np.int64)
        with pytest.raises(ValueError, match="non-negative"):
            build_fuzzy_onehot(2, bounds, [-1])

    def test_rejects_empty_depths(self) -> None:
        bounds = np.array([10, 10], dtype=np.int64)
        with pytest.raises(ValueError, match="at least one value"):
            build_fuzzy_onehot(2, bounds, [])

    def test_rejects_zero_bound(self) -> None:
        bounds = np.array([10, 0, 10], dtype=np.int64)
        with pytest.raises(ValueError, match=">= 1"):
            build_fuzzy_onehot(3, bounds, [5])

    def test_rejects_wrong_mask_length(self) -> None:
        bounds = np.array([10, 10, 10], dtype=np.int64)
        mask = np.array([True, False], dtype=np.bool_)
        with pytest.raises(ValueError, match="does not match"):
            build_fuzzy_onehot(3, bounds, [5], gene_mask=mask)

    def test_rejects_non_bool_mask(self) -> None:
        bounds = np.array([10, 10, 10], dtype=np.int64)
        mask = np.array([1, 0, 1], dtype=np.int64)
        with pytest.raises(ValueError, match="dtype bool"):
            build_fuzzy_onehot(3, bounds, [5], gene_mask=mask)  # type: ignore[arg-type]

    def test_rejects_non_positive_n_genes(self) -> None:
        bounds = np.array([10], dtype=np.int64)
        with pytest.raises(ValueError, match="positive int"):
            build_fuzzy_onehot(0, bounds, [5])


# ===========================================================================
# build_precise_scan
# ===========================================================================


class TestPreciseScan:
    """Verify Stage-2 precise-scan seed-matrix construction."""

    def test_basic_shape_and_content(self) -> None:
        bounds = np.full(5, 26, dtype=np.int64)
        awake = np.array([True, False, True, False, False], dtype=np.bool_)
        depths = [5, 15, 25]

        matrix = build_precise_scan(awake, bounds, depths)

        # sum(awake)=2; 2 * 3 depths = 6 rows; no zero row.
        assert matrix.shape == (6, 5)
        # No all-zero row should be present.
        assert not np.any((matrix == 0).all(axis=1))

        # Every row has exactly one non-zero entry at an awake gene.
        awake_indices = {0, 2}
        for row in matrix:
            nonzero = np.flatnonzero(row)
            assert nonzero.size == 1
            assert int(nonzero[0]) in awake_indices

    def test_depth_ordering(self) -> None:
        """Rows are ordered depth-major: all genes at depth 0, then depth 1, ..."""
        bounds = np.full(3, 20, dtype=np.int64)
        awake = np.array([True, True, True], dtype=np.bool_)
        depths = [1, 9]

        matrix = build_precise_scan(awake, bounds, depths)

        assert matrix.shape == (6, 3)
        # Depth 1 block.
        np.testing.assert_array_equal(matrix[0], [1, 0, 0])
        np.testing.assert_array_equal(matrix[1], [0, 1, 0])
        np.testing.assert_array_equal(matrix[2], [0, 0, 1])
        # Depth 9 block.
        np.testing.assert_array_equal(matrix[3], [9, 0, 0])
        np.testing.assert_array_equal(matrix[4], [0, 9, 0])
        np.testing.assert_array_equal(matrix[5], [0, 0, 9])

    def test_no_awake_genes_returns_empty(self) -> None:
        bounds = np.full(4, 10, dtype=np.int64)
        awake = np.zeros(4, dtype=np.bool_)
        matrix = build_precise_scan(awake, bounds, [5])
        assert matrix.shape == (0, 4)
        assert matrix.dtype == np.int64

    def test_clamping_propagates(self) -> None:
        """Clamping rules from build_fuzzy_onehot apply here too."""
        bounds = np.array([3, 10], dtype=np.int64)  # max = 2, 9
        awake = np.array([True, True], dtype=np.bool_)
        matrix = build_precise_scan(awake, bounds, [9])

        # Gene 0 (awake, bound 3): clamped to 2.
        np.testing.assert_array_equal(matrix[0], [2, 0])
        np.testing.assert_array_equal(matrix[1], [0, 9])

    def test_output_is_contiguous_int64(self) -> None:
        bounds = np.full(3, 10, dtype=np.int64)
        awake = np.array([True, False, True], dtype=np.bool_)
        matrix = build_precise_scan(awake, bounds, [5])
        assert matrix.dtype == np.int64
        assert matrix.flags["C_CONTIGUOUS"]

    def test_rejects_mismatched_shapes(self) -> None:
        bounds = np.array([10, 10, 10], dtype=np.int64)
        awake = np.array([True, False], dtype=np.bool_)
        with pytest.raises(ValueError, match="does not match"):
            build_precise_scan(awake, bounds, [5])

    def test_rejects_non_bool_mask(self) -> None:
        bounds = np.array([10, 10], dtype=np.int64)
        awake = np.array([1, 0], dtype=np.int64)
        with pytest.raises(ValueError, match="dtype bool"):
            build_precise_scan(awake, bounds, [5])  # type: ignore[arg-type]


# ===========================================================================
# build_pareto_init
# ===========================================================================


class TestParetoInit:
    """Verify Stage-3 Pareto-seeded population construction."""

    def test_pop_size_lt_n_pareto(self) -> None:
        rng = np.random.default_rng(0)
        bounds = np.full(4, 10, dtype=np.int64)
        pareto = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 9, 0]], dtype=np.int64,
        )
        out = build_pareto_init(pareto, pop_size=2, gene_bounds=bounds, rng=rng)

        assert out.shape == (2, 4)
        # Exactly the first two Pareto rows, unchanged.
        np.testing.assert_array_equal(out, pareto[:2])

    def test_pop_size_eq_n_pareto(self) -> None:
        rng = np.random.default_rng(0)
        bounds = np.full(3, 10, dtype=np.int64)
        pareto = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        out = build_pareto_init(pareto, pop_size=2, gene_bounds=bounds, rng=rng)

        assert out.shape == (2, 3)
        np.testing.assert_array_equal(out, pareto)

    def test_pop_size_gt_n_pareto_preserves_originals(self) -> None:
        """The first n_pareto rows are exact copies of the input Pareto front."""
        rng = np.random.default_rng(42)
        bounds = np.full(5, 10, dtype=np.int64)
        pareto = np.array(
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]], dtype=np.int64,
        )
        out = build_pareto_init(
            pareto,
            pop_size=6,
            gene_bounds=bounds,
            perturbation_prob=0.5,
            rng=rng,
        )
        assert out.shape == (6, 5)
        np.testing.assert_array_equal(out[:2], pareto)

    def test_pop_size_gt_n_pareto_extras_within_bounds(self) -> None:
        rng = np.random.default_rng(0)
        bounds = np.array([3, 5, 8, 26], dtype=np.int64)
        pareto = np.array([[0, 0, 0, 0]], dtype=np.int64)
        out = build_pareto_init(
            pareto,
            pop_size=50,
            gene_bounds=bounds,
            perturbation_prob=1.0,  # force maximum perturbation
            rng=rng,
        )

        assert out.shape == (50, 4)
        # Every gene must obey its bound.
        for i, ub in enumerate(bounds):
            assert out[:, i].min() >= 0
            assert out[:, i].max() < ub

    def test_zero_perturbation_yields_cycled_copies(self) -> None:
        """With perturbation_prob=0, extras are exact cyclic repeats."""
        rng = np.random.default_rng(0)
        bounds = np.full(3, 10, dtype=np.int64)
        pareto = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)

        out = build_pareto_init(
            pareto,
            pop_size=5,
            gene_bounds=bounds,
            perturbation_prob=0.0,
            rng=rng,
        )
        assert out.shape == (5, 3)
        np.testing.assert_array_equal(out[0], [1, 2, 3])
        np.testing.assert_array_equal(out[1], [4, 5, 6])
        # Cyclic fill: 0, 1, 0 for rows 2, 3, 4.
        np.testing.assert_array_equal(out[2], [1, 2, 3])
        np.testing.assert_array_equal(out[3], [4, 5, 6])
        np.testing.assert_array_equal(out[4], [1, 2, 3])

    def test_full_perturbation_makes_extras_differ(self) -> None:
        """With perturbation_prob=1.0 and a wide bound, extras almost surely differ."""
        rng = np.random.default_rng(7)
        bounds = np.full(10, 100, dtype=np.int64)
        pareto = np.array([[0] * 10], dtype=np.int64)

        out = build_pareto_init(
            pareto,
            pop_size=20,
            gene_bounds=bounds,
            perturbation_prob=1.0,
            rng=rng,
        )
        # Preserve original.
        np.testing.assert_array_equal(out[0], np.zeros(10, dtype=np.int64))
        # At least one extra row is non-zero (vanishingly unlikely to be all-zero).
        n_nonzero_rows = int(np.sum(np.any(out[1:] != 0, axis=1)))
        assert n_nonzero_rows >= 18  # allow a tiny slack for randomness

    def test_deterministic_with_seeded_rng(self) -> None:
        bounds = np.full(4, 10, dtype=np.int64)
        pareto = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int64)

        out_a = build_pareto_init(
            pareto, pop_size=10, gene_bounds=bounds,
            perturbation_prob=0.3, rng=np.random.default_rng(123),
        )
        out_b = build_pareto_init(
            pareto, pop_size=10, gene_bounds=bounds,
            perturbation_prob=0.3, rng=np.random.default_rng(123),
        )
        np.testing.assert_array_equal(out_a, out_b)

    def test_output_is_contiguous_int64(self) -> None:
        rng = np.random.default_rng(0)
        bounds = np.full(3, 10, dtype=np.int64)
        pareto = np.array([[1, 2, 3]], dtype=np.int64)
        out = build_pareto_init(pareto, 5, bounds, rng=rng)
        assert out.dtype == np.int64
        assert out.flags["C_CONTIGUOUS"]

    # --- invalid input -----------------------------------------------------

    def test_rejects_non_ndarray_pareto(self) -> None:
        bounds = np.array([10, 10], dtype=np.int64)
        with pytest.raises(ValueError, match="np.ndarray"):
            build_pareto_init([[1, 2]], pop_size=3, gene_bounds=bounds)  # type: ignore[arg-type]

    def test_rejects_1d_pareto(self) -> None:
        bounds = np.array([10, 10], dtype=np.int64)
        pareto = np.array([1, 2], dtype=np.int64)
        with pytest.raises(ValueError, match="2-D"):
            build_pareto_init(pareto, pop_size=3, gene_bounds=bounds)

    def test_rejects_float_pareto(self) -> None:
        bounds = np.array([10, 10], dtype=np.int64)
        pareto = np.array([[1.0, 2.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="integer dtype"):
            build_pareto_init(pareto, pop_size=3, gene_bounds=bounds)

    def test_rejects_empty_pareto(self) -> None:
        bounds = np.array([10, 10], dtype=np.int64)
        pareto = np.zeros((0, 2), dtype=np.int64)
        with pytest.raises(ValueError, match="at least one row"):
            build_pareto_init(pareto, pop_size=3, gene_bounds=bounds)

    def test_rejects_mismatched_gene_dims(self) -> None:
        bounds = np.array([10, 10, 10], dtype=np.int64)
        pareto = np.array([[1, 2]], dtype=np.int64)
        with pytest.raises(ValueError, match="genes but gene_bounds"):
            build_pareto_init(pareto, pop_size=3, gene_bounds=bounds)

    def test_rejects_non_positive_pop_size(self) -> None:
        bounds = np.array([10, 10], dtype=np.int64)
        pareto = np.array([[1, 2]], dtype=np.int64)
        with pytest.raises(ValueError, match="positive int"):
            build_pareto_init(pareto, pop_size=0, gene_bounds=bounds)

    def test_rejects_invalid_perturbation_prob(self) -> None:
        bounds = np.array([10, 10], dtype=np.int64)
        pareto = np.array([[1, 2]], dtype=np.int64)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            build_pareto_init(
                pareto, pop_size=3, gene_bounds=bounds, perturbation_prob=1.5,
            )
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            build_pareto_init(
                pareto, pop_size=3, gene_bounds=bounds, perturbation_prob=-0.1,
            )

    def test_rejects_pareto_exceeding_bounds(self) -> None:
        bounds = np.array([5, 5], dtype=np.int64)
        pareto = np.array([[1, 6]], dtype=np.int64)
        with pytest.raises(ValueError, match=">= gene_bounds"):
            build_pareto_init(pareto, pop_size=3, gene_bounds=bounds)

    def test_rejects_negative_pareto(self) -> None:
        bounds = np.array([5, 5], dtype=np.int64)
        pareto = np.array([[-1, 2]], dtype=np.int64)
        with pytest.raises(ValueError, match="negative"):
            build_pareto_init(pareto, pop_size=3, gene_bounds=bounds)
