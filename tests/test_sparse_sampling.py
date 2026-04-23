"""Tests for :mod:`src.optimizer.sparse_sampling`.

Sparse init-population sampler for the SMOO boundary optimizer. Tests
exercise the pure NumPy behaviour of ``SparseSampling._do``; no PyMoo
optimizer round-trip needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from src.optimizer.sparse_sampling import SparseSampling


# ---------------------------------------------------------------------------
# Test fixture — minimal Problem stand-in (n_var + xu)
# ---------------------------------------------------------------------------


@dataclass
class _FakeProblem:
    n_var: int
    xu: np.ndarray


def make_problem(n_image: int, text_dim: int, image_bound: int = 25,
                 text_bound: int = 24) -> _FakeProblem:
    xu = np.concatenate([
        np.full(n_image, image_bound, dtype=np.int64),
        np.full(text_dim, text_bound, dtype=np.int64),
    ])
    return _FakeProblem(n_var=n_image + text_dim, xu=xu)


# ===========================================================================
# Parameter validation
# ===========================================================================


class TestConstructor:
    """Parameter bounds-checking."""

    def test_valid_defaults(self) -> None:
        s = SparseSampling(text_dim=3)
        assert s.text_dim == 3
        assert s.p_active == 0.03
        assert s.geometric_rate == 0.5

    def test_negative_text_dim(self) -> None:
        with pytest.raises(ValueError, match="text_dim"):
            SparseSampling(text_dim=-1)

    def test_p_active_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="p_active"):
            SparseSampling(text_dim=3, p_active=-0.1)
        with pytest.raises(ValueError, match="p_active"):
            SparseSampling(text_dim=3, p_active=1.5)

    def test_geometric_rate_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="geometric_rate"):
            SparseSampling(text_dim=3, geometric_rate=0.0)
        with pytest.raises(ValueError, match="geometric_rate"):
            SparseSampling(text_dim=3, geometric_rate=1.5)

    def test_zero_anchor_fraction_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="zero_anchor"):
            SparseSampling(text_dim=3, zero_anchor_fraction=-0.1)

    def test_fraction_sum_exceeds_one(self) -> None:
        with pytest.raises(ValueError, match="must be"):
            SparseSampling(
                text_dim=3,
                zero_anchor_fraction=0.6,
                uniform_fallback_fraction=0.5,
            )


# ===========================================================================
# Output shape & bounds
# ===========================================================================


class TestShapeAndBounds:
    """Output array shape, dtype, and per-gene bounds."""

    def test_output_shape(self) -> None:
        s = SparseSampling(text_dim=3, seed=0)
        problem = make_problem(n_image=228, text_dim=3)
        samples = s._do(problem, n_samples=100)

        assert samples.shape == (100, 231)
        assert samples.dtype == np.int64

    def test_within_bounds(self) -> None:
        s = SparseSampling(text_dim=3, seed=0)
        problem = make_problem(n_image=50, text_dim=3, image_bound=25,
                               text_bound=10)
        samples = s._do(problem, n_samples=500)

        # Every gene is in [0, xu_i]
        for j in range(problem.n_var):
            assert samples[:, j].min() >= 0
            assert samples[:, j].max() <= problem.xu[j]

    def test_text_dim_too_large(self) -> None:
        s = SparseSampling(text_dim=10, seed=0)
        problem = make_problem(n_image=5, text_dim=10)
        # problem.n_var=15 but 15 > text_dim(10); should work
        s._do(problem, n_samples=10)

        # problem.n_var==text_dim: no image block → error
        bad = make_problem(n_image=0, text_dim=5)
        s2 = SparseSampling(text_dim=5, seed=0)
        with pytest.raises(ValueError, match="text_dim"):
            s2._do(bad, n_samples=10)


# ===========================================================================
# Sparsity distribution
# ===========================================================================


class TestSparsity:
    """Empirical sparsity matches E[Binomial(n_image, p_active)] closely."""

    def test_expected_n_active(self) -> None:
        p_active = 0.03
        n_image = 228
        # Use all-geometric to isolate sparsity (no zero anchor, no uniform)
        s = SparseSampling(
            text_dim=3,
            p_active=p_active,
            zero_anchor_fraction=0.0,
            uniform_fallback_fraction=0.0,
            seed=42,
        )
        problem = make_problem(n_image=n_image, text_dim=3)
        samples = s._do(problem, n_samples=2000)

        # n_active per individual, image block only
        n_active = (samples[:, :n_image] != 0).sum(axis=1)
        mean_active = n_active.mean()

        expected = n_image * p_active          # 6.84
        # Binomial std = sqrt(n p (1-p)) / sqrt(N_samples)
        # ~2.58 / sqrt(2000) ≈ 0.058
        # 4σ window
        std_mean = np.sqrt(n_image * p_active * (1 - p_active)) / np.sqrt(2000)
        assert abs(mean_active - expected) < 4 * std_mean, (
            f"mean n_active {mean_active:.3f} outside 4σ of {expected:.3f}"
        )

    def test_zero_p_active_gives_all_zero_image(self) -> None:
        s = SparseSampling(
            text_dim=3,
            p_active=0.0,
            zero_anchor_fraction=0.0,
            uniform_fallback_fraction=0.0,
            seed=0,
        )
        problem = make_problem(n_image=50, text_dim=3)
        samples = s._do(problem, n_samples=20)

        assert samples[:, :50].sum() == 0   # all image-block zero
        # Text block may still be non-zero (uniform)

    def test_full_p_active_gives_dense_image(self) -> None:
        s = SparseSampling(
            text_dim=3,
            p_active=1.0,
            zero_anchor_fraction=0.0,
            uniform_fallback_fraction=0.0,
            seed=0,
        )
        problem = make_problem(n_image=30, text_dim=3)
        samples = s._do(problem, n_samples=50)

        # With p=1.0, every image gene is active (Bernoulli mask=True everywhere)
        # Geometric depth ≥ 1, so every image gene is ≥ 1
        assert (samples[:, :30] >= 1).all()


# ===========================================================================
# Zero anchors
# ===========================================================================


class TestZeroAnchors:
    """Exact-zero rows in the image block."""

    def test_zero_anchor_count(self) -> None:
        s = SparseSampling(
            text_dim=3,
            zero_anchor_fraction=0.20,
            uniform_fallback_fraction=0.0,
            seed=0,
        )
        problem = make_problem(n_image=30, text_dim=3)
        samples = s._do(problem, n_samples=100)

        # First 20 rows should have all-zero image block.
        image_block = samples[:, :30]
        is_zero = (image_block.sum(axis=1) == 0)
        n_zero_rows = is_zero[:20].sum()
        assert n_zero_rows == 20

    def test_zero_anchor_text_uniform(self) -> None:
        """Text block of zero-anchor rows uses uniform distribution, not zero."""
        s = SparseSampling(
            text_dim=3,
            zero_anchor_fraction=1.0,
            uniform_fallback_fraction=0.0,
            seed=0,
        )
        problem = make_problem(n_image=30, text_dim=3, text_bound=10)
        samples = s._do(problem, n_samples=500)

        # All image blocks are zero
        assert samples[:, :30].sum() == 0
        # Text block: at least some non-zero rows (uniform over 11 values)
        text_block = samples[:, 30:]
        n_nonzero_text_rows = (text_block.sum(axis=1) > 0).sum()
        # P(all 3 text genes zero) = (1/11)^3 ≈ 0.00075 → ~0.4 out of 500
        assert n_nonzero_text_rows > 400


# ===========================================================================
# Depth distribution (geometric vs uniform)
# ===========================================================================


class TestDepthDistribution:
    """Geometric bias: p50 of active values < uniform equivalent."""

    def test_geometric_p50_is_shallow(self) -> None:
        """With rate=0.5, median active depth should be 1 or 2, not codebook-center."""
        s = SparseSampling(
            text_dim=3,
            p_active=1.0,
            geometric_rate=0.5,
            zero_anchor_fraction=0.0,
            uniform_fallback_fraction=0.0,
            seed=42,
        )
        problem = make_problem(n_image=100, text_dim=3, image_bound=16383)
        samples = s._do(problem, n_samples=200)

        image_values = samples[:, :100].ravel()
        active = image_values[image_values > 0]
        p50 = np.median(active)

        # Geometric(0.5) has mean=2, so p50 should be ≤ 3 with wide margin
        assert p50 <= 3, f"geometric p50 too high: {p50}"

    def test_uniform_p50_is_roughly_half_bound(self) -> None:
        """With uniform_fallback=1.0, median active depth ≈ bound/2."""
        image_bound = 16383
        s = SparseSampling(
            text_dim=3,
            p_active=1.0,
            zero_anchor_fraction=0.0,
            uniform_fallback_fraction=1.0,
            seed=42,
        )
        problem = make_problem(n_image=100, text_dim=3, image_bound=image_bound)
        samples = s._do(problem, n_samples=200)

        image_values = samples[:, :100].ravel()
        active = image_values[image_values > 0]
        p50 = np.median(active)

        # Uniform [1, 16383] has median ≈ 8192
        assert 7000 < p50 < 9500, f"uniform p50 not near mid-bound: {p50}"


# ===========================================================================
# Text block
# ===========================================================================


class TestTextBlock:
    """Text-block is uniform across all sub-samplers, unaffected by sparsity."""

    def test_text_block_uniform_chi2(self) -> None:
        from scipy.stats import chisquare

        s = SparseSampling(
            text_dim=1,
            p_active=0.03,
            seed=0,
        )
        problem = make_problem(n_image=30, text_dim=1, text_bound=9)
        samples = s._do(problem, n_samples=2000)

        text_col = samples[:, 30]
        counts = np.bincount(text_col, minlength=10)
        # χ²-test for uniformity over 10 values
        stat, pval = chisquare(counts)
        assert pval > 0.01, f"text block not uniform: p={pval}"

    def test_text_dim_zero_ok(self) -> None:
        """text_dim=0 means no text block; image-only genotype."""
        s = SparseSampling(text_dim=0, seed=0)
        problem = make_problem(n_image=20, text_dim=0)
        samples = s._do(problem, n_samples=30)

        assert samples.shape == (30, 20)


# ===========================================================================
# Reproducibility
# ===========================================================================


class TestReproducibility:
    """Same seed → identical output."""

    def test_same_seed_same_output(self) -> None:
        s1 = SparseSampling(text_dim=3, seed=123)
        s2 = SparseSampling(text_dim=3, seed=123)
        problem = make_problem(n_image=50, text_dim=3)

        out1 = s1._do(problem, n_samples=20)
        out2 = s2._do(problem, n_samples=20)

        np.testing.assert_array_equal(out1, out2)

    def test_different_seeds_different_output(self) -> None:
        s1 = SparseSampling(text_dim=3, seed=1)
        s2 = SparseSampling(text_dim=3, seed=2)
        problem = make_problem(n_image=50, text_dim=3)

        out1 = s1._do(problem, n_samples=20)
        out2 = s2._do(problem, n_samples=20)

        assert not np.array_equal(out1, out2)


# ===========================================================================
# Round-trip with real Problem
# ===========================================================================


class TestPyMooRoundTrip:
    """SparseSampling plugs into PyMoo's Problem directly without adapters."""

    def test_with_real_pymoo_problem(self) -> None:
        from pymoo.core.problem import Problem

        n_image, text_dim = 20, 3
        n_var = n_image + text_dim
        xl = np.zeros(n_var, dtype=np.int64)
        xu = np.concatenate([
            np.full(n_image, 25, dtype=np.int64),
            np.full(text_dim, 10, dtype=np.int64),
        ])
        problem = Problem(n_var=n_var, n_obj=3, xl=xl, xu=xu, vtype=int)

        s = SparseSampling(text_dim=text_dim, seed=0)
        samples = s._do(problem, n_samples=30)

        assert samples.shape == (30, n_var)
        # All within bounds
        for j in range(n_var):
            assert samples[:, j].min() >= 0
            assert samples[:, j].max() <= xu[j]
