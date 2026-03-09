"""Tests for the image manipulator types, selection, and genotype logic.

These tests exercise the pure data structures and functions without
requiring a VQGAN model. The codec and full pipeline are tested
separately with a model checkpoint.
"""

import numpy as np
import pytest

from src.manipulator.image.types import (
    CandidateStrategy,
    CodeGrid,
    ManipulationContext,
    PatchSelection,
    PatchStrategy,
)
from src.manipulator.image.selection import (
    build_codebook_knn,
    build_patch_selection,
    select_candidates,
    select_patches,
)
from src.manipulator.image.manipulator import apply_genotype


# ---------------------------------------------------------------------------
# CodeGrid
# ---------------------------------------------------------------------------


class TestCodeGrid:
    def test_creation_and_immutability(self):
        indices = np.array([[0, 1], [2, 3]], dtype=np.int64)
        grid = CodeGrid(indices)
        assert grid.shape == (2, 2)
        assert grid.n_tokens == 4
        with pytest.raises(ValueError):
            grid.indices[0, 0] = 99

    def test_defensive_copy_breaks_aliasing(self):
        arr = np.array([[0, 1], [2, 3]], dtype=np.int64)
        grid = CodeGrid(arr)
        arr[0, 0] = 999  # mutate caller's array
        assert grid.indices[0, 0] == 0  # grid is unaffected

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError):
            CodeGrid(np.array([1, 2, 3], dtype=np.int64))

    def test_replace_returns_new_grid(self):
        grid = CodeGrid(np.array([[0, 1], [2, 3]], dtype=np.int64))
        rows = np.array([0, 1], dtype=np.intp)
        cols = np.array([0, 1], dtype=np.intp)
        codes = np.array([10, 20], dtype=np.int64)
        new = grid.replace(rows, cols, codes)

        assert new.indices[0, 0] == 10
        assert new.indices[1, 1] == 20
        assert grid.indices[0, 0] == 0  # original unchanged
        assert grid.indices[1, 1] == 3

    def test_fingerprint_deterministic(self):
        a = CodeGrid(np.array([[1, 2], [3, 4]], dtype=np.int64))
        b = CodeGrid(np.array([[1, 2], [3, 4]], dtype=np.int64))
        assert a.fingerprint == b.fingerprint

    def test_fingerprint_changes(self):
        a = CodeGrid(np.array([[1, 2], [3, 4]], dtype=np.int64))
        b = CodeGrid(np.array([[1, 2], [3, 5]], dtype=np.int64))
        assert a.fingerprint != b.fingerprint


# ---------------------------------------------------------------------------
# Patch selection
# ---------------------------------------------------------------------------


class TestPatchSelection:
    def test_select_all(self):
        grid = CodeGrid(np.zeros((4, 4), dtype=np.int64))
        pos = select_patches(grid, PatchStrategy.ALL)
        assert pos.shape == (16, 2)

    def test_select_frequency_respects_ratio(self):
        # Grid with 4 unique codes: 0 appears 8x, 1 appears 4x, 2 appears 2x, 3 appears 2x
        indices = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [2, 2, 3, 3],
        ], dtype=np.int64)
        grid = CodeGrid(indices)

        # ratio=0.25 of 4 unique codes → top 1 code (code 0, 8 positions)
        pos = select_patches(grid, PatchStrategy.FREQUENCY, ratio=0.25)
        assert len(pos) == 8
        for r, c in pos:
            assert grid.indices[r, c] == 0

    def test_select_frequency_at_least_one(self):
        grid = CodeGrid(np.arange(16, dtype=np.int64).reshape(4, 4))
        # All unique — ratio=0.01 should still select at least 1
        pos = select_patches(grid, PatchStrategy.FREQUENCY, ratio=0.01)
        assert len(pos) >= 1


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------


class TestCandidateSelection:
    def setup_method(self):
        self.neighbors = np.arange(100, dtype=np.int64)

    def test_knn_takes_first_k(self):
        result = select_candidates(self.neighbors, CandidateStrategy.KNN, k=5)
        np.testing.assert_array_equal(result, [0, 1, 2, 3, 4])

    def test_kfn_takes_last_k(self):
        result = select_candidates(self.neighbors, CandidateStrategy.KFN, k=5)
        np.testing.assert_array_equal(result, [95, 96, 97, 98, 99])

    def test_uniform_spans_range(self):
        result = select_candidates(self.neighbors, CandidateStrategy.UNIFORM, k=3)
        assert result[0] == 0
        assert result[-1] == 99
        assert len(result) == 3

    def test_k_clamped_to_available(self):
        short = np.array([10, 20], dtype=np.int64)
        result = select_candidates(short, CandidateStrategy.KNN, k=5)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Codebook KNN
# ---------------------------------------------------------------------------


class TestCodebookKNN:
    def test_knn_shape_and_self_excluded(self):
        # Small codebook: 5 vectors in 3D
        rng = np.random.default_rng(42)
        codebook = rng.standard_normal((5, 3)).astype(np.float32)
        knn = build_codebook_knn(codebook)

        assert knn.shape == (5, 4)  # 5 codes, 4 neighbors each (self excluded)
        # No codeword is its own neighbor
        for i in range(5):
            assert i not in knn[i]

    def test_nearest_neighbor_is_correct(self):
        # Two nearly identical vectors, one far away
        codebook = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.01, 0.0],  # very close to 0
            [-1.0, 0.0, 0.0],  # opposite direction
        ], dtype=np.float32)
        knn = build_codebook_knn(codebook)

        assert knn[0, 0] == 1  # nearest to 0 is 1
        assert knn[1, 0] == 0  # nearest to 1 is 0

    def test_caching(self, tmp_path):
        codebook = np.eye(4, dtype=np.float32)
        cache = tmp_path / "knn.npz"

        knn1 = build_codebook_knn(codebook, cache_path=cache)
        assert cache.exists()

        knn2 = build_codebook_knn(codebook, cache_path=cache)
        np.testing.assert_array_equal(knn1, knn2)


# ---------------------------------------------------------------------------
# Genotype application
# ---------------------------------------------------------------------------


class TestApplyGenotype:
    def setup_method(self):
        self.grid = CodeGrid(np.array([
            [10, 20],
            [30, 40],
        ], dtype=np.int64))

        self.selection = PatchSelection(
            positions=np.array([[0, 0], [1, 1]], dtype=np.intp),
            candidates=(
                np.array([11, 12, 13], dtype=np.int64),
                np.array([41, 42, 43], dtype=np.int64),
            ),
            original_codes=np.array([10, 40], dtype=np.int64),
        )

    def test_zero_genotype_preserves_original(self):
        genotype = np.array([0, 0], dtype=np.int64)
        result = apply_genotype(self.grid, self.selection, genotype)
        np.testing.assert_array_equal(result.indices, self.grid.indices)

    def test_single_mutation(self):
        genotype = np.array([2, 0], dtype=np.int64)  # replace patch 0 with candidate[1]
        result = apply_genotype(self.grid, self.selection, genotype)
        assert result.indices[0, 0] == 12
        assert result.indices[1, 1] == 40  # unchanged

    def test_full_mutation(self):
        genotype = np.array([1, 3], dtype=np.int64)
        result = apply_genotype(self.grid, self.selection, genotype)
        assert result.indices[0, 0] == 11
        assert result.indices[1, 1] == 43

    def test_original_grid_unchanged(self):
        genotype = np.array([1, 1], dtype=np.int64)
        apply_genotype(self.grid, self.selection, genotype)
        assert self.grid.indices[0, 0] == 10
        assert self.grid.indices[1, 1] == 40

    def test_wrong_genotype_length_raises(self):
        with pytest.raises(ValueError, match="Genotype length"):
            apply_genotype(self.grid, self.selection, np.array([1], dtype=np.int64))


# ---------------------------------------------------------------------------
# ManipulationContext
# ---------------------------------------------------------------------------


class TestPatchSelectionValidation:
    def test_mismatched_candidates_length_raises(self):
        with pytest.raises(ValueError, match="candidates"):
            PatchSelection(
                positions=np.array([[0, 0], [1, 1]], dtype=np.intp),
                candidates=(np.array([1], dtype=np.int64),),  # 1, not 2
                original_codes=np.array([0, 0], dtype=np.int64),
            )

    def test_mismatched_original_codes_length_raises(self):
        with pytest.raises(ValueError, match="original_codes"):
            PatchSelection(
                positions=np.array([[0, 0], [1, 1]], dtype=np.intp),
                candidates=(
                    np.array([1], dtype=np.int64),
                    np.array([2], dtype=np.int64),
                ),
                original_codes=np.array([0], dtype=np.int64),  # 1, not 2
            )


class TestManipulationContext:
    def test_genotype_properties(self):
        selection = PatchSelection(
            positions=np.array([[0, 0], [1, 1], [2, 2]], dtype=np.intp),
            candidates=(
                np.array([1, 2], dtype=np.int64),
                np.array([3, 4, 5], dtype=np.int64),
                np.array([6], dtype=np.int64),
            ),
            original_codes=np.array([0, 0, 0], dtype=np.int64),
        )
        grid = CodeGrid(np.zeros((4, 4), dtype=np.int64))
        ctx = ManipulationContext(original_grid=grid, selection=selection)

        assert ctx.genotype_dim == 3
        np.testing.assert_array_equal(ctx.gene_bounds, [3, 4, 2])

    def test_zero_genotype(self):
        selection = PatchSelection(
            positions=np.array([[0, 0]], dtype=np.intp),
            candidates=(np.array([5], dtype=np.int64),),
            original_codes=np.array([0], dtype=np.int64),
        )
        grid = CodeGrid(np.zeros((2, 2), dtype=np.int64))
        ctx = ManipulationContext(original_grid=grid, selection=selection)

        g = ctx.zero_genotype()
        assert len(g) == 1
        assert g[0] == 0

    def test_random_genotype_within_bounds(self):
        selection = PatchSelection(
            positions=np.array([[0, 0], [0, 1]], dtype=np.intp),
            candidates=(
                np.array([1, 2, 3], dtype=np.int64),
                np.array([4, 5], dtype=np.int64),
            ),
            original_codes=np.array([0, 0], dtype=np.int64),
        )
        grid = CodeGrid(np.zeros((2, 2), dtype=np.int64))
        ctx = ManipulationContext(original_grid=grid, selection=selection)

        rng = np.random.default_rng(42)
        for _ in range(100):
            g = ctx.random_genotype(rng)
            assert 0 <= g[0] < 4  # 3 candidates + 1 (original)
            assert 0 <= g[1] < 3  # 2 candidates + 1 (original)


# ---------------------------------------------------------------------------
# build_patch_selection integration
# ---------------------------------------------------------------------------


class TestBuildPatchSelection:
    def test_end_to_end(self):
        grid = CodeGrid(np.array([
            [0, 0, 1],
            [1, 2, 2],
            [3, 3, 3],
        ], dtype=np.int64))

        # Synthetic KNN: each code's neighbors are just the other codes in order
        knn = np.zeros((4, 3), dtype=np.int64)
        for i in range(4):
            knn[i] = [j for j in range(4) if j != i]

        sel = build_patch_selection(
            grid=grid,
            knn=knn,
            patch_strategy=PatchStrategy.FREQUENCY,
            patch_ratio=0.5,  # top 50% of unique codes → top 2 of 4
            candidate_strategy=CandidateStrategy.KNN,
            n_candidates=2,
        )

        # Frequencies: code 3 → 3x, code 0 → 2x, code 1 → 2x, code 2 → 2x
        # Deterministic tie-breaking: descending count, then ascending code.
        # Top 2 unique codes: code 3 (3x), code 0 (2x, lowest index among ties)
        assert sel.n_patches == 5  # 3 patches for code 3 + 2 for code 0
        for pos, code in zip(sel.positions, sel.original_codes):
            assert code in (0, 3)
