"""Tests for the image manipulator types, selection, and genotype logic.

These tests exercise the pure data structures and functions without
requiring a VQGAN model. The codec and full pipeline are tested
separately with a model checkpoint.
"""

import numpy as np
import pytest
from PIL import Image

from src.manipulator.image.cone_candidates import ConeCandidateFilter
from src.manipulator.image.manipulator import (
    ConeFilterConfig,
    ImageConfig,
    ImageManipulator,
    apply_genotype,
)
from src.manipulator.image.selection import (
    build_codebook_knn,
    build_cone_patch_selection,
    build_patch_selection,
    select_candidates,
    select_patches,
)
from src.manipulator.image.types import (
    CandidateStrategy,
    CodeGrid,
    ManipulationContext,
    PatchSelection,
    PatchStrategy,
)


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

    def test_select_frequency_rounds_down(self):
        grid = CodeGrid(np.arange(16, dtype=np.int64).reshape(4, 4))
        # 16 unique codes → ratio=0.01 * 16 = 0.16 → int(...) = 0.
        # The `max(1, ...)` floor was removed so modality=text_only can
        # produce a zero-patch selection by setting patch_ratio=0.
        pos = select_patches(grid, PatchStrategy.FREQUENCY, ratio=0.01)
        assert len(pos) == 0

    def test_select_frequency_zero_ratio(self):
        grid = CodeGrid(np.arange(16, dtype=np.int64).reshape(4, 4))
        pos = select_patches(grid, PatchStrategy.FREQUENCY, ratio=0.0)
        assert pos.shape == (0, 2)


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


# ---------------------------------------------------------------------------
# Cone-filter candidate path
# ---------------------------------------------------------------------------


def _stripe_codebook(n_codes: int = 8) -> np.ndarray:
    """Codebook on the 2-D x-axis: codeword i sits at (float(i), 0).

    Picking ``p_c = codebook[i]`` and ``p_t = codebook[j]`` for ``i < j``
    means the cone axis points along +x and on-axis codewords between
    them survive any non-degenerate alpha.
    """
    cb = np.zeros((n_codes, 2), dtype=np.float32)
    cb[:, 0] = np.arange(n_codes, dtype=np.float32)
    return cb


class TestBuildConePatchSelection:
    def setup_method(self):
        # 2x2 origin grid; codeword indices 0..3 sit on the x-axis.
        self.codebook = _stripe_codebook(n_codes=8)
        self.grid = CodeGrid(np.array([[0, 2], [4, 5]], dtype=np.int64))
        self.cone = ConeCandidateFilter(alpha_deg=45.0)

    def test_per_position_candidates_match_axis_geometry(self):
        # Target grid: every cell points at codeword 7 (the far end of the
        # stripe). For position (0, 0) the segment goes 0 -> 7 and admits
        # codewords [0..7] in tau order.
        target_grid = np.full((2, 2), 7, dtype=np.int64)
        sel = build_cone_patch_selection(
            grid=self.grid,
            target_grid=target_grid,
            codebook=self.codebook,
            cone_filter=self.cone,
            patch_strategy=PatchStrategy.ALL,
            patch_ratio=1.0,
        )

        assert sel.n_patches == 4

        # Map (row, col) -> candidate list for clarity.
        cand_by_pos: dict[tuple[int, int], np.ndarray] = {}
        for (r, c), cands in zip(sel.positions, sel.candidates):
            cand_by_pos[(int(r), int(c))] = cands

        # Position (0, 0): origin=0, target=7. tau-sorted survivors: 0..7.
        np.testing.assert_array_equal(
            cand_by_pos[(0, 0)],
            np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64),
        )
        # Position (1, 1): origin=5, target=7. tau-sorted: 5, 6, 7.
        np.testing.assert_array_equal(
            cand_by_pos[(1, 1)],
            np.array([5, 6, 7], dtype=np.int64),
        )

    def test_degenerate_axis_gives_empty_candidates(self):
        # All target == origin → every position is degenerate → gene_bounds
        # must be 1 (only "keep origin" is valid).
        target_grid = self.grid.indices.copy()
        sel = build_cone_patch_selection(
            grid=self.grid,
            target_grid=target_grid,
            codebook=self.codebook,
            cone_filter=self.cone,
            patch_strategy=PatchStrategy.ALL,
            patch_ratio=1.0,
        )
        for cands in sel.candidates:
            assert cands.size == 0
        np.testing.assert_array_equal(sel.gene_bounds, np.ones(4, dtype=np.int64))

    def test_mixed_degenerate_and_active_positions(self):
        # Two positions degenerate (0 == 0, 5 == 5), two active.
        target_grid = np.array([[0, 7], [3, 5]], dtype=np.int64)
        sel = build_cone_patch_selection(
            grid=self.grid,
            target_grid=target_grid,
            codebook=self.codebook,
            cone_filter=self.cone,
            patch_strategy=PatchStrategy.ALL,
            patch_ratio=1.0,
        )
        bounds = sel.gene_bounds
        # bounds = len(candidates) + 1; degenerate -> 1, active -> > 1.
        assert bounds.shape == (4,)
        # Find each (row, col) and check.
        pos = [tuple(int(x) for x in p) for p in sel.positions]
        order = {p: i for i, p in enumerate(pos)}
        assert bounds[order[(0, 0)]] == 1  # degenerate
        assert bounds[order[(1, 1)]] == 1  # degenerate
        assert bounds[order[(0, 1)]] > 1   # active: 2 -> 7
        assert bounds[order[(1, 0)]] > 1   # active: 4 -> 3

    def test_shape_mismatch_raises(self):
        bad_target = np.zeros((3, 3), dtype=np.int64)
        with pytest.raises(ValueError, match="shape"):
            build_cone_patch_selection(
                grid=self.grid,
                target_grid=bad_target,
                codebook=self.codebook,
                cone_filter=self.cone,
                patch_strategy=PatchStrategy.ALL,
                patch_ratio=1.0,
            )

    def test_zero_positions_gives_empty_selection(self):
        target_grid = np.full((2, 2), 7, dtype=np.int64)
        sel = build_cone_patch_selection(
            grid=self.grid,
            target_grid=target_grid,
            codebook=self.codebook,
            cone_filter=self.cone,
            patch_strategy=PatchStrategy.FREQUENCY,
            patch_ratio=0.0,
        )
        assert sel.n_patches == 0


# ---------------------------------------------------------------------------
# ImageManipulator.prepare() routing — knn vs cone
# ---------------------------------------------------------------------------


class _FakeCodec:
    """Tiny in-memory codec that maps PIL images to a hand-crafted grid.

    Bypasses VQGAN entirely so manipulator tests stay fast and offline.
    """

    def __init__(self, codebook: np.ndarray, grid: CodeGrid) -> None:
        self._codebook = codebook.astype(np.float32)
        self._grid = grid
        self._grid_size = grid.shape

    @property
    def codebook(self) -> np.ndarray:
        return self._codebook

    @property
    def grid_size(self) -> tuple[int, int]:
        return self._grid_size

    def encode(self, image: Image.Image) -> CodeGrid:
        return self._grid

    def decode_batch(self, grids):
        return [Image.new("RGB", (4, 4)) for _ in grids]


class _FixedModalBuilder:
    """Modal-builder stub returning a pre-computed grid for any class."""

    def __init__(self, target_grid: np.ndarray) -> None:
        self._target_grid = target_grid.astype(np.int64)
        self.ensure_calls: list[str] = []

    def ensure(self, class_name: str) -> np.ndarray:
        self.ensure_calls.append(class_name)
        return self._target_grid

    def populate_many(self, class_names) -> None:
        for name in class_names:
            self.ensure(name)


class TestImageManipulatorKNNPath:
    """Legacy KNN path: prepare(image) without target_class still works."""

    def setup_method(self):
        # 2x2 grid uniquely populated so FREQUENCY selects everything.
        self.codebook = _stripe_codebook(n_codes=8)
        grid = CodeGrid(np.array([[0, 1], [2, 3]], dtype=np.int64))
        self.codec = _FakeCodec(self.codebook, grid)
        self.config = ImageConfig(
            patch_strategy=PatchStrategy.ALL,
            patch_ratio=1.0,
            n_candidates=3,
            candidate_strategy=CandidateStrategy.KNN,
        )
        self.manipulator = ImageManipulator(self.codec, self.config)

    def test_legacy_config_round_trips(self):
        # ImageConfig() default doesn't enable cone_filter; manipulator
        # should construct without a builder.
        m = ImageManipulator(self.codec, ImageConfig())
        assert not m.cone_filter_enabled
        assert m.modal_builder is None

    def test_prepare_without_target_class_uses_knn(self):
        ctx = self.manipulator.prepare(Image.new("RGB", (4, 4)))
        assert ctx.candidate_strategy == "knn"
        assert ctx.target_class is None
        # Gene bounds are n_candidates + 1 since KNN truncates to k.
        assert (ctx.gene_bounds == 4).all()

    def test_prepare_with_target_class_records_metadata_on_knn_path(self):
        # Even on the legacy path, the target_class is propagated to the
        # context for trace metadata.
        ctx = self.manipulator.prepare(Image.new("RGB", (4, 4)), target_class="junco")
        assert ctx.candidate_strategy == "knn"
        assert ctx.target_class == "junco"


class TestImageManipulatorConePath:
    def setup_method(self):
        self.codebook = _stripe_codebook(n_codes=8)
        # Origin grid: all zeros so cone axis is 0 -> target codeword.
        grid = CodeGrid(np.zeros((2, 2), dtype=np.int64))
        self.codec = _FakeCodec(self.codebook, grid)
        self.modal_grid = np.full((2, 2), 7, dtype=np.int64)
        self.builder = _FixedModalBuilder(self.modal_grid)
        self.config = ImageConfig(
            patch_strategy=PatchStrategy.ALL,
            patch_ratio=1.0,
            cone_filter=ConeFilterConfig(enabled=True, alpha_deg=45.0),
        )
        self.manipulator = ImageManipulator(
            self.codec,
            self.config,
            modal_builder=self.builder,
            target_classes=("junco",),
        )

    def test_prepopulates_target_class_in_constructor(self):
        # Constructor should call builder.populate_many on the target class.
        assert "junco" in self.builder.ensure_calls

    def test_cone_filter_enabled_property(self):
        assert self.manipulator.cone_filter_enabled

    def test_prepare_routes_through_cone(self):
        ctx = self.manipulator.prepare(
            Image.new("RGB", (4, 4)), target_class="junco",
        )
        assert ctx.candidate_strategy == "cone_filter"
        assert ctx.target_class == "junco"

    def test_cone_candidates_are_tau_sorted(self):
        ctx = self.manipulator.prepare(
            Image.new("RGB", (4, 4)), target_class="junco",
        )
        # Each patch has origin=0, target=7, expected survivors [0..7].
        for cands in ctx.selection.candidates:
            np.testing.assert_array_equal(
                cands, np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64),
            )

    def test_gene_zero_keeps_origin(self):
        ctx = self.manipulator.prepare(
            Image.new("RGB", (4, 4)), target_class="junco",
        )
        zero = ctx.zero_genotype()
        mutated = apply_genotype(ctx.original_grid, ctx.selection, zero)
        np.testing.assert_array_equal(mutated.indices, ctx.original_grid.indices)

    def test_gene_one_picks_first_cone_survivor(self):
        ctx = self.manipulator.prepare(
            Image.new("RGB", (4, 4)), target_class="junco",
        )
        gene = np.ones(ctx.genotype_dim, dtype=np.int64)
        mutated = apply_genotype(ctx.original_grid, ctx.selection, gene)
        # First survivor (k=1) for every (0->7) cone is codeword 0
        # — which equals origin. So the result equals origin.
        np.testing.assert_array_equal(
            mutated.indices, ctx.original_grid.indices,
        )

    def test_gene_max_picks_last_cone_survivor(self):
        ctx = self.manipulator.prepare(
            Image.new("RGB", (4, 4)), target_class="junco",
        )
        # max valid gene per position is gene_bounds - 1; here all bounds = 9.
        gene = ctx.gene_bounds.astype(np.int64) - 1
        mutated = apply_genotype(ctx.original_grid, ctx.selection, gene)
        # Every position now points at the last τ entry, which is the
        # target codeword (7).
        assert (mutated.indices == 7).all()

    def test_out_of_bound_gene_raises(self):
        ctx = self.manipulator.prepare(
            Image.new("RGB", (4, 4)), target_class="junco",
        )
        # Gene = bound is out of range; bound is exclusive.
        bad_gene = ctx.gene_bounds.astype(np.int64)
        with pytest.raises(IndexError):
            apply_genotype(ctx.original_grid, ctx.selection, bad_gene)

    def test_target_class_required_when_cone_enabled(self):
        with pytest.raises(ValueError, match="target_class"):
            self.manipulator.prepare(Image.new("RGB", (4, 4)))

    def test_cone_path_with_degenerate_target(self):
        # Build a manipulator whose target grid equals the origin grid
        # — every position is degenerate → gene_bounds all 1.
        degenerate_grid = np.zeros((2, 2), dtype=np.int64)
        builder = _FixedModalBuilder(degenerate_grid)
        manipulator = ImageManipulator(
            self.codec,
            self.config,
            modal_builder=builder,
        )
        ctx = manipulator.prepare(
            Image.new("RGB", (4, 4)), target_class="junco",
        )
        np.testing.assert_array_equal(ctx.gene_bounds, np.ones(4, dtype=np.int64))


class TestImageManipulatorAttachBuilder:
    def setup_method(self):
        self.codebook = _stripe_codebook(n_codes=8)
        grid = CodeGrid(np.zeros((2, 2), dtype=np.int64))
        self.codec = _FakeCodec(self.codebook, grid)
        self.config = ImageConfig(
            patch_strategy=PatchStrategy.ALL,
            patch_ratio=1.0,
            cone_filter=ConeFilterConfig(enabled=True, alpha_deg=45.0),
        )

    def test_attach_modal_builder_late_binding(self):
        # Build without a builder, then attach + precompute later — used
        # when seed generation runs in parallel with manipulator init.
        manipulator = ImageManipulator(self.codec, self.config)
        assert manipulator.modal_builder is None
        builder = _FixedModalBuilder(np.full((2, 2), 7, dtype=np.int64))
        manipulator.attach_modal_builder(builder)
        manipulator.precompute_targets(("junco",))
        assert "junco" in builder.ensure_calls

    def test_precompute_no_op_without_builder(self):
        # Default ImageConfig (no cone filter) — precompute should be silent.
        manipulator = ImageManipulator(self.codec, ImageConfig())
        manipulator.precompute_targets(("junco",))  # no exception expected

    def test_cone_enabled_without_builder_raises_on_prepare(self):
        manipulator = ImageManipulator(self.codec, self.config)
        with pytest.raises(RuntimeError, match="modal_builder"):
            manipulator.prepare(
                Image.new("RGB", (4, 4)), target_class="junco",
            )


# ---------------------------------------------------------------------------
# Legacy ImageConfig YAML round-trip
# ---------------------------------------------------------------------------


class TestLegacyConfigRoundTrip:
    """Legacy YAML configs (no ``image.cone_filter``) must still load."""

    def test_default_image_config_has_disabled_cone(self):
        cfg = ImageConfig()
        assert not cfg.cone_filter.enabled
        assert cfg.cone_filter.alpha_deg == 20.0
        assert cfg.cone_filter.target_m == 100

    def test_dacite_yaml_load_without_cone_section(self):
        # Mirrors the runner's dacite usage on a minimal YAML override.
        import dacite

        from src.config import ExperimentConfig

        raw = {
            "name": "legacy_test",
            "categories": ["a", "b"],
            "image": {"patch_ratio": 0.25},
        }
        exp = dacite.from_dict(ExperimentConfig, raw, config=dacite.Config(cast=[tuple, frozenset]))
        assert exp.image.patch_ratio == 0.25
        # The cone_filter sub-block fills with defaults; cone path remains off.
        assert not exp.image.cone_filter.enabled

    def test_dacite_yaml_load_with_cone_section(self):
        import dacite

        from src.config import ExperimentConfig

        raw = {
            "name": "cone_test",
            "categories": ["a", "b"],
            "image": {
                "cone_filter": {
                    "enabled": True,
                    "alpha_deg": 15.5,
                    "target_m": 50,
                },
            },
        }
        exp = dacite.from_dict(ExperimentConfig, raw, config=dacite.Config(cast=[tuple, frozenset]))
        assert exp.image.cone_filter.enabled
        assert exp.image.cone_filter.alpha_deg == 15.5
        assert exp.image.cone_filter.target_m == 50
