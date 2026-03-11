"""Tests for VLMManipulator: the multi-modal wrapper.

All tests use synthetic data — no real VQGAN or FastText models are loaded.
Fake sub-manipulators delegate to the real pure ``apply_genotype`` functions
with synthetic ManipulationContexts built from known data.
"""

import numpy as np
import pytest
from gensim.models import KeyedVectors
from PIL import Image

from src.manipulator.image.manipulator import apply_genotype as apply_image_genotype
from src.manipulator.image.types import (
    CodeGrid,
    ManipulationContext as ImageManipulationContext,
    PatchSelection,
)
from src.manipulator.text.manipulator import apply_genotype as apply_text_genotype
from src.manipulator.text.types import (
    ManipulationContext as TextManipulationContext,
    TokenSequence,
    WordSelection,
)
from src.manipulator.vlm_manipulator import VLMManipulator


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------


def _make_embeddings(vocab: dict[str, list[float]]) -> KeyedVectors:
    """Build a small gensim KeyedVectors from a dict."""
    dim = len(next(iter(vocab.values())))
    kv = KeyedVectors(vector_size=dim)
    words = list(vocab.keys())
    vectors = np.array(list(vocab.values()), dtype=np.float32)
    kv.add_vectors(words, vectors)
    return kv


def _make_tokens(*items: tuple[str, str, str]) -> TokenSequence:
    """Build a TokenSequence from (word, pos, whitespace) triples."""
    return TokenSequence(
        tokens=tuple(w for w, _, _ in items),
        pos_tags=tuple(p for _, p, _ in items),
        whitespace=tuple(s for _, _, s in items),
    )


def _make_image_context() -> ImageManipulationContext:
    """Synthetic image context: 2x2 grid with 2 selected patches."""
    grid = CodeGrid(np.array([[10, 20], [30, 40]], dtype=np.int64))
    selection = PatchSelection(
        positions=np.array([[0, 0], [1, 1]], dtype=np.intp),
        candidates=(
            np.array([11, 12, 13], dtype=np.int64),
            np.array([41, 42], dtype=np.int64),
        ),
        original_codes=np.array([10, 40], dtype=np.int64),
    )
    return ImageManipulationContext(original_grid=grid, selection=selection)


def _make_text_context() -> TextManipulationContext:
    """Synthetic text context: 4 tokens, 2 mutable words."""
    tokens = _make_tokens(
        ("The", "DET", " "),
        ("quick", "ADJ", " "),
        ("brown", "ADJ", " "),
        ("fox", "NOUN", ""),
    )
    selection = WordSelection(
        positions=np.array([1, 3], dtype=np.intp),
        candidates=(("fast", "rapid"), ("wolf", "dog", "cat")),
        original_words=("quick", "fox"),
    )
    return TextManipulationContext(original_tokens=tokens, selection=selection)


# Embedding vocab: words used in text context candidates + originals.
VOCAB = {
    "quick": [1.0, 0.0, 0.0],
    "fast": [0.95, 0.05, 0.0],   # close to quick
    "rapid": [0.85, 0.15, 0.0],  # a bit farther
    "fox": [0.0, 1.0, 0.0],
    "wolf": [0.0, 0.9, 0.1],     # close to fox
    "dog": [0.0, 0.8, 0.2],      # moderate
    "cat": [0.0, 0.6, 0.4],      # farther
}


# ---------------------------------------------------------------------------
# Fake sub-manipulators
# ---------------------------------------------------------------------------


class FakeImageManipulator:
    """Minimal image manipulator for testing.

    Uses the real ``apply_genotype`` on a synthetic grid, then returns
    a solid-colour PIL image whose red channel encodes the genotype sum.
    """

    def prepare(self, image: Image.Image) -> ImageManipulationContext:
        return _make_image_context()

    def apply(self, ctx: ImageManipulationContext, genotype) -> Image.Image:
        mutated_grid = apply_image_genotype(
            ctx.original_grid, ctx.selection, genotype
        )
        # Encode genotype sum in red channel for easy assertion
        r = int(np.sum(genotype)) % 256
        return Image.new("RGB", (8, 8), (r, 0, 0))


class FakeTextManipulator:
    """Minimal text manipulator for testing."""

    def __init__(self, kv: KeyedVectors) -> None:
        self._embeddings = kv

    @property
    def embeddings(self) -> KeyedVectors:
        return self._embeddings

    def prepare(self, text: str, exclude_words=None) -> TextManipulationContext:
        return _make_text_context()

    def apply(self, ctx: TextManipulationContext, genotype) -> str:
        mutated = apply_text_genotype(
            ctx.original_tokens, ctx.selection, genotype
        )
        return mutated.text


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def embeddings():
    return _make_embeddings(VOCAB)


@pytest.fixture()
def vlm(embeddings):
    """Return a prepared VLMManipulator."""
    m = VLMManipulator(
        image_manipulator=FakeImageManipulator(),
        text_manipulator=FakeTextManipulator(embeddings),
    )
    m.prepare(Image.new("RGB", (8, 8)), "The quick brown fox")
    return m


@pytest.fixture()
def unprepared(embeddings):
    """Return a VLMManipulator that has NOT been prepared."""
    return VLMManipulator(
        image_manipulator=FakeImageManipulator(),
        text_manipulator=FakeTextManipulator(embeddings),
    )


# ---------------------------------------------------------------------------
# TestVLMManipulatorProperties
# ---------------------------------------------------------------------------


class TestVLMManipulatorProperties:
    def test_genotype_dim_is_sum(self, vlm):
        assert vlm.genotype_dim == vlm.image_dim + vlm.text_dim

    def test_gene_bounds_concatenated(self, vlm):
        img_bounds = vlm.image_context.gene_bounds
        txt_bounds = vlm.text_context.gene_bounds
        expected = np.concatenate([img_bounds, txt_bounds])
        np.testing.assert_array_equal(vlm.gene_bounds, expected)

    def test_image_dim_text_dim(self, vlm):
        # Image context has 2 patches, text context has 2 words
        assert vlm.image_dim == 2
        assert vlm.text_dim == 2

    def test_is_prepared_false_before_prepare(self, unprepared):
        assert not unprepared.is_prepared

    def test_is_prepared_true_after_prepare(self, vlm):
        assert vlm.is_prepared


# ---------------------------------------------------------------------------
# TestVLMManipulatorManipulate
# ---------------------------------------------------------------------------


class TestVLMManipulatorManipulate:
    def test_zero_genotype_produces_original(self, vlm):
        weights = vlm.zero_genotype().reshape(1, -1)
        images, texts = vlm.manipulate(candidates=None, weights=weights)

        assert len(images) == 1
        assert len(texts) == 1
        # Zero genotype => original text preserved
        assert texts[0] == "The quick brown fox"
        # Zero genotype => red channel = 0
        assert images[0].getpixel((0, 0))[0] == 0

    def test_single_image_mutation(self, vlm):
        g = vlm.zero_genotype()
        g[0] = 1  # mutate first image gene only
        weights = g.reshape(1, -1)
        images, texts = vlm.manipulate(candidates=None, weights=weights)

        assert texts[0] == "The quick brown fox"  # text unchanged
        assert images[0].getpixel((0, 0))[0] == 1  # red = sum of img genes

    def test_single_text_mutation(self, vlm):
        g = vlm.zero_genotype()
        g[vlm.image_dim] = 1  # mutate first text gene only
        weights = g.reshape(1, -1)
        images, texts = vlm.manipulate(candidates=None, weights=weights)

        assert "fast" in texts[0].lower()  # first candidate for "quick"
        assert images[0].getpixel((0, 0))[0] == 0  # image unchanged

    def test_batched_manipulate(self, vlm):
        pop_size = 5
        weights = np.zeros((pop_size, vlm.genotype_dim), dtype=np.int64)
        images, texts = vlm.manipulate(candidates=None, weights=weights)

        assert len(images) == pop_size
        assert len(texts) == pop_size

    def test_manipulate_before_prepare_raises(self, unprepared):
        with pytest.raises(RuntimeError, match="prepare"):
            unprepared.manipulate(
                candidates=None,
                weights=np.zeros((1, 4), dtype=np.int64),
            )


# ---------------------------------------------------------------------------
# TestTextCandidateDistances
# ---------------------------------------------------------------------------


class TestTextCandidateDistances:
    def test_distances_computed_during_prepare(self, vlm):
        assert vlm.text_candidate_distances is not None

    def test_distances_shape_matches_text_context(self, vlm):
        dists = vlm.text_candidate_distances
        assert len(dists) == vlm.text_dim
        # Each entry has as many distances as candidates for that word
        for i, d in enumerate(dists):
            n_cands = len(vlm.text_context.selection.candidates[i])
            assert len(d) == n_cands

    def test_distances_are_cosine(self, vlm, embeddings):
        """Verify distances match manual cosine distance computation."""
        dists = vlm.text_candidate_distances
        for i, (orig, cands) in enumerate(
            zip(
                vlm.text_context.selection.original_words,
                vlm.text_context.selection.candidates,
            )
        ):
            for k, cand in enumerate(cands):
                vec_o = embeddings[orig.lower()]
                vec_c = embeddings[cand.lower()]
                cos_sim = np.dot(vec_o, vec_c) / (
                    np.linalg.norm(vec_o) * np.linalg.norm(vec_c)
                )
                expected_dist = 1.0 - cos_sim
                np.testing.assert_allclose(
                    dists[i][k], expected_dist, atol=1e-6
                )

    def test_distances_sorted_ascending(self, vlm):
        """Candidates are sorted by cosine distance, so distances should be ascending."""
        for d in vlm.text_candidate_distances:
            for j in range(len(d) - 1):
                assert d[j] <= d[j + 1] + 1e-9, (
                    f"Distances not ascending: {d}"
                )


# ---------------------------------------------------------------------------
# TestGenotypeHelpers
# ---------------------------------------------------------------------------


class TestGenotypeHelpers:
    def test_zero_genotype_all_zeros(self, vlm):
        g = vlm.zero_genotype()
        np.testing.assert_array_equal(g, np.zeros(vlm.genotype_dim, dtype=np.int64))

    def test_zero_genotype_correct_length(self, vlm):
        g = vlm.zero_genotype()
        assert len(g) == vlm.genotype_dim

    def test_random_genotype_within_bounds(self, vlm):
        rng = np.random.default_rng(42)
        bounds = vlm.gene_bounds
        for _ in range(50):
            g = vlm.random_genotype(rng)
            assert len(g) == vlm.genotype_dim
            for j in range(len(g)):
                assert 0 <= g[j] < bounds[j], (
                    f"Gene {j}: value {g[j]} not in [0, {bounds[j]})"
                )
