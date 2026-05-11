"""Tests for VLMManipulator: the multi-modal wrapper.

Uses synthetic FakeImageManipulator + FakeCompositeTextManipulator —
no real VQGAN, ModernBERT, or spaCy load.
"""

import numpy as np
import pytest
from PIL import Image

from src.manipulator.vlm_manipulator import VLMManipulator

from conftest import FakeCompositeTextManipulator, FakeImageManipulator


@pytest.fixture()
def vlm():
    m = VLMManipulator(
        image_manipulator=FakeImageManipulator(),
        text_manipulator=FakeCompositeTextManipulator(),
    )
    m.prepare(Image.new("RGB", (8, 8)), "The quick brown fox")
    return m


@pytest.fixture()
def unprepared():
    return VLMManipulator(
        image_manipulator=FakeImageManipulator(),
        text_manipulator=FakeCompositeTextManipulator(),
    )


class TestVLMManipulatorProperties:
    def test_genotype_dim_is_sum(self, vlm):
        assert vlm.genotype_dim == vlm.image_dim + vlm.text_dim

    def test_gene_bounds_concatenated(self, vlm):
        img_bounds = vlm.image_context.gene_bounds
        txt_bounds = vlm.text_context.gene_bounds
        expected = np.concatenate([img_bounds, txt_bounds])
        np.testing.assert_array_equal(vlm.gene_bounds, expected)

    def test_image_dim_text_dim(self, vlm):
        # Image: 2 patches; text: 2 mutable positions in stub composite
        assert vlm.image_dim == 2
        assert vlm.text_dim == 2

    def test_is_prepared_false_before_prepare(self, unprepared):
        assert not unprepared.is_prepared

    def test_is_prepared_true_after_prepare(self, vlm):
        assert vlm.is_prepared


class TestVLMManipulatorManipulate:
    def test_zero_genotype_produces_original(self, vlm):
        weights = vlm.zero_genotype().reshape(1, -1)
        images, texts = vlm.manipulate(candidates=None, weights=weights)
        assert len(images) == 1
        assert len(texts) == 1
        assert texts[0] == "The quick brown fox"
        assert images[0].getpixel((0, 0))[0] == 0

    def test_single_image_mutation(self, vlm):
        g = vlm.zero_genotype()
        g[0] = 1
        weights = g.reshape(1, -1)
        images, texts = vlm.manipulate(candidates=None, weights=weights)
        assert texts[0] == "The quick brown fox"
        assert images[0].getpixel((0, 0))[0] == 1

    def test_single_text_mutation(self, vlm):
        g = vlm.zero_genotype()
        g[vlm.image_dim] = 1  # mutate first text gene
        weights = g.reshape(1, -1)
        images, texts = vlm.manipulate(candidates=None, weights=weights)
        assert "fast" in texts[0].lower()
        assert images[0].getpixel((0, 0))[0] == 0

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
