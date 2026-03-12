"""Tests for VLMBoundaryTester, NormalizedGenomeDistance, and ArchiveSparsity.

Uses synthetic sub-manipulators and a fake SUT — no real VLM or VQGAN
models are loaded.  The full optimisation loop runs with tiny parameters
(pop_size=5, generations=3) to verify wiring and data flow.
"""

import json

import numpy as np
import pytest
import torch
from PIL import Image

from src.manipulator.vlm_manipulator import VLMManipulator
from src.objectives import (
    ArchiveSparsity,
    Concentration,
    CriterionCollection,
    MatrixDistance,
    NormalizedGenomeDistance,
    TargetedBalance,
    TextReplacementDistance,
)
from src.optimizer.discrete_pymoo_optimizer import DiscretePymooOptimizer
from src.config import ExperimentConfig, SeedTriple
from src.tester.vlm_boundary_tester import (
    VLMBoundaryTester,
    _pil_to_tensor,
)

from conftest import (
    FakeImageManipulator,
    FakeTextManipulator,
    make_embeddings,
    VOCAB,
)


# ---------------------------------------------------------------------------
# Test-local constants and fakes
# ---------------------------------------------------------------------------

CATEGORIES = ("cat_a", "cat_b", "cat_c")
PROMPT_TEMPLATE = "Classify this"
ANSWER_FORMAT = " from: {categories}."


class FakeSUT:
    """Returns deterministic logprobs favouring cat_a."""

    def process_input(self, image, text=None, categories=None):
        return torch.tensor([-0.1, -2.0, -5.0])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def embeddings():
    return make_embeddings(VOCAB)


@pytest.fixture()
def manipulator(embeddings):
    return VLMManipulator(
        image_manipulator=FakeImageManipulator(),
        text_manipulator=FakeTextManipulator(embeddings),
    )


@pytest.fixture()
def objectives():
    """4 batched objectives."""
    return CriterionCollection(
        MatrixDistance(),
        TextReplacementDistance(),
        TargetedBalance(),
        Concentration(),
    )


@pytest.fixture()
def config(tmp_path):
    return ExperimentConfig(
        categories=CATEGORIES,
        prompt_template=PROMPT_TEMPLATE,
        answer_format=ANSWER_FORMAT,
        generations=3,
        pop_size=5,
        save_dir=tmp_path / "runs",
        name="test_exp",
    )


@pytest.fixture()
def seed():
    return SeedTriple(
        image=Image.new("RGB", (8, 8), (128, 64, 32)),
        class_a="cat_a",
        class_b="cat_b",
    )


# ---------------------------------------------------------------------------
# TestNormalizedGenomeDistance
# ---------------------------------------------------------------------------


class TestNormalizedGenomeDistance:
    def test_identical_is_zero(self):
        g = np.array([5, 10, 15])
        bounds = np.array([10, 20, 30])
        metric = NormalizedGenomeDistance(bounds)
        assert metric.evaluate(images=[g, g.copy()]) == 0.0

    def test_max_distance_is_one(self):
        bounds = np.array([11, 21])  # max diffs = 10, 20
        metric = NormalizedGenomeDistance(bounds)
        a = np.array([0, 0])
        b = np.array([10, 20])
        np.testing.assert_allclose(
            metric.evaluate(images=[a, b]), 1.0, atol=1e-9,
        )

    def test_intermediate_distance(self):
        bounds = np.array([11, 11])  # max diffs = 10, 10
        metric = NormalizedGenomeDistance(bounds)
        a = np.array([0, 0])
        b = np.array([5, 5])
        np.testing.assert_allclose(
            metric.evaluate(images=[a, b]), 0.5, atol=1e-9,
        )

    def test_handles_bound_of_one(self):
        """Gene with bound=1 has max_diff=1 (clamped), avoids div-by-zero."""
        bounds = np.array([1, 11])
        metric = NormalizedGenomeDistance(bounds)
        a = np.array([0, 0])
        b = np.array([0, 10])
        assert metric.evaluate(images=[a, b]) == pytest.approx(0.5)

    def test_asymmetric_bounds(self):
        bounds = np.array([4, 26])  # max diffs = 3, 25
        metric = NormalizedGenomeDistance(bounds)
        a = np.array([0, 0])
        b = np.array([3, 25])
        np.testing.assert_allclose(
            metric.evaluate(images=[a, b]), 1.0, atol=1e-9,
        )


# ---------------------------------------------------------------------------
# TestArchiveSparsityIntegration
# ---------------------------------------------------------------------------


class TestArchiveSparsityIntegration:
    def test_with_genome_distance_metric(self):
        bounds = np.array([10, 10, 10])
        metric = NormalizedGenomeDistance(bounds)
        sparsity = ArchiveSparsity(metric=metric, regime="min", on_genomes=True)

        target = np.array([5, 5, 5])
        archive = [np.array([0, 0, 0]), np.array([9, 9, 9])]

        val = sparsity.evaluate(
            images=[None, None],
            solution_archive=[],
            genome_target=target,
            genome_archive=archive,
        )
        # min dist to archive: dist to [9,9,9] = mean(4/9, 4/9, 4/9) ≈ 0.444
        # sparsity = 1 - 0.444 ≈ 0.556
        assert 0.0 < val < 1.0

    def test_identical_to_archive_gives_one(self):
        bounds = np.array([10, 10])
        metric = NormalizedGenomeDistance(bounds)
        sparsity = ArchiveSparsity(metric=metric, regime="min", on_genomes=True)

        target = np.array([3, 3])
        archive = [target.copy()]

        val = sparsity.evaluate(
            images=[None, None],
            solution_archive=[],
            genome_target=target,
            genome_archive=archive,
        )
        # dist = 0, sparsity = 1 - 0 = 1.0 (bad: not diverse)
        assert val == pytest.approx(1.0)

    def test_far_from_archive_gives_low_value(self):
        bounds = np.array([11, 11])  # max diff = 10
        metric = NormalizedGenomeDistance(bounds)
        sparsity = ArchiveSparsity(metric=metric, regime="min", on_genomes=True)

        target = np.array([10, 10])
        archive = [np.array([0, 0])]

        val = sparsity.evaluate(
            images=[None, None],
            solution_archive=[],
            genome_target=target,
            genome_archive=archive,
        )
        # dist = 1.0, sparsity = 1 - 1.0 = 0.0 (good: very diverse)
        assert val == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestUtilities
# ---------------------------------------------------------------------------


class TestUtilities:
    def test_pil_to_tensor_shape(self):
        img = Image.new("RGB", (16, 8))
        t = _pil_to_tensor(img)
        assert t.shape == (3, 8, 16)  # C, H, W

    def test_pil_to_tensor_range(self):
        img = Image.new("RGB", (4, 4), (255, 128, 0))
        t = _pil_to_tensor(img)
        assert t.min() >= 0.0
        assert t.max() <= 1.0

    def test_pil_to_tensor_grayscale(self):
        img = Image.new("L", (4, 4))
        t = _pil_to_tensor(img)
        assert t.shape == (1, 4, 4)


# ---------------------------------------------------------------------------
# TestVLMBoundaryTester
# ---------------------------------------------------------------------------


class TestVLMBoundaryTester:
    def _make_tester(self, config, manipulator, objectives):
        optimizer = DiscretePymooOptimizer(
            gene_bounds=np.ones(1, dtype=np.int64) * 2,
            num_objectives=5,  # 4 batched + 1 sparsity
            pop_size=config.pop_size,
        )
        return VLMBoundaryTester(
            sut=FakeSUT(),
            manipulator=manipulator,
            optimizer=optimizer,
            objectives=objectives,
            config=config,
        )

    def test_full_loop_runs(self, config, manipulator, objectives, seed):
        tester = self._make_tester(config, manipulator, objectives)
        tester.test([seed])

    def test_trace_parquet_exists(self, config, manipulator, objectives, seed):
        tester = self._make_tester(config, manipulator, objectives)
        tester.test([seed])

        run_dirs = list(config.save_dir.iterdir())
        assert len(run_dirs) == 1
        assert (run_dirs[0] / "trace.parquet").exists()

    def test_trace_row_count(self, config, manipulator, objectives, seed):
        tester = self._make_tester(config, manipulator, objectives)
        tester.test([seed])

        import pandas as pd

        run_dir = next(config.save_dir.iterdir())
        df = pd.read_parquet(run_dir / "trace.parquet")
        expected = config.generations * config.pop_size
        assert len(df) == expected

    def test_trace_has_all_fitness_columns(
        self, config, manipulator, objectives, seed,
    ):
        tester = self._make_tester(config, manipulator, objectives)
        tester.test([seed])

        import pandas as pd

        run_dir = next(config.save_dir.iterdir())
        df = pd.read_parquet(run_dir / "trace.parquet")

        # 4 batched + 1 ArchiveSparsity
        fitness_cols = [c for c in df.columns if c.startswith("fitness_")]
        assert len(fitness_cols) == 5
        assert "fitness_ArchiveSparsity" in df.columns

    def test_trace_has_required_columns(
        self, config, manipulator, objectives, seed,
    ):
        tester = self._make_tester(config, manipulator, objectives)
        tester.test([seed])

        import pandas as pd

        run_dir = next(config.save_dir.iterdir())
        df = pd.read_parquet(run_dir / "trace.parquet")

        required = {
            "seed_id", "generation", "individual",
            "genotype", "logprobs", "decoded_text",
            "predicted_class", "p_class_a", "p_class_b",
        }
        assert required.issubset(set(df.columns))

    def test_stats_json(self, config, manipulator, objectives, seed):
        tester = self._make_tester(config, manipulator, objectives)
        tester.test([seed])

        run_dir = next(config.save_dir.iterdir())
        with open(run_dir / "stats.json") as f:
            stats = json.load(f)

        assert stats["class_a"] == "cat_a"
        assert stats["class_b"] == "cat_b"
        assert stats["generations"] == config.generations
        assert stats["pop_size"] == config.pop_size

    def test_context_json(self, config, manipulator, objectives, seed):
        tester = self._make_tester(config, manipulator, objectives)
        tester.test([seed])

        run_dir = next(config.save_dir.iterdir())
        with open(run_dir / "context.json") as f:
            ctx = json.load(f)

        assert "image_patch_positions" in ctx
        assert "text_original_words" in ctx
        assert "text_candidate_distances" in ctx

    def test_origin_image_saved(self, config, manipulator, objectives, seed):
        tester = self._make_tester(config, manipulator, objectives)
        tester.test([seed])

        run_dir = next(config.save_dir.iterdir())
        assert (run_dir / "origin.png").exists()

    def test_pareto_candidates_saved(
        self, config, manipulator, objectives, seed,
    ):
        tester = self._make_tester(config, manipulator, objectives)
        tester.test([seed])

        run_dir = next(config.save_dir.iterdir())
        pareto_pngs = list(run_dir.glob("pareto_*.png"))
        pareto_jsons = list(run_dir.glob("pareto_*.json"))
        assert len(pareto_pngs) > 0
        assert len(pareto_pngs) == len(pareto_jsons)

    def test_two_seeds_produce_two_dirs(
        self, config, manipulator, objectives, seed,
    ):
        seed2 = SeedTriple(
            image=Image.new("RGB", (8, 8), (0, 0, 0)),
            class_a="cat_a",
            class_b="cat_c",
        )
        tester = self._make_tester(config, manipulator, objectives)
        tester.test([seed, seed2])

        run_dirs = list(config.save_dir.iterdir())
        assert len(run_dirs) == 2
