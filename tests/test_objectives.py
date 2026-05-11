"""Tests for TargetedBalance and TextEmbeddingDistance criteria.

All tests use real tensors and arrays with known values — no mocks.

TargetedBalance returns ``|lp_A - lp_B|`` — the raw log-prob gap, not a
softmax-prob difference. See ``src/objectives/targeted_balance.py`` and
Diary 2026-04-15-Exp05-Numerical-Floor-Discovery for the rationale.
"""

import numpy as np
import pytest
import torch

from src.objectives import (
    CriterionCollection,
    TargetedBalance,
    TextEmbeddingDistance,
)


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

TENCH, GOLDFISH, GREAT_WHITE, TIGER_SHARK = 0, 1, 2, 3
HAMMERHEAD, ELECTRIC_RAY, STINGRAY, COCK, HEN, OSTRICH = 4, 5, 6, 7, 8, 9
N_CLASSES = 10

EASY_LOGITS = torch.tensor([
    -0.0007, -5.9734, -6.2444, -6.3307, -9.2230,
    -10.5205, -12.0000, -14.8340, -15.8164, -17.7656,
])

HARD_LOGITS = torch.tensor([
    -1.6758, -2.8166, -3.4982, -5.1875, -4.1200,
    -6.3240, -1.6476, -9.8438, -8.1172, -8.6211,
])


def _gap(logits: torch.Tensor, a: int, b: int) -> float:
    return abs(float(logits[a]) - float(logits[b]))


# ===========================================================================
# TextEmbeddingDistance
# ===========================================================================


class TestTextEmbeddingDistance:
    def test_pass_through_single(self):
        crit = TextEmbeddingDistance()
        result = crit.evaluate(text_distances=np.array([0.42]))
        assert result == pytest.approx([0.42])

    def test_pass_through_batched(self):
        crit = TextEmbeddingDistance()
        result = crit.evaluate(text_distances=np.array([0.0, 0.1, 0.5]), batch_dim=0)
        assert result == pytest.approx([0.0, 0.1, 0.5])

    def test_zero_distance(self):
        crit = TextEmbeddingDistance()
        result = crit.evaluate(text_distances=np.zeros(3), batch_dim=0)
        assert result == [0.0, 0.0, 0.0]

    def test_name(self):
        assert TextEmbeddingDistance().name == "TextDist"


# ===========================================================================
# TargetedBalance
# ===========================================================================


class TestTargetedBalance:

    def test_perfect_balance(self):
        logits = torch.full((N_CLASSES,), -100.0)
        logits[TENCH] = 0.0
        logits[STINGRAY] = 0.0
        crit = TargetedBalance()
        result = crit.evaluate(
            logits=logits, target_classes=(TENCH, STINGRAY), batch_dim=None,
        )
        assert len(result) == 1
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_complete_imbalance(self):
        logits = torch.full((N_CLASSES,), -100.0)
        logits[TENCH] = 0.0
        logits[STINGRAY] = -100.0
        crit = TargetedBalance()
        result = crit.evaluate(
            logits=logits, target_classes=(TENCH, STINGRAY), batch_dim=None,
        )
        expected = _gap(logits, TENCH, STINGRAY)
        assert result[0] == pytest.approx(expected, abs=1e-6)
        assert result[0] == pytest.approx(100.0, abs=1e-6)

    def test_realistic_easy_case(self):
        crit = TargetedBalance()
        result = crit.evaluate(
            logits=EASY_LOGITS, target_classes=(TENCH, GOLDFISH), batch_dim=None,
        )
        expected = _gap(EASY_LOGITS, TENCH, GOLDFISH)
        assert result[0] == pytest.approx(expected, abs=1e-6)
        assert result[0] > 5.0  # large log-prob gap on the easy case

    def test_realistic_hard_case(self):
        crit = TargetedBalance()
        result = crit.evaluate(
            logits=HARD_LOGITS, target_classes=(STINGRAY, TENCH), batch_dim=None,
        )
        expected = _gap(HARD_LOGITS, STINGRAY, TENCH)
        assert result[0] == pytest.approx(expected, abs=1e-6)
        assert result[0] < 0.05  # near the boundary on the hard case

    def test_softmax_applied_correctly(self):
        logits = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
        crit = TargetedBalance()
        result = crit.evaluate(
            logits=logits, target_classes=(0, 1), batch_dim=None,
        )
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_batched_evaluation(self):
        logits = torch.stack([EASY_LOGITS, HARD_LOGITS])
        crit = TargetedBalance()
        results = crit.evaluate(
            logits=logits, target_classes=(TENCH, STINGRAY), batch_dim=0,
        )
        assert isinstance(results, list)
        assert len(results) == 2
        expected_easy = _gap(EASY_LOGITS, TENCH, STINGRAY)
        expected_hard = _gap(HARD_LOGITS, TENCH, STINGRAY)
        assert results[0] == pytest.approx(expected_easy, abs=1e-6)
        assert results[1] == pytest.approx(expected_hard, abs=1e-6)

    def test_uniform_distribution(self):
        logits = torch.zeros(N_CLASSES)
        crit = TargetedBalance()
        result = crit.evaluate(
            logits=logits, target_classes=(TENCH, STINGRAY), batch_dim=None,
        )
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_criterion_name(self):
        assert TargetedBalance().name == "TgtBal"


# ===========================================================================
# CriterionCollection integration
# ===========================================================================


class TestCriterionInterface:
    def test_collection_evaluate_all(self):
        text_crit = TextEmbeddingDistance()
        balance_crit = TargetedBalance()
        collection = CriterionCollection(text_crit, balance_crit)
        logits = torch.stack([EASY_LOGITS, HARD_LOGITS])
        collection.evaluate_all(
            logits=logits,
            target_classes=(TENCH, STINGRAY),
            text_distances=np.array([0.13, 0.07]),
            batch_dim=0,
        )
        results = collection.results
        assert "TextDist" in results
        assert "TgtBal" in results
        assert len(results["TextDist"]) == 2
        assert len(results["TgtBal"]) == 2

    def test_collection_names(self):
        collection = CriterionCollection(
            TextEmbeddingDistance(), TargetedBalance(),
        )
        assert collection.names == ["TextDist", "TgtBal"]

    def test_results_retrieval(self):
        text_crit = TextEmbeddingDistance()
        balance_crit = TargetedBalance()
        collection = CriterionCollection(text_crit, balance_crit)
        logits = HARD_LOGITS.unsqueeze(0)
        collection.evaluate_all(
            logits=logits,
            target_classes=(STINGRAY, TENCH),
            text_distances=np.array([0.13]),
            batch_dim=0,
        )
        text_result = collection.results["TextDist"]
        assert text_result[0] == pytest.approx(0.13)
