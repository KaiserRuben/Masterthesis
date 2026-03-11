"""Tests for TextReplacementDistance, TargetedBalance, and Concentration criteria.

All tests use real tensors and arrays with known values -- no mocks.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

# Ensure project root is importable
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.objectives import (
    Concentration,
    CriterionCollection,
    TargetedBalance,
    TextReplacementDistance,
)


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

# Index assignments for a 10-class vocabulary.
TENCH, GOLDFISH, GREAT_WHITE, TIGER_SHARK = 0, 1, 2, 3
HAMMERHEAD, ELECTRIC_RAY, STINGRAY, COCK, HEN, OSTRICH = 4, 5, 6, 7, 8, 9
N_CLASSES = 10

# Realistic log-prob vectors from VLM scoring.
EASY_LOGITS = torch.tensor([
    -0.0007,   # tench       (dominant)
    -5.9734,   # goldfish
    -6.2444,   # great_white_shark
    -6.3307,   # tiger_shark
    -9.2230,   # hammerhead
    -10.5205,  # electric_ray
    -12.0000,  # stingray
    -14.8340,  # cock
    -15.8164,  # hen
    -17.7656,  # ostrich
])

HARD_LOGITS = torch.tensor([
    -1.6758,   # tench       (near-boundary with stingray)
    -2.8166,   # goldfish
    -3.4982,   # great_white_shark
    -5.1875,   # tiger_shark
    -4.1200,   # hammerhead
    -6.3240,   # electric_ray
    -1.6476,   # stingray
    -9.8438,   # cock
    -8.1172,   # hen
    -8.6211,   # ostrich
])

# Text criterion test data: precomputed cosine distances per word position.
# Position 0 ("dog"): 3 candidates at distances 0.05, 0.15, 1.0
# Position 1 ("big"): 2 candidates at distances 0.08, 0.20
TEXT_DISTANCES: tuple[np.ndarray, ...] = (
    np.array([0.05, 0.15, 1.0]),
    np.array([0.08, 0.20]),
)


def _softmax(logits: torch.Tensor) -> torch.Tensor:
    """Compute softmax for reference calculations."""
    return F.softmax(logits, dim=-1)


# ===========================================================================
# TextReplacementDistance
# ===========================================================================


class TestTextReplacementDistance:

    def test_zero_genotype_returns_zero(self):
        """All genes = 0 means no replacements, distance = 0."""
        criterion = TextReplacementDistance()
        geno = np.array([0, 0], dtype=np.int64)
        result = criterion.evaluate(
            text_genotypes=geno,
            text_candidate_distances=TEXT_DISTANCES,
            batch_dim=None,
        )
        assert result == [0.0]

    def test_single_gene_active(self):
        """Gene = 1 at position 0: uses nearest candidate."""
        criterion = TextReplacementDistance()
        geno = np.array([1, 0], dtype=np.int64)
        result = criterion.evaluate(
            text_genotypes=geno,
            text_candidate_distances=TEXT_DISTANCES,
            batch_dim=None,
        )
        assert len(result) == 1
        assert result[0] == pytest.approx(0.05)

    def test_multiple_genes_active(self):
        """Genes active at both positions: sum of individual distances."""
        criterion = TextReplacementDistance()
        geno = np.array([1, 2], dtype=np.int64)
        result = criterion.evaluate(
            text_genotypes=geno,
            text_candidate_distances=TEXT_DISTANCES,
            batch_dim=None,
        )
        expected = 0.05 + 0.20  # pos0 cand1 + pos1 cand2
        assert result[0] == pytest.approx(expected)

    def test_batched_evaluation(self):
        """Batch of genotypes returns a list with one distance per individual."""
        criterion = TextReplacementDistance()
        genos = np.array([
            [0, 0],  # no change
            [1, 0],  # pos0 cand1 only
            [0, 1],  # pos1 cand1 only
            [2, 2],  # pos0 cand2 + pos1 cand2
            [3, 1],  # pos0 cand3 + pos1 cand1
        ], dtype=np.int64)
        results = criterion.evaluate(
            text_genotypes=genos,
            text_candidate_distances=TEXT_DISTANCES,
            batch_dim=0,
        )

        assert len(results) == 5
        assert results[0] == pytest.approx(0.0)
        assert results[1] == pytest.approx(0.05)
        assert results[2] == pytest.approx(0.08)
        assert results[3] == pytest.approx(0.15 + 0.20)
        assert results[4] == pytest.approx(1.0 + 0.08)

    def test_max_candidate_index(self):
        """Gene value at the maximum candidate index works correctly."""
        criterion = TextReplacementDistance()
        # Position 0 has 3 candidates (max gene = 3), position 1 has 2 (max gene = 2)
        geno = np.array([3, 2], dtype=np.int64)
        result = criterion.evaluate(
            text_genotypes=geno,
            text_candidate_distances=TEXT_DISTANCES,
            batch_dim=None,
        )
        expected = 1.0 + 0.20  # pos0 cand3 + pos1 cand2
        assert result[0] == pytest.approx(expected)

    def test_1d_genotype_batch_dim_none(self):
        """A 1-D genotype with batch_dim=None returns list of one float."""
        criterion = TextReplacementDistance()
        geno = np.array([2, 1], dtype=np.int64)
        result = criterion.evaluate(
            text_genotypes=geno,
            text_candidate_distances=TEXT_DISTANCES,
            batch_dim=None,
        )
        expected = 0.15 + 0.08
        assert len(result) == 1
        assert result[0] == pytest.approx(expected)

    def test_criterion_name(self):
        """Name is TextDist."""
        criterion = TextReplacementDistance()
        assert criterion.name == "TextDist"


# ===========================================================================
# TargetedBalance
# ===========================================================================


class TestTargetedBalance:

    def test_perfect_balance(self):
        """Two targets have equal high logits, others very low -> balance ~ 0."""
        logits = torch.full((N_CLASSES,), -100.0)
        logits[TENCH] = 0.0
        logits[STINGRAY] = 0.0

        criterion = TargetedBalance()
        result = criterion.evaluate(
            logits=logits,
            target_classes=(TENCH, STINGRAY),
            batch_dim=None,
        )

        assert len(result) == 1
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_complete_imbalance(self):
        """One target class dominates -> balance near 1.0."""
        logits = torch.full((N_CLASSES,), -100.0)
        logits[TENCH] = 0.0
        logits[STINGRAY] = -100.0

        criterion = TargetedBalance()
        result = criterion.evaluate(
            logits=logits,
            target_classes=(TENCH, STINGRAY),
            batch_dim=None,
        )

        probs = _softmax(logits)
        expected = abs(probs[TENCH].item() - probs[STINGRAY].item())
        assert result[0] == pytest.approx(expected, abs=1e-6)
        assert result[0] > 0.99

    def test_realistic_easy_case(self):
        """Tench at -0.0007, goldfish at -5.97 -> highly imbalanced."""
        criterion = TargetedBalance()
        result = criterion.evaluate(
            logits=EASY_LOGITS,
            target_classes=(TENCH, GOLDFISH),
            batch_dim=None,
        )

        probs = _softmax(EASY_LOGITS)
        expected = abs(probs[TENCH].item() - probs[GOLDFISH].item())
        assert result[0] == pytest.approx(expected, abs=1e-6)
        assert result[0] > 0.95

    def test_realistic_hard_case(self):
        """Stingray at -1.6476, tench at -1.6758 -> nearly balanced."""
        criterion = TargetedBalance()
        result = criterion.evaluate(
            logits=HARD_LOGITS,
            target_classes=(STINGRAY, TENCH),
            batch_dim=None,
        )

        probs = _softmax(HARD_LOGITS)
        expected = abs(probs[STINGRAY].item() - probs[TENCH].item())
        assert result[0] == pytest.approx(expected, abs=1e-6)
        assert result[0] < 0.05

    def test_softmax_applied_correctly(self):
        """Verify softmax is actually applied (not treating inputs as probs)."""
        logits = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
        criterion = TargetedBalance()
        result = criterion.evaluate(
            logits=logits,
            target_classes=(0, 1),
            batch_dim=None,
        )
        # Uniform distribution -> P(0) = P(1) = 0.2, balance = 0
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_batched_evaluation(self):
        """Multiple logit vectors -> list of correct results."""
        logits = torch.stack([EASY_LOGITS, HARD_LOGITS])
        criterion = TargetedBalance()
        results = criterion.evaluate(
            logits=logits,
            target_classes=(TENCH, STINGRAY),
            batch_dim=0,
        )

        assert isinstance(results, list)
        assert len(results) == 2

        probs_easy = _softmax(EASY_LOGITS)
        probs_hard = _softmax(HARD_LOGITS)
        expected_easy = abs(probs_easy[TENCH].item() - probs_easy[STINGRAY].item())
        expected_hard = abs(probs_hard[TENCH].item() - probs_hard[STINGRAY].item())

        assert results[0] == pytest.approx(expected_easy, abs=1e-6)
        assert results[1] == pytest.approx(expected_hard, abs=1e-6)

    def test_uniform_distribution(self):
        """All classes equal -> P(A)=P(B)=1/N, balance = 0."""
        logits = torch.zeros(N_CLASSES)
        criterion = TargetedBalance()
        result = criterion.evaluate(
            logits=logits,
            target_classes=(TENCH, STINGRAY),
            batch_dim=None,
        )
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_criterion_name(self):
        """Name is TgtBal."""
        criterion = TargetedBalance()
        assert criterion.name == "TgtBal"


# ===========================================================================
# Concentration
# ===========================================================================


class TestConcentration:

    def test_all_mass_on_target_pair(self):
        """All mass on target pair -> concentration ~ 0."""
        logits = torch.full((N_CLASSES,), -100.0)
        logits[TENCH] = 0.0
        logits[STINGRAY] = 0.0

        criterion = Concentration()
        result = criterion.evaluate(
            logits=logits,
            target_classes=(TENCH, STINGRAY),
            batch_dim=None,
        )

        assert len(result) == 1
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_all_mass_on_non_target(self):
        """All probability on non-target class -> concentration ~ 1.0."""
        logits = torch.full((N_CLASSES,), -100.0)
        logits[GOLDFISH] = 0.0

        criterion = Concentration()
        result = criterion.evaluate(
            logits=logits,
            target_classes=(TENCH, STINGRAY),
            batch_dim=None,
        )

        assert result[0] > 0.99

    def test_uniform_distribution(self):
        """All classes equal -> concentration = (N-2)/N."""
        logits = torch.zeros(N_CLASSES)
        criterion = Concentration()
        result = criterion.evaluate(
            logits=logits,
            target_classes=(TENCH, STINGRAY),
            batch_dim=None,
        )

        expected = (N_CLASSES - 2) / N_CLASSES  # 8/10 = 0.8
        assert result[0] == pytest.approx(expected, abs=1e-6)

    def test_realistic_easy_case(self):
        """Tench dominant -> almost all mass on tench, low concentration."""
        criterion = Concentration()
        result = criterion.evaluate(
            logits=EASY_LOGITS,
            target_classes=(TENCH, GOLDFISH),
            batch_dim=None,
        )

        probs = _softmax(EASY_LOGITS)
        outside = probs.sum().item() - probs[TENCH].item() - probs[GOLDFISH].item()
        assert result[0] == pytest.approx(outside, abs=1e-6)
        assert result[0] < 0.01

    def test_realistic_hard_case(self):
        """Near-boundary with leakage to goldfish -> moderate concentration."""
        criterion = Concentration()
        result = criterion.evaluate(
            logits=HARD_LOGITS,
            target_classes=(STINGRAY, TENCH),
            batch_dim=None,
        )

        probs = _softmax(HARD_LOGITS)
        outside = probs.sum().item() - probs[STINGRAY].item() - probs[TENCH].item()
        assert result[0] == pytest.approx(outside, abs=1e-6)
        assert result[0] > 0.1

    def test_probability_sums_to_one(self):
        """Verify: concentration + P(A) + P(B) ~ 1.0."""
        criterion = Concentration()
        result = criterion.evaluate(
            logits=HARD_LOGITS,
            target_classes=(STINGRAY, TENCH),
            batch_dim=None,
        )

        probs = _softmax(HARD_LOGITS)
        p_a = probs[STINGRAY].item()
        p_b = probs[TENCH].item()
        assert result[0] + p_a + p_b == pytest.approx(1.0, abs=1e-6)

    def test_batched_evaluation(self):
        """Multiple logit vectors -> list of correct results."""
        logits = torch.stack([EASY_LOGITS, HARD_LOGITS])
        criterion = Concentration()
        results = criterion.evaluate(
            logits=logits,
            target_classes=(TENCH, STINGRAY),
            batch_dim=0,
        )

        assert isinstance(results, list)
        assert len(results) == 2

        for i, single_logits in enumerate([EASY_LOGITS, HARD_LOGITS]):
            probs = _softmax(single_logits)
            expected = probs.sum().item() - probs[TENCH].item() - probs[STINGRAY].item()
            assert results[i] == pytest.approx(expected, abs=1e-6)

    def test_criterion_name(self):
        """Name is Conc."""
        criterion = Concentration()
        assert criterion.name == "Conc"


# ===========================================================================
# CriterionCollection integration
# ===========================================================================


class TestCriterionInterface:

    def test_all_criteria_with_collection(self):
        """All three criteria work with CriterionCollection.evaluate_all().

        Passes all kwargs at once; each criterion picks up its own.
        """
        text_crit = TextReplacementDistance()
        balance_crit = TargetedBalance()
        conc_crit = Concentration()

        collection = CriterionCollection(text_crit, balance_crit, conc_crit)

        # Shared kwargs — each criterion takes what it needs.
        logits = torch.stack([EASY_LOGITS, HARD_LOGITS])
        text_genos = np.array([[1, 0], [0, 1]], dtype=np.int64)

        collection.evaluate_all(
            logits=logits,
            target_classes=(TENCH, STINGRAY),
            text_genotypes=text_genos,
            text_candidate_distances=TEXT_DISTANCES,
            batch_dim=0,
        )

        results = collection.results
        assert "TextDist" in results
        assert "TgtBal" in results
        assert "Conc" in results

        # Verify shapes
        assert isinstance(results["TextDist"], list)
        assert len(results["TextDist"]) == 2
        assert isinstance(results["TgtBal"], list)
        assert len(results["TgtBal"]) == 2
        assert isinstance(results["Conc"], list)
        assert len(results["Conc"]) == 2

    def test_criterion_names(self):
        """Verify .name returns the correct string for each criterion."""
        assert TextReplacementDistance().name == "TextDist"
        assert TargetedBalance().name == "TgtBal"
        assert Concentration().name == "Conc"

    def test_collection_names(self):
        """CriterionCollection.names returns all three names."""
        collection = CriterionCollection(
            TextReplacementDistance(),
            TargetedBalance(),
            Concentration(),
        )
        assert collection.names == ["TextDist", "TgtBal", "Conc"]

    def test_results_retrieval(self):
        """CriterionCollection.results allows retrieval by criterion name."""
        text_crit = TextReplacementDistance()
        balance_crit = TargetedBalance()
        conc_crit = Concentration()

        collection = CriterionCollection(text_crit, balance_crit, conc_crit)

        logits = HARD_LOGITS.unsqueeze(0)
        text_genos = np.array([[1, 1]], dtype=np.int64)

        collection.evaluate_all(
            logits=logits,
            target_classes=(STINGRAY, TENCH),
            text_genotypes=text_genos,
            text_candidate_distances=TEXT_DISTANCES,
            batch_dim=0,
        )

        results = collection.results
        text_result = results["TextDist"]
        assert isinstance(text_result, list)
        assert text_result[0] == pytest.approx(0.05 + 0.08)

        # TargetedBalance and Concentration also present
        assert isinstance(results["TgtBal"], list)
        assert isinstance(results["Conc"], list)

        # Verify concentration + P(A) + P(B) ~ 1.0
        probs = _softmax(HARD_LOGITS)
        p_a = probs[STINGRAY].item()
        p_b = probs[TENCH].item()
        assert results["Conc"][0] + p_a + p_b == pytest.approx(1.0, abs=1e-6)
