"""Tests for the boundary-pair (evolutionary → PDQ) pipeline.

Unit-level coverage:

* Stage-2 passes with non-zero ``target`` snap toward the target and
  reduce delta-rank-sum / delta-sparsity (not absolute rank_sum).
* ``rank_sum_delta`` input distance produces the expected L1.
* Anchor selection respects ``anchor_selection.k``.
* Pair-softmax label assignment picks the larger pair entry.
* Boundary-pair config projections to per-stage configs preserve the
  shared fields verbatim.
* Archive parquet declares the new PDQ v5 provenance columns.

End-to-end testing (real SUT, real manipulator, real seeds) is covered
by the smoke run — see :mod:`src.boundary_pair` README.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from src.pdq.artifacts import ARCHIVE_COLUMNS
from src.pdq.config import VALID_D_I
from src.pdq.distances.input import rank_sum, rank_sum_delta
from src.pdq.metric import INPUT_DISTANCES
from src.pdq.search.stage2 import (
    CheckResult,
    Stage2Result,
    greedy_zeroing,
    minimise_flip,
    random_subset,
    rank_reduction,
)
from src.pdq.config import (
    PassesConfig,
    RandomSubsetPassConfig,
    RankPassConfig,
    Stage2Config,
    ZeroPassConfig,
)


# ---------------------------------------------------------------------------
# Stage-2 with non-zero target
# ---------------------------------------------------------------------------


def _always_flipped(_geno: np.ndarray) -> CheckResult:
    """Mock check fn that accepts every step (flip always preserved)."""
    return CheckResult(
        still_flipped=True,
        label="other",
        sut_call_id=0,
        wall_time_cum=0.0,
    )


def _flipped_unless_matches_target(target: np.ndarray):
    """Mock factory: flip breaks the moment partner == target everywhere."""

    def check(geno: np.ndarray) -> CheckResult:
        still = bool(np.any(geno != target))
        return CheckResult(
            still_flipped=still,
            label="other" if still else "anchor",
            sut_call_id=0,
            wall_time_cum=0.0,
        )

    return check


class TestGreedyZeroingWithTarget:
    """Pass A — snap-to-target semantics."""

    def test_target_none_matches_legacy_zero(self) -> None:
        """target=None behaves identically to zero-target (legacy PDQ)."""
        g = np.array([3, 0, 2, 5, 0, 1], dtype=np.int64)
        g_a, _ = greedy_zeroing(g, _always_flipped, budget=10)
        g_b, _ = greedy_zeroing(
            g, _always_flipped, budget=10, target=np.zeros_like(g),
        )
        assert np.array_equal(g_a, g_b)
        assert int(np.sum(g_a)) == 0

    def test_target_anchor_snaps_genes_to_anchor(self) -> None:
        anchor = np.array([2, 5, 0, 3, 7, 1], dtype=np.int64)
        flipped = np.array([4, 1, 6, 0, 7, 9], dtype=np.int64)

        out, traj = greedy_zeroing(
            flipped, _always_flipped, budget=10, target=anchor,
        )
        # Position 4 starts equal — never touched. Rest snap to anchor.
        assert np.array_equal(out, anchor)
        n_differing = int(np.sum(flipped != anchor))
        assert len(traj) == n_differing
        for step in traj:
            gene_idx = step["target_gene"]
            assert step["new_value"] == int(anchor[gene_idx])

    def test_target_anchor_delta_rank_sum_decreases(self) -> None:
        anchor = np.array([2, 5, 0, 3], dtype=np.int64)
        flipped = np.array([0, 1, 4, 7], dtype=np.int64)

        _, traj = greedy_zeroing(
            flipped, _always_flipped, budget=10, target=anchor,
        )
        delta_before = int(np.sum(np.abs(flipped - anchor)))
        assert traj[0]["rank_sum_before"] == delta_before
        assert traj[-1]["rank_sum_after"] == 0

    def test_rejected_step_leaves_gene_unchanged(self) -> None:
        anchor = np.array([0, 0, 0], dtype=np.int64)
        flipped = np.array([3, 2, 1], dtype=np.int64)

        out, _ = greedy_zeroing(
            flipped,
            _flipped_unless_matches_target(anchor),
            budget=10,
            target=anchor,
        )
        assert int(np.sum(out != anchor)) >= 1


class TestRankReductionWithTarget:
    def test_step_toward_target_below(self) -> None:
        # gene 0: |3-1| = 2 (decrement direction)
        # gene 1: |2-3| = 1 (increment direction)
        anchor = np.array([1, 3], dtype=np.int64)
        flipped = np.array([3, 2], dtype=np.int64)

        out, traj = rank_reduction(
            flipped, _always_flipped, budget=20, step_size=1, target=anchor,
        )
        assert out[0] == anchor[0]
        assert out[1] == anchor[1]
        assert traj[0]["target_gene"] == 0
        assert traj[0]["old_value"] == 3 and traj[0]["new_value"] == 2

    def test_step_toward_target_above(self) -> None:
        anchor = np.array([10], dtype=np.int64)
        flipped = np.array([7], dtype=np.int64)

        out, traj = rank_reduction(
            flipped, _always_flipped, budget=20, step_size=1, target=anchor,
        )
        assert out[0] == 10
        assert traj[0]["old_value"] == 7 and traj[0]["new_value"] == 8


class TestRandomSubsetWithTarget:
    def test_subset_snaps_to_target(self) -> None:
        anchor = np.array([5, 5, 5, 5], dtype=np.int64)
        flipped = np.array([1, 2, 3, 4], dtype=np.int64)
        rng = np.random.default_rng(0)

        out, _ = random_subset(
            flipped, _always_flipped,
            budget=20, rng=rng,
            subset_sizes=(2,), n_trials_per_size=5,
            target=anchor,
        )
        delta_after = int(np.sum(np.abs(out - anchor)))
        assert delta_after < int(np.sum(np.abs(flipped - anchor)))


class TestMinimiseFlipWithAnchor:
    def test_anchor_geno_threads_to_passes(self) -> None:
        anchor = np.array([3, 7, 1, 0], dtype=np.int64)
        flipped = np.array([5, 2, 4, 3], dtype=np.int64)
        cfg = Stage2Config(
            budget_sut_calls_per_flip=20,
            passes=PassesConfig(
                zero=ZeroPassConfig(enabled=True),
                rank=RankPassConfig(enabled=False),
                random_subset=RandomSubsetPassConfig(enabled=False),
            ),
        )
        rng = np.random.default_rng(0)

        result = minimise_flip(
            flipped_geno=flipped,
            sut_check_fn=_always_flipped,
            anchor_label="anchor",
            budget_total=20,
            rng=rng,
            cfg=cfg,
            input_distance_fn=lambda g, a: float(rank_sum_delta(g, a)),
            stage1_flip_label="other",
            seed_idx=0,
            flip_id=0,
            anchor_geno=anchor,
        )
        assert np.array_equal(result.genotype_min, anchor)
        assert result.d_i_min == 0.0

    def test_default_anchor_is_zeros(self) -> None:
        """Omitting anchor_geno reproduces legacy zero-anchor PDQ."""
        flipped = np.array([3, 0, 2, 5], dtype=np.int64)
        cfg = Stage2Config(
            budget_sut_calls_per_flip=20,
            passes=PassesConfig(
                zero=ZeroPassConfig(enabled=True),
                rank=RankPassConfig(enabled=False),
                random_subset=RandomSubsetPassConfig(enabled=False),
            ),
        )
        rng = np.random.default_rng(0)

        result = minimise_flip(
            flipped_geno=flipped,
            sut_check_fn=_always_flipped,
            anchor_label="anchor",
            budget_total=20,
            rng=rng,
            cfg=cfg,
            input_distance_fn=lambda g, a: float(rank_sum_delta(g, a)),
            stage1_flip_label="other",
            seed_idx=0,
            flip_id=0,
        )
        assert int(np.sum(result.genotype_min)) == 0


# ---------------------------------------------------------------------------
# rank_sum_delta metric + registry
# ---------------------------------------------------------------------------


class TestRankSumDelta:
    def test_matches_rank_sum_when_anchor_is_zero(self) -> None:
        g = np.array([0, 3, 5, 0, 1], dtype=np.int64)
        zero = np.zeros_like(g)
        assert rank_sum_delta(g, zero) == rank_sum(g)

    def test_l1_distance_with_nonzero_anchor(self) -> None:
        g = np.array([2, 5, 7], dtype=np.int64)
        anchor = np.array([5, 5, 3], dtype=np.int64)
        assert rank_sum_delta(g, anchor) == 7

    def test_registry_exposes_rank_sum_delta(self) -> None:
        assert "rank_sum_delta" in INPUT_DISTANCES
        assert "rank_sum_delta" in VALID_D_I
        fn = INPUT_DISTANCES["rank_sum_delta"]
        g = np.array([2, 5, 7], dtype=np.int64)
        anchor = np.array([5, 5, 3], dtype=np.int64)
        assert fn(g, anchor) == 7.0


# ---------------------------------------------------------------------------
# Anchor selection helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakePareto:
    """Minimal stand-in for an SMOO Pareto candidate."""

    solution: np.ndarray
    fitness: np.ndarray


class TestSelectAnchors:
    def _make(self) -> list[_FakePareto]:
        return [
            _FakePareto(np.array([1, 0]), np.array([0.10, 0.50])),
            _FakePareto(np.array([2, 0]), np.array([0.05, 0.10])),  # best TgtBal
            _FakePareto(np.array([3, 0]), np.array([0.20, 0.30])),
        ]

    def test_k_none_returns_all(self) -> None:
        from src.boundary_pair.config import AnchorSelectionConfig
        from src.boundary_pair.runner import _select_anchors

        pareto = self._make()
        selected = _select_anchors(pareto, AnchorSelectionConfig(k=None))
        assert len(selected) == 3
        assert [i for i, _ in selected] == [0, 1, 2]

    def test_top_k_picks_best_tgtbal(self) -> None:
        from src.boundary_pair.config import AnchorSelectionConfig
        from src.boundary_pair.runner import _select_anchors

        pareto = self._make()
        selected = _select_anchors(pareto, AnchorSelectionConfig(k=1))
        assert len(selected) == 1
        assert selected[0][0] == 1

    def test_unsupported_source_raises(self) -> None:
        from src.boundary_pair.config import AnchorSelectionConfig
        from src.boundary_pair.runner import _select_anchors

        with pytest.raises(ValueError):
            _select_anchors([], AnchorSelectionConfig(source="trace_threshold"))


class TestPairSoftmax:
    def test_argmax_picks_higher_pair_class(self) -> None:
        from src.boundary_pair.runner import _pair_softmax_argmax

        categories = ("alpha", "bravo", "charlie")
        logprobs = [-1.0, -3.0, -5.0]
        label, p_a, p_b = _pair_softmax_argmax(
            logprobs, categories, "alpha", "bravo",
        )
        assert label == "alpha"
        assert p_a > p_b
        assert abs((p_a + p_b) - 1.0) < 1e-6

    def test_ties_go_to_class_a(self) -> None:
        from src.boundary_pair.runner import _pair_softmax_argmax

        categories = ("alpha", "bravo")
        logprobs = [-2.0, -2.0]
        label, _, _ = _pair_softmax_argmax(
            logprobs, categories, "alpha", "bravo",
        )
        assert label == "alpha"


# ---------------------------------------------------------------------------
# Boundary-pair config projections
# ---------------------------------------------------------------------------


class TestBoundaryPairConfigProjections:
    def test_shared_fields_thread_through(self) -> None:
        from src.boundary_pair.config import (
            BoundaryPairExperimentConfig,
            to_evolutionary_config,
            to_pdq_config,
        )

        cfg = BoundaryPairExperimentConfig(
            device="cuda",
            n_categories=15,
            name="exp_pair",
            save_dir=Path("/tmp/runs"),
        )
        evo = to_evolutionary_config(cfg)
        pdq = to_pdq_config(cfg)

        for shared in (
            "device", "categories", "n_categories",
            "prompt_template", "answer_format",
        ):
            assert getattr(evo, shared) == getattr(cfg, shared)
            assert getattr(pdq, shared) == getattr(cfg, shared)

        assert str(evo.save_dir).endswith(f"{cfg.name}/evolutionary")
        assert str(pdq.save_dir).endswith(f"{cfg.name}/pdq")

    def test_default_d_i_primary_is_rank_sum_delta(self) -> None:
        from src.boundary_pair.config import BoundaryPairExperimentConfig

        cfg = BoundaryPairExperimentConfig()
        assert cfg.pdq.distances.d_i_primary == "rank_sum_delta"

    def test_operator_knobs_thread_to_evolutionary_config(self) -> None:
        """evolutionary.optimizer.{mutation,crossover} reach the projection."""
        from src.boundary_pair.config import (
            BoundaryPairExperimentConfig,
            EvolutionaryStageConfig,
            to_evolutionary_config,
        )
        from src.config import CrossoverConfig, MutationConfig, OptimizerConfig

        cfg = BoundaryPairExperimentConfig(
            evolutionary=EvolutionaryStageConfig(
                optimizer=OptimizerConfig(
                    mutation=MutationConfig(prob=0.3, eta=1.0),
                    crossover=CrossoverConfig(prob=0.8, eta=5.0),
                ),
            ),
        )
        evo = to_evolutionary_config(cfg)
        assert evo.optimizer.mutation.prob == 0.3
        assert evo.optimizer.mutation.eta == 1.0
        assert evo.optimizer.crossover.prob == 0.8
        assert evo.optimizer.crossover.eta == 5.0

        # Defaults stay at the historical hardcoded operator values.
        default_evo = to_evolutionary_config(BoundaryPairExperimentConfig())
        assert default_evo.optimizer.mutation == MutationConfig(
            prob=None, eta=3.0
        )
        assert default_evo.optimizer.crossover == CrossoverConfig(
            prob=0.9, eta=3.0
        )

    def test_load_boundary_pair_config_from_template(self) -> None:
        from src.boundary_pair.config import load_boundary_pair_config

        template = (
            Path(__file__).resolve().parent.parent
            / "configs/templates/boundary_pair_template.yaml"
        )
        cfg = load_boundary_pair_config(template)
        assert cfg.name == "boundary_pair"
        assert cfg.anchor_selection.source == "pareto_front"
        assert cfg.pdq.distances.d_i_primary == "rank_sum_delta"


# ---------------------------------------------------------------------------
# Archive schema (PDQ v5)
# ---------------------------------------------------------------------------


class TestArchiveSchemaV5:
    def test_new_provenance_cols_present(self) -> None:
        for col in ("pareto_idx", "evolutionary_gen", "anchor_source"):
            assert col in ARCHIVE_COLUMNS

    def test_build_archive_row_emits_new_cols(self) -> None:
        from PIL import Image

        from src.pdq.archive import build_archive_row_stage2
        from src.pdq.search.base import Candidate, ScoredCandidate

        cand = Candidate(
            genotype=np.array([3, 0, 2], dtype=np.int64),
            strategy="dense_uniform",
            seed_idx=0,
            gen_step=0,
        )
        img = Image.new("RGB", (4, 4))
        sc = ScoredCandidate(
            candidate=cand,
            candidate_id=0,
            label="other",
            logprobs=[-1.0, -2.0],
            sut_call_id=0,
            rendered_text="x",
            rendered_image=img,
            discovery_wall_time_cum=0.0,
            d_i=5.0,
            d_o=1.0,
            pdq_score=0.2,
            flipped=True,
            img_sparsity=2,
            txt_sparsity=0,
            total_sparsity=2,
            img_rank_sum=5,
            txt_rank_sum=0,
            total_rank_sum=5,
            hamming_to_anchor=2,
            image_pixel_L2=10.0,
            text_cosine_sum=0.0,
        )
        s2 = Stage2Result(
            genotype_min=np.array([1, 0, 0], dtype=np.int64),
            d_i_min=1.0,
            sut_calls_used=3,
            trajectory_per_pass={},
            stopped_reason="all_passes_complete",
            final_label="other",
        )

        row = build_archive_row_stage2(
            sc=sc,
            flip_id=0,
            seed_id="seed_0000",
            run_id="test",
            anchor_geno_list=[0, 0, 0],
            anchor_logprobs=[0.0, 0.0],
            anchor_label="anchor",
            stage1_sut_calls=10,
            stage2_result=s2,
            pareto_idx=7,
            evolutionary_gen=42,
            anchor_source="evolutionary",
        )
        assert row["pareto_idx"] == 7
        assert row["evolutionary_gen"] == 42
        assert row["anchor_source"] == "evolutionary"

    def test_legacy_row_defaults_to_zero_source(self) -> None:
        from PIL import Image

        from src.pdq.archive import build_archive_row_stage2
        from src.pdq.search.base import Candidate, ScoredCandidate

        cand = Candidate(
            genotype=np.array([3, 0, 2], dtype=np.int64),
            strategy="dense_uniform",
            seed_idx=0,
            gen_step=0,
        )
        img = Image.new("RGB", (4, 4))
        sc = ScoredCandidate(
            candidate=cand,
            candidate_id=0,
            label="other",
            logprobs=[-1.0, -2.0],
            sut_call_id=0,
            rendered_text="x",
            rendered_image=img,
            discovery_wall_time_cum=0.0,
            d_i=5.0, d_o=1.0, pdq_score=0.2, flipped=True,
            img_sparsity=2, txt_sparsity=0, total_sparsity=2,
            img_rank_sum=5, txt_rank_sum=0, total_rank_sum=5,
            hamming_to_anchor=2,
            image_pixel_L2=10.0, text_cosine_sum=0.0,
        )
        s2 = Stage2Result(
            genotype_min=np.zeros(3, dtype=np.int64),
            d_i_min=0.0,
            sut_calls_used=2,
            trajectory_per_pass={},
            stopped_reason="all_passes_complete",
            final_label="other",
        )

        row = build_archive_row_stage2(
            sc=sc, flip_id=0, seed_id="seed_0000", run_id="test",
            anchor_geno_list=[0, 0, 0], anchor_logprobs=[0.0, 0.0],
            anchor_label="anchor", stage1_sut_calls=10, stage2_result=s2,
        )
        assert row["pareto_idx"] is None
        assert row["evolutionary_gen"] is None
        assert row["anchor_source"] == "zero"
