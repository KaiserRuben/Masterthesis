"""Tests for PDQ flip policies (``any_non_anchor`` vs ``pair_target``).

Covers:

* ``make_flip_predicate`` semantics — pair-restricted flip detection
  true/false cases, tie handling, anchor-on-target-side mirroring, and
  the off-pair-attractor scenario from Exp-100 (full-category argmax
  lands on a generic attractor class while the pair criterion varies).
* Stage-1 ``score_candidate`` integration: ``flipped`` follows the
  predicate while ``label`` stays the full-category argmax.
* Stage-2 preservation: ``minimise_flip`` keeps only steps that hold
  the pair criterion even when the 6-cat label never equals the anchor.
* Config defaults (``pair_target``) and registry membership.
* Standalone runner fail-fast on ``pair_target``.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.pdq.config import (
    PDQExperimentConfig,
    Stage1Config,
    Stage2Config,
    VALID_FLIP_POLICIES,
    validate_config,
)
from src.pdq.flip_policy import make_flip_predicate
from src.pdq.runner import PDQRunner
from src.pdq.search.base import Candidate, score_candidate
from src.pdq.search.stage2 import CheckResult, minimise_flip
from src.pdq.config import (
    PassesConfig,
    RandomSubsetPassConfig,
    RankPassConfig,
    ZeroPassConfig,
)

CATEGORIES = ("junco", "ostrich", "boa constrictor")
PAIR = ("junco", "ostrich")
# Anchor on the junco (anchor-class) side of the pair boundary.
ANCHOR_LOGPROBS = [-1.0, -2.0, -5.0]


def _pair_predicate(anchor_logprobs=ANCHOR_LOGPROBS):
    return make_flip_predicate(
        "pair_target",
        categories=CATEGORIES,
        anchor_label="junco",
        anchor_logprobs=anchor_logprobs,
        pair_classes=PAIR,
    )


# ---------------------------------------------------------------------------
# make_flip_predicate — pair_target semantics
# ---------------------------------------------------------------------------


class TestPairTargetPredicate:
    def test_flip_when_target_beats_anchor(self) -> None:
        fn = _pair_predicate()
        # lp[ostrich] > lp[junco] → flip, even though argmax is junco-free.
        assert fn([-3.0, -2.0, -4.0], "ostrich") is True

    def test_no_flip_when_anchor_side_holds(self) -> None:
        fn = _pair_predicate()
        assert fn([-1.0, -2.0, -4.0], "junco") is False

    def test_attractor_argmax_is_not_a_flip_on_anchor_side(self) -> None:
        """Exp-100 scenario: 6-cat argmax = 'boa constrictor' (≠ anchor
        label, so any_non_anchor would call it a flip) but the pair
        criterion says anchor side → not a flip."""
        fn = _pair_predicate()
        lp = [-2.0, -3.0, -0.5]  # argmax boa; junco still beats ostrich
        assert fn(lp, "boa constrictor") is False

    def test_attractor_argmax_is_a_flip_on_target_side(self) -> None:
        """seed_0006 recoverable-signal case: argmax boa, but
        lp[ostrich] > lp[junco] → pair flip."""
        fn = _pair_predicate()
        lp = [-3.0, -2.0, -0.5]
        assert fn(lp, "boa constrictor") is True

    def test_tie_goes_to_class_a_side(self) -> None:
        """Tie rule matches _pair_softmax_argmax: lp[a] >= lp[b] → a-side.
        With the anchor on the a-side, a tie is not a flip."""
        fn = _pair_predicate()
        assert fn([-2.0, -2.0, -4.0], "junco") is False

    def test_anchor_on_target_side_mirrors_criterion(self) -> None:
        """When the anchor itself sits on the target (class_b) side, a
        flip means crossing back to the class_a side."""
        fn = _pair_predicate(anchor_logprobs=[-2.5, -1.0, -5.0])
        assert fn([-1.0, -2.0, -4.0], "junco") is True   # crossed to a-side
        assert fn([-3.0, -2.0, -4.0], "ostrich") is False  # still b-side

    def test_predicate_ignores_full_category_label(self) -> None:
        fn = _pair_predicate()
        lp = [-3.0, -2.0, -4.0]
        assert fn(lp, "junco") == fn(lp, "boa constrictor")


class TestAnyNonAnchorPredicate:
    def test_label_comparison_only(self) -> None:
        fn = make_flip_predicate(
            "any_non_anchor",
            categories=CATEGORIES,
            anchor_label="junco",
            anchor_logprobs=ANCHOR_LOGPROBS,
            pair_classes=None,
        )
        assert fn([-9.0, -9.0, -0.1], "boa constrictor") is True
        assert fn([-0.1, -9.0, -9.0], "junco") is False


class TestPredicateErrors:
    def test_pair_target_without_pair_raises(self) -> None:
        with pytest.raises(ValueError, match="pair_target"):
            make_flip_predicate(
                "pair_target",
                categories=CATEGORIES,
                anchor_label="junco",
                anchor_logprobs=ANCHOR_LOGPROBS,
                pair_classes=None,
            )

    def test_pair_class_missing_from_categories_raises(self) -> None:
        with pytest.raises(ValueError, match="not in the configured categories"):
            make_flip_predicate(
                "pair_target",
                categories=CATEGORIES,
                anchor_label="junco",
                anchor_logprobs=ANCHOR_LOGPROBS,
                pair_classes=("junco", "salamander"),
            )

    def test_unknown_policy_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown flip policy"):
            make_flip_predicate(
                "nearest_neighbour",
                categories=CATEGORIES,
                anchor_label="junco",
                anchor_logprobs=ANCHOR_LOGPROBS,
                pair_classes=PAIR,
            )


# ---------------------------------------------------------------------------
# Stage-1 score_candidate integration
# ---------------------------------------------------------------------------


def _score(logprobs: list[float], flip_predicate) -> object:
    cand = Candidate(
        genotype=np.array([1, 0], dtype=np.int64),
        strategy="dense_uniform",
        seed_idx=0,
        gen_step=0,
    )
    img = Image.new("RGB", (4, 4))
    return score_candidate(
        cand=cand,
        anchor_label="junco",
        anchor_geno=np.zeros(2, dtype=np.int64),
        anchor_image_arr=np.array(img),
        text_distance_fn=lambda _t: 0.0,
        image_dim=1,
        categories=CATEGORIES,
        sut_call_fn=lambda _g: (logprobs, 0, img, "prompt", 0.0),
        input_distance_fn=lambda g, a: float(np.sum(np.abs(g - a))),
        output_distance_fn=lambda a, b: float(a != b),
        flip_predicate=flip_predicate,
    )


class TestScoreCandidateFlipPredicate:
    def test_pair_flip_true_label_stays_full_argmax(self) -> None:
        sc = _score([-3.0, -2.0, -0.5], _pair_predicate())
        assert sc.flipped is True
        assert sc.label == "boa constrictor"  # full-category argmax kept

    def test_pair_flip_false_despite_non_anchor_argmax(self) -> None:
        sc = _score([-2.0, -3.0, -0.5], _pair_predicate())
        assert sc.flipped is False
        assert sc.label == "boa constrictor"

    def test_none_predicate_falls_back_to_any_non_anchor(self) -> None:
        sc = _score([-2.0, -3.0, -0.5], None)
        assert sc.flipped is True  # label 'boa constrictor' != 'junco'


# ---------------------------------------------------------------------------
# Stage-2 preservation under pair_target
# ---------------------------------------------------------------------------


def _fake_logprobs(geno: np.ndarray) -> list[float]:
    """Synthetic SUT: argmax is always the off-pair attractor 'boa';
    pair side flips to ostrich iff sum(geno) >= 4.5 (i.e. >= 5)."""
    s = float(np.sum(geno))
    return [-1.0 - s, -10.0 + s, -0.1]


def _make_check(predicate):
    def check(geno: np.ndarray) -> CheckResult:
        lp = _fake_logprobs(geno)
        label = CATEGORIES[int(np.argmax(lp))]
        return CheckResult(predicate(lp, label), label, 0, 0.0)
    return check


def _zero_only_cfg() -> Stage2Config:
    return Stage2Config(
        budget_sut_calls_per_flip=30,
        flip_preserve_policy="pair_target",
        passes=PassesConfig(
            zero=ZeroPassConfig(enabled=True),
            rank=RankPassConfig(enabled=False),
            random_subset=RandomSubsetPassConfig(enabled=False),
        ),
    )


class TestStage2PreservesPairTarget:
    def test_shrink_stops_at_pair_boundary(self) -> None:
        """sum >= 5 keeps the pair flip; zeroing must not cross below."""
        flipped = np.array([3, 2, 2], dtype=np.int64)  # sum 7, pair-flipped
        predicate = _pair_predicate()
        result = minimise_flip(
            flipped_geno=flipped,
            sut_check_fn=_make_check(predicate),
            anchor_label="junco",
            budget_total=30,
            rng=np.random.default_rng(0),
            cfg=_zero_only_cfg(),
            input_distance_fn=lambda g, a: float(np.sum(np.abs(g - a))),
            stage1_flip_label="boa constrictor",
            seed_idx=0,
            flip_id=0,
        )
        # Every accepted step preserved sum > 4; minimum reachable is 5.
        assert float(np.sum(result.genotype_min)) == 5.0
        assert predicate(_fake_logprobs(result.genotype_min), "boa constrictor")
        # Rejected steps are exactly those that would break the criterion.
        steps = result.trajectory_per_pass["zero"]
        for step in steps:
            assert step["accepted"] == step["still_flipped"]

    def test_any_non_anchor_would_collapse_to_zero(self) -> None:
        """Contrast: under any_non_anchor the attractor label ('boa')
        differs from the anchor everywhere, so zeroing runs to the
        anchor — the Exp-100 artifact behaviour."""
        flipped = np.array([3, 2, 2], dtype=np.int64)
        predicate = make_flip_predicate(
            "any_non_anchor",
            categories=CATEGORIES,
            anchor_label="junco",
            anchor_logprobs=ANCHOR_LOGPROBS,
            pair_classes=None,
        )
        result = minimise_flip(
            flipped_geno=flipped,
            sut_check_fn=_make_check(predicate),
            anchor_label="junco",
            budget_total=30,
            rng=np.random.default_rng(0),
            cfg=_zero_only_cfg(),
            input_distance_fn=lambda g, a: float(np.sum(np.abs(g - a))),
            stage1_flip_label="boa constrictor",
            seed_idx=0,
            flip_id=0,
        )
        assert float(np.sum(result.genotype_min)) == 0.0


# ---------------------------------------------------------------------------
# Config defaults + validation
# ---------------------------------------------------------------------------


class TestFlipPolicyConfig:
    def test_registry_contains_both_policies(self) -> None:
        assert VALID_FLIP_POLICIES == {"any_non_anchor", "pair_target"}

    def test_defaults_are_pair_target(self) -> None:
        assert Stage1Config().flip_policy == "pair_target"
        assert Stage2Config().flip_preserve_policy == "pair_target"

    def test_boundary_pair_default_is_pair_target(self) -> None:
        from src.boundary_pair.config import BoundaryPairExperimentConfig

        cfg = BoundaryPairExperimentConfig()
        assert cfg.pdq.stage1.flip_policy == "pair_target"
        assert cfg.pdq.stage2.flip_preserve_policy == "pair_target"

    def test_validate_config_accepts_pair_target(self) -> None:
        validate_config(PDQExperimentConfig())  # defaults → no raise

    def test_boundary_pair_template_uses_pair_target(self) -> None:
        from pathlib import Path

        from src.boundary_pair.config import load_boundary_pair_config

        template = (
            Path(__file__).resolve().parent.parent
            / "configs/templates/boundary_pair_template.yaml"
        )
        cfg = load_boundary_pair_config(template)
        assert cfg.pdq.stage1.flip_policy == "pair_target"
        assert cfg.pdq.stage2.flip_preserve_policy == "pair_target"


# ---------------------------------------------------------------------------
# Standalone runner fail-fast
# ---------------------------------------------------------------------------


class TestStandaloneFailFast:
    def test_default_config_raises_before_any_loading(self) -> None:
        with pytest.raises(ValueError, match="pair_target"):
            PDQRunner(PDQExperimentConfig()).run()

    def test_stage2_only_pair_target_also_raises(self) -> None:
        cfg = PDQExperimentConfig(
            stage1=Stage1Config(flip_policy="any_non_anchor"),
        )
        with pytest.raises(
            ValueError, match="stage2.flip_preserve_policy",
        ):
            PDQRunner(cfg).run()
