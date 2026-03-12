"""Tests for the VLM system-under-test components.

Uses a concrete FakeScorer (not a mock) to test VLMSUT without loading
a real 10GB VLM model.  All tests use real objects and deterministic values.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from src.config import (
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_ANSWER_FORMAT,
    ExperimentConfig,
    SUTConfig,
)

# Test-only category list matching the keys in _FAKE_RESULTS.
_TEST_CATEGORIES: tuple[str, ...] = (
    "macaw", "peacock", "flamingo", "monarch butterfly", "jellyfish",
    "chameleon", "toucan", "leopard", "red panda", "lionfish",
    "coral reef", "volcano", "castle", "mosque", "palace",
)
from src.sut.scorer import VLMScorer

# ---------------------------------------------------------------------------
# FakeScorer -- concrete VLMScorer that returns deterministic values
# ---------------------------------------------------------------------------

# Fixed scoring data: category -> (log_prob, log_prob_norm, n_tokens).
_FAKE_RESULTS: dict[str, tuple[float, float, int]] = {
    "macaw": (-0.50, -0.25, 2),
    "peacock": (-1.20, -0.60, 2),
    "flamingo": (-2.00, -0.67, 3),
    "monarch butterfly": (-3.50, -0.88, 4),
    "jellyfish": (-4.00, -1.33, 3),
    "chameleon": (-5.00, -1.67, 3),
    "toucan": (-6.00, -3.00, 2),
    "leopard": (-7.00, -3.50, 2),
    "red panda": (-8.00, -2.67, 3),
    "lionfish": (-9.00, -3.00, 3),
    "coral reef": (-10.00, -3.33, 3),
    "volcano": (-11.00, -3.67, 3),
    "castle": (-12.00, -6.00, 2),
    "mosque": (-13.00, -6.50, 2),
    "palace": (-14.00, -7.00, 2),
    # Extra labels for override tests.
    "cat": (-0.10, -0.10, 1),
    "dog": (-0.80, -0.40, 2),
    "bird": (-1.50, -0.75, 2),
}


class FakeScorer(VLMScorer):
    """Concrete scorer that returns deterministic values without a model.

    Overrides :meth:`score_categories` to return fixed results from
    ``_FAKE_RESULTS``.  Does not call ``super().__init__`` -- no model
    is loaded.
    """

    def __init__(self) -> None:
        # Deliberately skip super().__init__ to avoid loading a model.
        self._device = torch.device("cpu")
        self._enable_thinking = False
        self._max_thinking_tokens = 0

    def _prepare_inputs(self, image, prompt, enable_thinking):  # type: ignore[override]
        raise NotImplementedError("FakeScorer does not prepare real inputs")

    def score_categories(
        self,
        image,  # type: ignore[override]
        prompt,
        categories,
        thinking_ids=None,
    ) -> list[tuple[str, float, float, int]]:
        scored = []
        for cat in categories:
            if cat in _FAKE_RESULTS:
                lp, lp_norm, n = _FAKE_RESULTS[cat]
                scored.append((cat, lp, lp_norm, n))
            else:
                scored.append((cat, float("-inf"), float("-inf"), 0))
        return sorted(scored, key=lambda x: x[2], reverse=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sut(
    config: ExperimentConfig | None = None,
) -> "FakeSUT":
    """Build a VLMSUT-like object backed by FakeScorer.

    We import VLMSUT here so the module-level SMOO import runs inside
    the test, and then monkey-patch the scorer.
    """
    from src.sut.vlm_sut import VLMSUT

    class FakeSUT(VLMSUT):
        """VLMSUT subclass that injects a FakeScorer instead of loading
        a real model."""

        def __init__(self, config: ExperimentConfig | None = None) -> None:
            self._config = config or ExperimentConfig(categories=_TEST_CATEGORIES)
            self._device = torch.device(self._config.device)
            self._scorer = FakeScorer()
            self._prompt = (
                self._config.prompt_template
                + self._config.answer_format.format(
                    categories=", ".join(self._config.categories),
                )
            )

    return FakeSUT(config)


def _dummy_image() -> "Image.Image":
    """Create a tiny 8x8 red PIL image for testing."""
    from PIL import Image

    return Image.new("RGB", (8, 8), color=(255, 0, 0))


# =========================================================================
# TestExperimentConfig (formerly TestVLMSUTConfig)
# =========================================================================


class TestExperimentConfig:
    """Configuration dataclass validation."""

    def test_default_values(self) -> None:
        cfg = ExperimentConfig()
        assert cfg.sut.model_id == "Qwen/Qwen3.5-9B"
        assert cfg.device == "cpu"
        assert cfg.categories == ()
        assert cfg.sut.enable_thinking is False
        assert cfg.sut.max_thinking_tokens == 2000
        assert cfg.sut.max_pixels is None

    def test_custom_values(self) -> None:
        cats = ("cat", "dog")
        cfg = ExperimentConfig(
            device="cuda",
            categories=cats,
            prompt_template="Pick one:",
            answer_format=" {categories}",
            sut=SUTConfig(
                model_id="test/model",
                enable_thinking=True,
                max_thinking_tokens=500,
                max_pixels=1024,
            ),
        )
        assert cfg.sut.model_id == "test/model"
        assert cfg.device == "cuda"
        assert cfg.categories == cats
        assert cfg.prompt_template == "Pick one:"
        assert cfg.answer_format == " {categories}"
        assert cfg.sut.enable_thinking is True
        assert cfg.sut.max_thinking_tokens == 500
        assert cfg.sut.max_pixels == 1024

    def test_frozen_immutability(self) -> None:
        cfg = ExperimentConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.device = "cuda"  # type: ignore[misc]

    def test_prompt_template_and_answer_format(self) -> None:
        """Default template + answer_format together contain {categories}."""
        assert "{categories}" in DEFAULT_ANSWER_FORMAT
        full = DEFAULT_PROMPT_TEMPLATE + DEFAULT_ANSWER_FORMAT
        assert "What is the main subject" in full
        assert "{categories}" in full


# =========================================================================
# TestPromptBuilding
# =========================================================================


class TestPromptBuilding:
    """Verify prompt construction from template + categories."""

    def test_default_prompt(self) -> None:
        sut = _make_sut()
        expected = (
            DEFAULT_PROMPT_TEMPLATE
            + DEFAULT_ANSWER_FORMAT.format(
                categories=", ".join(_TEST_CATEGORIES),
            )
        )
        assert sut._prompt == expected

    def test_custom_template(self) -> None:
        cfg = ExperimentConfig(
            prompt_template="Choose:",
            answer_format=" {categories}.",
            categories=("cat", "dog", "bird"),
        )
        sut = _make_sut(cfg)
        assert sut._prompt == "Choose: cat, dog, bird."

    def test_category_subset(self) -> None:
        cfg = ExperimentConfig(categories=("macaw", "peacock"))
        sut = _make_sut(cfg)
        assert "macaw" in sut._prompt
        assert "peacock" in sut._prompt
        # Labels not in the subset should not appear.
        assert "flamingo" not in sut._prompt


# =========================================================================
# TestScorerOutputConversion
# =========================================================================


class TestScorerOutputConversion:
    """Verify score_categories_tensor returns correct shape and order."""

    def test_tensor_shape(self) -> None:
        scorer = FakeScorer()
        cats = ("macaw", "peacock", "flamingo")
        tensor = scorer.score_categories_tensor(
            _dummy_image(), "prompt", cats
        )
        assert tensor.shape == (3,)

    def test_tensor_ordering(self) -> None:
        """tensor[i] must equal log_prob_norm for categories[i]."""
        scorer = FakeScorer()
        cats = ("flamingo", "macaw", "peacock")
        tensor = scorer.score_categories_tensor(
            _dummy_image(), "prompt", cats
        )
        assert tensor[0].item() == pytest.approx(-0.67, abs=1e-6)  # flamingo
        assert tensor[1].item() == pytest.approx(-0.25, abs=1e-6)  # macaw
        assert tensor[2].item() == pytest.approx(-0.60, abs=1e-6)  # peacock

    def test_single_category(self) -> None:
        scorer = FakeScorer()
        tensor = scorer.score_categories_tensor(
            _dummy_image(), "prompt", ("macaw",)
        )
        assert tensor.shape == (1,)
        assert tensor[0].item() == pytest.approx(-0.25, abs=1e-6)


# =========================================================================
# TestProcessInput
# =========================================================================


class TestProcessInput:
    """VLMSUT.process_input integration via FakeScorer."""

    def test_returns_correct_shape(self) -> None:
        cats = ("cat", "dog", "bird")
        sut = _make_sut(ExperimentConfig(categories=cats))
        result = sut.process_input(_dummy_image())
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3,)

    def test_category_ordering(self) -> None:
        """tensor[i] corresponds to categories[i], not sorted order."""
        cats = ("dog", "cat", "bird")
        sut = _make_sut(ExperimentConfig(categories=cats))
        result = sut.process_input(_dummy_image())
        assert result[0].item() == pytest.approx(-0.40, abs=1e-6)  # dog
        assert result[1].item() == pytest.approx(-0.10, abs=1e-6)  # cat
        assert result[2].item() == pytest.approx(-0.75, abs=1e-6)  # bird

    def test_category_override(self) -> None:
        """Passing categories= overrides config categories."""
        sut = _make_sut(ExperimentConfig(categories=("macaw", "peacock")))
        # Override with different categories.
        result = sut.process_input(
            _dummy_image(), categories=("cat", "dog")
        )
        assert result.shape == (2,)
        assert result[0].item() == pytest.approx(-0.10, abs=1e-6)  # cat
        assert result[1].item() == pytest.approx(-0.40, abs=1e-6)  # dog

    def test_text_override(self) -> None:
        """Passing text= overrides the config prompt."""
        sut = _make_sut(ExperimentConfig(categories=("macaw",)))
        # FakeScorer ignores the prompt, but VLMSUT should pass it through.
        result = sut.process_input(_dummy_image(), text="Custom prompt")
        assert result.shape == (1,)
        assert result[0].item() == pytest.approx(-0.25, abs=1e-6)


# =========================================================================
# TestInputValid
# =========================================================================


class TestInputValid:
    """VLMSUT.input_valid correctness checks."""

    def test_correct_prediction(self) -> None:
        """Top prediction matches condition -> is_valid=True."""
        # "macaw" has the highest log_prob_norm among defaults.
        sut = _make_sut()
        is_valid, logprobs = sut.input_valid(_dummy_image(), "macaw")
        assert is_valid is True
        assert isinstance(logprobs, torch.Tensor)
        assert logprobs.shape == (len(_TEST_CATEGORIES),)

    def test_wrong_prediction(self) -> None:
        """Top prediction does not match condition -> is_valid=False."""
        sut = _make_sut()
        is_valid, logprobs = sut.input_valid(_dummy_image(), "palace")
        assert is_valid is False
        assert isinstance(logprobs, torch.Tensor)

    def test_tuple_input(self) -> None:
        """input_valid accepts (image, text) tuple."""
        sut = _make_sut()
        is_valid, logprobs = sut.input_valid(
            (_dummy_image(), "some prompt"), "macaw"
        )
        assert is_valid is True
        assert isinstance(logprobs, torch.Tensor)

    def test_tuple_input_with_none_text(self) -> None:
        """input_valid accepts (image, None) tuple."""
        sut = _make_sut()
        is_valid, _ = sut.input_valid((_dummy_image(), None), "macaw")
        assert is_valid is True
