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
    PMIConfig,
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

    def encode_text(self, texts):  # type: ignore[override]
        # Stub for the abstract method. Tests don't exercise text-embedding
        # paths; returning a zero vector keeps the contract.
        import numpy as np
        return np.zeros((len(texts), 1), dtype=np.float32)

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
            # Production VLMSUT.__init__ sets these; tests bypass that
            # path, so seed defaults that disable the Redis cache and
            # text-embedder paths.
            self._redis = None
            self._cache_hits = 0
            self._cache_misses = 0
            self._last_call_cached = False
            self._text_embedder = None

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
        assert cfg.device == "mps"
        assert cfg.sut.backend == "torch"
        assert cfg.sut.processor_id is None
        assert cfg.sut.ov_device == "GPU"
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


# =========================================================================
# PMI calibration (PMIConfig / Exp-104)
# =========================================================================


class ImageSensitiveScorer(FakeScorer):
    """FakeScorer whose norm log-probs depend on the image, so the PMI
    subtraction is observable. Detects the content-neutral null image
    (uniform gray 128) and returns a distinct baseline. Counts calls so
    baseline caching can be asserted. Knows only labels ``"a"`` / ``"b"``.
    """

    _NULL = {"a": -0.5, "b": -2.5}    # baseline scored on the gray null image
    _SIGNAL = {"a": -1.0, "b": -1.2}  # any other (real) image

    def __init__(self) -> None:
        super().__init__()
        self.n_calls = 0

    def score_categories(self, image, prompt, categories, thinking_ids=None):  # type: ignore[override]
        import numpy as np

        self.n_calls += 1
        arr = np.asarray(image)
        table = self._NULL if bool((arr == 128).all()) else self._SIGNAL
        scored = [(c, table[c] * 2.0, table[c], 2) for c in categories]
        return sorted(scored, key=lambda x: x[2], reverse=True)


def _make_pmi_sut(scorer: "VLMScorer", pmi: PMIConfig) -> "VLMSUT":
    """FakeSUT injecting *scorer* + a PMI config. Deliberately does NOT
    pre-set the PMI caches, exercising the lazy-init fallback that keeps
    __init__-bypassing test doubles working."""
    from src.sut.vlm_sut import VLMSUT

    class FakeSUT(VLMSUT):
        def __init__(self) -> None:
            self._config = ExperimentConfig(categories=("a", "b"), pmi=pmi)
            self._device = torch.device("cpu")
            self._scorer = scorer
            self._prompt = (
                self._config.prompt_template
                + self._config.answer_format.format(categories="a, b")
            )
            self._redis = None
            self._cache_hits = 0
            self._cache_misses = 0
            self._last_call_cached = False
            self._text_embedder = None

    return FakeSUT()


class TestPMIConfig:
    """PMIConfig defaults and validation."""

    def test_defaults(self) -> None:
        p = PMIConfig()
        assert p.enabled is False
        assert p.null_image == "gray"
        assert p.null_image_size == 448
        assert p.null_image_seed == 0
        # Present on ExperimentConfig, default-off (no behaviour change).
        assert ExperimentConfig().pmi.enabled is False

    def test_invalid_null_image_raises(self) -> None:
        with pytest.raises(ValueError):
            PMIConfig(null_image="rainbow")

    def test_null_image_variants_valid(self) -> None:
        for name in ("gray", "black", "white", "noise"):
            assert PMIConfig(null_image=name).null_image == name

    def test_apply_to_seedgen_default_on(self) -> None:
        # Default preserves whole-system behaviour; opt-out is explicit.
        assert PMIConfig().apply_to_seedgen is True
        assert PMIConfig(apply_to_seedgen=False).apply_to_seedgen is False


class TestNullImage:
    """The content-neutral baseline image builder."""

    def _null(self, name: str):
        sut = _make_pmi_sut(
            ImageSensitiveScorer(),
            PMIConfig(enabled=True, null_image=name, null_image_size=8),
        )
        return sut._null_image()

    def test_solid_colours(self) -> None:
        assert self._null("gray").getpixel((0, 0)) == (128, 128, 128)
        assert self._null("black").getpixel((0, 0)) == (0, 0, 0)
        assert self._null("white").getpixel((0, 0)) == (255, 255, 255)

    def test_noise_is_not_solid(self) -> None:
        import numpy as np

        arr = np.asarray(self._null("noise"))
        assert arr.shape == (8, 8, 3)
        assert arr.std() > 0  # actual noise, not a constant fill


class TestPMICalibration:
    """process_input under the PMI flag."""

    def test_disabled_is_identity(self) -> None:
        """PMI off → raw scores returned, no baseline call (comparability)."""
        scorer = ImageSensitiveScorer()
        sut = _make_pmi_sut(scorer, PMIConfig(enabled=False))
        out = sut.process_input(_dummy_image())
        assert out[0].item() == pytest.approx(-1.0)   # a, raw SIGNAL
        assert out[1].item() == pytest.approx(-1.2)   # b, raw SIGNAL
        assert scorer.n_calls == 1                    # no null-image baseline

    def test_baseline_subtracted(self) -> None:
        """PMI on → returns SIGNAL − NULL, per class, in input order."""
        scorer = ImageSensitiveScorer()
        sut = _make_pmi_sut(scorer, PMIConfig(enabled=True))
        out = sut.process_input(_dummy_image())
        # a: -1.0 - (-0.5) = -0.5 ; b: -1.2 - (-2.5) = +1.3
        assert out[0].item() == pytest.approx(-0.5)
        assert out[1].item() == pytest.approx(1.3)

    def test_baseline_cached(self) -> None:
        """Baseline is scored once per category-tuple, then reused."""
        scorer = ImageSensitiveScorer()
        sut = _make_pmi_sut(scorer, PMIConfig(enabled=True))
        sut.process_input(_dummy_image())   # signal + baseline  → 2
        sut.process_input(_dummy_image())   # signal only         → +1
        assert scorer.n_calls == 3

    def test_pmi_baseline_accessor(self) -> None:
        scorer = ImageSensitiveScorer()
        sut = _make_pmi_sut(scorer, PMIConfig(enabled=True))
        assert sut.pmi_baseline(("a", "b")) == pytest.approx([-0.5, -2.5])
        # Disabled → None (nothing to persist).
        off = _make_pmi_sut(ImageSensitiveScorer(), PMIConfig(enabled=False))
        assert off.pmi_baseline(("a", "b")) is None

    def test_force_raw_returns_raw_while_pmi_enabled(self) -> None:
        """Inside force_raw(), process_input is raw even with PMI on; the
        override is restored on exit (seed-gen-freeze mechanism)."""
        scorer = ImageSensitiveScorer()
        sut = _make_pmi_sut(scorer, PMIConfig(enabled=True))
        with sut.force_raw():
            out = sut.process_input(_dummy_image())
            assert out[0].item() == pytest.approx(-1.0)  # raw SIGNAL, no subtract
            assert out[1].item() == pytest.approx(-1.2)
        # Restored: PMI subtraction back in effect.
        out2 = sut.process_input(_dummy_image())
        assert out2[0].item() == pytest.approx(-0.5)
        assert out2[1].item() == pytest.approx(1.3)

    def test_force_raw_noop_when_disabled(self) -> None:
        scorer = ImageSensitiveScorer()
        sut = _make_pmi_sut(scorer, PMIConfig(enabled=False))
        with sut.force_raw():
            out = sut.process_input(_dummy_image())
        assert out[0].item() == pytest.approx(-1.0)
        assert scorer.n_calls == 1  # no null-image baseline call


class _FaultyRedis:
    """Redis double whose ops raise, simulating a mid-run connection reset."""

    def __init__(self) -> None:
        self.calls = 0

    def get(self, *_a, **_k):
        self.calls += 1
        raise OSError("connection reset by peer")

    def set(self, *_a, **_k):
        self.calls += 1
        raise OSError("connection reset by peer")


class TestCacheFaultTolerance:
    """A mid-run Redis fault degrades to no-cache, never crashes the run."""

    def _sut_with_redis(self, redis_obj):
        from src.sut.vlm_sut import VLMSUT

        class FakeSUT(VLMSUT):
            def __init__(self) -> None:
                self._config = ExperimentConfig(categories=("a", "b"))
                self._device = torch.device("cpu")
                self._scorer = FakeScorer()
                self._prompt = "p"
                self._redis = redis_obj
                self._cache_hits = 0
                self._cache_misses = 0
                self._last_call_cached = False
                self._text_embedder = None

        return FakeSUT()

    def test_get_fault_returns_score_and_disables_cache(self) -> None:
        r = _FaultyRedis()
        sut = self._sut_with_redis(r)
        out = sut.process_input(_dummy_image())          # must NOT raise
        assert out.shape == (2,)
        assert sut._redis is None                        # cache disabled after fault
        assert r.calls == 1                              # faulted once, then no more

    def test_set_fault_disables_cache(self) -> None:
        class GetOkSetFail(_FaultyRedis):
            def get(self, *_a, **_k):
                return None                              # miss → proceeds to compute + set
        r = GetOkSetFail()
        sut = self._sut_with_redis(r)
        out = sut.process_input(_dummy_image())          # must NOT raise
        assert out.shape == (2,)
        assert sut._redis is None


class TestPMICacheKey:
    """Redis key must isolate PMI-corrected from raw results."""

    def test_pmi_tag_distinguishes(self) -> None:
        from src.sut.vlm_sut import _cache_key

        img = _dummy_image()
        legacy = _cache_key("m", img, "p", ("a", "b"))
        empty_tag = _cache_key("m", img, "p", ("a", "b"), pmi_tag="")
        pmi = _cache_key("m", img, "p", ("a", "b"), pmi_tag="pmi:gray:448:0")
        assert legacy == empty_tag    # empty tag == pre-PMI key (cache reuse)
        assert pmi != legacy          # corrected results never collide w/ raw
