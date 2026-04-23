"""Tests for :mod:`src.utils.pair_resolver`.

Uses a fake SUT and a fake DataSource (no real VLM or HuggingFace calls).
The fakes return deterministic samples and logprobs so that
:func:`src.evolutionary.generate_seeds` emits a predictable, reproducible pool
for every test.
"""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from src.config import ExperimentConfig, SUTConfig, SeedConfig
from src.data.imagenet import ImageSample
from src.utils.pair_resolver import (
    PairSpec,
    _closest_pairs,
    _levenshtein,
    _parse_pair,
    clear_pool_cache,
    resolve_pair,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeDataSource:
    """Minimal :class:`~src.data.DataSource` implementation.

    Emits ``n_per_class`` deterministic :class:`ImageSample` per category.
    Each sample's image is a solid-colour PIL image keyed on its global
    index so the tester's eventual consumption also sees unique bytes.
    """

    def __init__(self, categories: list[str]) -> None:
        self._categories = list(categories)
        self.load_calls = 0

    def labels(self) -> list[str]:
        return list(self._categories)

    def load_samples(
        self, categories: list[str], n_per_class: int,
    ) -> list[ImageSample]:
        self.load_calls += 1
        samples: list[ImageSample] = []
        for ci, name in enumerate(categories):
            for k in range(n_per_class):
                shade = (ci * 37 + k * 11) % 256
                img = Image.new("RGB", (8, 8), (shade, shade, shade))
                samples.append(ImageSample(
                    image=img, class_idx=ci, class_name=name,
                ))
        return samples


class ScriptedSUT:
    """Fake VLMSUT that returns scripted logprobs per (class_name, k_in_class).

    ``scripts[class_name]`` is a list of logprob vectors, one per sample
    of that class. Cycles if the list is shorter than the number of
    samples passed for that class.
    """

    def __init__(
        self,
        scripts: dict[str, list[list[float]]],
        categories: list[str],
    ) -> None:
        self._scripts = scripts
        self._categories = list(categories)
        self._counters: dict[str, int] = {c: 0 for c in self._categories}
        self.score_calls = 0

    def process_input(self, image, text=None, categories=None):
        self.score_calls += 1
        # Figure out which class this image belongs to by matching the
        # pixel shade produced by FakeDataSource.
        shade = image.getpixel((0, 0))[0]
        # Decode (class_idx, k_in_class) from shade by inversion; the
        # generator used shade = (ci*37 + k*11) % 256. A brute lookup is
        # fine — ci * 5 variants per test is tiny.
        cls_name = None
        k_in_class = 0
        for ci, name in enumerate(self._categories):
            for k in range(50):
                if (ci * 37 + k * 11) % 256 == shade and name in self._scripts:
                    cls_name = name
                    k_in_class = k
                    break
            if cls_name is not None:
                break

        if cls_name is None:
            # Default: return a low logprob vector favouring index 0.
            return torch.tensor([-0.1] + [-10.0] * (len(self._categories) - 1))

        script = self._scripts[cls_name]
        vec = script[k_in_class % len(script)]
        self._counters[cls_name] += 1
        return torch.tensor(vec, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


CATEGORIES = ("junco", "chickadee", "robin")


def _make_config(n_per_class: int = 3, gap: float = 2.0) -> ExperimentConfig:
    return ExperimentConfig(
        categories=CATEGORIES,
        sut=SUTConfig(model_id="fake/model-x"),
        seeds=SeedConfig(n_per_class=n_per_class, max_logprob_gap=gap),
    )


def _make_scripts() -> dict[str, list[list[float]]]:
    """Scripted logprobs where each class's top-1 is itself.

    - junco samples: top-1 junco, runner-up chickadee (small gap, in pool)
    - chickadee samples: top-1 chickadee, runner-up robin (small gap)
    - robin samples: top-1 robin, runner-up junco (small gap)

    Indices in the returned vectors correspond to CATEGORIES.
    """
    return {
        "junco":     [[0.0, -0.5, -5.0]],  # junco > chickadee > robin
        "chickadee": [[-5.0, 0.0, -0.5]],  # chickadee > robin > junco
        "robin":     [[-0.5, -5.0, 0.0]],  # robin > junco > chickadee
    }


# ---------------------------------------------------------------------------
# Unit tests — pure helpers
# ---------------------------------------------------------------------------


class TestParsePair:
    def test_arrow_separator(self):
        spec = _parse_pair("junco->chickadee")
        assert spec.class_a == "junco"
        assert spec.class_b == "chickadee"

    def test_dash_separator(self):
        spec = _parse_pair("junco-chickadee")
        assert spec.class_a == "junco"
        assert spec.class_b == "chickadee"

    def test_strips_whitespace(self):
        spec = _parse_pair("  junco  ->  chickadee  ")
        assert spec.class_a == "junco"
        assert spec.class_b == "chickadee"

    def test_arrow_preferred_over_dash(self):
        # Arrow should be preferred so hyphenated names stay intact.
        spec = _parse_pair("band-tailed pigeon->junco")
        assert spec.class_a == "band-tailed pigeon"
        assert spec.class_b == "junco"

    def test_underscore_normalisation(self):
        spec = _parse_pair("great_white_shark->tiger_shark")
        assert spec.class_a == "great_white_shark"
        assert spec.class_a_normalised == "great white shark"
        assert spec.class_b_normalised == "tiger shark"

    def test_matches_exact_and_normalised(self):
        spec = _parse_pair("great_white_shark->tiger_shark")
        # Exact (preserving underscores) — labels with real underscores.
        assert spec.matches("great_white_shark", "tiger_shark")
        # Normalised (spaces) — the common ImageNet case.
        assert spec.matches("great white shark", "tiger shark")
        # Asymmetric — order matters.
        assert not spec.matches("tiger shark", "great white shark")

    def test_missing_separator_raises(self):
        with pytest.raises(ValueError, match="missing a separator"):
            _parse_pair("junco_chickadee")

    def test_empty_half_raises(self):
        with pytest.raises(ValueError, match="empty class name"):
            _parse_pair("junco->")


class TestLevenshtein:
    def test_identical(self):
        assert _levenshtein("abc", "abc") == 0

    def test_empty(self):
        assert _levenshtein("", "abc") == 3
        assert _levenshtein("abc", "") == 3

    def test_substitution(self):
        assert _levenshtein("cat", "bat") == 1

    def test_mixed(self):
        assert _levenshtein("kitten", "sitting") == 3


class TestClosestPairs:
    def test_returns_closest_first(self):
        spec = _parse_pair("junco->chickadee")
        pool_pairs = [
            ("robin", "junco"),
            ("junco", "chickaadee"),   # typo — closest to target
            ("chickadee", "robin"),
        ]
        out = _closest_pairs(spec, pool_pairs, k=3)
        assert out[0] == ("junco", "chickaadee")

    def test_k_limits_results(self):
        spec = _parse_pair("a->b")
        pool_pairs = [("a", "b"), ("c", "d"), ("e", "f")]
        assert len(_closest_pairs(spec, pool_pairs, k=2)) == 2


# ---------------------------------------------------------------------------
# Integration tests — resolve_pair end-to-end with fakes
# ---------------------------------------------------------------------------


class TestResolvePair:
    def setup_method(self):
        clear_pool_cache()

    # (a) exact-match resolution
    def test_exact_match_returns_indices(self):
        config = _make_config(n_per_class=3)
        data = FakeDataSource(list(CATEGORIES))
        sut = ScriptedSUT(_make_scripts(), list(CATEGORIES))

        out = resolve_pair(
            "junco->chickadee",
            config=config, sut=sut, data_source=data,
            replicates=2,
        )
        assert len(out) == 2
        # Sorted ascending (stable, reproducible).
        assert out == sorted(out)
        # Indices are 0-based pool positions; the junco samples come
        # first (category order 0 → 1 → 2), so these are small.
        assert all(i < 3 for i in out)

    def test_returns_exactly_replicates(self):
        config = _make_config(n_per_class=5)
        data = FakeDataSource(list(CATEGORIES))
        sut = ScriptedSUT(_make_scripts(), list(CATEGORIES))

        out = resolve_pair(
            "junco->chickadee",
            config=config, sut=sut, data_source=data,
            replicates=3,
        )
        assert len(out) == 3

    # (b) underscore normalisation
    def test_underscore_normalisation(self):
        categories = ("great white shark", "tiger shark")
        scripts = {
            "great white shark": [[0.0, -0.5]],
            "tiger shark":       [[-0.5, 0.0]],
        }
        config = ExperimentConfig(
            categories=categories,
            sut=SUTConfig(model_id="fake/normalise"),
            seeds=SeedConfig(n_per_class=2, max_logprob_gap=2.0),
        )
        data = FakeDataSource(list(categories))
        sut = ScriptedSUT(scripts, list(categories))

        out = resolve_pair(
            "great_white_shark->tiger_shark",
            config=config, sut=sut, data_source=data,
            replicates=1,
        )
        assert len(out) == 1

    # (c) insufficient-replicates error text
    def test_insufficient_replicates_error(self):
        config = _make_config(n_per_class=2)  # only 2 junco samples → 2 pool seeds
        data = FakeDataSource(list(CATEGORIES))
        sut = ScriptedSUT(_make_scripts(), list(CATEGORIES))

        with pytest.raises(ValueError) as exc_info:
            resolve_pair(
                "junco->chickadee",
                config=config, sut=sut, data_source=data,
                replicates=10,
            )
        msg = str(exc_info.value)
        assert "only 2 pool seed" in msg
        assert "replicates=10" in msg
        # Remedy hint present.
        assert "N <= 2" in msg
        # Pool params named.
        assert "max_logprob_gap" in msg
        assert "n_per_class" in msg

    # (d) absent-pair error lists closest pairs
    def test_absent_pair_lists_suggestions(self):
        config = _make_config(n_per_class=3)
        data = FakeDataSource(list(CATEGORIES))
        sut = ScriptedSUT(_make_scripts(), list(CATEGORIES))

        with pytest.raises(ValueError) as exc_info:
            resolve_pair(
                "junco->chickaadee",  # typo
                config=config, sut=sut, data_source=data,
                replicates=1,
            )
        msg = str(exc_info.value)
        assert "not found" in msg
        assert "Did you mean" in msg
        # The actual pool pair must appear as a suggestion.
        assert "junco->chickadee" in msg
        # Pool size + params present.
        assert "unique pairs" in msg
        assert "max_logprob_gap" in msg

    # (e) in-process cache hit avoids re-generating
    def test_cache_hit_avoids_regeneration(self):
        config = _make_config(n_per_class=3)
        data = FakeDataSource(list(CATEGORIES))
        sut = ScriptedSUT(_make_scripts(), list(CATEGORIES))

        resolve_pair(
            "junco->chickadee",
            config=config, sut=sut, data_source=data,
            replicates=1,
        )
        first_calls = data.load_calls
        assert first_calls == 1

        # Same (categories, n_per_class, max_gap, model_id) → cache hit.
        resolve_pair(
            "chickadee->robin",
            config=config, sut=sut, data_source=data,
            replicates=1,
        )
        assert data.load_calls == first_calls, (
            "generate_seeds should not have re-loaded samples"
        )

    def test_cache_miss_on_different_params(self):
        data = FakeDataSource(list(CATEGORIES))
        sut = ScriptedSUT(_make_scripts(), list(CATEGORIES))

        config_a = _make_config(n_per_class=3, gap=2.0)
        config_b = _make_config(n_per_class=3, gap=1.0)  # different gap

        resolve_pair(
            "junco->chickadee",
            config=config_a, sut=sut, data_source=data,
            replicates=1,
        )
        resolve_pair(
            "junco->chickadee",
            config=config_b, sut=sut, data_source=data,
            replicates=1,
        )
        # Each distinct key forces a fresh generate_seeds.
        assert data.load_calls == 2

    # Supplementary guards on public-surface invariants.
    def test_replicates_zero_raises(self):
        config = _make_config(n_per_class=3)
        data = FakeDataSource(list(CATEGORIES))
        sut = ScriptedSUT(_make_scripts(), list(CATEGORIES))
        with pytest.raises(ValueError, match="replicates must be"):
            resolve_pair(
                "junco->chickadee",
                config=config, sut=sut, data_source=data,
                replicates=0,
            )

    def test_empty_pool_error_message(self):
        # All classes mis-predict → nothing enters the pool.
        scripts = {
            "junco":     [[-5.0, 0.0, -10.0]],  # top-1 chickadee (wrong)
            "chickadee": [[0.0, -5.0, -10.0]],  # top-1 junco (wrong)
            "robin":     [[0.0, -5.0, -10.0]],  # top-1 junco (wrong)
        }
        config = _make_config(n_per_class=2)
        data = FakeDataSource(list(CATEGORIES))
        sut = ScriptedSUT(scripts, list(CATEGORIES))

        with pytest.raises(ValueError) as exc_info:
            resolve_pair(
                "junco->chickadee",
                config=config, sut=sut, data_source=data,
                replicates=1,
            )
        msg = str(exc_info.value)
        assert "seed pool is empty" in msg
        assert "max_logprob_gap" in msg


class TestPairSpecDataclass:
    """Guard frozen-dataclass invariants expected by downstream callers."""

    def test_is_frozen(self):
        spec = PairSpec("a", "b", "a", "b")
        with pytest.raises(Exception):
            spec.class_a = "x"  # type: ignore[misc]
