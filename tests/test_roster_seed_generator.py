"""Tests for src.common.roster_seed_generator.

Uses fake SUT + DataSource to avoid loading any model. The fake SUT
deterministically scores images by index encoded in their pixel value,
so we can prescribe acceptance / rejection per image.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
import torch
from PIL import Image

from src.common.roster_seed_generator import roster_seeds
from src.config import (
    AbstractionConfig,
    ExperimentConfig,
    GapFilterConfig,
    RosterConfig,
    SeedConfig,
    SUTConfig,
)
from src.data.imagenet import ImageSample


# -- Fakes -----------------------------------------------------------------


class FakeDataSource:
    """Returns deterministic ImageSample list for the given categories.

    Each image is a tiny PIL whose single pixel value encodes a per-class
    counter. The fake SUT decodes that counter to determine acceptance.
    """

    def __init__(self, all_labels: list[str]) -> None:
        self._labels = list(all_labels)

    def labels(self) -> list[str]:
        return list(self._labels)

    def load_samples(
        self, categories: list[str], n_per_class: int,
    ) -> list[ImageSample]:
        out = []
        for cat in categories:
            cls_idx = self._labels.index(cat)
            for k in range(n_per_class):
                # Pixel value (cls_idx, k, 0) so the fake SUT can decode
                # which (class, candidate) this image came from.
                img = Image.new("RGB", (1, 1), color=(cls_idx, k, 0))
                out.append(ImageSample(
                    image=img, class_idx=cls_idx, class_name=cat,
                ))
        return out


class FakeSUT:
    """Returns a logprob tensor where one chosen class index is the argmax.

    Acceptance is controlled by a (cls_idx, k) -> (top_idx, top_logprob)
    table. Defaults: top_idx = cls_idx (correct), top_logprob = 1.0.
    """

    def __init__(
        self,
        all_labels: list[str],
        overrides: dict[tuple[int, int], tuple[int, float]] | None = None,
    ) -> None:
        self._labels = list(all_labels)
        self._overrides = overrides or {}

    def process_input(
        self,
        image: Image.Image,
        text: str | None = None,
        categories: tuple[str, ...] | None = None,
    ) -> torch.Tensor:
        cls_idx, k, _ = image.getpixel((0, 0))
        top_idx, top_logprob = self._overrides.get(
            (cls_idx, k), (cls_idx, 1.0),
        )
        cats = categories if categories is not None else tuple(self._labels)
        # Build a flat logprob vector: -10 everywhere except top_idx.
        out = torch.full((len(cats),), -10.0, dtype=torch.float32)
        # Map top_idx (a position in self._labels) into the cats tuple.
        # Test fixture controls categories=full label tuple, so positions match.
        out[top_idx] = top_logprob
        return out


def _make_config(
    class_list: list[str],
    seeds_per_class: int = 2,
    min_anchor_confidence: float = 0.5,
) -> ExperimentConfig:
    return ExperimentConfig(
        device="cpu",
        categories=tuple(class_list),
        seeds=SeedConfig(
            mode="roster",
            roster=RosterConfig(
                class_list=tuple(class_list),
                seeds_per_class=seeds_per_class,
                min_anchor_confidence=min_anchor_confidence,
                abstraction=AbstractionConfig(),
            ),
        ),
        sut=SUTConfig(),
    )


# -- Tests -----------------------------------------------------------------


class TestRosterSeeds:
    def test_happy_path_collects_n_per_class(self) -> None:
        labels = ["junco", "Beagle", "great white shark"]
        ds = FakeDataSource(labels)
        sut = FakeSUT(labels)
        cfg = _make_config(labels, seeds_per_class=2,
                           min_anchor_confidence=0.5)

        out = roster_seeds(sut, cfg, ds)
        assert len(out) == 6  # 3 classes × 2 seeds

        # Order: class-by-class in class_list order; seed_idx 0..N-1.
        assert [s.class_concrete for s in out] == [
            "junco", "junco",
            "Beagle", "Beagle",
            "great white shark", "great white shark",
        ]
        assert [s.seed_idx_in_class for s in out] == [0, 1, 0, 1, 0, 1]

    def test_below_threshold_filtered(self) -> None:
        labels = ["junco", "Beagle"]
        ds = FakeDataSource(labels)
        # Threshold 0.5 → reject any logprob < -0.5. Force junco's first 3
        # candidates to a strongly-negative logprob (correctly classified
        # but not confident enough). k=3 onward keep the FakeSUT default
        # top_logprob=1.0 (>= -0.5) → accepted.
        overrides = {
            (0, k): (0, -2.0)
            for k in range(3)
        }
        sut = FakeSUT(labels, overrides=overrides)
        cfg = _make_config(labels, seeds_per_class=2,
                           min_anchor_confidence=0.5)
        # oversample_factor = 4, seeds_per_class=2 → 8 candidates per class.
        out = roster_seeds(sut, cfg, ds)
        assert len(out) == 4

    def test_misclassified_filtered(self) -> None:
        labels = ["junco", "Beagle"]
        ds = FakeDataSource(labels)
        # Force junco's first 5 candidates to be mis-classified as Beagle.
        overrides = {(0, k): (1, 1.0) for k in range(5)}  # top_idx=1=Beagle
        sut = FakeSUT(labels, overrides=overrides)
        cfg = _make_config(labels, seeds_per_class=2,
                           min_anchor_confidence=0.5)
        out = roster_seeds(sut, cfg, ds)
        # 2 junco seeds (k=5,6) + 2 Beagle seeds (k=0,1) accepted.
        assert len(out) == 4

    def test_pool_exhaustion_raises(self) -> None:
        labels = ["junco", "Beagle"]
        ds = FakeDataSource(labels)
        # All junco candidates mis-classified — pool exhausts.
        overrides = {(0, k): (1, 1.0) for k in range(100)}
        sut = FakeSUT(labels, overrides=overrides)
        cfg = _make_config(labels, seeds_per_class=2,
                           min_anchor_confidence=0.5)
        with pytest.raises(RuntimeError, match="pool exhaustion"):
            roster_seeds(sut, cfg, ds)

    def test_invalid_class_in_roster_raises(self) -> None:
        labels = ["junco", "made_up_class_xyz"]
        ds = FakeDataSource(labels)
        sut = FakeSUT(labels)
        cfg = _make_config(labels, seeds_per_class=1)
        # validate_class_list catches taxonomy-unknowns first.
        with pytest.raises(ValueError, match="unknown to taxonomy"):
            roster_seeds(sut, cfg, ds)

    def test_empty_class_list_raises(self) -> None:
        labels = ["junco", "Beagle"]
        ds = FakeDataSource(labels)
        sut = FakeSUT(labels)
        cfg = _make_config([], seeds_per_class=1)
        # Empty class list rejected at the roster_seeds entry.
        with pytest.raises(ValueError, match="non-empty"):
            roster_seeds(sut, cfg, ds)

    def test_wrong_mode_raises(self) -> None:
        labels = ["junco"]
        ds = FakeDataSource(labels)
        sut = FakeSUT(labels)
        cfg = ExperimentConfig(
            device="cpu",
            categories=tuple(labels),
            seeds=SeedConfig(
                mode="gap_filter",
                gap_filter=GapFilterConfig(),
            ),
            sut=SUTConfig(),
        )
        with pytest.raises(ValueError, match="seeds.mode='roster'"):
            roster_seeds(sut, cfg, ds)
