"""Tests for the modal target grid builder.

Uses tiny in-process stubs for the codec and the data source so the
tests neither load VQGAN nor hit ImageNet / Redis.
"""

from __future__ import annotations

import dataclasses
import io
from typing import Iterable

import numpy as np
import pytest
from PIL import Image

from src.common.redis_cache import BytesRedisCache
from src.manipulator.image.class_target import (
    CLASS_TARGET_KEY_PREFIX,
    ModalTargetBuilder,
    _codebook_hash,
    _grid_from_bytes,
    _grid_to_bytes,
    _safe_class_name,
    build_modal_grid,
)
from src.manipulator.image.types import CodeGrid


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubCodec:
    """Minimal codec that returns a pre-supplied :class:`CodeGrid` per image.

    Maps the input PIL image's first pixel to a slot in an internal table.
    Lets a test drive the codec deterministically by colouring the input
    images differently.
    """

    def __init__(self, codebook: np.ndarray, table: dict[int, CodeGrid]) -> None:
        self._codebook = codebook.astype(np.float32)
        h, w = next(iter(table.values())).shape
        self._grid_size = (h, w)
        self._table = table

    @property
    def codebook(self) -> np.ndarray:
        return self._codebook

    @property
    def grid_size(self) -> tuple[int, int]:
        return self._grid_size

    def encode(self, image: Image.Image) -> CodeGrid:
        # Use the RED channel of the (0, 0) pixel as a key. Tests build
        # images with a unique colour per exemplar so encode is one-to-one.
        key = int(image.getpixel((0, 0))[0])
        return self._table[key]


@dataclasses.dataclass(frozen=True)
class _Sample:
    image: Image.Image
    class_idx: int
    class_name: str


class _StubDataSource:
    """Tiny data source that yields pre-built exemplars per class."""

    def __init__(self, per_class: dict[str, list[Image.Image]]) -> None:
        self._per_class = per_class

    def labels(self) -> list[str]:
        return list(self._per_class.keys())

    def load_samples(
        self, categories: list[str], n_per_class: int,
    ) -> list[_Sample]:
        out: list[_Sample] = []
        for cat in categories:
            for img in self._per_class.get(cat, [])[:n_per_class]:
                out.append(_Sample(image=img, class_idx=0, class_name=cat))
        return out


class _InMemoryRedis:
    """Trivial in-memory key/value store mimicking the ``BytesRedisCache`` surface.

    Lets a test exercise the L2 round-trip without spinning up Redis.
    """

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}
        self.get_calls: int = 0
        self.set_calls: int = 0

    def get(self, key: str) -> bytes | None:
        self.get_calls += 1
        return self.store.get(key)

    def set(self, key: str, value: bytes) -> None:
        self.set_calls += 1
        self.store[key] = value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _img(red: int) -> Image.Image:
    """Build an 8x8 PIL image whose (0, 0) red channel is ``red``."""
    return Image.new("RGB", (8, 8), (red, 0, 0))


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestSafeClassName:
    def test_lowercases_and_replaces_spaces(self) -> None:
        assert _safe_class_name("Great White Shark") == "great_white_shark"

    def test_idempotent(self) -> None:
        out = _safe_class_name("great_white_shark")
        assert _safe_class_name(out) == out


class TestCodebookHash:
    def test_stable_under_round_trip(self) -> None:
        rng = np.random.default_rng(0)
        cb = rng.standard_normal((32, 4)).astype(np.float32)
        h1 = _codebook_hash(cb)
        h2 = _codebook_hash(cb.copy())
        assert h1 == h2
        assert len(h1) == 16

    def test_changes_with_content(self) -> None:
        rng = np.random.default_rng(0)
        cb1 = rng.standard_normal((32, 4)).astype(np.float32)
        cb2 = cb1.copy()
        cb2[0, 0] += 1.0
        assert _codebook_hash(cb1) != _codebook_hash(cb2)


class TestGridBytesRoundTrip:
    def test_round_trip(self) -> None:
        grid = np.arange(9, dtype=np.int64).reshape(3, 3)
        recovered = _grid_from_bytes(_grid_to_bytes(grid))
        np.testing.assert_array_equal(grid, recovered)
        assert recovered.dtype == np.int64


# ---------------------------------------------------------------------------
# build_modal_grid
# ---------------------------------------------------------------------------


class TestBuildModalGrid:
    def setup_method(self) -> None:
        # Grid 2x2, codebook size 4.
        self.cb = np.eye(4, dtype=np.float32)
        # Three exemplars; per-position codeword:
        #   pos (0,0): [1, 1, 2]  → argmax 1
        #   pos (0,1): [3, 0, 0]  → argmax 0
        #   pos (1,0): [2, 2, 2]  → argmax 2
        #   pos (1,1): [0, 1, 1]  → argmax 1
        grids = [
            CodeGrid(np.array([[1, 3], [2, 0]], dtype=np.int64)),
            CodeGrid(np.array([[1, 0], [2, 1]], dtype=np.int64)),
            CodeGrid(np.array([[2, 0], [2, 1]], dtype=np.int64)),
        ]
        self.codec = _StubCodec(
            self.cb,
            table={i: grids[i] for i in range(3)},
        )
        self.imgs = [_img(i) for i in range(3)]

    def test_per_position_argmax(self) -> None:
        modal = build_modal_grid(self.codec, self.imgs, n_codes=4)
        np.testing.assert_array_equal(modal, [[1, 0], [2, 1]])
        assert modal.dtype == np.int64

    def test_deterministic(self) -> None:
        a = build_modal_grid(self.codec, self.imgs, n_codes=4)
        b = build_modal_grid(self.codec, list(reversed(self.imgs)), n_codes=4)
        # Argmax of the histogram is order-independent.
        np.testing.assert_array_equal(a, b)

    def test_m_equals_one(self) -> None:
        modal = build_modal_grid(self.codec, [self.imgs[0]], n_codes=4)
        # With a single exemplar the modal grid is just that grid.
        np.testing.assert_array_equal(modal, [[1, 3], [2, 0]])

    def test_empty_iterable_raises(self) -> None:
        with pytest.raises(ValueError, match="empty image iterable"):
            build_modal_grid(self.codec, [], n_codes=4)


# ---------------------------------------------------------------------------
# ModalTargetBuilder
# ---------------------------------------------------------------------------


class TestModalTargetBuilder:
    def setup_method(self) -> None:
        self.cb = np.eye(4, dtype=np.float32)
        grids = {
            0: CodeGrid(np.array([[1, 0], [2, 1]], dtype=np.int64)),
            1: CodeGrid(np.array([[1, 0], [2, 1]], dtype=np.int64)),
        }
        self.codec = _StubCodec(self.cb, table=grids)
        self.data = _StubDataSource(per_class={"junco": [_img(0), _img(1)]})

    def _builder(
        self, *, redis_cache: BytesRedisCache | None = None,
    ) -> ModalTargetBuilder:
        return ModalTargetBuilder(
            codec=self.codec,
            data_source=self.data,
            preset="test-preset",
            target_m=2,
            redis_cache=redis_cache,
        )

    def test_cache_key_uses_safe_class_name_and_preset(self) -> None:
        b = self._builder()
        key = b.cache_key("Junco")
        assert key.startswith(CLASS_TARGET_KEY_PREFIX + ":test-preset:")
        assert ":junco:m2" in key
        # Codebook hash component is between preset and class name.
        assert _codebook_hash(self.cb) in key

    def test_ensure_builds_then_hits_l1(self) -> None:
        b = self._builder()
        first = b.ensure("junco")
        # Same instance returned on the second call (L1 hit).
        second = b.get("junco")
        assert second is not None
        np.testing.assert_array_equal(first, second)

    def test_ensure_writes_l2_and_l2_hit_skips_build(self) -> None:
        redis = _InMemoryRedis()
        cache = BytesRedisCache(redis)
        b = self._builder(redis_cache=cache)
        first = b.ensure("junco")
        assert redis.set_calls == 1
        key = b.cache_key("junco")
        assert key in redis.store

        # New builder with a SAME redis (shared L2) and an empty L1.
        b2 = self._builder(redis_cache=cache)
        # Hydrate L1 via L2 — must not invoke the codec (no fresh build).
        before_get_calls = redis.get_calls
        hydrated = b2.ensure("junco")
        np.testing.assert_array_equal(first, hydrated)
        assert redis.get_calls == before_get_calls + 1
        assert redis.set_calls == 1  # no extra write

    def test_get_returns_none_when_uncached(self) -> None:
        b = self._builder()
        assert b.get("junco") is None

    def test_populate_many_pre_fills_l1(self) -> None:
        b = self._builder()
        b.populate_many(["junco"])
        assert b.get("junco") is not None

    def test_class_with_no_exemplars_raises(self) -> None:
        b = ModalTargetBuilder(
            codec=self.codec,
            data_source=_StubDataSource(per_class={}),
            preset="test-preset",
            target_m=2,
        )
        with pytest.raises(RuntimeError, match="No exemplars"):
            b.ensure("missing_class")

    def test_redis_disconnected_falls_back_to_l1(self) -> None:
        # ``BytesRedisCache(None)`` simulates a Redis that never connects.
        cache = BytesRedisCache(None)
        b = self._builder(redis_cache=cache)
        grid = b.ensure("junco")
        # Build still completes, L1 populated, even though L2 is dead.
        assert b.get("junco") is not None
        # And a second ensure returns the same array (no rebuild).
        np.testing.assert_array_equal(grid, b.ensure("junco"))
