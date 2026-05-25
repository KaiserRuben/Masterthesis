"""Modal target grid: per-position argmax of codeword histograms over class exemplars.

Given a target class ``T`` and ``m`` exemplar images, encode each via the VQGAN
codec, then for every grid position take the most-frequent codeword across the
``m`` encodings. The resulting (H, W) integer array is the "modal" target
representation of the class in codebook-index space.

Two-layer cache:

* L1 (in-memory) — ``dict[target_class_safe, ndarray]`` held on the builder.
* L2 (Redis) — bytes payload under the key
  ``{CLASS_TARGET_KEY_PREFIX}:{preset}:{codebook_hash}:{class_safe}:m{m}``.
  Optional; degrades to L1-only when Redis is unreachable.

No on-disk fallback by design — the grids are <10 KB and (re)building them
costs ~``m`` VQGAN encodes, which is bounded and acceptable.
"""

from __future__ import annotations

import hashlib
import io
import logging
from typing import TYPE_CHECKING, Iterable, Protocol

import numpy as np
from numpy.typing import NDArray
from PIL import Image

if TYPE_CHECKING:
    from src.common.redis_cache import BytesRedisCache
    from src.data import DataSource

    from .codec import VQGANCodec

logger = logging.getLogger(__name__)


# Top-level key prefix for the Redis modal-grid cache. Surfaced as a constant
# so future schema bumps (e.g. encoder change, class-name normalisation
# change) can be expressed by adjusting this single literal.
CLASS_TARGET_KEY_PREFIX = "vqgan:class_target"


class _SupportsEncodeMany(Protocol):
    """Minimal interface the builder needs from a codec.

    Real ``VQGANCodec`` satisfies this; tests inject a stub.
    """

    @property
    def codebook(self) -> NDArray[np.float32]: ...

    @property
    def grid_size(self) -> tuple[int, int]: ...

    def encode(self, image: Image.Image): ...


def _safe_class_name(category: str) -> str:
    """Normalise a class name for cache keys.

    Matches :meth:`src.data.imagenet.ImageNetCache._safe_name` so the same
    spelling normalises consistently across the codebase.
    """
    return category.replace(" ", "_").lower()


def _codebook_hash(codebook: NDArray[np.float32]) -> str:
    """First 16 hex chars of sha256(codebook bytes). Same convention as KNN cache."""
    return hashlib.sha256(codebook.tobytes()).hexdigest()[:16]


def _grid_to_bytes(grid: NDArray[np.int64]) -> bytes:
    """Serialise an int64 modal grid via ``np.save`` (no .npy header on disk)."""
    buf = io.BytesIO()
    np.save(buf, grid.astype(np.int64), allow_pickle=False)
    return buf.getvalue()


def _grid_from_bytes(blob: bytes) -> NDArray[np.int64]:
    return np.load(io.BytesIO(blob), allow_pickle=False).astype(np.int64)


def build_modal_grid(
    codec: _SupportsEncodeMany,
    images: Iterable[Image.Image],
    n_codes: int,
) -> NDArray[np.int64]:
    """Per-position argmax of the codeword histogram over ``images``.

    :param codec: Object exposing ``encode(image)`` returning a ``CodeGrid``
        and ``grid_size``.
    :param images: Iterable of PIL exemplar images.
    :param n_codes: Codebook vocabulary size.
    :returns: ``(H, W)`` int64 array — the modal codeword per grid position.
    :raises ValueError: If no images are provided.
    """
    h, w = codec.grid_size
    counts = np.zeros((h, w, n_codes), dtype=np.int64)
    n_seen = 0
    for img in images:
        grid = codec.encode(img)
        idx = grid.indices  # (h, w) int64
        # Vectorised histogram update: counts[r, c, idx[r, c]] += 1.
        rows = np.arange(h)[:, None]
        cols = np.arange(w)[None, :]
        counts[rows, cols, idx] += 1
        n_seen += 1
    if n_seen == 0:
        raise ValueError(
            "build_modal_grid() received an empty image iterable — "
            "cannot build a modal grid without at least one exemplar."
        )
    # argmax along the codebook axis. Ties broken by lowest index (np.argmax
    # is stable in that direction), which is deterministic across runs.
    return counts.argmax(axis=2).astype(np.int64)


class ModalTargetBuilder:
    """Build + cache per-class modal target grids.

    Uses a two-layer cache: L1 (per-instance dict) hit returns immediately;
    L2 (Redis bytes) hit hydrates L1; miss encodes ``m`` exemplars from
    ``data_source.load_samples([class_name], n_per_class=m)`` and writes
    both layers. Parallel workers calling :meth:`ensure` concurrently
    converge on the same result without locking — first writer wins on
    Redis (idempotent), and the L1 cache is per-instance so each worker
    keeps its own copy.

    :param codec: The VQGAN codec — used to encode exemplars and to derive
        the codebook hash / grid size.
    :param data_source: Provides ``load_samples`` to fetch class exemplars.
    :param preset: VQGAN preset name (e.g. ``"f8-16384"``). Used in the L2
        cache key so different models do not share modal grids.
    :param redis_cache: Optional bytes-mode cache facade (see
        :class:`src.common.redis_cache.BytesRedisCache`). Pass ``None`` to
        disable L2.
    :param target_m: Number of exemplars used per class.
    """

    def __init__(
        self,
        codec: _SupportsEncodeMany,
        data_source: "DataSource",
        *,
        preset: str,
        target_m: int,
        redis_cache: "BytesRedisCache | None" = None,
    ) -> None:
        self._codec = codec
        self._data_source = data_source
        self._preset = preset
        self._target_m = target_m
        self._redis = redis_cache
        self._cb_hash = _codebook_hash(np.asarray(codec.codebook))
        self._n_codes = int(np.asarray(codec.codebook).shape[0])
        self._l1: dict[str, NDArray[np.int64]] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def grid_size(self) -> tuple[int, int]:
        return self._codec.grid_size

    @property
    def target_m(self) -> int:
        return self._target_m

    # ------------------------------------------------------------------
    # Cache key + lookup
    # ------------------------------------------------------------------

    def cache_key(self, class_name: str) -> str:
        """Return the canonical L2 (Redis) cache key for ``class_name``."""
        safe = _safe_class_name(class_name)
        return (
            f"{CLASS_TARGET_KEY_PREFIX}:{self._preset}:{self._cb_hash}:"
            f"{safe}:m{self._target_m}"
        )

    def _l1_key(self, class_name: str) -> str:
        return _safe_class_name(class_name)

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def get(self, class_name: str) -> NDArray[np.int64] | None:
        """Return cached modal grid for ``class_name`` if present.

        Checks L1 first; on miss, peeks at L2 and hydrates L1 (no build).
        Returns ``None`` when neither layer has the grid.
        """
        l1k = self._l1_key(class_name)
        cached = self._l1.get(l1k)
        if cached is not None:
            return cached
        if self._redis is not None:
            blob = self._redis.get(self.cache_key(class_name))
            if blob is not None:
                grid = _grid_from_bytes(blob)
                self._l1[l1k] = grid
                return grid
        return None

    def ensure(self, class_name: str) -> NDArray[np.int64]:
        """Return the modal grid for ``class_name``, building it on miss.

        Build path: L1 miss → L2 miss → encode ``target_m`` exemplars → write
        L2 → write L1. Redis I/O failures degrade gracefully (logged in the
        Redis client; the build still completes and L1 is populated).
        """
        cached = self.get(class_name)
        if cached is not None:
            return cached

        logger.info(
            "Modal target cache miss for %r — encoding %d exemplars...",
            class_name, self._target_m,
        )
        samples = self._data_source.load_samples(
            categories=[class_name],
            n_per_class=self._target_m,
        )
        if not samples:
            raise RuntimeError(
                f"No exemplars available for class {class_name!r}; "
                "cannot build modal target grid."
            )
        grid = build_modal_grid(
            self._codec,
            (s.image for s in samples),
            n_codes=self._n_codes,
        )
        if self._redis is not None:
            self._redis.set(self.cache_key(class_name), _grid_to_bytes(grid))
        self._l1[self._l1_key(class_name)] = grid
        return grid

    def populate_many(self, class_names: Iterable[str]) -> None:
        """Pre-populate L1 (via L2 or build) for a batch of class names.

        Idempotent and lock-free: parallel workers calling this with the
        same class lists either all hit L2, or one writes and the others
        observe the write on their next L2 lookup. Failure to fetch from
        L2 for an individual class triggers a build for that class only.
        """
        for name in class_names:
            self.ensure(name)


__all__ = [
    "CLASS_TARGET_KEY_PREFIX",
    "ModalTargetBuilder",
    "build_modal_grid",
]
