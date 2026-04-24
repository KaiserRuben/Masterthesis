"""Cached text-embedding + anchor-distance helper on top of a VLMScorer.

Scope is intentionally narrow: caching + cosine arithmetic. The actual
forward pass lives on :meth:`~src.sut.scorer.VLMScorer.encode_text`, so
this class is model-agnostic and trivially testable against any object
that exposes that method.

Backends:
    * Redis (when a client is passed in) — shared namespace with the SUT's
      inference cache, keys prefixed ``txtemb:<sha256(model_id,text)>``.
      Values are JSON-encoded float lists, matching the inference cache's
      convention so one ``decode_responses=True`` client serves both.
    * In-process LRU (fallback) — size 16k strings.
"""

from __future__ import annotations

import hashlib
import json
import logging
from functools import lru_cache
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class _SupportsEncodeText(Protocol):
    def encode_text(self, texts: list[str]) -> np.ndarray: ...


def _cache_key(model_id: str, text: str) -> str:
    h = hashlib.sha256()
    h.update(model_id.encode())
    h.update(b"\x1f")
    h.update(text.encode())
    return "txtemb:" + h.hexdigest()


class TextEmbedder:
    """Cached wrapper around ``scorer.encode_text`` with anchor-distance helper.

    :param scorer: Any object exposing
        :meth:`~src.sut.scorer.VLMScorer.encode_text` — usually the
        ``VLMSUT``'s internal scorer so model weights are shared.
    :param model_id: Model identifier used as cache namespace.
    :param redis_client: Optional Redis client; falls back to an LRU when ``None``.
    """

    def __init__(
        self,
        scorer: _SupportsEncodeText,
        model_id: str,
        redis_client=None,
    ) -> None:
        self._scorer = scorer
        self._model_id = model_id
        self._redis = redis_client
        self._hits = 0
        self._misses = 0
        if self._redis is None:
            self._embed_mem = lru_cache(maxsize=16384)(self._miss_single)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def cache_stats(self) -> dict[str, int]:
        return {"hits": self._hits, "misses": self._misses}

    def embed(self, text: str) -> np.ndarray:
        """Return a 1-D float32 embedding of *text* (cached)."""
        if self._redis is None:
            return self._embed_mem(text).copy()
        key = _cache_key(self._model_id, text)
        cached = self._redis.get(key)
        if cached is not None:
            self._hits += 1
            return np.asarray(json.loads(cached), dtype=np.float32)
        self._misses += 1
        vec = self._miss_many([text])[0]
        self._redis.set(key, json.dumps(vec.tolist()))
        return vec

    def embed_many(self, texts: list[str]) -> np.ndarray:
        """Return ``(N, D)`` float32 embeddings, batching the misses."""
        if self._redis is None:
            return np.stack([self.embed(t) for t in texts], axis=0)

        keys = [_cache_key(self._model_id, t) for t in texts]
        cached = self._redis.mget(keys)

        out: list[np.ndarray | None] = [None] * len(texts)
        misses_idx: list[int] = []
        misses_txt: list[str] = []
        for i, blob in enumerate(cached):
            if blob is None:
                misses_idx.append(i)
                misses_txt.append(texts[i])
            else:
                self._hits += 1
                out[i] = np.asarray(json.loads(blob), dtype=np.float32)
        self._misses += len(misses_idx)

        if misses_txt:
            vecs = self._miss_many(misses_txt)
            pipe = self._redis.pipeline()
            for slot, v in zip(misses_idx, vecs):
                pipe.set(keys[slot], json.dumps(v.tolist()))
                out[slot] = v
            pipe.execute()

        return np.stack(out, axis=0)

    def cosine_distances_to(
        self, anchor_embedding: np.ndarray, texts: list[str],
    ) -> np.ndarray:
        """Cosine distance (``1 - cos_sim``) of each text to *anchor_embedding*.

        Clamped to ``[0, inf)`` so identical texts return exactly 0.0
        instead of drifting to ``-0.0`` through float error.
        """
        vecs = self.embed_many(texts)
        vn = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
        an = anchor_embedding / (np.linalg.norm(anchor_embedding) + 1e-12)
        sim = vn @ an
        return np.clip(1.0 - sim, 0.0, None).astype(np.float32)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _miss_single(self, text: str) -> np.ndarray:
        return self._miss_many([text])[0]

    def _miss_many(self, texts: list[str]) -> np.ndarray:
        return self._scorer.encode_text(texts)
