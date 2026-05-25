"""Lightweight Redis client wrapper with graceful in-memory fallback.

Existing call sites (``src.sut.vlm_sut``, ``src.sut.text_embedder``) talk to
Redis through ``decode_responses=True`` clients holding JSON-encoded payloads.
The cone-filter modal-target cache stores raw bytes instead (numpy arrays
serialized via ``np.save``), so it needs a sibling helper that talks bytes:
``connect_bytes_redis()`` plus a tiny ``BytesRedisCache`` facade with
``get/set`` semantics and ``decode_responses=False``.

The cache degrades gracefully: when Redis is unreachable, calls return
``None`` on ``get`` and silently no-op on ``set``. Higher layers fall back to
their L1 (in-memory) caches without blocking experiments.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def connect_bytes_redis(url: str) -> Optional[object]:
    """Return a binary-mode ``redis.Redis`` client, or ``None`` on failure.

    Returns a client constructed with ``decode_responses=False`` so callers
    can store and retrieve raw bytes. Network/auth failures are caught and
    logged at INFO; experiments keep running with cache disabled.
    """
    if not url:
        return None
    try:
        import redis

        client = redis.Redis.from_url(url, decode_responses=False)
        client.ping()
        logger.info("Bytes-mode Redis cache connected at %s", url)
        return client
    except Exception:  # noqa: BLE001
        logger.info("Redis unavailable at %s — falling back to in-memory only", url)
        return None


class BytesRedisCache:
    """Thin bytes-typed wrapper around a Redis client.

    Holds the client (or ``None``) and exposes ``get/set`` that never raise:
    when no client is present, ``get`` returns ``None`` and ``set`` is a
    no-op. This keeps cache-hit/-miss logic identical regardless of whether
    Redis is reachable.
    """

    __slots__ = ("_client",)

    def __init__(self, client: Optional[object]) -> None:
        self._client = client

    @classmethod
    def from_url(cls, url: str) -> "BytesRedisCache":
        return cls(connect_bytes_redis(url))

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    def get(self, key: str) -> bytes | None:
        if self._client is None:
            return None
        try:
            return self._client.get(key)
        except Exception:  # noqa: BLE001
            logger.warning("Redis GET failed for %s — treating as miss", key)
            return None

    def set(self, key: str, value: bytes) -> None:
        if self._client is None:
            return
        try:
            self._client.set(key, value)
        except Exception:  # noqa: BLE001
            logger.warning("Redis SET failed for %s — cache miss persists", key)


__all__ = ["BytesRedisCache", "connect_bytes_redis"]
