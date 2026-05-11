"""SMOO-compatible VLM system-under-test.

Wraps a :class:`~src.sut.scorer.VLMScorer` into SMOO's
:class:`SUT` interface so the boundary tester can call
``sut.process_input(image, text)`` and get back a log-prob tensor.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import torch
from PIL import Image
from smoo.sut import SUT
from torch import Tensor

from src.config import ExperimentConfig

from .scorer import create_scorer
from .text_embedder import TextEmbedder

logger = logging.getLogger(__name__)


def _cache_key(
    model_id: str,
    image: Image.Image,
    prompt: str,
    categories: tuple[str, ...],
) -> str:
    """Build a deterministic cache key from the exact inference inputs."""
    h = hashlib.sha256()
    h.update(model_id.encode())
    h.update(image.tobytes())
    h.update(f"{image.size}|{image.mode}".encode())
    h.update(prompt.encode())
    h.update("\0".join(categories).encode())
    return h.hexdigest()


class VLMSUT(SUT):
    """VLM system-under-test with teacher-forced scoring.

    Wraps a VLM scorer into SMOO's SUT interface.  For each input
    (image + optional text), runs teacher-forced decoding to produce a
    log-prob vector over the category set.

    If a Redis server is reachable at ``config.sut.redis_url``, inference
    results are cached by exact input hash.  If Redis is unavailable the
    SUT works normally without caching.

    :param config: Configuration object.  Uses defaults when ``None``.
    """

    def __init__(
        self,
        config: ExperimentConfig | None = None,
        *,
        scorer=None,
        text_embedder=None,
        redis_client=None,
    ) -> None:
        """Build a VLMSUT.

        :param config: Experiment configuration. Uses defaults when ``None``.
        :param scorer: Pre-loaded :class:`~src.sut.scorer.VLMScorer` to
            share across worker threads. ``None`` (default) loads a
            fresh scorer using ``config.sut`` — the legacy single-process
            path. When provided, ``config.sut.*`` model-loading fields
            are ignored (the caller already loaded the model).
        :param text_embedder: Pre-built :class:`TextEmbedder` paired
            with *scorer*. Pass alongside *scorer* when sharing across
            threads so the LRU/Redis cache is unified.
        :param redis_client: Pre-connected Redis client to share. When
            ``None`` (and *scorer* is also ``None``), the constructor
            connects on its own using ``config.sut.redis_url``.
        """
        self._config = config or ExperimentConfig()
        self._device = torch.device(self._config.device)
        if scorer is None:
            self._scorer = create_scorer(
                model_id=self._config.sut.model_id,
                device=self._config.device,
                enable_thinking=self._config.sut.enable_thinking,
                max_thinking_tokens=self._config.sut.max_thinking_tokens,
                max_pixels=self._config.sut.max_pixels,
                load_in_8bit=self._config.sut.load_in_8bit,
                load_in_4bit=self._config.sut.load_in_4bit,
                backend=self._config.sut.backend,
                processor_id=self._config.sut.processor_id,
                ov_device=self._config.sut.ov_device,
            )
        else:
            self._scorer = scorer
        self._prompt = (
            self._config.prompt_template
            + self._config.answer_format.format(
                categories=", ".join(self._config.categories),
            )
        )
        self._redis = (
            redis_client
            if redis_client is not None
            else _connect_redis(self._config.sut.redis_url)
        )
        self._cache_hits = 0
        self._cache_misses = 0
        self._last_call_cached = False
        if text_embedder is not None:
            self._text_embedder = text_embedder
        else:
            self._text_embedder = _maybe_make_text_embedder(
                self._scorer,
                self._config.sut.model_id,
                self._redis,
                self.device_str,
            )

    # ------------------------------------------------------------------
    # SUT interface
    # ------------------------------------------------------------------

    def process_input(
        self,
        image: Image.Image,
        text: str | None = None,
        categories: tuple[str, ...] | None = None,
    ) -> Tensor:
        """Score an image against categories.

        Results are cached in Redis (when available) keyed on the exact
        pixel content, prompt, and category tuple.

        :param image: PIL image to classify.
        :param text: Override prompt text.  If ``None``, uses the prompt
            built from ``config.prompt_template`` and ``config.categories``.
            If provided, should be the complete prompt (template already
            filled).
        :param categories: Override categories for this call.  If ``None``,
            uses ``config.categories``.  This changes which labels are
            force-decoded.
        :returns: Tensor of shape ``(n_categories,)`` with *log_prob_norm*
            values.  ``tensor[i]`` corresponds to ``categories[i]``.
        """
        cats = categories if categories is not None else self._config.categories
        prompt = text if text is not None else self._prompt

        # --- cache lookup ---
        if self._redis is not None:
            key = _cache_key(
                self._config.sut.model_id, image, prompt, cats,
            )
            cached = self._redis.get(key)
            if cached is not None:
                self._cache_hits += 1
                self._last_call_cached = True
                return torch.tensor(json.loads(cached), dtype=torch.float32)
            self._cache_misses += 1

        self._last_call_cached = False
        result = self._scorer.score_categories_tensor(image, prompt, cats)

        # --- cache store ---
        if self._redis is not None:
            self._redis.set(key, json.dumps(result.tolist()))

        return result

    def input_valid(self, inpt: Any, cond: Any) -> tuple[bool, Any]:
        """Validate that the VLM's top prediction matches the condition.

        :param inpt: Either a PIL image, or a tuple ``(image, text)``
            where *text* is the prompt override (may be ``None``).
        :param cond: Expected top-1 category (``str``).
        :returns: ``(is_valid, logprobs_tensor)`` where *is_valid* is
            ``True`` when the highest log-prob category equals *cond*.
        """
        if isinstance(inpt, tuple):
            image, text = inpt
        else:
            image, text = inpt, None

        logprobs = self.process_input(image, text=text)
        top_idx = int(logprobs.argmax().item())
        top_label = self._config.categories[top_idx]
        return top_label == cond, logprobs

    @property
    def text_embedder(self) -> TextEmbedder | None:
        """Text-only sentence embedder sharing this SUT's backbone.

        ``None`` when the scorer doesn't support ``encode_text`` (currently
        the OpenVINO backend). Callers must handle ``None`` and substitute
        a zero-distance fallback for the :class:`TextEmbeddingDistance`
        objective.
        """
        return self._text_embedder

    @property
    def scorer(self):
        """Underlying :class:`~src.sut.scorer.VLMScorer`.

        Exposed so the parent runner can build per-thread VLMSUT
        wrappers that share one set of model weights.
        """
        return self._scorer

    @property
    def redis_client(self):
        """Shared Redis client (or ``None``). Exposed for thread fan-out."""
        return self._redis

    @property
    def device_str(self) -> str:
        """String form of the SUT device — used as the distlock key.

        Torch backends return e.g. ``"mps"`` / ``"cuda:0"``. OpenVINO
        backends report the OV device (``"ov:GPU"`` / ``"ov:CPU"``) so
        the lock key reflects the actual physical accelerator, not the
        ignored torch device.
        """
        if self._config.sut.backend == "openvino":
            return f"ov:{self._config.sut.ov_device}"
        return str(self._device)

    @property
    def cache_stats(self) -> dict[str, int]:
        """Return cumulative cache hit/miss counts."""
        return {"hits": self._cache_hits, "misses": self._cache_misses}

    @property
    def last_call_cached(self) -> bool:
        """Whether the most recent ``process_input`` call was served from cache."""
        return self._last_call_cached


# -----------------------------------------------------------------------
# TextEmbedder factory -- skip when scorer can't encode_text
# -----------------------------------------------------------------------


def _maybe_make_text_embedder(scorer, model_id, redis, device_str):
    """Build a :class:`TextEmbedder` only if the scorer supports it.

    The OpenVINO backend exports the IR with logits-only outputs, so its
    ``encode_text`` raises :class:`NotImplementedError`. We probe with a
    one-token call; on failure, return ``None``.
    """
    try:
        scorer.encode_text(["x"])
    except NotImplementedError:
        logger.info(
            "Scorer does not support encode_text -- TextEmbeddingDistance "
            "will receive zeros (effective 2-objective optimization)."
        )
        return None
    return TextEmbedder(scorer, model_id, redis, device_str=device_str)


# -----------------------------------------------------------------------
# Redis helper
# -----------------------------------------------------------------------


def _connect_redis(url: str):
    """Try to connect to Redis.  Return client or ``None`` on failure."""
    if not url:
        return None
    try:
        import redis

        client = redis.Redis.from_url(url, decode_responses=True)
        client.ping()
        logger.info("Inference cache connected to Redis at %s", url)
        return client
    except Exception:  # noqa: BLE001
        logger.info("Redis unavailable at %s — running without cache", url)
        return None