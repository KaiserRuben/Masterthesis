"""PDQ SUT adapter.

Wraps :class:`~src.sut.vlm_sut.VLMSUT` to record every call with
timing, stage, and cache-hit information.  The adapter is the single
choke-point through which all VLM calls flow — Stage 1, Stage 2, and
the anchor call all go through it.

Call records are accumulated in memory and drained via
:meth:`pop_records` for parquet logging.
"""

from __future__ import annotations

import hashlib
import logging
from time import time
from typing import TYPE_CHECKING, Any

import torch
from PIL import Image
from torch import Tensor

if TYPE_CHECKING:
    from src.sut.vlm_sut import VLMSUT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _image_hash(image: Image.Image) -> str:
    """Quick content-based image hash (first 16 hex chars of SHA-256)."""
    h = hashlib.sha256()
    h.update(image.tobytes())
    h.update(f"{image.size}|{image.mode}".encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class SUTAdapter:
    """Thin wrapper around VLMSUT that records per-call metadata.

    Every call to :meth:`call` is logged with:

    - monotonic ``call_id``
    - ``stage`` tag (``"anchor"``, ``"stage1"``, ``"stage2"``)
    - wall time (per-call and cumulative)
    - cache-hit flag (read from ``VLMSUT.last_call_cached``)
    - logprobs, top-1/top-2 labels, logprob gap

    Records are buffered internally and drained via :meth:`pop_records`.

    :param sut: The underlying VLMSUT instance.
    """

    def __init__(self, sut: VLMSUT) -> None:
        self._sut = sut
        self._call_counter: int = 0
        self._miss_counter: int = 0
        self._wall_cumulative: float = 0.0
        self._records: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def call(
        self,
        image: Image.Image,
        text: str,
        categories: tuple[str, ...],
        stage: str,
        candidate_id: int | None = None,
    ) -> tuple[Tensor, int]:
        """Make one SUT call and record all metadata.

        :param image: PIL image to score.
        :param text: Full prompt text (template + answer options).
        :param categories: Category tuple (same order as in VLMSUT).
        :param stage: Call-site label (``"anchor"`` / ``"stage1"`` /
            ``"stage2"``).
        :param candidate_id: Candidate row id from ``candidates.parquet``
            (``None`` for the anchor call).
        :returns: ``(logprobs_tensor, call_id)`` where *logprobs_tensor*
            is shape ``(n_categories,)`` and *call_id* is the monotonic
            index of this call.
        """
        call_id = self._call_counter
        self._call_counter += 1

        t0 = time()
        logprobs: Tensor = self._sut.process_input(
            image, text=text, categories=categories,
        )
        wall_time = time() - t0

        self._wall_cumulative += wall_time

        cache_hit = self._sut.last_call_cached
        if not cache_hit:
            self._miss_counter += 1

        # Compute top-1 and top-2 indices.
        n = logprobs.shape[0]
        top1_idx = int(logprobs.argmax().item())
        if n > 1:
            top2_idx = int(
                logprobs.topk(2).indices[1].item()
            )
        else:
            top2_idx = top1_idx
        logprob_gap = float(logprobs[top1_idx] - logprobs[top2_idx])

        cats = list(categories)
        record: dict[str, Any] = {
            "call_id": call_id,
            "candidate_id": candidate_id,
            "stage": stage,
            "prompt_hash": hashlib.sha256(text.encode()).hexdigest()[:16],
            "image_hash": _image_hash(image),
            "cache_hit": cache_hit,
            "logprobs": logprobs.tolist(),
            "categories": cats,
            "top1_idx": top1_idx,
            "top1_label": cats[top1_idx],
            "top2_idx": top2_idx,
            "top2_label": cats[top2_idx],
            "logprob_gap_1v2": logprob_gap,
            "wall_time_s": wall_time,
            "wall_time_cumulative_s": self._wall_cumulative,
            "sut_call_cumulative": self._call_counter,
            "cache_miss_cumulative": self._miss_counter,
        }
        self._records.append(record)

        logger.debug(
            "SUT[%s] call_id=%d  top1=%s  gap=%.3f  cached=%s  t=%.2fs",
            stage, call_id, cats[top1_idx], logprob_gap, cache_hit, wall_time,
        )

        return logprobs, call_id

    def pop_records(self) -> list[dict[str, Any]]:
        """Drain and return all accumulated call records.

        The internal buffer is cleared after this call.

        :returns: List of call-record dicts (one per :meth:`call`
            invocation since the last drain).
        """
        recs = self._records
        self._records = []
        return recs

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def call_count(self) -> int:
        """Total number of SUT calls made through this adapter."""
        return self._call_counter

    @property
    def miss_count(self) -> int:
        """Number of cache-miss calls (actual model inferences)."""
        return self._miss_counter

    @property
    def wall_time_cumulative(self) -> float:
        """Total wall time spent in SUT calls so far (seconds)."""
        return self._wall_cumulative

    @property
    def cache_stats(self) -> dict[str, int]:
        """Cumulative cache hit/miss counts from the underlying VLMSUT.

        Forwarded verbatim so per-seed ``stats.json`` can report the
        same aggregate as the SMOO tester without duplicating the
        counter logic.
        """
        return dict(self._sut.cache_stats)
