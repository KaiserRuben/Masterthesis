"""Process-local device mutex.

Single-holder lock keyed by device string. Used to serialise GPU access
across worker threads inside one process: each device gets one
``threading.Lock``; whichever worker holds it has exclusive use of the
device until it releases. Different device strings never contend.

The single-process design (one model per device, N evaluation threads)
makes a cross-process distributed lock unnecessary — a plain
``threading.Lock`` is enough and adds no Redis round-trip per GPU call.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager, nullcontext
from typing import Iterator

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Module-level lock registry
# ----------------------------------------------------------------------
#
# Keyed by device string ("mps", "cuda:0", "cpu", "ov:GPU"). Locks are
# created lazily on first use; the registry itself is guarded by a
# bootstrap mutex so concurrent first-use is safe.

_LOCKS: dict[str, threading.Lock] = {}
_REGISTRY_LOCK = threading.Lock()
_ENABLED = False


def configure(enable: bool) -> None:
    """Toggle device locking process-wide.

    When disabled (default), :func:`lock` returns a no-op context. The
    runner enables it whenever ``parallel.workers > 1`` so single-worker
    runs pay zero overhead.
    """
    global _ENABLED
    _ENABLED = bool(enable)


def _get_lock(device: str) -> threading.Lock:
    held = _LOCKS.get(device)
    if held is not None:
        return held
    with _REGISTRY_LOCK:
        held = _LOCKS.get(device)
        if held is None:
            held = threading.Lock()
            _LOCKS[device] = held
        return held


@contextmanager
def lock(device: str) -> Iterator[None]:
    """Acquire the mutex for *device* for the duration of the ``with`` block.

    Same device string → same lock → serialised access. Different device
    strings → no contention. No-op when locking is disabled (single-
    worker mode).
    """
    if not _ENABLED:
        with nullcontext():
            yield
        return
    held = _get_lock(device)
    held.acquire()
    try:
        yield
    finally:
        held.release()
