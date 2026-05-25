"""Generic N-worker seed-dispatch helper.

Shared by every pipeline runner: each builds its own per-worker bundle
type, but the round-robin slicing + ``ThreadPoolExecutor`` pattern is
identical, so it lives here.

``workers == 1`` short-circuits to sequential execution (no thread, no
lock churn).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Sequence, TypeVar

logger = logging.getLogger(__name__)


B = TypeVar("B")  # per-worker bundle type
S = TypeVar("S")  # indexed-seed tuple type, typically (int, SeedTriple)


def dispatch_workers(
    *,
    workers: int,
    indexed_seeds: Sequence[S],
    build_bundle: Callable[[int], B],
    run_slice: Callable[[Sequence[S], B], None],
    thread_name_prefix: str = "seedw",
) -> None:
    """Run *run_slice* over a round-robin slicing of *indexed_seeds*.

    *build_bundle* receives the worker_id (0…N−1) and constructs the
    thin per-worker state (VLMSUT thread wrapper, manipulator,
    optimiser, …).  *run_slice* receives a slice and the matching
    bundle and runs the per-seed loop.

    With ``workers == 1`` no thread is spawned; ``build_bundle(0)`` and
    ``run_slice`` run on the calling thread.

    :param workers: Number of worker threads (clamped to ≥ 1).
    :param indexed_seeds: List of ``(seed_idx, seed)`` tuples, typically
        from :func:`src.common.apply_seed_filter`.
    :param build_bundle: ``worker_id → bundle`` factory.
    :param run_slice: ``(slice, bundle) → None`` per-worker loop.
    :param thread_name_prefix: Thread name prefix for debuggers.
    """
    n = max(1, int(workers))
    total = len(indexed_seeds)

    if n == 1:
        logger.info("Worker 0/1: %d seed(s) assigned", total)
        run_slice(list(indexed_seeds), build_bundle(0))
        return

    bundles = [build_bundle(i) for i in range(n)]
    with ThreadPoolExecutor(
        max_workers=n, thread_name_prefix=thread_name_prefix,
    ) as ex:
        futs = []
        for i in range(n):
            slice_i = list(indexed_seeds[i::n])
            if not slice_i:
                continue
            logger.info(
                "Worker %d/%d: %d/%d seed(s) assigned",
                i, n, len(slice_i), total,
            )
            futs.append(ex.submit(run_slice, slice_i, bundles[i]))
        for f in futs:
            f.result()


__all__ = ["dispatch_workers"]
