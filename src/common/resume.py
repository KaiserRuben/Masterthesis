"""Shared resume support for the boundary-test runners.

An interrupted run leaves finished seed dirs on disk. Both the evolutionary
runner (``run_boundary_test.py``) and the boundary-pair runner
(``run_boundary_pair_test.py``) can skip those and re-run only the unfinished
complement. The two pipelines write different layouts and completion markers,
so *discovering* the seed dirs is injected by the caller via
:class:`SeedDirProbe`; everything downstream — partial cleanup, index-drift
sanity-checking against the regenerated seed pool, and narrowing
``filter_indices`` to the unfinished complement — is shared here.

On-disk contract per pipeline:

==============  =====================================  =================
pipeline        seed dir                               completion marker
==============  =====================================  =================
evolutionary    ``<save_dir>/<name>_seed_<idx>_<ts>/``  ``stats.json``
boundary-pair   ``<save_dir>/<name>/seed_<idx>_<ts>/``  ``manifest.json``
==============  =====================================  =================

The ``seed_metadata`` block used for drift detection lives in ``stats.json``
for both (boundary-pair reads the sibling ``evolutionary/stats.json``).
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.config import SeedTriple

logger = logging.getLogger(__name__)

# Concrete ``seed_metadata`` fields compared between a finished seed dir and
# the regenerated pool to catch silent seed-index drift: the config changed
# since the original run, so index N now maps to a different seed.
DEFAULT_SANITY_FIELDS: tuple[str, ...] = (
    "anchor_class_concrete",
    "target_class_concrete",
    "level_anchor",
    "level_target",
    "seed_idx_in_class",
)

# Read failures that mean "this finished dir is unusable — re-run the seed"
# rather than "abort the whole resume": missing/locked files (OSError), bad
# JSON (ValueError, which JSONDecodeError subclasses), absent keys (KeyError),
# or a non-int seed_idx (TypeError). Anything else propagates.
_READ_ERRORS = (OSError, ValueError, KeyError, TypeError)


@dataclass(frozen=True)
class SeedDirProbe:
    """Layout adapter telling the scanner how to read one pipeline's dirs.

    :param iter_seed_dirs: Yields candidate seed dirs for ``(save_dir, name)``.
        Non-directories among the results are ignored by the scanner.
    :param is_finished: True iff the dir holds its completion marker.
    :param read_seed_idx: Persisted 0-based seed index for a finished dir.
    :param read_metadata: The dir's ``seed_metadata`` mapping, or ``None`` if
        the marker exists but metadata is absent.

    ``read_seed_idx`` / ``read_metadata`` may raise on a corrupt dir; the
    scanner catches it and re-runs that seed (see :data:`_READ_ERRORS`).
    """

    iter_seed_dirs: Callable[[Path, str], Iterable[Path]]
    is_finished: Callable[[Path], bool]
    read_seed_idx: Callable[[Path], int]
    read_metadata: Callable[[Path], dict[str, Any] | None]


def default_seed_probe(name: str) -> SeedDirProbe:
    """Probe for the standard layout written by ``VLMBoundaryTester``.

    Seed dirs are ``<save_dir>/<name>_seed_<idx>_<ts>/`` with completion
    marker ``stats.json`` (carrying the ``seed_metadata`` block). This is the
    layout the evolutionary runner produces directly; the boundary-pair runner
    nests it under a per-seed wrapper and supplies its own probe instead.
    """
    # idx is parsed from the dir name; the trailing group is the launch ts.
    pattern = re.compile(rf"^{re.escape(name)}_seed_(\d+)_\d+$")

    def read_seed_idx(seed_dir: Path) -> int:
        match = pattern.match(seed_dir.name)
        if match is None:
            raise ValueError(f"unexpected seed dir name: {seed_dir.name!r}")
        return int(match.group(1))

    def read_metadata(seed_dir: Path) -> dict[str, Any] | None:
        return json.loads((seed_dir / "stats.json").read_text()).get("seed_metadata")

    return SeedDirProbe(
        iter_seed_dirs=lambda save_dir, run_name: save_dir.glob(f"{run_name}_seed_*"),
        is_finished=lambda seed_dir: (seed_dir / "stats.json").exists(),
        read_seed_idx=read_seed_idx,
        read_metadata=read_metadata,
    )


@dataclass
class ScanResult:
    """Seed dirs found on disk, bucketed by completion state."""

    finished: dict[int, dict[str, Any]] = field(default_factory=dict)
    partial_dirs: list[Path] = field(default_factory=list)
    unreadable: list[tuple[Path, str]] = field(default_factory=list)
    duplicate_idxs: list[int] = field(default_factory=list)


def _scan(probe: SeedDirProbe, save_dir: Path, name: str) -> ScanResult:
    """Walk the run's seed dirs and classify each by completion state."""
    result = ScanResult()
    for seed_dir in sorted(probe.iter_seed_dirs(save_dir, name)):
        if not seed_dir.is_dir():
            continue
        if not probe.is_finished(seed_dir):
            result.partial_dirs.append(seed_dir)
            continue
        try:
            idx = probe.read_seed_idx(seed_dir)
            metadata = probe.read_metadata(seed_dir)
        except _READ_ERRORS as exc:
            result.unreadable.append((seed_dir, f"parse error: {exc}"))
            continue
        if metadata is None:
            result.unreadable.append((seed_dir, "missing seed_metadata"))
            continue
        # Multiple finished dirs for one idx (e.g. a re-resume): keep the
        # first; they should be equivalent under the drift check below.
        if idx in result.finished:
            result.duplicate_idxs.append(idx)
            continue
        result.finished[idx] = metadata
    return result


def _log_scan(scan: ScanResult, *, clean_partials: bool) -> None:
    """Report (and optionally remove) partial / duplicate / unreadable dirs."""
    if scan.partial_dirs:
        logger.info(
            "Resume: found %d partial seed dir(s) (no completion marker — "
            "interrupted mid-run).",
            len(scan.partial_dirs),
        )
        if clean_partials:
            for partial in scan.partial_dirs:
                shutil.rmtree(partial)
            logger.info(
                "Resume: removed %d partial seed dir(s).", len(scan.partial_dirs)
            )
        else:
            for partial in scan.partial_dirs[:5]:
                logger.info("  partial: %s", partial.name)
            if len(scan.partial_dirs) > 5:
                logger.info("  ... +%d more", len(scan.partial_dirs) - 5)
            logger.info(
                "Resume: pass --clean-partials to remove them; otherwise they "
                "stay on disk (harmless — analysis keys off the completion marker)."
            )

    if scan.duplicate_idxs:
        unique = sorted(set(scan.duplicate_idxs))
        sample = ", ".join(str(i) for i in unique[:5])
        more = max(0, len(unique) - 5)
        logger.warning(
            "Resume: %d duplicate finished dir(s) across %d seed_idx value(s) "
            "(sample: %s%s) — kept first encountered. Consider archiving the rest.",
            len(scan.duplicate_idxs),
            len(unique),
            sample,
            f", +{more} more" if more else "",
        )

    if scan.unreadable:
        logger.warning(
            "Resume: %d finished dir(s) had a marker but unreadable metadata; "
            "treating as not-done and re-running.",
            len(scan.unreadable),
        )
        for bad_dir, reason in scan.unreadable[:5]:
            logger.warning("  unreadable: %s — %s", bad_dir.name, reason)
        if len(scan.unreadable) > 5:
            logger.warning("  ... +%d more", len(scan.unreadable) - 5)


def _check_drift(
    finished: dict[int, dict[str, Any]],
    seeds: Sequence["SeedTriple"],
    sanity_fields: tuple[str, ...],
) -> None:
    """Abort if any finished idx no longer maps to the same seed it ran on."""
    for idx in sorted(finished):
        if idx >= len(seeds):
            raise RuntimeError(
                f"Resume drift: persisted seed_idx={idx} exceeds the "
                f"regenerated pool size ({len(seeds)}). The config (categories "
                f"/ seeds_per_class / abstraction / seed_int) has shrunk the "
                f"pool since the original run."
            )
        live = seeds[idx].metadata or {}
        persisted = finished[idx]
        for field_name in sanity_fields:
            if live.get(field_name) != persisted.get(field_name):
                raise RuntimeError(
                    f"Resume drift at seed_idx={idx}: persisted "
                    f"{field_name}={persisted.get(field_name)!r}, regenerated "
                    f"{live.get(field_name)!r}. The config (categories / "
                    f"seeds_per_class / abstraction / seed_int) changed since "
                    f"the original run; aborting to prevent silent index "
                    f"drift.\npersisted metadata: {persisted}\n"
                    f"regenerated metadata: {live}"
                )


def compute_resume_filter(
    probe: SeedDirProbe,
    save_dir: Path,
    name: str,
    seeds: Sequence["SeedTriple"],
    existing_filter_indices: tuple[int, ...],
    *,
    clean_partials: bool = False,
    sanity_fields: tuple[str, ...] = DEFAULT_SANITY_FIELDS,
) -> tuple[int, ...]:
    """Return the unfinished seed indices to run, narrowing any existing filter.

    Scans ``save_dir`` via *probe*, logs (and with *clean_partials* removes)
    partial dirs, sanity-checks each finished seed against the regenerated
    *seeds* pool (drift raises :class:`RuntimeError`), and returns
    ``range(len(seeds)) − finished`` intersected with
    *existing_filter_indices* (an empty tuple means no pre-existing filter).

    A returned empty tuple means every requested seed is already complete.
    Callers MUST treat that as "nothing to run": handing it to
    :func:`apply_seed_filter` would be read as "keep all" and re-run the
    finished seeds.
    """
    scan = _scan(probe, Path(save_dir), name)
    _log_scan(scan, clean_partials=clean_partials)
    _check_drift(scan.finished, seeds, sanity_fields)

    remaining = [i for i in range(len(seeds)) if i not in scan.finished]
    if existing_filter_indices:
        requested = set(existing_filter_indices)
        remaining = [i for i in remaining if i in requested]
    logger.info(
        "Resume: %d/%d seed(s) already complete (sanity-checked); %d to run.",
        len(scan.finished),
        len(seeds),
        len(remaining),
    )
    return tuple(remaining)
