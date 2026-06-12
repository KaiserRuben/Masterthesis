"""Tests for src.common.resume — the shared runner resume support.

Covers:
- compute_resume_filter: skip finished, re-run partial/unreadable, dedup,
  intersect with an existing filter, empty-when-all-done, partial cleanup
- drift sanity check: metadata mismatch / out-of-range idx abort
- default_seed_probe: the standard <name>_seed_<idx>_<ts>/stats.json layout
- the engine is layout-agnostic via a custom SeedDirProbe
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.common.resume import (
    SeedDirProbe,
    compute_resume_filter,
    default_seed_probe,
)

NAME = "exp_demo"


class _Seed:
    """Minimal stand-in for SeedTriple — resume only reads ``.metadata``."""

    def __init__(self, **metadata: object) -> None:
        self.metadata = dict(metadata)


def _meta(idx_in_class: int = 0, anchor: str = "junco") -> dict[str, object]:
    """A seed_metadata block covering every DEFAULT_SANITY_FIELDS key."""
    return {
        "anchor_class_concrete": anchor,
        "target_class_concrete": "boa constrictor",
        "level_anchor": 0,
        "level_target": 1,
        "seed_idx_in_class": idx_in_class,
    }


def _pool(n: int) -> list[_Seed]:
    return [_Seed(**_meta(i)) for i in range(n)]


def _finished(root: Path, idx: int, ts: int, metadata: dict | None) -> Path:
    """Standard-layout finished seed dir (stats.json present)."""
    seed_dir = root / f"{NAME}_seed_{idx}_{ts}"
    seed_dir.mkdir(parents=True)
    payload = {} if metadata is None else {"seed_metadata": metadata}
    (seed_dir / "stats.json").write_text(json.dumps(payload))
    return seed_dir


def _partial(root: Path, idx: int, ts: int) -> Path:
    """Interrupted seed dir — no stats.json."""
    seed_dir = root / f"{NAME}_seed_{idx}_{ts}"
    seed_dir.mkdir(parents=True)
    (seed_dir / "convergence.parquet").write_bytes(b"partial")
    return seed_dir


def _resume(tmp_path, seeds, existing=(), *, clean_partials=False):
    return compute_resume_filter(
        default_seed_probe(NAME),
        tmp_path,
        NAME,
        seeds,
        existing,
        clean_partials=clean_partials,
    )


# --- compute_resume_filter, standard layout --------------------------------


def test_skips_finished_runs_partial_and_unstarted(tmp_path):
    seeds = _pool(5)
    _finished(tmp_path, 0, 100, seeds[0].metadata)
    _finished(tmp_path, 1, 200, seeds[1].metadata)
    _partial(tmp_path, 2, 300)  # interrupted → must re-run
    # 3 and 4 never started
    assert _resume(tmp_path, seeds) == (2, 3, 4)


def test_no_run_dir_runs_all(tmp_path):
    assert _resume(tmp_path, _pool(3)) == (0, 1, 2)


def test_intersects_existing_filter(tmp_path):
    seeds = _pool(5)
    _finished(tmp_path, 0, 100, seeds[0].metadata)
    _finished(tmp_path, 1, 200, seeds[1].metadata)
    # remaining {2,3,4} ∩ requested {1,2,3} = {2,3}
    assert _resume(tmp_path, seeds, existing=(1, 2, 3)) == (2, 3)


def test_all_finished_returns_empty(tmp_path):
    seeds = _pool(3)
    for i in range(3):
        _finished(tmp_path, i, 100 + i, seeds[i].metadata)
    assert _resume(tmp_path, seeds) == ()


def test_duplicate_finished_idx_skipped_once(tmp_path):
    seeds = _pool(3)
    _finished(tmp_path, 0, 100, seeds[0].metadata)
    _finished(tmp_path, 0, 200, seeds[0].metadata)  # re-resume duplicate
    assert _resume(tmp_path, seeds) == (1, 2)


def test_unreadable_metadata_reruns(tmp_path):
    seeds = _pool(3)
    _finished(tmp_path, 0, 100, None)  # stats.json without seed_metadata
    assert 0 in _resume(tmp_path, seeds)


def test_corrupt_stats_json_reruns(tmp_path):
    seeds = _pool(3)
    seed_dir = tmp_path / f"{NAME}_seed_0_100"
    seed_dir.mkdir()
    (seed_dir / "stats.json").write_text("{not valid json")
    assert 0 in _resume(tmp_path, seeds)


# --- partial cleanup -------------------------------------------------------


def test_clean_partials_removes_dirs(tmp_path):
    seeds = _pool(3)
    partial = _partial(tmp_path, 2, 300)
    remaining = _resume(tmp_path, seeds, clean_partials=True)
    assert not partial.exists()
    assert 2 in remaining


def test_partials_kept_without_flag(tmp_path):
    seeds = _pool(3)
    partial = _partial(tmp_path, 2, 300)
    _resume(tmp_path, seeds, clean_partials=False)
    assert partial.exists()


# --- drift sanity check ----------------------------------------------------


def test_drift_metadata_mismatch_aborts(tmp_path):
    seeds = _pool(3)  # seeds[0] anchor = "junco"
    _finished(tmp_path, 0, 100, _meta(0, anchor="ostrich"))  # diverged
    with pytest.raises(RuntimeError, match="Resume drift at seed_idx=0"):
        _resume(tmp_path, seeds)


def test_drift_idx_out_of_range_aborts(tmp_path):
    seeds = _pool(3)
    _finished(tmp_path, 9, 100, _meta(9))  # pool only has 0..2
    with pytest.raises(RuntimeError, match="exceeds the regenerated pool size"):
        _resume(tmp_path, seeds)


# --- engine is layout-agnostic ---------------------------------------------


def test_engine_supports_custom_layout(tmp_path):
    for idx in (0, 2):
        seed_dir = tmp_path / f"run{idx}"
        seed_dir.mkdir()
        (seed_dir / "DONE").write_text("")
        (seed_dir / "meta.json").write_text(json.dumps(_meta(idx)))
    probe = SeedDirProbe(
        iter_seed_dirs=lambda save_dir, name: save_dir.glob("run*"),
        is_finished=lambda seed_dir: (seed_dir / "DONE").exists(),
        read_seed_idx=lambda seed_dir: int(seed_dir.name.removeprefix("run")),
        read_metadata=lambda seed_dir: json.loads(
            (seed_dir / "meta.json").read_text()
        ),
    )
    remaining = compute_resume_filter(probe, tmp_path, "ignored", _pool(4), ())
    assert remaining == (1, 3)
