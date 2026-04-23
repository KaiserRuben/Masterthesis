"""Load and normalise PDQ run data into a common schema.

Each PDQ run directory contains per-seed folders with:
- archive.parquet: VV-validated boundary points
- candidates.parquet: all Stage 1 candidates
- stage1_flips.parquet: discovered flips
- stage2_trajectories.parquet: minimisation steps
- sut_calls.parquet: every SUT invocation (with N-dim logprobs)
- stats.json: seed metadata
- config.json: full experiment config

Multiple runs (timestamps) may share the same seed index.
This loader selects only the LATEST run per seed index.

Schema versions
---------------
PDQ always stored N-dim logprobs in its parquet files, so the v1→v2
bump is a marker-only change:

- **v1** (pre-refactor): no ``schema_version`` marker anywhere; the
  full category list lives in ``config.json['categories']``.
- **v2** (trace-complete): parquet files stamped with
  ``schema_version=2`` in file-level metadata; ``config.json`` and
  ``stats.json`` both carry ``schema_version``, and ``stats.json``
  now also includes a ``categories`` field for parity with the SMOO
  side.

:func:`load_archive` and :func:`load_sut_calls` transparently handle
both versions and always return the canonical categories alongside
the DataFrame so a single file is enough to decode the N-dim
``logprobs_*`` columns.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Schema-version detection + unified loaders
# ---------------------------------------------------------------------------

from analysis.core.parquet_utils import read_parquet_metadata as _read_parquet_metadata


def _load_categories(
    parquet_path: Path,
    stats: dict[str, Any] | None,
    config: dict[str, Any] | None,
) -> list[str]:
    """Recover the category list for a PDQ parquet.

    Lookup order:
    1. parquet file metadata key ``categories`` (v2)
    2. ``stats.json['categories']`` (v2)
    3. ``config.json['categories']`` (both v1 and v2)
    """
    meta = _read_parquet_metadata(parquet_path)
    raw = meta.get("categories")
    if raw:
        try:
            return list(json.loads(raw))
        except Exception:  # noqa: BLE001
            pass
    if stats and stats.get("categories"):
        return list(stats["categories"])
    if config and config.get("categories"):
        return list(config["categories"])
    return []


def detect_schema_version(parquet_path: Path) -> int:
    """Detect the on-disk schema version of a PDQ parquet.

    v1 had no marker; v2 stamps ``schema_version`` in file-level
    metadata. Absence → assume v1.
    """
    raw = _read_parquet_metadata(parquet_path).get("schema_version")
    if raw is None:
        return 1
    try:
        return int(raw)
    except ValueError:
        return 1


def load_archive(
    archive_path: Path | str,
    stats_path: Path | str | None = None,
    config_path: Path | str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a PDQ ``archive.parquet`` of either schema version.

    The DataFrame is returned as-is — no column normalisation. The
    metadata dict always includes ``schema_version``, ``categories``
    (index→class mapping for the N-dim ``logprobs_*`` columns), and
    the two source-file paths so the caller can re-read anything
    extra.

    :param archive_path: Path to ``archive.parquet``.
    :param stats_path: Optional; defaults to ``<parent>/stats.json``.
    :param config_path: Optional; defaults to ``<parent>/config.json``.
    :returns: ``(archive_df, meta)``.
    """
    archive_path = Path(archive_path)
    parent = archive_path.parent
    if stats_path is None:
        stats_path = parent / "stats.json"
    else:
        stats_path = Path(stats_path)
    if config_path is None:
        config_path = parent / "config.json"
    else:
        config_path = Path(config_path)

    df = pd.read_parquet(archive_path)

    stats: dict[str, Any] = {}
    if stats_path.exists():
        try:
            with open(stats_path) as f:
                stats = json.load(f)
        except Exception:  # noqa: BLE001
            stats = {}
    config: dict[str, Any] = {}
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
        except Exception:  # noqa: BLE001
            config = {}

    # Version resolution: stats.json > config.json > parquet metadata.
    version: int
    if "schema_version" in stats:
        version = int(stats["schema_version"])
    elif "schema_version" in config:
        version = int(config["schema_version"])
    else:
        version = detect_schema_version(archive_path)

    categories = _load_categories(archive_path, stats, config)

    meta: dict[str, Any] = {
        "schema_version": version,
        "categories": categories,
        "archive_path": archive_path,
        "stats_path": stats_path,
        "config_path": config_path,
    }
    return df, meta


def load_sut_calls(
    sut_calls_path: Path | str,
    stats_path: Path | str | None = None,
    config_path: Path | str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a PDQ ``sut_calls.parquet`` of either schema version.

    Parallel of :func:`load_archive`; returns the raw DataFrame plus
    the canonical category list recovered from the first available
    source.
    """
    sut_calls_path = Path(sut_calls_path)
    parent = sut_calls_path.parent
    if stats_path is None:
        stats_path = parent / "stats.json"
    else:
        stats_path = Path(stats_path)
    if config_path is None:
        config_path = parent / "config.json"
    else:
        config_path = Path(config_path)

    df = pd.read_parquet(sut_calls_path)
    stats: dict[str, Any] = {}
    if stats_path.exists():
        try:
            with open(stats_path) as f:
                stats = json.load(f)
        except Exception:  # noqa: BLE001
            stats = {}
    config: dict[str, Any] = {}
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
        except Exception:  # noqa: BLE001
            config = {}

    version: int
    if "schema_version" in stats:
        version = int(stats["schema_version"])
    elif "schema_version" in config:
        version = int(config["schema_version"])
    else:
        version = detect_schema_version(sut_calls_path)

    categories = _load_categories(sut_calls_path, stats, config)

    meta: dict[str, Any] = {
        "schema_version": version,
        "categories": categories,
        "sut_calls_path": sut_calls_path,
        "stats_path": stats_path,
        "config_path": config_path,
    }
    return df, meta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def latest_seeds(run_dir: Path) -> list[Path]:
    """Select only the latest timestamp per seed index.

    Seed dirs follow: seed_NNNN_TIMESTAMP. If multiple timestamps
    exist for the same seed index, keep only the highest timestamp.
    """
    pat = re.compile(r"^seed_(\d{4})_(\d+)$")
    by_idx: dict[int, tuple[int, Path]] = {}

    for d in run_dir.iterdir():
        if not d.is_dir():
            continue
        m = pat.match(d.name)
        if not m:
            continue
        idx, ts = int(m.group(1)), int(m.group(2))
        if idx not in by_idx or ts > by_idx[idx][0]:
            by_idx[idx] = (ts, d)

    return [path for _, path in sorted(by_idx.values(), key=lambda x: x[0])]


def safe_read_parquet(path: Path) -> pd.DataFrame | None:
    """Read parquet, return None on any error (e.g. corrupted footer)."""
    try:
        return pd.read_parquet(path)
    except Exception as e:
        print(f"  WARN: cannot read {path.name}: {e}")
        return None


# ---------------------------------------------------------------------------
# Seed-level loading
# ---------------------------------------------------------------------------

def load_seed(seed_dir: Path) -> dict | None:
    """Load one PDQ seed directory.  Returns None if stats.json is missing."""
    stats_path = seed_dir / "stats.json"
    if not stats_path.exists():
        return None
    try:
        with open(stats_path) as f:
            stats = json.load(f)
        if not stats:
            return None
    except (json.JSONDecodeError, ValueError):
        return None

    return {
        "stats": stats,
        "archive": safe_read_parquet(seed_dir / "archive.parquet"),
        "candidates": safe_read_parquet(seed_dir / "candidates.parquet"),
        "stage1_flips": safe_read_parquet(seed_dir / "stage1_flips.parquet"),
        "stage2_traj": safe_read_parquet(seed_dir / "stage2_trajectories.parquet"),
        "seed_dir": seed_dir,
    }


# ---------------------------------------------------------------------------
# Run-level aggregation
# ---------------------------------------------------------------------------

def load_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all latest seeds in a PDQ run directory.

    Returns:
        stats_df: one row per seed
        archive_df: all archive rows (boundary points)
        candidates_df: all Stage 1 candidates
        stage2_df: all Stage 2 trajectory steps
    """
    run_dir = Path(run_dir)
    seed_dirs = latest_seeds(run_dir)

    stats_rows = []
    archive_frames = []
    candidate_frames = []
    stage2_frames = []

    for sd in seed_dirs:
        data = load_seed(sd)
        if data is None:
            print(f"  SKIP {sd.name}: no valid stats.json")
            continue

        s = data["stats"]
        seed_idx = s["seed_idx"]
        run_name = run_dir.name

        stats_rows.append({
            "run": run_name,
            "seed_idx": seed_idx,
            "class_a": s["class_a"],
            "class_b": s["class_b"],
            "label_anchor": s["label_anchor"],
            "n_stage1_candidates": s["n_stage1_candidates"],
            "n_stage1_flips": s["n_stage1_flips"],
            "n_distinct_targets": s["n_distinct_targets"],
            "n_stage2_flips": s["n_stage2_flips"],
            "n_stage2_sut_calls": s["n_stage2_sut_calls"],
            "wall_time_s": s["wall_time_s"],
            "genotype_dim": s["genotype_dim"],
            "n_img_genes": s["n_img_genes"],
            "n_txt_genes": s["n_txt_genes"],
            "seed_dir": sd.name,
        })

        if data["archive"] is not None and not data["archive"].empty:
            ar = data["archive"].copy()
            ar["run"] = run_name
            ar["seed_idx"] = seed_idx
            ar["class_a"] = s["class_a"]
            ar["class_b"] = s["class_b"]
            archive_frames.append(ar)

        if data["candidates"] is not None and not data["candidates"].empty:
            cd = data["candidates"].copy()
            cd["run"] = run_name
            cd["seed_idx"] = seed_idx
            cd["class_a"] = s["class_a"]
            cd["label_anchor"] = s["label_anchor"]
            candidate_frames.append(cd)

        if data["stage2_traj"] is not None and not data["stage2_traj"].empty:
            st = data["stage2_traj"].copy()
            st["run"] = run_name
            st["seed_idx"] = seed_idx
            stage2_frames.append(st)

    stats_df = pd.DataFrame(stats_rows)
    archive_df = pd.concat(archive_frames, ignore_index=True) if archive_frames else pd.DataFrame()
    candidates_df = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    stage2_df = pd.concat(stage2_frames, ignore_index=True) if stage2_frames else pd.DataFrame()

    return stats_df, archive_df, candidates_df, stage2_df
