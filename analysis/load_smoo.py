"""Load and normalise SMOO run data into a common schema.

Each SMOO run directory contains per-seed folders with:
- trace.parquet: every individual in every generation
- pareto_N.json: Pareto-front solutions
- stats.json: seed metadata

This loader produces two DataFrames:
- trace_df: all evaluated individuals (generation × pop_size rows)
- pareto_df: Pareto-front solutions with genotype-level metrics

Schema versions
---------------
Two on-disk layouts are supported transparently:

- **v1** (pre-refactor): ``trace.parquet`` has a ``logprobs`` column of
  length 2, no ``cache_hit`` column, and ``stats.json`` stores only the
  pair under ``categories``. Files have no file-level metadata marker.
- **v2** (trace-complete): ``trace.parquet`` has a length-N
  ``logprobs`` column (full category list scored by the SUT) and a
  ``cache_hit`` column per row; ``stats.json`` stores the full N-list
  as ``categories`` plus ``pair`` / ``target_classes`` positions.
  Parquet files are stamped with ``schema_version=2`` in their
  file-level metadata.

:func:`load_trace` returns the DataFrame plus a dict of detected
metadata (``schema_version``, ``categories``, ``pair``,
``target_classes``). A v1 trace is exposed as-is (length-2
``logprobs``) together with the categories from the adjacent
``stats.json`` so downstream code can always map indices to class
names without ``git checkout``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Schema-version detection + unified trace loader
# ---------------------------------------------------------------------------

def _read_parquet_metadata(path: Path) -> dict[str, str]:
    """Read file-level key/value metadata from a parquet file.

    Returns a plain ``{str: str}`` dict (keys/values decoded from UTF-8).
    Missing or unreadable metadata yields an empty dict — the caller
    falls back to content inspection.
    """
    try:
        raw = pq.read_schema(path).metadata or {}
    except Exception:  # noqa: BLE001 — metadata is best-effort
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        try:
            out[k.decode("utf-8")] = v.decode("utf-8")
        except Exception:  # noqa: BLE001
            pass
    return out


def detect_schema_version(trace_path: Path) -> int:
    """Detect the on-disk schema version of a SMOO ``trace.parquet``.

    Preferred path: look at file-level metadata (``schema_version``
    written by the v2 writer). Fallback: read the first row and
    inspect its ``logprobs`` length — a length-2 value is v1, anything
    else is assumed v2 (v1 only supported pairs).

    :param trace_path: Path to ``trace.parquet``.
    :returns: ``1`` or ``2``.
    """
    meta = _read_parquet_metadata(trace_path)
    raw = meta.get("schema_version")
    if raw is not None:
        try:
            return int(raw)
        except ValueError:
            pass

    # Fallback: content sniff. v1 always wrote length-2 logprobs.
    try:
        head = pd.read_parquet(trace_path).head(1)
        if "logprobs" in head.columns and len(head):
            n = len(head.iloc[0]["logprobs"])
            return 1 if n == 2 else 2
    except Exception:  # noqa: BLE001
        pass
    return 1


def load_trace(
    trace_path: Path | str,
    stats_path: Path | str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a SMOO trace parquet of either schema version.

    v1 traces have ``logprobs`` of length 2 (target pair only) and no
    ``cache_hit`` column. v2 traces have length-N logprobs plus
    ``cache_hit``. The returned metadata dict always includes
    ``schema_version``, ``categories`` (list of class names), and
    ``pair``/``target_classes`` — values are reconstructed from
    ``stats.json`` if not present in the parquet metadata (v1 path).

    :param trace_path: Path to ``trace.parquet``.
    :param stats_path: Optional path to the adjacent ``stats.json``.
        Defaults to ``<parent>/stats.json``.
    :returns: ``(trace_df, meta)``. Caller owns both; no normalisation
        is applied to ``logprobs`` — v1 stays length-2, v2 stays
        length-N. ``meta['categories']`` is the canonical index→class
        mapping for the ``logprobs`` column in either version.
    """
    trace_path = Path(trace_path)
    if stats_path is None:
        stats_path = trace_path.parent / "stats.json"
    else:
        stats_path = Path(stats_path)

    df = pd.read_parquet(trace_path)
    file_meta = _read_parquet_metadata(trace_path)

    # Load stats.json for fallback category recovery (v1 writers did
    # not stamp the parquet; v2 writers do, but we still read stats so
    # the returned meta is consistent across versions).
    stats: dict[str, Any] = {}
    if stats_path.exists():
        try:
            with open(stats_path) as f:
                stats = json.load(f)
        except Exception:  # noqa: BLE001
            stats = {}

    # Version resolution: stats > parquet metadata > content sniff.
    version: int
    if "schema_version" in stats:
        version = int(stats["schema_version"])
    elif "schema_version" in file_meta:
        try:
            version = int(file_meta["schema_version"])
        except ValueError:
            version = detect_schema_version(trace_path)
    else:
        version = detect_schema_version(trace_path)

    # Category recovery. Parquet metadata is preferred (v2); otherwise
    # fall back to stats.json. For v1 stats.json contains only the
    # pair, which is also the canonical index map for length-2
    # logprobs.
    categories: list[str] = []
    raw_cats = file_meta.get("categories")
    if raw_cats:
        try:
            categories = list(json.loads(raw_cats))
        except Exception:  # noqa: BLE001
            categories = []
    if not categories:
        categories = list(stats.get("categories") or [])

    pair: list[str] = []
    raw_pair = file_meta.get("pair")
    if raw_pair:
        try:
            pair = list(json.loads(raw_pair))
        except Exception:  # noqa: BLE001
            pair = []
    if not pair:
        pair = list(stats.get("pair") or [])
    if not pair and "class_a" in stats and "class_b" in stats:
        pair = [stats["class_a"], stats["class_b"]]

    target_classes: list[int] = []
    raw_tc = file_meta.get("target_classes")
    if raw_tc:
        try:
            target_classes = list(json.loads(raw_tc))
        except Exception:  # noqa: BLE001
            target_classes = []
    if not target_classes and stats.get("target_classes"):
        target_classes = list(stats["target_classes"])
    if not target_classes and pair and categories:
        try:
            target_classes = [categories.index(p) for p in pair]
        except ValueError:
            # v1 fallback: pair IS the full categories, so positions
            # are (0, 1).
            target_classes = [0, 1]

    meta: dict[str, Any] = {
        "schema_version": version,
        "categories": categories,
        "pair": pair,
        "target_classes": target_classes,
        "trace_path": trace_path,
        "stats_path": stats_path,
    }
    return df, meta


# ---------------------------------------------------------------------------
# Seed-level loading
# ---------------------------------------------------------------------------

def load_seed(seed_dir: Path) -> dict:
    """Load one SMOO seed directory.

    Returns dict with keys ``stats``, ``trace``, ``pareto`` and
    ``meta`` (the unified metadata from :func:`load_trace`). The
    ``trace`` value is the raw DataFrame — v1 files still expose a
    length-2 ``logprobs`` column, v2 files expose the full N-dim
    column. ``meta['categories']`` is always populated with the
    canonical index→class mapping for that trace's ``logprobs``.
    """
    with open(seed_dir / "stats.json") as f:
        stats = json.load(f)

    trace, meta = load_trace(seed_dir / "trace.parquet", seed_dir / "stats.json")

    # Load all pareto_N.json
    pareto_files = sorted(
        seed_dir.glob("pareto_*.json"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    pareto_solutions = []
    for pf in pareto_files:
        with open(pf) as f:
            sol = json.load(f)
        sol["pareto_idx"] = int(pf.stem.split("_")[1])
        pareto_solutions.append(sol)

    return {
        "stats": stats,
        "trace": trace,
        "pareto": pareto_solutions,
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# Run-level aggregation
# ---------------------------------------------------------------------------

def _genotype_metrics(geno: list[int]) -> dict:
    """Compute rank_sum and sparsity from a genotype list."""
    arr = np.array(geno, dtype=np.int64)
    return {
        "rank_sum": int(arr.sum()),
        "sparsity": int(np.count_nonzero(arr)),
        "genotype_dim": len(arr),
    }


def load_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all seeds in a SMOO run directory.

    Returns:
        stats_df: one row per seed (metadata)
        trace_df: all generation traces (tagged with seed info)
        pareto_df: all Pareto-front solutions (tagged with seed info)
    """
    run_dir = Path(run_dir)
    seed_dirs = sorted(
        [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("vlm_boundary_seed_")],
        key=lambda d: int(d.name.split("_")[3]),
    )

    stats_rows = []
    trace_frames = []
    pareto_rows = []

    for sd in seed_dirs:
        try:
            data = load_seed(sd)
        except Exception as e:
            print(f"  SKIP {sd.name}: {e}")
            continue

        s = data["stats"]
        seed_idx = s["seed_idx"]
        run_name = run_dir.name

        stats_rows.append({
            "run": run_name,
            "seed_idx": seed_idx,
            "class_a": s["class_a"],
            "class_b": s["class_b"],
            "n_pareto": s["n_pareto"],
            "generations": s["generations"],
            "pop_size": s.get("pop_size", 20),
            "runtime_s": s["runtime_seconds"],
            "image_dim": s["image_dim"],
            "text_dim": s["text_dim"],
            "genotype_dim": s["image_dim"] + s["text_dim"],
            "archive_size": s.get("archive_size", 0),
            "seed_dir": sd.name,
        })

        # Trace
        tr = data["trace"].copy()
        tr["run"] = run_name
        tr["class_a"] = s["class_a"]
        tr["class_b"] = s["class_b"]
        trace_frames.append(tr)

        # Pareto solutions
        for sol in data["pareto"]:
            gm = _genotype_metrics(sol["genotype"])
            pareto_rows.append({
                "run": run_name,
                "seed_idx": seed_idx,
                "class_a": s["class_a"],
                "class_b": s["class_b"],
                "pareto_idx": sol["pareto_idx"],
                "text": sol["text"],
                "fitness": sol["fitness"],
                "rank_sum": gm["rank_sum"],
                "sparsity": gm["sparsity"],
                "genotype_dim": gm["genotype_dim"],
            })

    stats_df = pd.DataFrame(stats_rows)
    trace_df = pd.concat(trace_frames, ignore_index=True) if trace_frames else pd.DataFrame()
    pareto_df = pd.DataFrame(pareto_rows)

    return stats_df, trace_df, pareto_df


def load_all_runs(runs_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all SMOO runs (directories not starting with 'pdq')."""
    runs_dir = Path(runs_dir)
    all_stats, all_trace, all_pareto = [], [], []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("pdq") or run_dir.name.startswith("."):
            continue
        # Check it's a SMOO run (has vlm_boundary_seed_* subdirs)
        if not any(d.name.startswith("vlm_boundary_seed_") for d in run_dir.iterdir() if d.is_dir()):
            continue

        print(f"Loading SMOO run: {run_dir.name}")
        stats_df, trace_df, pareto_df = load_run(run_dir)
        all_stats.append(stats_df)
        if not trace_df.empty:
            all_trace.append(trace_df)
        if not pareto_df.empty:
            all_pareto.append(pareto_df)

    return (
        pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame(),
        pd.concat(all_trace, ignore_index=True) if all_trace else pd.DataFrame(),
        pd.concat(all_pareto, ignore_index=True) if all_pareto else pd.DataFrame(),
    )
