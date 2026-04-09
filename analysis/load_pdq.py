"""Load and normalise PDQ run data into a common schema.

Each PDQ run directory contains per-seed folders with:
- archive.parquet: VV-validated boundary points
- candidates.parquet: all Stage 1 candidates
- stage1_flips.parquet: discovered flips
- stage2_trajectories.parquet: minimisation steps
- stats.json: seed metadata

Multiple runs (timestamps) may share the same seed index.
This loader selects only the LATEST run per seed index.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _latest_seeds(run_dir: Path) -> list[Path]:
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


def _safe_read_parquet(path: Path) -> pd.DataFrame | None:
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
        "archive": _safe_read_parquet(seed_dir / "archive.parquet"),
        "candidates": _safe_read_parquet(seed_dir / "candidates.parquet"),
        "stage1_flips": _safe_read_parquet(seed_dir / "stage1_flips.parquet"),
        "stage2_traj": _safe_read_parquet(seed_dir / "stage2_trajectories.parquet"),
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
    seed_dirs = _latest_seeds(run_dir)

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
