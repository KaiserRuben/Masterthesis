"""
I/O Utilities

Helpers for loading config, embeddings, and incremental operations.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """
    Load pipeline configuration from YAML file.

    Args:
        path: Path to config file. Defaults to pipeline/config.yaml

    Returns:
        Configuration dict
    """
    if path is None:
        # Default: pipeline/config.yaml (lib/io.py -> parents[1] is pipeline/)
        path = Path(__file__).parents[1] / "config.yaml"
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        return yaml.safe_load(f)


def resolve_path(rel_path: str, base_dir: Path | None = None) -> Path:
    """
    Resolve a relative path from config to absolute path.

    Args:
        rel_path: Relative path string (e.g., "data/pipeline/scenes.parquet")
        base_dir: Base directory. Defaults to repo root.

    Returns:
        Absolute Path
    """
    if base_dir is None:
        base_dir = Path(__file__).parents[2]  # repo root (pipeline/lib/io.py -> Masterarbeit/)
    return base_dir / rel_path


def load_embeddings(path: Path | str) -> np.ndarray:
    """
    Load embeddings from NPZ file.

    Args:
        path: Path to embeddings.npz

    Returns:
        Embeddings array of shape (N, dim)
    """
    path = Path(path)

    if not path.exists():
        return np.array([])

    data = np.load(path)
    return data["embeddings"]


def save_embeddings(embeddings: np.ndarray, path: Path | str) -> None:
    """
    Save embeddings to NPZ file.

    Args:
        embeddings: Embeddings array of shape (N, dim)
        path: Path to save to
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, embeddings=embeddings)


def append_embeddings(path: Path | str, new_embeddings: np.ndarray) -> int:
    """
    Append new embeddings to existing NPZ file.

    Args:
        path: Path to embeddings.npz
        new_embeddings: New embeddings to append, shape (M, dim)

    Returns:
        Start index of the newly appended embeddings
    """
    path = Path(path)

    existing = load_embeddings(path)

    if len(existing) == 0:
        # No existing embeddings
        save_embeddings(new_embeddings, path)
        return 0

    # Verify dimensions match
    if existing.shape[1] != new_embeddings.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: existing {existing.shape[1]}, "
            f"new {new_embeddings.shape[1]}"
        )

    # Append
    start_index = len(existing)
    combined = np.vstack([existing, new_embeddings])
    save_embeddings(combined, path)

    return start_index


def merge_inference_results(
    scenes_df: pd.DataFrame,
    results: list[dict],
) -> pd.DataFrame:
    """
    Merge inference results into scenes DataFrame.

    Args:
        scenes_df: Existing scenes DataFrame
        results: List of inference result dicts, each with:
            - clip_id
            - ade (or min_ade)
            - coc_reasoning
            - inference_timestamp
            - traj_direction, traj_speed, traj_lateral (optional)

    Returns:
        Updated DataFrame
    """
    if not results:
        return scenes_df

    # Build update DataFrame
    updates = []
    for r in results:
        if "error" in r:
            continue

        clip_id = r["clip_id"]
        update = {
            "clip_id": clip_id,
            "ade": r.get("ade") or r.get("min_ade"),
            "coc_reasoning": r.get("coc_reasoning", ""),
            "has_ade": True,
            "inference_timestamp": r.get("inference_timestamp", ""),
        }

        # Add trajectory classes if present
        if "traj_direction" in r:
            update["traj_direction"] = r["traj_direction"]
        if "traj_speed" in r:
            update["traj_speed"] = r["traj_speed"]
        if "traj_lateral" in r:
            update["traj_lateral"] = r["traj_lateral"]

        updates.append(update)

    if not updates:
        return scenes_df

    updates_df = pd.DataFrame(updates)

    # Merge by clip_id
    merged = scenes_df.merge(
        updates_df,
        on="clip_id",
        how="left",
        suffixes=("", "_new"),
    )

    # Update columns where new data exists
    for col in ["ade", "coc_reasoning", "has_ade", "inference_timestamp",
                "traj_direction", "traj_speed", "traj_lateral"]:
        if f"{col}_new" in merged.columns:
            merged[col] = merged[f"{col}_new"].combine_first(merged[col])
            merged = merged.drop(columns=[f"{col}_new"])

    return merged


def get_repo_root() -> Path:
    """Get the repository root directory."""
    # pipeline/lib/io.py -> parents[2] is repo root (Masterarbeit/)
    return Path(__file__).parents[2]


def get_git_hash() -> str:
    """Get current git commit hash, or 'unknown' if not available."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=get_repo_root(),
        )
        if result.returncode == 0:
            return result.stdout.strip()[:7]
    except Exception:
        pass
    return "unknown"
