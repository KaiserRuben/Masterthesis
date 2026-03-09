"""
I/O Utilities

Helpers for loading config, embeddings, and incremental operations.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .models import PipelineConfig


def load_config(path: Path | str | None = None) -> PipelineConfig:
    """
    Load pipeline configuration from YAML file.

    Args:
        path: Path to config file. Defaults to pipeline/config.yaml

    Returns:
        PipelineConfig model with validated configuration
    """
    if path is None:
        # Default: pipeline/config.yaml (lib/io.py -> parents[1] is pipeline/)
        path = Path(__file__).parents[1] / "config.yaml"
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    return PipelineConfig.model_validate(data)


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


def load_embeddings(path: Path | str, key: str = "embeddings") -> np.ndarray:
    """
    Load embeddings from NPZ file.

    Args:
        path: Path to embeddings.npz
        key: Key to load from NPZ file (default: "embeddings" for image embeddings)

    Returns:
        Embeddings array of shape (N, dim)
    """
    path = Path(path)

    if not path.exists():
        return np.array([])

    data = np.load(path)
    if key not in data:
        return np.array([])
    return data[key]


def load_text_vocabulary(path: Path | str) -> tuple[np.ndarray, dict[str, dict[str, int]]]:
    """
    Load text embedding vocabulary from NPZ file.

    Args:
        path: Path to embeddings.npz

    Returns:
        Tuple of (embeddings array, vocabulary mapping)
        - embeddings: shape (V, dim) where V is vocabulary size
        - vocab_map: {key: {value: index}} mapping
    """
    path = Path(path)

    if not path.exists():
        return np.array([]), {}

    data = np.load(path, allow_pickle=True)

    if "text_embeddings" not in data:
        return np.array([]), {}

    text_emb = data["text_embeddings"]
    vocab_map = data["text_vocab_map"].item() if "text_vocab_map" in data else {}

    return text_emb, vocab_map


def save_embeddings(
    path: Path | str,
    embeddings: np.ndarray | None = None,
    text_embeddings: np.ndarray | None = None,
    text_vocab_map: dict | None = None,
) -> None:
    """
    Save embeddings to NPZ file (supports multiple modalities).

    Args:
        path: Path to save to
        embeddings: Image embeddings array of shape (N, dim)
        text_embeddings: Text vocabulary embeddings of shape (V, dim)
        text_vocab_map: Vocabulary mapping {key: {value: index}}
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing data to preserve other arrays
    existing_data = {}
    if path.exists():
        with np.load(path, allow_pickle=True) as data:
            for key in data.files:
                existing_data[key] = data[key]

    # Update with new data
    if embeddings is not None:
        existing_data["embeddings"] = embeddings
    if text_embeddings is not None:
        existing_data["text_embeddings"] = text_embeddings
    if text_vocab_map is not None:
        existing_data["text_vocab_map"] = text_vocab_map

    np.savez_compressed(path, **existing_data)


def append_embeddings(path: Path | str, new_embeddings: np.ndarray) -> int:
    """
    Append new image embeddings to existing NPZ file.

    Args:
        path: Path to embeddings.npz
        new_embeddings: New embeddings to append, shape (M, dim)

    Returns:
        Start index of the newly appended embeddings
    """
    path = Path(path)

    existing = load_embeddings(path, key="embeddings")

    if len(existing) == 0:
        # No existing embeddings
        save_embeddings(path, embeddings=new_embeddings)
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
    save_embeddings(path, embeddings=combined)

    return start_index


def merge_inference_results(
    scenes_df: pd.DataFrame,
    results: list,
) -> pd.DataFrame:
    """
    Merge inference results into scenes DataFrame.

    Args:
        scenes_df: Existing scenes DataFrame
        results: List of InferenceResult models or dicts, each with:
            - clip_id
            - ade (or min_ade)
            - coc_reasoning
            - inference_timestamp
            - traj_direction, traj_speed, traj_lateral (optional)

    Returns:
        Updated DataFrame
    """
    from .models import InferenceResult

    if not results:
        return scenes_df

    # Helper to get attribute from model or dict
    def get_attr(obj, key, default=None):
        if isinstance(obj, InferenceResult):
            return getattr(obj, key, default)
        return obj.get(key, default)

    def has_attr(obj, key):
        if isinstance(obj, InferenceResult):
            return hasattr(obj, key)
        return key in obj

    # Build update DataFrame
    updates = []
    for r in results:
        # Skip error entries (only applicable for dicts)
        if isinstance(r, dict) and "error" in r:
            continue

        clip_id = get_attr(r, "clip_id")
        update = {
            "clip_id": clip_id,
            "ade": get_attr(r, "ade") or get_attr(r, "min_ade"),
            "coc_reasoning": get_attr(r, "coc_reasoning", ""),
            "has_ade": True,
            "inference_timestamp": get_attr(r, "inference_timestamp", ""),
        }

        # Add trajectory classes if present
        if has_attr(r, "traj_direction"):
            update["traj_direction"] = get_attr(r, "traj_direction")
        if has_attr(r, "traj_speed"):
            update["traj_speed"] = get_attr(r, "traj_speed")
        if has_attr(r, "traj_lateral"):
            update["traj_lateral"] = get_attr(r, "traj_lateral")

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
