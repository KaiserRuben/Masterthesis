"""
Parquet Schema for scenes.parquet

Defines the schema and provides load/save utilities with correct dtypes.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# Classification keys (top 6 from EMB-001)
CLASSIFICATION_KEYS = [
    "weather",
    "time_of_day",
    "depth_complexity",
    "occlusion_level",
    "road_type",
    "required_action",
]

# Column definitions with dtypes
COLUMNS = {
    # Step 0: Sampling
    "clip_id": "string",
    "is_anchor": "boolean",
    "sample_seed": "Int64",  # nullable int

    # Step 1: Embedding
    "emb_index": "Int64",  # nullable int, index into embeddings.npz
    "has_embedding": "boolean",

    # Step 2: Classification (one column per key)
    "weather": "string",
    "time_of_day": "string",
    "depth_complexity": "string",
    "occlusion_level": "string",
    "road_type": "string",
    "required_action": "string",
    "label_source": "string",  # "vlm" or "propagated"
    "label_confidence": "Float64",  # nullable float

    # Step 3: Inference
    "ade": "Float64",  # nullable float, Average Displacement Error
    "coc_reasoning": "string",  # Chain-of-Cluster reasoning text
    "has_ade": "boolean",
    "inference_timestamp": "string",  # ISO timestamp

    # Trajectory classes (computed from raw trajectory)
    "traj_direction": "string",  # turn_left, turn_right, straight, slight_curve
    "traj_speed": "string",  # accelerate, decelerate, constant
    "traj_lateral": "string",  # lane_change_left, lane_change_right, lane_keep
}


def _get_metadata(df: pd.DataFrame) -> dict[str, Any]:
    """Extract pipeline metadata from DataFrame."""
    if hasattr(df, "attrs") and "pipeline_metadata" in df.attrs:
        return df.attrs["pipeline_metadata"]
    return {}


def _set_metadata(df: pd.DataFrame, metadata: dict[str, Any]) -> None:
    """Set pipeline metadata on DataFrame."""
    df.attrs["pipeline_metadata"] = metadata


def load_scenes(path: Path | str) -> pd.DataFrame:
    """
    Load scenes.parquet with correct dtypes.

    Returns empty DataFrame with correct schema if file doesn't exist.
    """
    path = Path(path)

    if not path.exists():
        # Return empty DataFrame with correct schema
        df = pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in COLUMNS.items()})
        return df

    # Read parquet with PyArrow
    table = pq.read_table(path)
    df = table.to_pandas()

    # Read custom metadata
    if table.schema.metadata and b"pipeline" in table.schema.metadata:
        metadata = json.loads(table.schema.metadata[b"pipeline"])
        _set_metadata(df, metadata)

    # Ensure correct dtypes for existing columns
    for col, dtype in COLUMNS.items():
        if col in df.columns:
            if dtype == "boolean":
                df[col] = df[col].astype("boolean")
            elif dtype == "Int64":
                df[col] = df[col].astype("Int64")
            elif dtype == "Float64":
                df[col] = df[col].astype("Float64")
            elif dtype == "string":
                df[col] = df[col].astype("string")

    return df


def save_scenes(df: pd.DataFrame, path: Path | str) -> None:
    """
    Save DataFrame to scenes.parquet with correct schema.

    Preserves pipeline metadata in parquet file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure correct dtypes
    for col, dtype in COLUMNS.items():
        if col in df.columns:
            if dtype == "boolean":
                df[col] = df[col].astype("boolean")
            elif dtype == "Int64":
                df[col] = df[col].astype("Int64")
            elif dtype == "Float64":
                df[col] = df[col].astype("Float64")
            elif dtype == "string":
                df[col] = df[col].astype("string")

    # Convert to PyArrow table
    table = pa.Table.from_pandas(df)

    # Add custom metadata
    metadata = _get_metadata(df)
    if metadata:
        existing_metadata = table.schema.metadata or {}
        new_metadata = {**existing_metadata, b"pipeline": json.dumps(metadata).encode()}
        table = table.replace_schema_metadata(new_metadata)

    # Write parquet
    pq.write_table(table, path)


def update_scenes(updates: pd.DataFrame, path: Path | str) -> pd.DataFrame:
    """
    Merge updates into existing scenes.parquet by clip_id.

    Args:
        updates: DataFrame with clip_id and columns to update
        path: Path to scenes.parquet

    Returns:
        Updated DataFrame
    """
    path = Path(path)
    df = load_scenes(path)

    if df.empty:
        save_scenes(updates, path)
        return updates

    # Set index for efficient merge
    df = df.set_index("clip_id")
    updates = updates.set_index("clip_id")

    # Update existing rows
    df.update(updates)

    # Add new rows
    new_ids = updates.index.difference(df.index)
    if len(new_ids) > 0:
        df = pd.concat([df, updates.loc[new_ids]])

    # Reset index
    df = df.reset_index()

    # Preserve metadata
    existing_df = load_scenes(path)
    if hasattr(existing_df, "attrs"):
        df.attrs = existing_df.attrs

    save_scenes(df, path)
    return df


def get_sample_metadata(path: Path | str) -> tuple[int | None, int | None]:
    """
    Get sampling metadata (n, seed) from existing scenes.parquet.

    Returns (None, None) if file doesn't exist or has no metadata.
    """
    path = Path(path)

    if not path.exists():
        return None, None

    df = load_scenes(path)
    metadata = _get_metadata(df)

    return metadata.get("n"), metadata.get("seed")


def set_sample_metadata(df: pd.DataFrame, n: int, seed: int) -> None:
    """Set sampling metadata on DataFrame."""
    metadata = _get_metadata(df)
    metadata["n"] = n
    metadata["seed"] = seed
    _set_metadata(df, metadata)
