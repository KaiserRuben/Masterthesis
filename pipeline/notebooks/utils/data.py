"""
Data loading and preprocessing for boundary analysis.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# ADE discretization
ADE_BINS = [0, 1.0, 2.5, 5.0, float("inf")]
ADE_LABELS = ["low", "medium", "high", "critical"]

# Semantic classification keys
CLASSIFICATION_KEYS = [
    "weather",
    "time_of_day",
    "depth_complexity",
    "occlusion_level",
    "road_type",
    "required_action",
]


def classify_ade(ade: float) -> str | None:
    """
    Discretize ADE into semantic classes.

    Thresholds based on trajectory prediction error interpretation:
    - low (<1.0m): Good prediction, within lane tolerance
    - medium (1.0-2.5m): Acceptable, minor deviation
    - high (2.5-5.0m): Poor, significant error
    - critical (>5.0m): Failure, unsafe prediction

    Args:
        ade: Average Displacement Error in meters

    Returns:
        ADE class label or None if input is NaN
    """
    if pd.isna(ade):
        return None
    if ade < 1.0:
        return "low"
    elif ade < 2.5:
        return "medium"
    elif ade < 5.0:
        return "high"
    return "critical"


@dataclass
class PipelineData:
    """Container for all pipeline data."""

    scenes: pd.DataFrame
    pairs: pd.DataFrame
    embeddings: np.ndarray
    umap_3d: np.ndarray
    stability: dict
    summary: dict

    @property
    def n_scenes(self) -> int:
        return len(self.scenes)

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    @property
    def n_with_ade(self) -> int:
        return self.scenes["ade"].notna().sum()

    @property
    def n_pairs_with_ade(self) -> int:
        return self.pairs["rel_delta_ade"].notna().sum()


def load_pipeline_data(
    data_dir: Path | str | None = None,
    umap_path: Path | str | None = None,
) -> PipelineData:
    """
    Load all pipeline data for visualization.

    Args:
        data_dir: Path to pipeline data directory. If None, auto-detect.
        umap_path: Path to pre-computed 3D UMAP. If None, use /tmp/emb_3d.npy.

    Returns:
        PipelineData container with all loaded data
    """
    # Auto-detect data directory
    if data_dir is None:
        # Try common locations
        candidates = [
            Path.cwd() / "data" / "pipeline",
            Path.cwd().parent / "data" / "pipeline",
            Path.cwd().parent.parent / "data" / "pipeline",
        ]
        for candidate in candidates:
            if (candidate / "scenes.parquet").exists():
                data_dir = candidate
                break
        if data_dir is None:
            raise FileNotFoundError("Could not locate pipeline data directory")

    data_dir = Path(data_dir)

    # Load core data
    scenes = pd.read_parquet(data_dir / "scenes.parquet")
    pairs = pd.read_parquet(data_dir / "results" / "pairs.parquet")
    embeddings = np.load(data_dir / "embeddings.npz")["embeddings"]

    with open(data_dir / "results" / "stability_map.json") as f:
        stability = json.load(f)

    with open(data_dir / "results" / "summary.json") as f:
        summary = json.load(f)

    # Load or compute 3D UMAP
    umap_path = Path(umap_path) if umap_path else Path("/tmp/emb_3d.npy")
    if umap_path.exists():
        umap_3d = np.load(umap_path)
    else:
        from umap import UMAP
        reducer = UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
        umap_3d = reducer.fit_transform(embeddings)
        np.save(umap_path, umap_3d)

    # Augment scenes with UMAP coordinates
    scenes["umap_x"] = umap_3d[:, 0]
    scenes["umap_y"] = umap_3d[:, 1]
    scenes["umap_z"] = umap_3d[:, 2]
    scenes["ade_class"] = scenes["ade"].apply(classify_ade)

    # Augment pairs with ADE classes
    pairs["ade_class_a"] = pairs["ade_a"].apply(classify_ade)
    pairs["ade_class_b"] = pairs["ade_b"].apply(classify_ade)
    pairs["ade_class_changed"] = (
        pairs["ade_class_a"].notna() &
        pairs["ade_class_b"].notna() &
        (pairs["ade_class_a"] != pairs["ade_class_b"])
    )

    return PipelineData(
        scenes=scenes,
        pairs=pairs,
        embeddings=embeddings,
        umap_3d=umap_3d,
        stability=stability,
        summary=summary,
    )


def get_idx_to_umap(scenes: pd.DataFrame) -> dict[int, tuple[float, float, float]]:
    """Build lookup from embedding index to UMAP coordinates."""
    return {
        row["emb_index"]: (row["umap_x"], row["umap_y"], row["umap_z"])
        for _, row in scenes.iterrows()
    }


def get_clip_to_umap(scenes: pd.DataFrame) -> dict[str, tuple[float, float, float]]:
    """Build lookup from clip_id to UMAP coordinates."""
    return {
        row["clip_id"]: (row["umap_x"], row["umap_y"], row["umap_z"])
        for _, row in scenes.iterrows()
    }
