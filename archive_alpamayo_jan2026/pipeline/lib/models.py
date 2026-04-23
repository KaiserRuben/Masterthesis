"""
Pydantic Models for Pipeline Data

Structured, validated models for pipeline data contracts.
Replaces implicit dictionary contracts with typed, validated models.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


# =============================================================================
# TRAJECTORY CLASSIFICATION
# =============================================================================


class TrajectoryClassification(BaseModel):
    """Classification result from trajectory analysis.

    Produced by classify_trajectory() in lib/trajectory.py.
    36 possible classes (4 direction x 3 speed x 3 lateral).
    """

    direction: Literal["turn_left", "turn_right", "straight", "slight_curve"]
    speed: Literal["accelerate", "decelerate", "constant"]
    lateral: Literal["lane_change_left", "lane_change_right", "lane_keep"]
    combined: str  # "{direction}_{speed}_{lateral}"
    delta_theta: float  # heading change (degrees)
    delta_v: float  # velocity change (m/s)
    delta_y: float  # lateral displacement (m)


# =============================================================================
# INFERENCE RESULT
# =============================================================================


class InferenceResult(BaseModel):
    """Result from running Alpamayo inference on a single scene.

    Produced by run_inference() in step_3_infer.py.
    """

    clip_id: str
    t0_us: int
    ade: float  # Average Displacement Error (meters)
    coc_reasoning: str  # Chain-of-Thought reasoning from model
    traj_direction: str
    traj_speed: str
    traj_lateral: str
    load_time_s: float
    inference_time_s: float
    inference_timestamp: str  # ISO format timestamp


# =============================================================================
# PIPELINE CONFIG
# =============================================================================


class DatasetConfig(BaseModel):
    """Dataset sampling configuration."""

    default_n: int
    default_seed: int
    t0_us: int  # Timestamp offset (microseconds)


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""

    model: str
    pretrained: str
    dim: int
    batch_size: int


class ClassificationConfig(BaseModel):
    """Classification keys configuration."""

    keys: list[str]


class InferenceConfig(BaseModel):
    """Inference model configuration."""

    model_id: str
    checkpoint_interval: int


class AnalysisConfig(BaseModel):
    """Analysis parameters configuration."""

    k_neighbors: int
    max_key_diff: int
    min_confidence: float


class PathsConfig(BaseModel):
    """File paths configuration."""

    data_dir: str
    scenes_file: str
    embeddings_file: str
    results_dir: str
    anchor_file: str
    image_cache: str


class PipelineConfig(BaseModel):
    """Root configuration for the entire pipeline.

    Loaded from config.yaml via load_config() in lib/io.py.
    """

    dataset: DatasetConfig
    embedding: EmbeddingConfig
    classification: ClassificationConfig
    inference: InferenceConfig
    analysis: AnalysisConfig
    paths: PathsConfig
