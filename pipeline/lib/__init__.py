"""Pipeline library modules."""

from .schema import (
    load_scenes,
    save_scenes,
    update_scenes,
    COLUMNS,
    CLASSIFICATION_KEYS,
)
from .composites import create_composite, ensure_composites
from .trajectory import classify_trajectory
from .io import (
    load_config,
    load_embeddings,
    append_embeddings,
    merge_inference_results,
)
from .models import (
    TrajectoryClassification,
    InferenceResult,
    PipelineConfig,
    DatasetConfig,
    EmbeddingConfig,
    ClassificationConfig,
    InferenceConfig,
    AnalysisConfig,
    PathsConfig,
)

__all__ = [
    # schema
    "load_scenes",
    "save_scenes",
    "update_scenes",
    "COLUMNS",
    "CLASSIFICATION_KEYS",
    # composites
    "create_composite",
    "ensure_composites",
    # trajectory
    "classify_trajectory",
    # io
    "load_config",
    "load_embeddings",
    "append_embeddings",
    "merge_inference_results",
    # models
    "TrajectoryClassification",
    "InferenceResult",
    "PipelineConfig",
    "DatasetConfig",
    "EmbeddingConfig",
    "ClassificationConfig",
    "InferenceConfig",
    "AnalysisConfig",
    "PathsConfig",
]
