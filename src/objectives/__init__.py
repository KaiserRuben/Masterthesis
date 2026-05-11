"""VLM boundary-testing objectives."""

from smoo.objectives import CriterionCollection
from smoo.objectives.image_criteria import MatrixDistance

from .targeted_balance import TargetedBalance
from .text_embedding_distance import TextEmbeddingDistance

__all__ = [
    "CriterionCollection",
    "MatrixDistance",
    "TargetedBalance",
    "TextEmbeddingDistance",
]
