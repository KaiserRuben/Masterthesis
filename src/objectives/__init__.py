"""VLM boundary-testing objectives."""

from smoo.objectives import CriterionCollection
from smoo.objectives.image_criteria import MatrixDistance

from .targeted_balance import TargetedBalance
from .text_embedding_distance import TextEmbeddingDistance
from .text_replacement_distance import TextReplacementDistance

__all__ = [
    "CriterionCollection",
    "MatrixDistance",
    "TargetedBalance",
    "TextEmbeddingDistance",
    "TextReplacementDistance",
]
