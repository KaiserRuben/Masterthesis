"""VLM boundary-testing objectives."""

from smoo.objectives import CriterionCollection
from smoo.objectives.auxiliary_criteria import ArchiveSparsity
from smoo.objectives.image_criteria import MatrixDistance

from .concentration import Concentration
from .genome_distance import NormalizedGenomeDistance
from .targeted_balance import TargetedBalance
from .text_replacement_distance import TextReplacementDistance

__all__ = [
    "ArchiveSparsity",
    "Concentration",
    "CriterionCollection",
    "MatrixDistance",
    "NormalizedGenomeDistance",
    "TargetedBalance",
    "TextReplacementDistance",
]
