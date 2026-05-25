"""Boundary-pair discovery pipeline (evolutionary → PDQ).

Produces ``(anchor, partner)`` pairs across the VLM decision boundary
with minimal genome delta ``|partner − anchor|``:

* Evolutionary stage finds a Pareto front of near-boundary individuals
  per seed.
* Each Pareto member is selected as a PDQ anchor; PDQ Stage 1 + Stage 2
  produce a partner with min ``rank_sum_delta`` from that anchor.

Output per seed is one ``archive.parquet`` with paired
``(genotype_anchor, genotype_min)`` rows tagged by ``pareto_idx`` and
``anchor_source="evolutionary"`` — the canonical Boundary Value
Analysis characterisation extended to VLMs.

Layered atop the two existing pipelines without violating their
boundaries: ``src.boundary_pair`` depends on both ``src.evolutionary``
and ``src.pdq``; neither depends on this package or on each other.
"""

from .config import (
    AnchorSelectionConfig,
    BoundaryPairExperimentConfig,
    EvolutionaryStageConfig,
    PDQStageConfig,
    load_boundary_pair_config,
    to_evolutionary_config,
    to_pdq_config,
)

__all__ = [
    "AnchorSelectionConfig",
    "BoundaryPairExperimentConfig",
    "EvolutionaryStageConfig",
    "PDQStageConfig",
    "load_boundary_pair_config",
    "to_evolutionary_config",
    "to_pdq_config",
]
