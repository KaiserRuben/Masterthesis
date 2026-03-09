"""Discrete image manipulation via VQGAN codebook swaps.

A VQGAN encodes any image into a spatial grid of discrete codebook
indices — typically 16×16 = 256 tokens, each drawn from a vocabulary
of 1024+ codewords. Manipulation means replacing selected tokens
with semantically nearby alternatives from the codebook.

The design separates three concerns:

    codec       — encode/decode between PIL images and code grids
    selection   — which patches to mutate, with which candidates
    manipulator — orchestrate prepare → apply cycles for the optimizer

Usage::

    from src.manipulator.image import ImageManipulator, ImageManipulatorConfig

    m = ImageManipulator.from_preset("f8-16384", device="mps")

    ctx = m.prepare(seed_image)
    genotype = ctx.random_genotype(rng=np.random.default_rng(42))
    mutated = m.apply(ctx, genotype)
"""

from .codec import VQGANCodec
from .manipulator import ImageManipulator, ImageManipulatorConfig, apply_genotype
from .types import (
    CandidateStrategy,
    CodeGrid,
    ManipulationContext,
    PatchSelection,
    PatchStrategy,
)

__all__ = [
    "CandidateStrategy",
    "CodeGrid",
    "ImageManipulator",
    "ImageManipulatorConfig",
    "ManipulationContext",
    "PatchSelection",
    "PatchStrategy",
    "VQGANCodec",
    "apply_genotype",
]
