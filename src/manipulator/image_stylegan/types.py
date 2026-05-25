"""Immutable data types for StyleGAN-based image manipulation.

Mirrors the surface of :mod:`src.manipulator.image.types` so the
:class:`VLMManipulator` bridge can hold either backend's context via the
:class:`src.manipulator.image_backend.ImageManipulationContextLike`
protocol.

The StyleGAN context carries:

* ``origin_w`` — the class-conditional w-tensor for the seed's origin
  class, accepted by the downstream SUT (one tensor per seed).
* ``target_w`` — the class-modal w-tensor for the seed's target class
  (one tensor per class, shared across seeds of that class).
* ``origin_image`` — the synthetic origin image actually scored by the SUT
  during precheck. Kept on the context so trace logic that expects an
  origin pixel buffer (objective input, visualization, FPS init) has
  something concrete to read.
* ``gene_bounds`` — derived from the generator's ``num_ws`` and the
  configured ``kappa_quant_levels``.
* ``target_class`` / ``candidate_strategy`` — propagate to trace metadata
  identically to the VQGAN context's same-named fields.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image
import torch


@dataclass(frozen=True)
class StyleGANManipulationContext:
    """Per-seed StyleGAN-XL manipulation context.

    All tensors live on the device chosen at manipulator construction;
    fragments / numpy arrays use the same shape conventions as the
    VQGAN context for downstream symmetry.

    :param origin_w: Origin w-tensor, shape ``(1, num_ws, w_dim)``. The
        synthetic origin image (``origin_image``) is rendered from this
        w via the generator's synthesis network.
    :param target_w: Target class-modal w-tensor, shape ``(1, num_ws, w_dim)``.
    :param origin_image: The PIL image actually accepted by the SUT
        precheck for this seed.
    :param origin_class: Origin (anchor) class label, stored for trace
        purposes — does not influence apply().
    :param target_class: Target class label (mirrors VQGAN context).
    :param kappa_quant_levels: Quantization level count (Q). Genes take
        values in ``[0, Q]`` inclusive; κ = gene/Q ∈ ``[0, 1]``.
    :param num_ws: Number of style layers (= len(genome)).
    :param candidate_strategy: Always ``"stylegan_interp"`` for this
        context; recorded so trace readers can interpret gene semantics.
    """

    origin_w: torch.Tensor
    target_w: torch.Tensor
    origin_image: Image.Image
    origin_class: str
    target_class: str | None
    kappa_quant_levels: int
    num_ws: int
    candidate_strategy: str = "stylegan_interp"

    @property
    def genotype_dim(self) -> int:
        """Number of integer genes the optimizer must provide."""
        return self.num_ws

    @property
    def gene_bounds(self) -> NDArray[np.int64]:
        """Exclusive upper bound per gene: ``kappa_quant_levels + 1``.

        Gene values are integers in ``[0, kappa_quant_levels]`` inclusive
        (i.e. ``[0, Q + 1)``), matching the per-gene upper-bound contract
        the discrete optimizer expects.
        """
        return np.full(
            self.num_ws, self.kappa_quant_levels + 1, dtype=np.int64,
        )

    def zero_genotype(self) -> NDArray[np.int64]:
        """All-zero genotype: κ = 0 everywhere → reproduces origin."""
        return np.zeros(self.num_ws, dtype=np.int64)

    def random_genotype(self, rng: np.random.Generator) -> NDArray[np.int64]:
        """Uniformly random integer genotype within bounds."""
        return rng.integers(
            0, self.kappa_quant_levels + 1, size=self.num_ws, dtype=np.int64,
        )


__all__ = ["StyleGANManipulationContext"]
