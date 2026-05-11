"""Multi-modal VLM manipulator wrapping image + composite-text sub-manipulators.

Bridges the two-phase (prepare / apply) lifecycle of the individual
manipulators with SMOO's ``Manipulator`` interface. The optimizer works
with a single concatenated genotype ``[image_genes | text_genes]`` and
this class splits and dispatches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from smoo.manipulator import Manipulator

from .image.manipulator import ImageManipulator
from .text.composite import CompositeTextManipulator

if TYPE_CHECKING:
    from .image.types import ManipulationContext as ImageManipulationContext
    from .text.composite import CompositeManipulationContext


class VLMManipulator(Manipulator):
    """Multi-modal manipulator wrapping image + composite-text sub-manipulators.

    Lifecycle::

        manipulator = VLMManipulator(image_manip, text_manip)
        manipulator.prepare(seed_image, seed_text)

        # In optimizer loop:
        images, texts = manipulator.manipulate(candidates=None, weights=genotypes)
    """

    def __init__(
        self,
        image_manipulator: ImageManipulator,
        text_manipulator: CompositeTextManipulator,
    ) -> None:
        self._image = image_manipulator
        self._text = text_manipulator
        self._image_ctx: ImageManipulationContext | None = None
        self._text_ctx: CompositeManipulationContext | None = None

    # -- lifecycle -----------------------------------------------------------

    def prepare(
        self,
        image: Image.Image,
        text: str,
        exclude_words: frozenset[str] | None = None,
    ) -> None:
        """Prepare both manipulators for a seed (image, text) pair.

        Call once per seed.

        :param image: Seed PIL image.
        :param text: Seed prompt text.
        :param exclude_words: Words to protect from text mutation
            (case-insensitive). Typically the category labels so the
            optimizer cannot trivially remove the correct answer.
        """
        self._image_ctx = self._image.prepare(image)
        self._text_ctx = self._text.prepare(text, exclude_words=exclude_words)

    # -- properties ----------------------------------------------------------

    @property
    def is_prepared(self) -> bool:
        return self._image_ctx is not None and self._text_ctx is not None

    @property
    def genotype_dim(self) -> int:
        return self._image_ctx.genotype_dim + self._text_ctx.genotype_dim

    @property
    def image_dim(self) -> int:
        return self._image_ctx.genotype_dim

    @property
    def text_dim(self) -> int:
        return self._text_ctx.genotype_dim

    @property
    def gene_bounds(self) -> NDArray[np.int64]:
        return np.concatenate([
            self._image_ctx.gene_bounds,
            self._text_ctx.gene_bounds,
        ])

    @property
    def image_manipulator(self) -> ImageManipulator:
        """Underlying image sub-manipulator. Exposed for samplers that
        need access to the VQGAN codebook (e.g. embedding-FPS init)."""
        return self._image

    @property
    def image_context(self) -> ImageManipulationContext:
        return self._image_ctx

    @property
    def text_context(self) -> CompositeManipulationContext:
        return self._text_ctx

    # -- SMOO Manipulator interface ------------------------------------------

    def manipulate(self, candidates=None, *, weights, **kwargs):
        """Apply genotypes to produce mutated (images, texts) pairs.

        :param candidates: Unused (contexts stored from ``prepare()``).
        :param weights: ``NDArray`` of shape ``(pop_size, genotype_dim)``
            with integer genotypes. The first ``image_dim`` genes drive
            the image; the remaining ``text_dim`` drive the text.
        :returns: ``(images, texts)`` lists.
        """
        if not self.is_prepared:
            raise RuntimeError("Call prepare() before manipulate().")

        weights = np.asarray(weights, dtype=np.int64)
        img_genes = weights[:, : self.image_dim]
        txt_genes = weights[:, self.image_dim:]

        # One VQGAN forward for the whole population (was N batch-1 calls).
        images = self._image.apply_batch(self._image_ctx, img_genes)
        # Text apply is a candidate-table lookup — keep sequential.
        texts = [
            self._text.apply(self._text_ctx, txt_genes[i])
            for i in range(len(weights))
        ]
        return images, texts

    def get_images(self, z):
        raise NotImplementedError(
            "VLMManipulator produces (images, texts) via manipulate(). "
            "Use manipulate() instead."
        )

    # -- genotype helpers ----------------------------------------------------

    def zero_genotype(self) -> NDArray[np.int64]:
        return np.zeros(self.genotype_dim, dtype=np.int64)

    def random_genotype(self, rng: np.random.Generator) -> NDArray[np.int64]:
        return np.concatenate([
            self._image_ctx.random_genotype(rng),
            self._text_ctx.random_genotype(rng),
        ])
