"""Multi-modal VLM manipulator wrapping image + text sub-manipulators.

Bridges the two-phase (prepare / apply) lifecycle of the individual
manipulators with SMOO's ``Manipulator`` interface.  The optimizer
works with a single concatenated genotype ``[image_genes | text_genes]``
and this class splits and dispatches appropriately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from smoo.manipulator import Manipulator

from .image.manipulator import ImageManipulator
from .text.manipulator import TextManipulator

if TYPE_CHECKING:
    from .image.types import ManipulationContext as ImageManipulationContext
    from .text.types import ManipulationContext as TextManipulationContext


class VLMManipulator(Manipulator):
    """Multi-modal manipulator wrapping image + text sub-manipulators.

    Lifecycle::

        manipulator = VLMManipulator(image_manip, text_manip)
        manipulator.prepare(seed_image, seed_text)

        # In optimizer loop:
        images, texts = manipulator.manipulate(candidates=None, weights=genotypes)
    """

    def __init__(
        self,
        image_manipulator: ImageManipulator,
        text_manipulator: TextManipulator,
    ) -> None:
        self._image = image_manipulator
        self._text = text_manipulator
        self._image_ctx: ImageManipulationContext | None = None
        self._text_ctx: TextManipulationContext | None = None
        self._text_candidate_distances: tuple[np.ndarray, ...] | None = None

    # -- lifecycle -----------------------------------------------------------

    def prepare(
        self,
        image: Image.Image,
        text: str,
        exclude_words: frozenset[str] | None = None,
    ) -> None:
        """Prepare both manipulators for a seed (image, text) pair.

        Call once per seed.  Creates manipulation contexts and precomputes
        text candidate distances for the TextReplacementDistance objective.

        Args:
            image: Seed PIL image.
            text: Seed prompt text.
            exclude_words: Words to protect from text mutation
                (case-insensitive).  Typically the category labels so
                the optimizer cannot trivially remove the correct answer.
        """
        self._image_ctx = self._image.prepare(image)
        self._text_ctx = self._text.prepare(text, exclude_words=exclude_words)
        self._text_candidate_distances = self._compute_text_distances()

    def _compute_text_distances(self) -> tuple[np.ndarray, ...]:
        """Compute cosine distances between each original word and its candidates.

        Returns a tuple of 1-D arrays: ``distances[i][k]`` is the cosine
        distance for word *i*, candidate *k*.
        """
        embeddings = self._text.embeddings
        return tuple(
            np.array([float(embeddings.distance(orig.lower(), c)) for c in cands])
            for orig, cands in zip(
                self._text_ctx.selection.original_words,
                self._text_ctx.selection.candidates,
            )
        )

    # -- properties ----------------------------------------------------------

    @property
    def is_prepared(self) -> bool:
        """Whether ``prepare()`` has been called."""
        return self._image_ctx is not None and self._text_ctx is not None

    @property
    def genotype_dim(self) -> int:
        """Total genotype length: image genes + text genes."""
        return self._image_ctx.genotype_dim + self._text_ctx.genotype_dim

    @property
    def image_dim(self) -> int:
        """Number of image genes."""
        return self._image_ctx.genotype_dim

    @property
    def text_dim(self) -> int:
        """Number of text genes."""
        return self._text_ctx.genotype_dim

    @property
    def gene_bounds(self) -> NDArray[np.int64]:
        """Per-gene upper bounds (exclusive), concatenated ``[image | text]``."""
        return np.concatenate([
            self._image_ctx.gene_bounds,
            self._text_ctx.gene_bounds,
        ])

    @property
    def image_context(self) -> ImageManipulationContext:
        """The prepared image manipulation context."""
        return self._image_ctx

    @property
    def text_context(self) -> TextManipulationContext:
        """The prepared text manipulation context."""
        return self._text_ctx

    @property
    def text_candidate_distances(self) -> tuple[np.ndarray, ...]:
        """Precomputed cosine distances for TextReplacementDistance objective."""
        return self._text_candidate_distances

    # -- SMOO Manipulator interface ------------------------------------------

    def manipulate(self, candidates=None, *, weights, **kwargs):
        """Apply genotypes to produce mutated (images, texts) pairs.

        Args:
            candidates: Unused (contexts stored from ``prepare()``).
            weights: ``NDArray`` of shape ``(pop_size, genotype_dim)`` with
                integer genotypes.  The first ``image_dim`` genes control
                the image; the remaining ``text_dim`` genes control the text.

        Returns:
            Tuple ``(images, texts)`` where *images* is a list of
            ``PIL.Image`` and *texts* is a list of ``str``.
        """
        if not self.is_prepared:
            raise RuntimeError("Call prepare() before manipulate().")

        images: list[Image.Image] = []
        texts: list[str] = []
        for genotype in weights:
            img_genes = genotype[: self.image_dim].astype(np.int64)
            txt_genes = genotype[self.image_dim :].astype(np.int64)
            images.append(self._image.apply(self._image_ctx, img_genes))
            texts.append(self._text.apply(self._text_ctx, txt_genes))
        return images, texts

    def get_images(self, z):
        """Not applicable for VLM multi-modal testing."""
        raise NotImplementedError(
            "VLMManipulator produces (images, texts) via manipulate(). "
            "Use manipulate() instead."
        )

    # -- genotype helpers ----------------------------------------------------

    def zero_genotype(self) -> NDArray[np.int64]:
        """All-zero genotype: identity for both modalities."""
        return np.zeros(self.genotype_dim, dtype=np.int64)

    def random_genotype(self, rng: np.random.Generator) -> NDArray[np.int64]:
        """Uniformly random genotype within all bounds."""
        return np.concatenate([
            self._image_ctx.random_genotype(rng),
            self._text_ctx.random_genotype(rng),
        ])
