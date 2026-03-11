"""Image manipulator: the top-level orchestrator.

Composes codec, selection, and genotype application into a
two-phase workflow:

    prepare(image)        → ManipulationContext   (once per seed)
    apply(context, genes) → PIL.Image             (many times per seed)

The ``apply_genotype`` function is exposed separately as a pure
function for unit testing and direct use outside the class.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .codec import VQGANCodec
from .selection import build_codebook_knn, build_patch_selection
from .types import (
    CandidateStrategy,
    CodeGrid,
    ManipulationContext,
    PatchSelection,
    PatchStrategy,
)


@dataclass(frozen=True)
class ImageConfig:
    """Image manipulator settings (VQGAN codebook swaps).

    Defined here (not in ``src/config``) to avoid circular imports.
    Re-exported via ``src.config.ImageConfig``.
    """

    preset: str = "f8-16384"
    patch_ratio: float = 0.1
    patch_strategy: PatchStrategy = PatchStrategy.FREQUENCY
    n_candidates: int = 25
    candidate_strategy: CandidateStrategy = CandidateStrategy.KNN
    resolution: int = 256
    knn_cache_path: Path | None = None


class ImageManipulator:
    """Discrete image manipulation via VQGAN codebook swaps.

    Lifecycle::

        manipulator = ImageManipulator(codec, config)

        # For each seed image:
        ctx = manipulator.prepare(seed_image)
        for genotype in optimizer.population:
            mutated = manipulator.apply(ctx, genotype)
            score = evaluate(mutated)
    """

    __slots__ = ("_codec", "_config", "_knn")

    def __init__(
        self,
        codec: VQGANCodec,
        config: ImageConfig | None = None,
    ) -> None:
        self._codec = codec
        self._config = config or ImageConfig()
        self._knn = build_codebook_knn(
            codec.codebook,
            cache_path=self._config.knn_cache_path,
        )

    @classmethod
    def from_preset(
        cls,
        device: str = "cpu",
        config: ImageConfig | None = None,
    ) -> ImageManipulator:
        """Load a VQGAN by preset name and build the manipulator.

        Available presets: ``"f16-1024"``, ``"f16-16384"``, ``"f8-16384"``.
        Set via ``config.preset`` (default: ``"f8-16384"``).

        Args:
            device: Torch device string, e.g. ``"mps"`` or ``"cuda"``.
            config: Manipulator configuration.
        """
        from .loading import load_vqgan

        cfg = config or ImageConfig()
        model = load_vqgan(cfg.preset)
        codec = VQGANCodec(model, device=device, resolution=cfg.resolution)
        return cls(codec, cfg)

    @classmethod
    def from_huggingface(
        cls,
        repo_id: str,
        device: str = "cpu",
        config: ImageConfig | None = None,
    ) -> ImageManipulator:
        """Load a VQGAN from HuggingFace and build the manipulator.

        Args:
            repo_id: e.g. ``"thomwolf/vqgan_imagenet_f16_1024"``.
            device: Torch device string.
            config: Manipulator configuration.
        """
        from .loading import load_huggingface_vqgan

        cfg = config or ImageConfig()
        model = load_huggingface_vqgan(repo_id)
        codec = VQGANCodec(model, device=device, resolution=cfg.resolution)
        return cls(codec, cfg)

    @classmethod
    def from_checkpoint(
        cls,
        arch_config: dict,
        checkpoint_path: str | Path,
        device: str = "cpu",
        config: ImageConfig | None = None,
    ) -> ImageManipulator:
        """Load a VQGAN from architecture config + checkpoint file.

        Args:
            arch_config: Architecture params dict (same keys as HF config.json).
            checkpoint_path: Path to ``.ckpt`` weights file.
            device: Torch device string.
            config: Manipulator configuration.
        """
        from .loading import load_checkpoint_vqgan

        cfg = config or ImageConfig()
        model = load_checkpoint_vqgan(arch_config, Path(checkpoint_path))
        codec = VQGANCodec(model, device=device, resolution=cfg.resolution)
        return cls(codec, cfg)

    # -- properties ----------------------------------------------------------

    @property
    def codec(self) -> VQGANCodec:
        return self._codec

    @property
    def config(self) -> ImageConfig:
        return self._config

    # -- two-phase API -------------------------------------------------------

    def prepare(self, image: Image.Image) -> ManipulationContext:
        """Encode a seed image and build its search space.

        Call once per seed. The returned context holds the encoded
        grid and the patch selection — everything the optimizer needs
        to know about the genotype dimensions and bounds.
        """
        grid = self._codec.encode(image)

        selection = build_patch_selection(
            grid=grid,
            knn=self._knn,
            patch_strategy=self._config.patch_strategy,
            patch_ratio=self._config.patch_ratio,
            candidate_strategy=self._config.candidate_strategy,
            n_candidates=self._config.n_candidates,
        )

        return ManipulationContext(
            original_grid=grid,
            selection=selection,
        )

    def apply(
        self,
        ctx: ManipulationContext,
        genotype: NDArray[np.int64],
    ) -> Image.Image:
        """Apply a genotype to produce a manipulated image.

        Args:
            ctx: Prepared context from ``prepare()``.
            genotype: Integer array of length ``ctx.genotype_dim``.
                Each value ∈ [0, bound) where bound = gene_bounds[i].
                0 = keep original patch, k ≥ 1 = use candidate[k-1].

        Returns:
            Manipulated PIL image.
        """
        mutated = apply_genotype(ctx.original_grid, ctx.selection, genotype)
        return self._codec.decode(mutated)


# ---------------------------------------------------------------------------
# Pure genotype application
# ---------------------------------------------------------------------------


def apply_genotype(
    grid: CodeGrid,
    selection: PatchSelection,
    genotype: NDArray[np.int64],
) -> CodeGrid:
    """Map a genotype through a patch selection to produce a mutated grid.

    This is the core function the optimizer drives. It is pure:
    same inputs always produce the same output, no side effects.

    Gene encoding:
        0              → keep original codeword at that position
        k ∈ [1, K]     → replace with selection.candidates[i][k - 1]
    """
    n = selection.n_patches
    if len(genotype) != n:
        raise ValueError(
            f"Genotype length {len(genotype)} ≠ selection size {n}"
        )

    # Fast path: no mutations
    active = np.nonzero(genotype)[0]
    if len(active) == 0:
        return grid

    rows = selection.positions[active, 0]
    cols = selection.positions[active, 1]
    codes = np.array(
        [selection.candidates[i][genotype[i] - 1] for i in active],
        dtype=np.int64,
    )

    return grid.replace(rows, cols, codes)
