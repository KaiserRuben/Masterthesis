"""Image manipulator: the top-level orchestrator.

Composes codec, selection, and genotype application into a
two-phase workflow:

    prepare(image)        → ManipulationContext   (once per seed)
    apply(context, genes) → PIL.Image             (many times per seed)

The ``apply_genotype`` function is exposed separately as a pure
function for unit testing and direct use outside the class.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .class_target import (
    CLASS_TARGET_KEY_PREFIX,
    ModalTargetBuilder,
)
from .codec import VQGANCodec
from .cone_candidates import ConeCandidateFilter
from .selection import (
    build_codebook_knn,
    build_cone_patch_selection,
    build_patch_selection,
)
from .types import (
    CandidateStrategy,
    CodeGrid,
    ManipulationContext,
    PatchSelection,
    PatchStrategy,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConeFilterConfig:
    """Targeted-cone candidate-list parameters.

    When ``enabled`` is True and a ``target_class`` is supplied to
    :meth:`ImageManipulator.prepare`, per-patch candidate lists are built
    via the origin→target double-cone filter against the modal target
    grid for that class. Otherwise the legacy KNN-from-origin path is
    used unchanged.

    :param enabled: Master switch. False (default) keeps legacy behaviour.
    :param alpha_deg: Cone half-angle in degrees.
    :param target_m: Number of class exemplars used to build the modal
        target grid (per-position argmax of codeword histogram).
    """

    enabled: bool = False
    alpha_deg: float = 20.0
    target_m: int = 100


@dataclass(frozen=True)
class StyleGANConfig:
    """StyleGAN-XL image-backend parameters.

    Only meaningful when :attr:`ImageConfig.backend` is ``"stylegan_xl"``.
    Lives alongside :class:`ConeFilterConfig` rather than in a separate
    module so dacite can pick it up off the same YAML namespace
    (``image.stylegan: { ... }``).

    :param checkpoint_url: Source URL for the NVlabs StyleGAN-XL pickle.
        Pre-trained ImageNet-256 checkpoint is recommended.
    :param checkpoint_path: Local on-disk cache target. ``~`` expanded via
        :func:`pathlib.Path.expanduser`. Downloader is invoked lazily and
        only when missing.
    :param interpolate: Forwarded to the underlying
        :class:`StyleGANManipulator`. ``True`` (default) yields continuous
        κ-interpolation between origin and target w; ``False`` uses
        discrete layer-pick style mixing — kept here for future ablation.
    :param truncation_psi: Mapping-network truncation factor; ``1.0``
        retains full diversity. Lower values pull samples toward the
        class-conditional mean.
    :param truncation_cutoff: Layer above which truncation is no longer
        applied. ``0`` (default) truncates from the first layer.
    :param kappa_quant_levels: Quantization level count for κ ∈ [0, 1].
        Each layer gene takes values in ``{0, 1, …, Q}`` where ``Q`` is
        this number; κ for layer ``l`` equals ``gene[l] / Q``. ``20``
        (default) gives 21 discrete settings per layer.
    :param target_m: Number of class-conditional samples averaged to
        build the modal target w for a class. Larger = smoother class
        archetype at cost of cache build time.
    :param sut_precheck_max_attempts: Maximum SUT-precheck seeds tried
        before declaring an (origin, target) pair infeasible. The
        pairwise precheck (see
        :func:`src.manipulator.image_stylegan.class_target.find_pair_dominant_origin_seed`)
        accepts the first seed whose synthetic origin gets a higher SUT
        log-prob for the origin class than for the target class.
    :param synthesis_batch_size: Sub-batch size used inside SMOO's
        :meth:`StyleGANManipulator.get_images` to chunk a population
        render. Synthesis peak memory scales linearly with this number,
        so on CPU (no VRAM offload) the optimizer's ``pop_size`` cannot
        be the synthesis batch — a single 20-image 256×256 forward
        through StyleGAN-XL needs >20 GB of activations. The optimizer
        still asks for ``pop_size`` renders per generation; SMOO loops
        them in chunks of this size. ``0`` means no chunking (use the
        full batch, only safe on GPU with ample VRAM).
    """

    checkpoint_url: str = (
        "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/"
        "models/imagenet256.pkl"
    )
    checkpoint_path: Path = field(
        default_factory=lambda: Path("~/.cache/stylegan_xl/imagenet256.pkl")
    )
    interpolate: bool = True
    truncation_psi: float = 1.0
    truncation_cutoff: int = 0
    kappa_quant_levels: int = 20
    target_m: int = 100
    sut_precheck_max_attempts: int = 200
    synthesis_batch_size: int = 4


@dataclass(frozen=True)
class ImageConfig:
    """Image manipulator settings.

    Two backends are dispatched by :attr:`backend`:

    * ``"vqgan_codebook"`` (default) — discrete VQGAN patch swaps, all
      ``patch_*``, ``n_candidates``, ``candidate_strategy``,
      ``resolution``, ``knn_cache_path``, and ``cone_filter`` fields
      apply; the ``stylegan`` block is ignored.
    * ``"stylegan_xl"`` — continuous StyleGAN-XL style-mixing quantised to
      integer genes; only :attr:`stylegan` applies, the VQGAN-side fields
      are ignored.

    Defined here (not in ``src/config``) to avoid circular imports.
    Re-exported via ``src.config.ImageConfig``.

    :param backend: ``"vqgan_codebook"`` (default) or ``"stylegan_xl"``.
    :param preset: VQGAN preset name (vqgan_codebook only).
    :param patch_ratio: Fraction of patches eligible for swap (vqgan only).
    :param patch_strategy: Patch-selection strategy (vqgan only).
    :param n_candidates: Codebook-replacement count per patch (vqgan only).
    :param candidate_strategy: How candidates are picked from KNN (vqgan).
    :param resolution: VQGAN input resolution (vqgan only).
    :param decode_batch_size: Sub-batch size for the VQGAN decoder
        forward (vqgan only). The whole optimiser population (and, at
        seed finalize, the full Pareto front of 200+ candidates) is
        decoded through :meth:`ImageManipulator.apply_batch`; decoder
        activations scale linearly with this number. ``8`` caps the
        per-forward peak well under a 30-image batch (~4× lower) at a
        negligible throughput cost, which matters on MPS/CPU where the
        device heap is never returned once a high-water mark is hit.
        Decode output is identical regardless of batching, so search
        behaviour is unaffected. Raise on large-VRAM GPUs to trade
        memory for throughput.
    :param knn_cache_path: Optional KNN cache path (vqgan only).
    :param cone_filter: Cone-filter sub-config (vqgan only).
    :param stylegan: StyleGAN-XL sub-config (stylegan only).
    """

    backend: str = "vqgan_codebook"
    preset: str = "f8-16384"
    patch_ratio: float = 0.1
    patch_strategy: PatchStrategy = PatchStrategy.FREQUENCY
    n_candidates: int = 25
    candidate_strategy: CandidateStrategy = CandidateStrategy.KNN
    resolution: int = 256
    decode_batch_size: int = 8
    knn_cache_path: Path | None = None
    cone_filter: ConeFilterConfig = field(default_factory=ConeFilterConfig)
    stylegan: StyleGANConfig = field(default_factory=StyleGANConfig)

    def __post_init__(self) -> None:
        if self.backend not in ("vqgan_codebook", "stylegan_xl"):
            raise ValueError(
                f"image.backend must be 'vqgan_codebook' or "
                f"'stylegan_xl'; got {self.backend!r}"
            )


class ImageManipulator:
    """Discrete image manipulation via VQGAN codebook swaps.

    Lifecycle::

        manipulator = ImageManipulator(codec, config)

        # For each seed image:
        ctx = manipulator.prepare(seed_image)
        for genotype in optimizer.population:
            mutated = manipulator.apply(ctx, genotype)
            score = evaluate(mutated)

    Cone-filter mode (opt-in via ``config.cone_filter.enabled``):
    construct the manipulator with a ``ModalTargetBuilder`` and a list of
    ``target_classes`` to pre-populate. Each :meth:`prepare` call must
    then pass the seed's ``target_class``; the per-patch candidate list
    is built by filtering the codebook against the origin→target cone
    instead of the KNN-from-origin list.
    """

    __slots__ = (
        "_codec",
        "_config",
        "_knn",
        "_cone_filter",
        "_modal_builder",
    )

    def __init__(
        self,
        codec: VQGANCodec,
        config: ImageConfig | None = None,
        *,
        modal_builder: "ModalTargetBuilder | None" = None,
        target_classes: tuple[str, ...] | None = None,
    ) -> None:
        self._codec = codec
        self._config = config or ImageConfig()
        self._knn = build_codebook_knn(
            codec.codebook,
            cache_path=self._config.knn_cache_path,
        )
        self._cone_filter: ConeCandidateFilter | None = None
        self._modal_builder: ModalTargetBuilder | None = modal_builder

        if self._config.cone_filter.enabled:
            self._cone_filter = ConeCandidateFilter(
                alpha_deg=self._config.cone_filter.alpha_deg,
            )
            if modal_builder is None:
                logger.warning(
                    "cone_filter.enabled=True but no modal_builder supplied; "
                    "per-seed prepare(target_class=...) will be required and "
                    "will fail without a modal_builder."
                )
            elif target_classes:
                # Idempotent, lock-free pre-population: Redis-first, build on
                # miss. Parallel-worker processes either all hit L2 or one
                # writes while the others see the write on their next call.
                modal_builder.populate_many(target_classes)

    @classmethod
    def from_preset(
        cls,
        device: str = "cpu",
        config: ImageConfig | None = None,
        *,
        modal_builder: "ModalTargetBuilder | None" = None,
        target_classes: tuple[str, ...] | None = None,
    ) -> ImageManipulator:
        """Load a VQGAN by preset name and build the manipulator.

        Available presets: ``"f16-1024"``, ``"f16-16384"``, ``"f8-16384"``.
        Set via ``config.preset`` (default: ``"f8-16384"``).

        Args:
            device: Torch device string, e.g. ``"mps"`` or ``"cuda"``.
            config: Manipulator configuration.
            modal_builder: Optional builder for class-conditional modal
                target grids (cone-filter mode).
            target_classes: Optional list of target classes to pre-populate
                the modal-builder cache for.
        """
        from .loading import load_vqgan

        cfg = config or ImageConfig()
        model = load_vqgan(cfg.preset)
        codec = VQGANCodec(model, device=device, resolution=cfg.resolution)
        return cls(
            codec,
            cfg,
            modal_builder=modal_builder,
            target_classes=target_classes,
        )

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

    @property
    def cone_filter_enabled(self) -> bool:
        """Whether this manipulator routes through the cone-filter path."""
        return self._cone_filter is not None

    @property
    def modal_builder(self) -> "ModalTargetBuilder | None":
        return self._modal_builder

    def attach_modal_builder(self, builder: "ModalTargetBuilder") -> None:
        """Late-bind a modal-target builder after construction.

        Useful when the runner instantiates the manipulator in parallel
        with seed generation (so the target-class list is not yet known)
        and later needs to enable the cone path once seeds are ready.
        """
        self._modal_builder = builder

    def precompute_targets(self, target_classes: tuple[str, ...]) -> None:
        """Pre-populate the modal-target cache for ``target_classes``.

        No-op when no modal builder has been attached. Otherwise iterates
        the list, calling :meth:`ModalTargetBuilder.ensure` for each class
        (L2 lookup → build on miss → L1 write). Parallel-worker safe via
        the same idempotent-by-key Redis pattern as the KNN cache.
        """
        if self._modal_builder is None:
            return
        self._modal_builder.populate_many(target_classes)

    # -- two-phase API -------------------------------------------------------

    def prepare(
        self,
        image: Image.Image,
        target_class: str | None = None,
        origin_class: str | None = None,
    ) -> ManipulationContext:
        """Encode a seed image and build its search space.

        Call once per seed. The returned context holds the encoded
        grid, the patch selection, and (when cone-filter mode is active)
        the target class identity for trace metadata.

        :param image: Seed PIL image.
        :param target_class: Concrete L0 class name the cone filter aims
            toward. Required when ``config.cone_filter.enabled`` is True;
            ignored on the legacy KNN path.
        :param origin_class: Ignored by VQGAN (origin is determined by
            the seed image's own latent code). Accepted for protocol
            compatibility with the StyleGAN backend.
        """
        _ = origin_class  # explicit no-op (see docstring)
        grid = self._codec.encode(image)

        if self._cone_filter is not None:
            if target_class is None:
                raise ValueError(
                    "ImageManipulator.prepare() requires target_class when "
                    "config.cone_filter.enabled is True."
                )
            if self._modal_builder is None:
                raise RuntimeError(
                    "cone_filter.enabled is True but no modal_builder was "
                    "supplied at construction; cannot build modal target "
                    "grid for class %r." % (target_class,)
                )
            target_grid = self._modal_builder.ensure(target_class)
            selection = build_cone_patch_selection(
                grid=grid,
                target_grid=target_grid,
                codebook=self._codec.codebook,
                cone_filter=self._cone_filter,
                patch_strategy=self._config.patch_strategy,
                patch_ratio=self._config.patch_ratio,
            )
            return ManipulationContext(
                original_grid=grid,
                selection=selection,
                target_class=target_class,
                candidate_strategy="cone_filter",
            )

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
            target_class=target_class,
            candidate_strategy="knn",
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
        return self.apply_batch(ctx, genotype[None, :])[0]

    def baseline_image(self, ctx: ManipulationContext) -> Image.Image:
        """Return the κ=0 reference image (codec-roundtripped seed).

        At the all-zeros genotype no patches are mutated, so the
        output is ``decode(encode(seed.image))`` — what every individual
        is measured against by MatrixDistance. Goes through
        :meth:`apply` (single-image, batch=1), which VQGAN handles
        natively without any SMOO involvement.
        """
        return self.apply(ctx, ctx.zero_genotype())

    def apply_batch(
        self,
        ctx: ManipulationContext,
        genotypes: NDArray[np.int64],
    ) -> list[Image.Image]:
        """Apply N genotypes and decode them in a single VQGAN forward.

        Args:
            ctx: Prepared context from ``prepare()``.
            genotypes: Integer array of shape ``(N, ctx.genotype_dim)``.

        Returns:
            List of N manipulated PIL images, in genotype order.
        """
        if len(genotypes) == 0:
            return []
        grids = [
            apply_genotype(ctx.original_grid, ctx.selection, g)
            for g in genotypes
        ]
        return self._codec.decode_batch(
            grids, chunk_size=self._config.decode_batch_size
        )


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
