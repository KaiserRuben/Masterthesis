"""Backend factory: instantiate the right image manipulator from the config.

The two runners share the same dispatch logic — pull it here so both
import the single source. Returns an :class:`ImageBackend`-satisfying
object that the rest of the pipeline holds without backend branching.

VQGAN path is unchanged from the legacy
:meth:`ImageManipulator.from_preset` factory; StyleGAN path resolves
checkpoint, loads generator, builds the SMOO manipulator and class-target
builder, and wires the SUT precheck if a SUT is supplied.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from src.manipulator.image.manipulator import ImageConfig, ImageManipulator
from src.manipulator.image_backend import ImageBackend

if TYPE_CHECKING:
    from src.common import BytesRedisCache
    from src.sut import VLMSUT


logger = logging.getLogger(__name__)


def build_image_backend(
    *,
    image_config: ImageConfig,
    device: str,
    categories: tuple[str, ...] = (),
    sut: "VLMSUT | None" = None,
    redis_cache: "BytesRedisCache | None" = None,
    prompt: str | None = None,
    class_name_to_idx: dict[str, int] | None = None,
) -> ImageBackend:
    """Construct the image backend selected by ``image_config.backend``.

    For ``"vqgan_codebook"`` (the default), this is exactly the legacy
    :meth:`ImageManipulator.from_preset` call. Cone-filter init runs
    elsewhere (runner attaches the modal-builder afterwards).

    For ``"stylegan_xl"`` this resolves and loads the checkpoint, builds
    the SMOO manipulator, instantiates the class-target builder with
    the supplied SUT, and returns a
    :class:`StyleGANImageManipulator`. The SUT precheck happens later
    via :meth:`precompute_targets`; this factory only sets up the
    wiring so the precompute can run.

    :param image_config: :class:`ImageConfig` instance from the run cfg.
    :param device: Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
    :param categories: SUT contrast set, forwarded to the precheck.
    :param sut: Loaded :class:`VLMSUT`. Required when backend is
        ``"stylegan_xl"``; ignored otherwise.
    :param redis_cache: L2 cache for the StyleGAN class-target / origin
        caches. ``None`` disables L2 (L1-only).
    :param prompt: Full prompt used during the SUT precheck. Must be the
        same prompt the run will use, so the precheck filters origins
        against the run's actual classification semantics.
    :param class_name_to_idx: Optional mapping from string class names to
        the generator's integer class indices. Defaults to position in
        ``categories``; pass explicitly if categories ≠ ImageNet order.
    :raises ValueError: If StyleGAN is selected without the required
        runner-side dependencies.
    """
    backend = image_config.backend
    if backend == "vqgan_codebook":
        return ImageManipulator.from_preset(device=device, config=image_config)

    if backend == "stylegan_xl":
        if sut is None:
            raise ValueError(
                "image.backend='stylegan_xl' requires a loaded SUT for "
                "the origin precheck; sut=None at factory time."
            )
        if prompt is None:
            raise ValueError(
                "image.backend='stylegan_xl' requires the full SUT prompt "
                "(template + answer suffix); prompt=None at factory time."
            )
        return _build_stylegan_backend(
            image_config=image_config,
            device=device,
            categories=categories,
            sut=sut,
            redis_cache=redis_cache,
            prompt=prompt,
            class_name_to_idx=class_name_to_idx,
        )

    raise ValueError(
        f"Unknown image.backend={backend!r} — expected "
        "'vqgan_codebook' or 'stylegan_xl'."
    )


def _build_stylegan_backend(
    *,
    image_config: ImageConfig,
    device: str,
    categories: tuple[str, ...],
    sut: "VLMSUT",
    redis_cache: "BytesRedisCache | None",
    prompt: str,
    class_name_to_idx: dict[str, int] | None,
) -> ImageBackend:
    """Build a :class:`StyleGANImageManipulator` end-to-end.

    Implementation lives here (not inline in :func:`build_image_backend`)
    so callers that only ever use VQGAN never pay the import cost of
    pulling in the SMOO StyleGAN tree.
    """
    from smoo.manipulator.style_gan_manipulator import StyleGANManipulator

    from src.manipulator.image_stylegan import (
        StyleGANClassTargetBuilder,
        StyleGANImageManipulator,
        VLMSUTLogprobsAdapter,
        checkpoint_sha256_hex,
        ensure_checkpoint,
        sut_signature,
    )

    cfg = image_config.stylegan
    torch_device = torch.device(device)

    # 1. Resolve + download checkpoint.
    ckpt_path = ensure_checkpoint(cfg.checkpoint_url, cfg.checkpoint_path)
    ckpt_hash = checkpoint_sha256_hex(ckpt_path)
    logger.info("StyleGAN checkpoint hash prefix: %s", ckpt_hash)

    # 2. Load generator + build SMOO manipulator. ``mix_dims`` covers all
    # style layers so every layer participates in interpolation.
    smoo_manip = StyleGANManipulator(
        generator=str(ckpt_path),
        device=torch_device,
        mix_dims=(0, _peek_num_ws(str(ckpt_path), torch_device)),
        interpolate=cfg.interpolate,
        conditional=True,
        batch_size=cfg.synthesis_batch_size,
    )
    # Apply truncation settings (the class-level attribute is shared by
    # ``get_w`` and any cache rebuild).
    smoo_manip.trunc_psi = cfg.truncation_psi
    smoo_manip.trunc_cutoff = cfg.truncation_cutoff

    # 3. Class-target builder (modal-w + origin-seed caches).
    sut_sig = sut_signature(
        model_id=sut.scorer.model_id if hasattr(sut.scorer, "model_id") else "",
        categories=categories,
    )
    builder = StyleGANClassTargetBuilder(
        generator=smoo_manip,
        sut=VLMSUTLogprobsAdapter(sut, prompt=prompt),
        checkpoint_hash=ckpt_hash,
        sut_signature=sut_sig,
        categories=categories,
        target_m=cfg.target_m,
        truncation_psi=cfg.truncation_psi,
        truncation_cutoff=cfg.truncation_cutoff,
        max_attempts=cfg.sut_precheck_max_attempts,
        redis_cache=redis_cache,
        device=torch_device,
        class_name_to_idx=class_name_to_idx,
    )

    # 4. Wrap in the image-backend manipulator.
    return StyleGANImageManipulator(
        smoo_manipulator=smoo_manip,
        class_target_builder=builder,
        config=image_config,
        device=torch_device,
        categories=categories,
    )


def _peek_num_ws(checkpoint_path: str, device: torch.device) -> int:
    """Read ``num_ws`` off a checkpoint without keeping the generator around.

    SMOO's ``StyleGANManipulator`` constructor takes ``mix_dims`` *before*
    instantiating the generator from a path string, so we cannot read
    ``num_ws`` off the live manipulator. Loading is cheap once cached on
    disk; we do it once here, throw the module away, then let SMOO load
    the same pickle a second time. (Memory cost: one transient generator
    held through this call only.)
    """
    from smoo.manipulator.style_gan_manipulator import _load_stylegan

    gen = _load_stylegan.load_stylegan(checkpoint_path)
    try:
        if hasattr(gen, "num_ws"):
            return int(gen.num_ws)
        if hasattr(gen, "synthesis") and hasattr(gen.synthesis, "num_ws"):
            return int(gen.synthesis.num_ws)
        raise RuntimeError(
            "StyleGAN checkpoint exposes neither G.num_ws nor "
            "G.synthesis.num_ws; cannot derive mix_dims."
        )
    finally:
        del gen


__all__ = ["build_image_backend"]
