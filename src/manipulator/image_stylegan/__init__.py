"""StyleGAN-XL image manipulator backend.

Plug-compatible with :class:`src.manipulator.image.ImageManipulator` via
the :class:`src.manipulator.image_backend.ImageBackend` protocol. See
:mod:`src.manipulator.image_stylegan.manipulator` for the entry point.
"""

from .class_target import (
    ACCEPTED_SEED_KEY_PREFIX,
    MODAL_W_KEY_PREFIX,
    StyleGANClassTargetBuilder,
    build_class_modal_w,
    find_pair_dominant_origin_seed,
    safe_class_name,
    sut_signature,
    tensor_to_pil,
)
from .loading import (
    checkpoint_sha256_hex,
    ensure_checkpoint,
    load_generator,
    resolve_checkpoint_path,
)
from .manipulator import StyleGANImageManipulator
from .sut_adapter import VLMSUTLogprobsAdapter
from .types import StyleGANManipulationContext

__all__ = [
    "ACCEPTED_SEED_KEY_PREFIX",
    "MODAL_W_KEY_PREFIX",
    "StyleGANClassTargetBuilder",
    "StyleGANImageManipulator",
    "StyleGANManipulationContext",
    "VLMSUTLogprobsAdapter",
    "build_class_modal_w",
    "checkpoint_sha256_hex",
    "ensure_checkpoint",
    "find_pair_dominant_origin_seed",
    "load_generator",
    "resolve_checkpoint_path",
    "safe_class_name",
    "sut_signature",
    "tensor_to_pil",
]
