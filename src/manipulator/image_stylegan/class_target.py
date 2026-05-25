"""Class-target builders for the StyleGAN backend.

Two artefacts must be available before the optimizer can run for a seed:

1. **Modal target w** — the class-conditional mean of ``m`` w-samples for
   the seed's target class. This is what every (κ → 1) interpolation
   moves toward.
2. **Pair-dominant origin seed** — the ``seed_int`` whose class-conditional
   synthetic sample (with the same truncation settings) is scored higher
   for the origin class than for the target class by the *same* SUT that
   will score the run. This guarantees the κ=0 endpoint sits on the
   origin side of the (origin, target) decision boundary the optimizer is
   trying to cross — which is the only property the boundary objective
   (``TargetedBalance = |P(A) − P(B)|``) actually needs. The earlier
   strict ``class ∈ top-K(50)`` rule rejected synthetic origins that were
   perfectly valid κ=0 endpoints for the specific pair, just not 50-way
   winners.

Both artefacts are cached at two layers:

* L1 (in-memory dict) keyed by class name (modal-w) or by
  ``(origin_class, target_class)`` pair (accepted seed), on the builder
  instance.
* L2 (Redis bytes) keyed by ``(checkpoint_hash, …, sut_signature)``.

The functions in this module are pure-ish: they receive the generator
and SUT as *protocols* (see :class:`SupportsGetW` / :class:`SupportsLogprobs`),
so unit tests inject mocks. The wrapping class
:class:`StyleGANClassTargetBuilder` owns the L1 + L2 plumbing.

No defensive try/except wrappers: if a pair cannot pass the precheck
within ``max_attempts``, callers should surface that error. The Redis
helper is intentionally graceful (returns ``None`` on miss) — that is
the contract of :class:`BytesRedisCache`, not a workaround.
"""

from __future__ import annotations

import hashlib
import io
import logging
import struct
from typing import Iterable, Mapping, Protocol

import numpy as np
import torch

logger = logging.getLogger(__name__)


# Top-level key prefixes — bumped together with the schema version when
# the cache layout changes.
MODAL_W_KEY_PREFIX = "stylegan:class_target"
ACCEPTED_SEED_KEY_PREFIX = "stylegan:origin_seed"


# ---------------------------------------------------------------------------
# Protocols (test-injectable surfaces)
# ---------------------------------------------------------------------------


class SupportsGetW(Protocol):
    """Minimum the generator wrapper needs to expose.

    Real :class:`smoo.manipulator.style_gan_manipulator.StyleGANManipulator`
    satisfies this; tests inject a stub.
    """

    def get_w(self, seed: int, class_idx: int, batch_size: int = 1) -> torch.Tensor: ...

    def get_images(self, w: torch.Tensor) -> torch.Tensor: ...


class SupportsLogprobs(Protocol):
    """Minimum the SUT precheck needs.

    Implementations accept a PIL image and a candidate label set and
    return a mapping from candidate name to log-probability under the
    same prompt + scoring contract the optimizer will use. The pairwise
    precheck looks up only the origin and target entries, but the full
    contrast set is passed so the SUT's normalisation matches the
    optimizer's runtime scoring exactly.
    """

    def predict_logprobs(
        self, image, candidates: tuple[str, ...],
    ) -> Mapping[str, float]: ...


# ---------------------------------------------------------------------------
# Image conversion (StyleGAN tensor -> PIL)
# ---------------------------------------------------------------------------


def tensor_to_pil(img_tensor: torch.Tensor):
    """Convert a SMOO StyleGAN output (CxHxW, [0,1] float) to a PIL image.

    The SMOO generator's :meth:`get_images` returns ``BxCxHxW`` in
    ``[0, 1]``. Single-image conversion strips the batch dim and casts
    to 8-bit; deferred import of PIL.Image is kept local to avoid
    pulling Pillow into modules that don't need it.

    :param img_tensor: Tensor of shape ``(1, C, H, W)`` or ``(C, H, W)``.
    :returns: PIL Image in mode ``RGB``.
    """
    from PIL import Image  # noqa: PLC0415

    if img_tensor.ndim == 4:
        img_tensor = img_tensor[0]
    arr = (
        img_tensor.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .to(torch.uint8)
        .permute(1, 2, 0)  # CHW -> HWC
        .cpu()
        .numpy()
    )
    return Image.fromarray(arr, mode="RGB")


# ---------------------------------------------------------------------------
# Serialization (torch w tensors round-trip via numpy)
# ---------------------------------------------------------------------------


def _w_to_bytes(w: torch.Tensor) -> bytes:
    """Serialise a w-tensor via ``np.save`` (float32, shape preserved)."""
    buf = io.BytesIO()
    np.save(buf, w.detach().cpu().float().numpy(), allow_pickle=False)
    return buf.getvalue()


def _w_from_bytes(blob: bytes, device: torch.device) -> torch.Tensor:
    """Recover a w-tensor from the ``_w_to_bytes`` payload."""
    arr = np.load(io.BytesIO(blob), allow_pickle=False)
    return torch.from_numpy(arr).to(device=device, dtype=torch.float32)


def _seed_to_bytes(seed_int: int) -> bytes:
    """Pack an int64 seed as 8 bytes (big-endian for stable lex sort)."""
    return struct.pack(">q", int(seed_int))


def _seed_from_bytes(blob: bytes) -> int:
    """Unpack the :func:`_seed_to_bytes` payload."""
    return int(struct.unpack(">q", blob)[0])


# ---------------------------------------------------------------------------
# SUT signature (cache key disambiguator)
# ---------------------------------------------------------------------------


def sut_signature(model_id: str, categories: Iterable[str], extra: str = "") -> str:
    """Stable 16-char hash of the SUT's distinguishing config.

    The accepted-origin cache must NOT be shared between different SUT
    setups (different model id, different scoring contrast set). This
    helper builds a compact, deterministic suffix from the model id, the
    sorted category tuple, and any additional caller-supplied string
    (e.g. preprocessing config).
    """
    h = hashlib.sha256()
    h.update(model_id.encode())
    h.update(b"\0")
    h.update(",".join(sorted(categories)).encode())
    h.update(b"\0")
    h.update(extra.encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Safe class name (same convention as VQGAN cache)
# ---------------------------------------------------------------------------


def safe_class_name(category: str) -> str:
    """Match :func:`src.manipulator.image.class_target._safe_class_name`."""
    return category.replace(" ", "_").lower()


# ---------------------------------------------------------------------------
# Pure functions (mock-injectable)
# ---------------------------------------------------------------------------


def build_class_modal_w(
    *,
    generator: SupportsGetW,
    class_idx: int,
    m: int,
    rng_seed: int = 0,
) -> torch.Tensor:
    """Average ``m`` class-conditional w-samples for ``class_idx``.

    Each sample uses a deterministic seed sequence ``rng_seed, rng_seed+1,
    …``, so a cache-miss rebuild is reproducible. The averaging operates
    in w-space (not z-space) because the smoo manipulator already
    handles truncation inside ``get_w``.

    :param generator: Anything satisfying :class:`SupportsGetW`.
    :param class_idx: Integer class label passed to the generator.
    :param m: Number of samples to average. Must be ``>= 1``.
    :param rng_seed: Starting seed for the deterministic sample sequence.
    :returns: Tensor of shape ``(1, num_ws, w_dim)`` — the modal target w.
    :raises ValueError: If ``m < 1``.
    """
    if m < 1:
        raise ValueError(f"build_class_modal_w: m must be >= 1, got {m}")
    accum: torch.Tensor | None = None
    for i in range(m):
        w = generator.get_w(seed=rng_seed + i, class_idx=class_idx)
        accum = w.clone() if accum is None else accum + w
    assert accum is not None
    return accum / float(m)


def find_pair_dominant_origin_seed(
    *,
    generator: SupportsGetW,
    sut: SupportsLogprobs,
    origin_class: str,
    target_class: str,
    origin_class_idx: int,
    candidates: tuple[str, ...],
    max_attempts: int,
) -> tuple[int, torch.Tensor]:
    """Search for a seed_int whose synthetic sample beats ``target_class``.

    Iterates ``seed_int = 0, 1, 2, …``; for each one, generates the w
    via ``generator.get_w`` with the *origin* class index, renders the
    image, and asks ``sut.predict_logprobs`` for the per-candidate
    log-probabilities. Accepts the first seed where
    ``lp[origin_class] > lp[target_class]`` — i.e. the κ=0 endpoint sits
    on the origin side of the (origin, target) boundary, which is the
    only property the run actually needs.

    :param generator: :class:`SupportsGetW`-compatible object.
    :param sut: :class:`SupportsLogprobs`-compatible scorer.
    :param origin_class: Class label whose log-prob must exceed the
        target's at the accepted seed.
    :param target_class: Class label the origin must beat. Must differ
        from ``origin_class``.
    :param origin_class_idx: Integer class index the generator uses to
        condition the synthetic sample.
    :param candidates: Full scoring contrast set (typically the run's
        ``categories`` tuple). Both ``origin_class`` and ``target_class``
        must be in it.
    :param max_attempts: Maximum seeds to try before giving up.
    :returns: ``(accepted_seed_int, w_tensor)``.
    :raises RuntimeError: If no seed accepted within ``max_attempts``.
    :raises ValueError: For degenerate arguments.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
    if origin_class == target_class:
        raise ValueError(
            f"origin_class and target_class must differ; both were "
            f"{origin_class!r}. A same-class pair has no boundary to test."
        )
    if origin_class not in candidates or target_class not in candidates:
        raise ValueError(
            f"Both origin_class={origin_class!r} and target_class={target_class!r} "
            f"must be in candidates (got {len(candidates)} entries)."
        )

    for seed_int in range(max_attempts):
        w = generator.get_w(seed=seed_int, class_idx=origin_class_idx)
        imgs = generator.get_images(w)
        img = tensor_to_pil(imgs)
        lp = sut.predict_logprobs(img, candidates)
        if lp[origin_class] > lp[target_class]:
            return seed_int, w
    raise RuntimeError(
        f"StyleGAN pair precheck failed: no synthetic origin for "
        f"{origin_class!r} (class_idx={origin_class_idx}) beat the target "
        f"{target_class!r} after {max_attempts} attempts. Either raise "
        "sut_precheck_max_attempts or drop this pair from the seed set."
    )


# ---------------------------------------------------------------------------
# Cache layer (L1 dict + L2 BytesRedisCache)
# ---------------------------------------------------------------------------


class StyleGANClassTargetBuilder:
    """Build + cache class-modal w-vectors and pair-dominant origin seeds.

    Two-layer cache as in :class:`src.manipulator.image.class_target.ModalTargetBuilder`:

    * L1 (per-instance) — three dicts keyed by class name (modal-w) or
      by ``(origin, target)`` pair (accepted seed + materialised origin
      w + origin image).
    * L2 (Redis bytes) — keys namespaced by checkpoint hash + SUT signature
      so different generators / scorers do not share entries.

    :param generator: SMOO StyleGAN manipulator (object satisfying
        :class:`SupportsGetW`).
    :param sut: SUT wrapper satisfying :class:`SupportsLogprobs`.
    :param checkpoint_hash: SHA-256-prefix identifier of the loaded
        checkpoint (see :func:`.loading.checkpoint_sha256_hex`).
    :param sut_signature: Disambiguator used in the L2 keys for the
        accepted-origin cache (see :func:`sut_signature`).
    :param categories: SUT contrast set forwarded to the precheck.
    :param target_m: Modal-w sample count.
    :param truncation_psi: Forwarded to L2 cache key only (the generator
        already holds the value; the key namespaces caches by it).
    :param truncation_cutoff: Same — only used in the L2 key.
    :param max_attempts: Per-pair precheck attempt cap.
    :param redis_cache: Optional :class:`BytesRedisCache` (None disables L2).
    :param device: Torch device, used to re-hydrate cached w tensors.
    :param class_name_to_idx: Lookup from string label to generator class
        index. Used by the public :meth:`ensure_*` methods.
    """

    def __init__(
        self,
        *,
        generator: SupportsGetW,
        sut: SupportsLogprobs,
        checkpoint_hash: str,
        sut_signature: str,
        categories: tuple[str, ...],
        target_m: int,
        truncation_psi: float,
        truncation_cutoff: int,
        max_attempts: int,
        redis_cache=None,
        device: torch.device | None = None,
        class_name_to_idx: dict[str, int] | None = None,
    ) -> None:
        self._gen = generator
        self._sut = sut
        self._ckpt_hash = checkpoint_hash
        self._sut_sig = sut_signature
        self._categories = tuple(categories)
        self._target_m = int(target_m)
        self._trunc_psi = float(truncation_psi)
        self._trunc_cutoff = int(truncation_cutoff)
        self._max_attempts = int(max_attempts)
        self._redis = redis_cache
        self._device = device or torch.device("cpu")
        self._name_to_idx = dict(class_name_to_idx or {})

        # L1 caches.
        self._modal_w_l1: dict[str, torch.Tensor] = {}
        self._origin_seed_l1: dict[tuple[str, str], int] = {}
        self._origin_w_l1: dict[tuple[str, str], torch.Tensor] = {}
        self._origin_image_l1: dict[tuple[str, str], "object"] = {}  # PIL.Image.Image

    # ------------------------------------------------------------------
    # Class-index resolution
    # ------------------------------------------------------------------

    def class_idx(self, class_name: str) -> int:
        """Map a class label to the generator's integer class index.

        Falls back to the position in the ``categories`` tuple if the
        explicit ``class_name_to_idx`` map omits the class (works when
        the SUT contrast set IS the ImageNet label list in ImageNet order).
        """
        if class_name in self._name_to_idx:
            return self._name_to_idx[class_name]
        if class_name in self._categories:
            return self._categories.index(class_name)
        raise KeyError(
            f"StyleGAN class index unknown for {class_name!r}; supply "
            f"class_name_to_idx or ensure the label is in the categories tuple."
        )

    # ------------------------------------------------------------------
    # Cache keys
    # ------------------------------------------------------------------

    def modal_w_key(self, class_name: str) -> str:
        safe = safe_class_name(class_name)
        return (
            f"{MODAL_W_KEY_PREFIX}:{self._ckpt_hash}:{safe}:m{self._target_m}"
            f":trunc{self._trunc_psi}:cut{self._trunc_cutoff}"
        )

    def accepted_seed_key(self, origin_class: str, target_class: str) -> str:
        safe_o = safe_class_name(origin_class)
        safe_t = safe_class_name(target_class)
        return (
            f"{ACCEPTED_SEED_KEY_PREFIX}:{self._ckpt_hash}:{safe_o}"
            f":vs_{safe_t}:{self._sut_sig}"
        )

    # ------------------------------------------------------------------
    # Modal w
    # ------------------------------------------------------------------

    def ensure_modal_w(self, class_name: str) -> torch.Tensor:
        """Return the modal target w for ``class_name``, building on miss."""
        if class_name in self._modal_w_l1:
            return self._modal_w_l1[class_name]
        if self._redis is not None:
            blob = self._redis.get(self.modal_w_key(class_name))
            if blob is not None:
                w = _w_from_bytes(blob, self._device)
                self._modal_w_l1[class_name] = w
                return w
        logger.info(
            "StyleGAN modal-w cache miss for %r — averaging %d samples...",
            class_name, self._target_m,
        )
        w = build_class_modal_w(
            generator=self._gen,
            class_idx=self.class_idx(class_name),
            m=self._target_m,
        ).to(self._device)
        if self._redis is not None:
            self._redis.set(self.modal_w_key(class_name), _w_to_bytes(w))
        self._modal_w_l1[class_name] = w
        return w

    # ------------------------------------------------------------------
    # Pair-dominant origin seed + materialised origin w + origin image
    # ------------------------------------------------------------------

    def ensure_origin(
        self, origin_class: str, target_class: str,
    ) -> tuple[int, torch.Tensor, "object"]:
        """Return ``(seed_int, origin_w, origin_image)`` for an (origin, target) pair.

        On L1 hit returns the cached triple. On L1 miss + L2 hit, replays
        the seed through the generator (so the w + image are reproducible
        without a fresh SUT round trip). On full miss, runs the precheck
        loop and writes both layers.
        """
        key = (origin_class, target_class)
        if key in self._origin_seed_l1:
            return (
                self._origin_seed_l1[key],
                self._origin_w_l1[key],
                self._origin_image_l1[key],
            )

        cached_seed: int | None = None
        if self._redis is not None:
            blob = self._redis.get(self.accepted_seed_key(origin_class, target_class))
            if blob is not None:
                cached_seed = _seed_from_bytes(blob)

        origin_class_idx = self.class_idx(origin_class)
        if cached_seed is not None:
            # Replay the cached seed to recover the w / image without SUT cost.
            logger.info(
                "StyleGAN accepted-seed cache hit for (%r vs %r): seed_int=%d",
                origin_class, target_class, cached_seed,
            )
            w = self._gen.get_w(seed=cached_seed, class_idx=origin_class_idx)
            imgs = self._gen.get_images(w)
            img = tensor_to_pil(imgs)
            seed_int = cached_seed
        else:
            logger.info(
                "StyleGAN accepted-seed cache miss for (%r vs %r) — "
                "running pairwise precheck (max %d attempts)...",
                origin_class, target_class, self._max_attempts,
            )
            seed_int, w = find_pair_dominant_origin_seed(
                generator=self._gen,
                sut=self._sut,
                origin_class=origin_class,
                target_class=target_class,
                origin_class_idx=origin_class_idx,
                candidates=self._categories,
                max_attempts=self._max_attempts,
            )
            imgs = self._gen.get_images(w)
            img = tensor_to_pil(imgs)
            if self._redis is not None:
                self._redis.set(
                    self.accepted_seed_key(origin_class, target_class),
                    _seed_to_bytes(seed_int),
                )

        w = w.to(self._device)
        self._origin_seed_l1[key] = seed_int
        self._origin_w_l1[key] = w
        self._origin_image_l1[key] = img
        return seed_int, w, img

    # ------------------------------------------------------------------
    # Bulk pre-population helpers (parity with ModalTargetBuilder)
    # ------------------------------------------------------------------

    def populate_modal_w(self, class_names: Iterable[str]) -> None:
        for name in class_names:
            self.ensure_modal_w(name)

    def populate_origins(self, pairs: Iterable[tuple[str, str]]) -> None:
        """Precompute origin seeds for each ``(origin, target)`` pair.

        Pairs are deduplicated; same-class pairs (no boundary to test) are
        silently skipped so callers can pass raw seed lists without filtering.
        """
        seen: set[tuple[str, str]] = set()
        for origin, target in pairs:
            if origin == target:
                continue
            if (origin, target) in seen:
                continue
            seen.add((origin, target))
            self.ensure_origin(origin, target)


__all__ = [
    "ACCEPTED_SEED_KEY_PREFIX",
    "MODAL_W_KEY_PREFIX",
    "StyleGANClassTargetBuilder",
    "SupportsGetW",
    "SupportsLogprobs",
    "build_class_modal_w",
    "find_pair_dominant_origin_seed",
    "safe_class_name",
    "sut_signature",
    "tensor_to_pil",
]
