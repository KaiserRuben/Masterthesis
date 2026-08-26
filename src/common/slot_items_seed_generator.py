"""Explicit sentence-slot seed source (Exp-105).

Turns hand-written ``(image, carrier template, fillers, pair)`` items into
:class:`SeedTriple` instances. Unlike the ImageNet-driven generators
(``gap_filter`` / ``roster``) nothing is searched or scored here: the
contrast set is written out in the config, so this module only resolves
image paths and instantiates the carrier sentences.

Candidate semantics: each filler is substituted into the carrier's
``{slot}``, producing a FULL sentence. Those sentences are the scored
categories — the shared prefix cancels exactly in the contrast, so the
suffix residue is the signal. ``class_a`` / ``class_b`` of the emitted
triple are the two sentences named by the item's ``pair``; the complete
instantiated list travels in ``metadata["candidates"]``, which the
evolutionary tester reads as the per-seed scoring scope (see
``src.evolutionary.vlm_boundary_tester.effective_candidates``).

One item → one seed, in config order, so ``seeds.filter_indices``
addresses items positionally.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from src.config import SLOT_PLACEHOLDER, ExperimentConfig, SeedTriple, SlotItem

logger = logging.getLogger(__name__)

# Repo root — relative item paths in a config are resolved against it so a
# config reads the same regardless of the CWD a run is launched from.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]


def instantiate(template: str, filler: str) -> str:
    """Substitute *filler* into the carrier template's ``{slot}``.

    Literal replace, not ``str.format`` — templates may contain other
    braces (JSON-ish carriers) without escaping them.
    """
    return template.replace(SLOT_PLACEHOLDER, filler)


def item_candidates(item: SlotItem) -> tuple[str, ...]:
    """All instantiated candidate sentences, in ``fillers`` order."""
    return tuple(instantiate(item.template, f) for f in item.fillers)


def item_pair(item: SlotItem) -> tuple[str, str]:
    """The two instantiated sentences named by the item's ``pair`` fillers."""
    a, b = item.pair
    return instantiate(item.template, a), instantiate(item.template, b)


def resolve_item_image(raw: Path | str) -> Path:
    """Resolve an item's image path, failing fast if it does not exist.

    Accepts absolute paths, ``~``-prefixed paths, and paths relative to
    the repo root (also tried relative to the CWD, so ad-hoc invocations
    from a subdirectory still work).

    :raises FileNotFoundError: With every location that was tried.
    """
    p = Path(raw).expanduser()
    if p.is_absolute():
        if p.is_file():
            return p
        raise FileNotFoundError(
            f"slot_items image not found: {p}"
        )
    tried: list[Path] = []
    for cand in (REPO_ROOT / p, Path.cwd() / p):
        if cand in tried:
            continue
        if cand.is_file():
            return cand
        tried.append(cand)
    raise FileNotFoundError(
        f"slot_items image {str(raw)!r} not found. Tried: "
        + ", ".join(str(t) for t in tried)
        + ". Paths are resolved relative to the repo root."
    )


def load_item_image(path: Path) -> Image.Image:
    """Load a seed image as RGB.

    No resizing here: the VQGAN codec resizes + center-crops to its own
    resolution on encode (see :meth:`VQGANCodec.preprocess`), exactly as
    for the refcocoplus adapter.
    """
    return Image.open(path).convert("RGB")


def build_seed_triples(
    items: Sequence[SlotItem],
    images: Sequence[Image.Image],
    image_paths: Sequence[Path] | None = None,
) -> list[SeedTriple]:
    """Pure transform: items + loaded images -> ``list[SeedTriple]``.

    :param items: Config items (already validated by :class:`SlotItem`).
    :param images: One loaded PIL image per item, same order.
    :param image_paths: Optional resolved paths, recorded in metadata for
        provenance. Falls back to the raw config paths.
    """
    if len(items) != len(images):
        raise ValueError(
            f"build_seed_triples: {len(items)} items but {len(images)} images."
        )
    paths = list(image_paths) if image_paths is not None else [
        Path(it.image) for it in items
    ]
    seeds: list[SeedTriple] = []
    for idx, (item, image, path) in enumerate(zip(items, images, paths)):
        candidates = item_candidates(item)
        cand_a, cand_b = item_pair(item)
        meta: dict[str, Any] = {
            # Read by the tester: per-seed scoring scope + answer options.
            "candidates": list(candidates),
            "template": item.template,
            "fillers": list(item.fillers),
            "pair_fillers": list(item.pair),
            "filler_a": item.pair[0],
            "filler_b": item.pair[1],
            "image_path": str(path),
            "item_idx": idx,
        }
        seeds.append(SeedTriple(
            image=image,
            class_a=cand_a,
            class_b=cand_b,
            metadata=meta,
        ))
    return seeds


def slot_items_seeds(
    sut: Any, exp_cfg: ExperimentConfig, data_source: Any,
) -> list[SeedTriple]:
    """Entry point matching the other seed generators' signature.

    *sut* and *data_source* are unused — slot items are fully specified in
    the config, so no scoring or ImageNet lookup happens.

    :raises ValueError: If the mode / block combination is wrong, or the
        image backend needs ImageNet class targets (cone filter, StyleGAN)
        which slot items cannot supply.
    :raises FileNotFoundError: If any item's image is missing.
    """
    if exp_cfg.seeds.mode != "slot_items":
        raise ValueError(
            f"slot_items_seeds() requires seeds.mode='slot_items'; "
            f"got {exp_cfg.seeds.mode!r}."
        )
    cfg = exp_cfg.seeds.slot_items
    if cfg is None:  # pragma: no cover — guarded by SeedConfig.__post_init__
        raise ValueError(
            "seeds.mode='slot_items' requires a seeds.slot_items config block."
        )
    _reject_class_target_backends(exp_cfg)

    items = list(cfg.items)
    paths = [resolve_item_image(it.image) for it in items]
    images = [load_item_image(p) for p in paths]
    seeds = build_seed_triples(items, images, paths)
    logger.info(
        "slot_items: %d seed(s) — %s",
        len(seeds),
        "; ".join(
            f"[{i}] {p.name}: {s.class_a!r} vs {s.class_b!r}"
            for i, (p, s) in enumerate(zip(paths, seeds))
        ),
    )
    return seeds


def _reject_class_target_backends(exp_cfg: ExperimentConfig) -> None:
    """Fail fast on image backends that need an ImageNet L0 target class.

    Both the VQGAN cone filter and StyleGAN-XL resolve the seed's target
    class against the ImageNet cache to build a modal target. A slot
    item's ``class_b`` is a carrier sentence, so those lookups would fail
    deep inside the precompute — much later, with a far worse message.
    """
    if (
        exp_cfg.image.backend == "vqgan_codebook"
        and exp_cfg.image.cone_filter.enabled
    ):
        raise ValueError(
            "seeds.mode='slot_items' is incompatible with "
            "image.cone_filter.enabled=true: the cone's modal target is "
            "built from an ImageNet class, but slot-item candidates are "
            "carrier sentences. Set image.cone_filter.enabled: false."
        )
    if exp_cfg.image.backend == "stylegan_xl":
        raise ValueError(
            "seeds.mode='slot_items' is incompatible with "
            "image.backend='stylegan_xl': StyleGAN synthesis is "
            "conditioned on an ImageNet class, but slot-item candidates "
            "are carrier sentences. Use image.backend: vqgan_codebook."
        )


__all__ = [
    "REPO_ROOT",
    "build_seed_triples",
    "instantiate",
    "item_candidates",
    "item_pair",
    "load_item_image",
    "resolve_item_image",
    "slot_items_seeds",
]
