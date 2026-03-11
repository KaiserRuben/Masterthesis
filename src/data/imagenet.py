"""ImageNet data loading utilities.

Shared by experiments and demos. Streams from HuggingFace
``ILSVRC/imagenet-1k`` (requires agreement to the license on HF Hub).

Images are cached locally by category so repeated requests for the
same categories avoid re-streaming the full 50k validation set.

Usage::

    from src.data.imagenet import load_samples, ImageSample

    samples = load_samples(["macaw", "peacock"], n_per_class=5)
    for s in samples:
        print(s.class_name, s.image.size)
"""

from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

IMAGENET_LABELS_URL = (
    "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels"
    "/master/imagenet-simple-labels.json"
)

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "imagenet"


@dataclass(frozen=True)
class ImageSample:
    """A single ImageNet validation image with its class metadata."""

    image: Image.Image
    class_idx: int
    class_name: str


def load_imagenet_labels(cache_dir: Path = _DEFAULT_CACHE_DIR) -> list[str]:
    """Download and cache the 1000 ImageNet class labels.

    Returns:
        List of 1000 human-readable label strings, indexed by class ID.
    """
    cache_file = cache_dir / "imagenet_labels.json"
    if not cache_file.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(IMAGENET_LABELS_URL, cache_file)
    return json.loads(cache_file.read_text())


# ---------------------------------------------------------------------------
# Cached category image loading
# ---------------------------------------------------------------------------


def _category_cache_dir(cache_dir: Path, category: str) -> Path:
    """Return the cache directory for a category, using a filesystem-safe name."""
    safe_name = category.replace(" ", "_").lower()
    return cache_dir / "category_images" / safe_name


def _load_cached(cache_dir: Path, category: str) -> list[Path]:
    """Return sorted list of cached PNG paths for a category."""
    cat_dir = _category_cache_dir(cache_dir, category)
    if not cat_dir.exists():
        return []
    return sorted(cat_dir.glob("*.png"))


def _save_to_cache(
    cache_dir: Path, category: str, image: Image.Image, index: int,
) -> Path:
    """Save an image to the category cache. Returns the saved path."""
    cat_dir = _category_cache_dir(cache_dir, category)
    cat_dir.mkdir(parents=True, exist_ok=True)
    path = cat_dir / f"{index:05d}.png"
    image.save(path)
    return path


def load_samples(
    categories: list[str],
    n_per_class: int,
    cache_dir: Path = _DEFAULT_CACHE_DIR,
) -> list[ImageSample]:
    """Load ImageNet validation images by category, with caching.

    Handles label loading internally. For each category, checks the
    cache first. Only streams from HuggingFace if any category needs
    more images than are cached.

    Args:
        categories: Category names to load (must be valid ImageNet labels).
        n_per_class: Number of images needed per category.
        cache_dir: Root cache directory.

    Returns:
        List of :class:`ImageSample` with image, class index, and name.
    """
    labels = load_imagenet_labels(cache_dir)
    cat_to_idx = {cat: labels.index(cat) for cat in categories}
    cached = {cat: _load_cached(cache_dir, cat) for cat in categories}
    need = {
        cat: n_per_class - len(paths)
        for cat, paths in cached.items()
        if len(paths) < n_per_class
    }

    if need:
        total_cached = sum(len(v) for v in cached.values())
        total_needed = sum(need.values())
        logger.info(
            f"Cache has {total_cached} images. "
            f"Need {total_needed} more across {len(need)} categories. "
            f"Streaming ImageNet..."
        )
        _stream_and_cache(labels, need, cached, cache_dir)
    else:
        logger.info(
            f"Cache hit: all {len(categories)} categories have "
            f">= {n_per_class} images."
        )

    return [
        ImageSample(
            image=Image.open(p).convert("RGB"),
            class_idx=cat_to_idx[cat],
            class_name=cat,
        )
        for cat in categories
        for p in cached[cat][:n_per_class]
    ]


def _stream_and_cache(
    labels: list[str],
    need: dict[str, int],
    cached: dict[str, list[Path]],
    cache_dir: Path,
) -> None:
    """Stream ImageNet val and cache images for categories that need more."""
    cat_indices = {labels.index(cat): cat for cat in need}
    remaining = dict(need)

    ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)

    for sample in tqdm(ds, desc="Streaming ImageNet val", total=50_000):
        label_idx = sample["label"]
        if label_idx not in cat_indices:
            continue

        cat = cat_indices[label_idx]
        if remaining[cat] <= 0:
            continue

        image = sample["image"].convert("RGB")
        next_idx = len(cached[cat])
        path = _save_to_cache(cache_dir, cat, image, next_idx)
        cached[cat].append(path)
        remaining[cat] -= 1

        if all(v <= 0 for v in remaining.values()):
            break


# ---------------------------------------------------------------------------
# Low-level streaming (no caching, for experiments that need full control)
# ---------------------------------------------------------------------------


def stream_imagenet_val(
    labels: list[str],
    *,
    categories: list[str] | None = None,
    max_images: int | None = None,
    max_per_class: int | None = None,
) -> list[ImageSample]:
    """Stream ImageNet validation images from HuggingFace (no caching).

    Prefer :func:`load_samples` for category-filtered loading with caching.

    Args:
        labels: Full label list from ``load_imagenet_labels()``.
        categories: If given, only yield images whose label is in this list.
        max_images: Stop after this many images total.
        max_per_class: Cap per class (useful for balanced sampling).

    Returns:
        List of :class:`ImageSample`.
    """
    if categories is not None:
        category_indices = {labels.index(c) for c in categories}
    else:
        category_indices = None

    ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)

    samples: list[ImageSample] = []
    class_counts: dict[int, int] = {}
    for sample in tqdm(ds, desc="Streaming ImageNet val", total=50_000):
        label_idx = sample["label"]

        if category_indices is not None and label_idx not in category_indices:
            continue

        count = class_counts.get(label_idx, 0)
        if max_per_class and count >= max_per_class:
            continue
        class_counts[label_idx] = count + 1

        samples.append(ImageSample(
            image=sample["image"].convert("RGB"),
            class_idx=label_idx,
            class_name=labels[label_idx],
        ))

        if max_images and len(samples) >= max_images:
            break

    return samples
