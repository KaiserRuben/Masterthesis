"""ImageNet data loading utilities.

Shared by experiments and demos. Streams from HuggingFace
``ILSVRC/imagenet-1k`` (requires agreement to the license on HF Hub).

:class:`ImageNetCache` manages one *primary* (writable) cache directory
and zero or more *fallback* (read-only) directories.  When loading
images the cache merges results from all locations (primary first),
deduplicating by filename.  New images are always written to the
primary directory.

Usage::

    from src.data.imagenet import ImageNetCache

    cache = ImageNetCache()                       # ~/.cache/imagenet
    cache = ImageNetCache(                        # with external drive
        fallbacks=[Path("/Volumes/SanDisk/Cache/imagenet")],
    )
    samples = cache.load_samples(["macaw", "peacock"], n_per_class=5)
"""

from __future__ import annotations

import json
import logging
import urllib.request
from collections.abc import Sequence
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


# ---------------------------------------------------------------------------
# Multi-directory cache
# ---------------------------------------------------------------------------


class ImageNetCache:
    """Multi-location ImageNet image cache.

    Scans all directories once at construction and builds an in-memory
    index ``{safe_category_name: [Path, ...]}``.  The first directory
    is the *primary* (writable); the rest are read-only fallbacks that
    are silently skipped when not mounted.

    :param dirs: Cache directories.  First = primary (writable),
        rest = read-only fallbacks.  Empty → ``".cache/imagenet"``
        relative to the current working directory.
    """

    def __init__(self, dirs: Sequence[Path] = ()) -> None:
        resolved = list(dirs) if dirs else [Path(".cache") / "imagenet"]
        self._primary = resolved[0]
        self._fallbacks = tuple(resolved[1:])
        self._index: dict[str, list[Path]] = self._scan()
        active = [d for d in self._fallbacks if d.exists()]
        if active:
            logger.info(
                "ImageNet cache fallbacks: %s",
                ", ".join(str(d) for d in active),
            )

    # -- Index build --------------------------------------------------------

    @staticmethod
    def _safe_name(category: str) -> str:
        return category.replace(" ", "_").lower()

    def _scan(self) -> dict[str, list[Path]]:
        """Scan all cache dirs once; build safe_name → sorted paths index."""
        index: dict[str, list[Path]] = {}
        for d in (self._primary, *self._fallbacks):
            images_root = d / "category_images"
            if not images_root.exists():
                continue
            for cat_dir in sorted(images_root.iterdir()):
                if not cat_dir.is_dir():
                    continue
                safe = cat_dir.name
                seen = {p.name for p in index.get(safe, [])}
                for p in sorted(cat_dir.glob("*.png")):
                    if p.name.startswith("._"):
                        continue
                    if p.name not in seen:
                        seen.add(p.name)
                        index.setdefault(safe, []).append(p)
        for paths in index.values():
            paths.sort(key=lambda p: p.name)
        return index

    def _cached(self, category: str) -> list[Path]:
        """Return the *live* list for *category* (creates one if absent)."""
        safe = self._safe_name(category)
        if safe not in self._index:
            self._index[safe] = []
        return self._index[safe]

    # -- Labels -------------------------------------------------------------

    def labels(self) -> list[str]:
        """Load the 1000 ImageNet class labels (download once to primary)."""
        for d in (self._primary, *self._fallbacks):
            f = d / "imagenet_labels.json"
            if f.exists():
                return json.loads(f.read_text())
        self._primary.mkdir(parents=True, exist_ok=True)
        dest = self._primary / "imagenet_labels.json"
        urllib.request.urlretrieve(IMAGENET_LABELS_URL, dest)
        return json.loads(dest.read_text())

    # -- Write (primary only) -----------------------------------------------

    def _save_image(
        self, category: str, image: Image.Image, index: int,
    ) -> Path:
        """Save *image* to the primary cache under *category*."""
        cat_dir = self._primary / "category_images" / self._safe_name(category)
        cat_dir.mkdir(parents=True, exist_ok=True)
        path = cat_dir / f"{index:05d}.png"
        image.save(path)
        return path

    # -- Public API ---------------------------------------------------------

    def load_samples(
        self,
        categories: list[str],
        n_per_class: int,
    ) -> list[ImageSample]:
        """Load ImageNet validation images by category, with caching.

        Merges images from all cache directories (primary + fallbacks).
        Streams from HuggingFace only if any category still needs more
        images than are available across all directories.
        """
        labels = self.labels()
        if not categories:
            categories = labels
        cat_to_idx = {cat: labels.index(cat) for cat in categories}
        cached = {cat: self._cached(cat) for cat in categories}
        need = {
            cat: n_per_class - len(paths)
            for cat, paths in cached.items()
            if len(paths) < n_per_class
        }

        if need:
            total_cached = sum(len(v) for v in cached.values())
            total_needed = sum(need.values())
            n_dirs = 1 + sum(1 for d in self._fallbacks if d.exists())
            logger.info(
                f"Cache has {total_cached} images across {n_dirs} dir(s). "
                f"Need {total_needed} more across {len(need)} categories. "
                f"Streaming ImageNet..."
            )
            self._stream_and_cache(labels, need, cached)
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
        self,
        labels: list[str],
        need: dict[str, int],
        cached: dict[str, list[Path]],
    ) -> None:
        """Stream ImageNet val and cache missing images to the primary dir.

        Appends to the *cached* lists in place — since ``_cached()``
        returns the live index list, the index stays in sync.
        """
        cat_indices = {labels.index(cat): cat for cat in need}
        remaining = dict(need)

        ds = load_dataset(
            "ILSVRC/imagenet-1k", split="validation", streaming=True,
        )

        for sample in tqdm(ds, desc="Streaming ImageNet val", total=50_000):
            label_idx = sample["label"]
            if label_idx not in cat_indices:
                continue

            cat = cat_indices[label_idx]
            if remaining[cat] <= 0:
                continue

            image = sample["image"].convert("RGB")
            next_idx = len(cached[cat])
            path = self._save_image(cat, image, next_idx)
            cached[cat].append(path)
            remaining[cat] -= 1

            if all(v <= 0 for v in remaining.values()):
                break


# ---------------------------------------------------------------------------
# Convenience wrappers (backward-compatible module-level API)
# ---------------------------------------------------------------------------


def load_imagenet_labels(cache_dir: Path = _DEFAULT_CACHE_DIR) -> list[str]:
    """Download and cache the 1000 ImageNet class labels."""
    return ImageNetCache(dirs=[cache_dir]).labels()


def load_samples(
    categories: list[str],
    n_per_class: int,
    cache_dir: Path = _DEFAULT_CACHE_DIR,
) -> list[ImageSample]:
    """Load samples using a single cache directory.

    For multi-directory support use :class:`ImageNetCache` directly.
    """
    return ImageNetCache(dirs=[cache_dir]).load_samples(categories, n_per_class)


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
