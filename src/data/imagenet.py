"""ImageNet data loading utilities.

Shared by experiments and demos. Streams from HuggingFace
``ILSVRC/imagenet-1k`` (requires agreement to the license on HF Hub).

Usage::

    labels = load_imagenet_labels()
    samples = stream_imagenet_val(labels, max_images=10)
    for s in samples:
        img, name = s["image"], labels[s["idx"]]
"""

from __future__ import annotations

import json
import urllib.request
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

IMAGENET_LABELS_URL = (
    "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels"
    "/master/imagenet-simple-labels.json"
)

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "imagenet"


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


def stream_imagenet_val(
    labels: list[str],
    *,
    categories: list[str] | None = None,
    max_images: int | None = None,
    max_per_class: int | None = None,
) -> list[dict]:
    """Stream ImageNet validation images from HuggingFace.

    Args:
        labels: Full label list from ``load_imagenet_labels()``.
        categories: If given, only yield images whose label is in this list.
        max_images: Stop after this many images total.
        max_per_class: Cap per class (useful for balanced sampling).

    Returns:
        List of dicts with ``"image"`` (PIL RGB) and ``"idx"`` (int class index).
    """
    if categories is not None:
        category_indices = {labels.index(c) for c in categories}
    else:
        category_indices = None

    ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)

    samples = []
    class_counts: dict[int, int] = defaultdict(int)
    for sample in tqdm(ds, desc="Streaming ImageNet val", total=50_000):
        label_idx = sample["label"]

        if category_indices is not None and label_idx not in category_indices:
            continue

        if max_per_class and class_counts[label_idx] >= max_per_class:
            continue
        class_counts[label_idx] += 1

        samples.append({
            "image": sample["image"].convert("RGB"),
            "idx": label_idx,
        })

        if max_images and len(samples) >= max_images:
            break

    return samples
