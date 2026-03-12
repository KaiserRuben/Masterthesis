"""Data source protocol for category-labelled image datasets."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .imagenet import ImageSample


@runtime_checkable
class DataSource(Protocol):
    """Interface for image datasets that provide labelled categories.

    Any data source (ImageNet, etc.) that can enumerate its labels
    and load samples by category should satisfy this protocol.
    """

    def labels(self) -> list[str]:
        """Return all available category labels."""
        ...

    def load_samples(
        self, categories: list[str], n_per_class: int,
    ) -> list[ImageSample]:
        """Load images for the given categories."""
        ...
