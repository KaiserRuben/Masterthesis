"""
Abstract Base Class for Embedding Providers

All providers must implement:
- name: Unique identifier
- embedding_dim: Output dimension
- embed_images: Batch image embedding
- embed_texts: Batch text embedding
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class EmbeddingResult:
    """Container for embedding results."""
    embeddings: np.ndarray  # (N, D)
    scene_ids: list[str]
    model_name: str
    embedding_dim: int


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this provider."""
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Output embedding dimension."""
        pass

    @abstractmethod
    def embed_images(self, image_paths: list[str], batch_size: int = 8) -> np.ndarray:
        """
        Embed batch of images.

        Args:
            image_paths: List of paths to images
            batch_size: Batch size for processing

        Returns:
            Normalized embeddings of shape (N, D)
        """
        pass

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embed batch of texts.

        Args:
            texts: List of text strings

        Returns:
            Normalized embeddings of shape (N, D)
        """
        pass
