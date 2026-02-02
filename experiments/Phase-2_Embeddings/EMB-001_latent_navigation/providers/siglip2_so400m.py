"""
SigLIP 2 SO400M Provider

Model: ViT-SO400M-14-SigLIP2 (400M params)
Source: open_clip (webli pretrained)
Embedding dim: 1152
Resolution: 224 (base)
"""

import numpy as np
import torch
import open_clip
from PIL import Image
from tqdm import tqdm

from .base import EmbeddingProvider


class SigLIP2Provider(EmbeddingProvider):
    """SigLIP 2 SO400M embedding provider (400M params).

    Note: Uses CPU due to MPS compatibility issues with SigLIP2.
    """

    MODEL_NAME = "ViT-SO400M-14-SigLIP2"
    RESOLUTION = 224

    def __init__(self, device: str = "cpu"):  # Force CPU due to MPS issues
        # SigLIP2 has MPS incompatibility, force CPU
        self.device = "cpu"
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._embedding_dim = None

    def _load_model(self):
        """Lazy load model on first use."""
        if self._model is not None:
            return

        print(f"Loading SigLIP2 SO400M to {self.device} (MPS incompatible)...")
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.MODEL_NAME,
            pretrained="webli",
            device=self.device,
        )
        self._tokenizer = open_clip.get_tokenizer(self.MODEL_NAME)
        self._model.eval()

        # Infer embedding dim from dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.RESOLUTION, self.RESOLUTION, device=self.device)
            self._embedding_dim = self._model.encode_image(dummy).shape[-1]

        print(f"Model loaded. Embedding dim: {self._embedding_dim}")

    @property
    def name(self) -> str:
        return "siglip2_so400m"

    @property
    def embedding_dim(self) -> int:
        self._load_model()
        return self._embedding_dim

    @torch.no_grad()
    def embed_images(self, image_paths: list[str], batch_size: int = 8) -> np.ndarray:
        """Embed images in batches."""
        self._load_model()
        embeddings = []

        for i in tqdm(range(0, len(image_paths), batch_size), desc=f"[{self.name}] Embedding images"):
            batch_paths = image_paths[i:i + batch_size]
            images = torch.stack([
                self._preprocess(Image.open(p).convert("RGB"))
                for p in batch_paths
            ]).to(self.device)

            emb = self._model.encode_image(images)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu().numpy())

        return np.vstack(embeddings)

    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed text strings."""
        self._load_model()
        tokens = self._tokenizer(texts).to(self.device)
        emb = self._model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()
