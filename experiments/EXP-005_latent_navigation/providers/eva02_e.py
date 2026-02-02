"""
EVA02-E-14-plus Provider

Model: EVA02-E-14-plus (4.4B params)
Pretrained: laion2b_s9b_b144k
Embedding dim: 1024
Resolution: 224
"""

import numpy as np
import torch
import open_clip
from PIL import Image
from tqdm import tqdm

from .base import EmbeddingProvider


class EVA02EProvider(EmbeddingProvider):
    """EVA02-E-14-plus embedding provider (4.4B params)."""

    def __init__(self, device: str = "mps"):
        self.device = device
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._embedding_dim = None

    def _load_model(self):
        """Lazy load model on first use."""
        if self._model is not None:
            return

        print(f"Loading EVA02-E-14-plus to {self.device}...")
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            "EVA02-E-14-plus",
            pretrained="laion2b_s9b_b144k",
            device=self.device,
        )
        self._tokenizer = open_clip.get_tokenizer("EVA02-E-14-plus")
        self._model.eval()

        # Infer embedding dim
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=self.device)
            self._embedding_dim = self._model.encode_image(dummy).shape[-1]

        print(f"Model loaded. Embedding dim: {self._embedding_dim}")

    @property
    def name(self) -> str:
        return "eva02_e"

    @property
    def embedding_dim(self) -> int:
        self._load_model()
        return self._embedding_dim

    @torch.no_grad()
    def embed_images(self, image_paths: list[str], batch_size: int = 4) -> np.ndarray:
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
