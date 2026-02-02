"""
Embedding Provider for Finetuned Multi-Task Encoder

Uses backbone features from trained model as embeddings.
For text embedding, falls back to a small CLIP model (OpenAI ViT-B/32).
"""

import sys
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from providers.base import EmbeddingProvider
from sub_finetuned_encoder.model import MultiTaskEncoder


class FinetunedEncoderProvider(EmbeddingProvider):
    """
    Provider for finetuned multi-task encoder.

    Uses:
    - Trained backbone for image embeddings
    - OpenAI CLIP ViT-B/32 for text embeddings (lightweight, compatible)
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str | None = None,
    ):
        """
        Initialize provider.

        Args:
            model_path: Path to trained model checkpoint (best_model.pt)
            device: Device to use (auto-detect if None)
        """
        self.model_path = Path(model_path)

        # Device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        config = checkpoint["config"]

        self._name = f"finetuned_{config['backbone']}_{config['key_mode']}"
        self._backbone = config["backbone"]
        self._key_mode = config["key_mode"]

        # Create model
        self.model = MultiTaskEncoder(
            backbone_name=config["backbone"],
            key_mode=config["key_mode"],
            pretrained=False,  # Will load from checkpoint
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self._embedding_dim = self.model.embedding_dim

        # Image transforms (validation mode)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Text encoder (lazy loaded)
        self._text_model = None
        self._text_tokenizer = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _load_text_model(self):
        """Lazy load CLIP text model."""
        if self._text_model is None:
            # Use lightweight CLIP for text
            self._text_model, _, _ = open_clip.create_model_and_transforms(
                "ViT-B-32",
                pretrained="openai",
                device=self.device,
            )
            self._text_tokenizer = open_clip.get_tokenizer("ViT-B-32")
            self._text_model.eval()

    @torch.no_grad()
    def embed_images(self, image_paths: list[str], batch_size: int = 8) -> np.ndarray:
        """Embed images using finetuned backbone."""
        embeddings = []

        for i in tqdm(range(0, len(image_paths), batch_size), desc=f"Embedding ({self.name})"):
            batch_paths = image_paths[i:i + batch_size]

            # Load and transform images
            images = []
            for path in batch_paths:
                img = Image.open(path).convert("RGB")
                img_tensor = self.transform(img)
                images.append(img_tensor)

            images = torch.stack(images).to(self.device)

            # Get embeddings (normalized backbone features)
            batch_emb = self.model.get_embeddings(images, normalize=True)
            embeddings.append(batch_emb.cpu().numpy())

        return np.vstack(embeddings)

    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embed texts using CLIP text encoder.

        Note: Text embeddings have different dimension than image embeddings.
        This is intentional - for text-image alignment, we project or use
        cosine similarity in the original spaces.
        """
        self._load_text_model()

        tokens = self._text_tokenizer(texts).to(self.device)
        text_emb = self._text_model.encode_text(tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        return text_emb.cpu().numpy()

    @torch.no_grad()
    def predict_keys(
        self,
        image_paths: list[str],
        batch_size: int = 8,
    ) -> dict[str, list]:
        """
        Predict semantic keys for images.

        Returns:
            {key: [predicted_values]} for each key the model was trained on
        """
        from sub_finetuned_encoder.model import KEY_VALUES

        predictions = {key: [] for key in self.model.all_keys}

        for i in tqdm(range(0, len(image_paths), batch_size), desc="Predicting"):
            batch_paths = image_paths[i:i + batch_size]

            images = []
            for path in batch_paths:
                img = Image.open(path).convert("RGB")
                img_tensor = self.transform(img)
                images.append(img_tensor)

            images = torch.stack(images).to(self.device)
            logits = self.model(images)

            for key in self.model.all_keys:
                preds = logits[key].argmax(dim=1).cpu().tolist()
                values = [KEY_VALUES[key][p] for p in preds]
                predictions[key].extend(values)

        return predictions


def load_provider(model_dir: str | Path, checkpoint: str = "best_model.pt") -> FinetunedEncoderProvider:
    """
    Load provider from model directory.

    Args:
        model_dir: Directory containing trained model
        checkpoint: Checkpoint file name

    Returns:
        Initialized provider
    """
    model_path = Path(model_dir) / checkpoint
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    return FinetunedEncoderProvider(model_path)


def list_available_models() -> list[dict]:
    """List all available trained models."""
    model_dir = Path(__file__).parents[3] / "data" / "EMB-001" / "finetuned_models"
    if not model_dir.exists():
        return []

    models = []
    for run_dir in model_dir.iterdir():
        if not run_dir.is_dir():
            continue

        best_model = run_dir / "best_model.pt"
        if best_model.exists():
            # Load config
            checkpoint = torch.load(best_model, map_location="cpu", weights_only=False)
            config = checkpoint.get("config", {})
            models.append({
                "path": str(run_dir),
                "name": run_dir.name,
                "backbone": config.get("backbone", "unknown"),
                "key_mode": config.get("key_mode", "unknown"),
                "val_acc": checkpoint.get("val_acc", 0),
            })

    return models
