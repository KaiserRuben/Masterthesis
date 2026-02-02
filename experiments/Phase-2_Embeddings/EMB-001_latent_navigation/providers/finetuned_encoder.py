"""
Finetuned Encoder Provider

Uses a custom-trained multi-task classification model as the image encoder.
For text embeddings, uses EVA-02-L to match dimension (1024-dim).

This provider also supports direct semantic key prediction from the trained heads.
"""

import sys
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from .base import EmbeddingProvider


# Import model architecture (lazy to avoid circular imports)
def _get_model_class():
    from sub_finetuned_encoder.model import MultiTaskEncoder
    return MultiTaskEncoder


def _get_key_values():
    from sub_finetuned_encoder.model import KEY_VALUES
    return KEY_VALUES


class FinetunedEncoderProvider(EmbeddingProvider):
    """
    Provider using finetuned multi-task classification backbone.

    Image embeddings: From trained ConvNeXt/EfficientNet backbone (1024-dim)
    Text embeddings: From EVA-02-L (1024-dim, for compatibility)

    Additional capability: Direct semantic key prediction
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str | None = None,
    ):
        """
        Initialize provider.

        Args:
            model_path: Path to trained model checkpoint (best_model.pt).
                       If None, auto-selects best available model.
            device: Device to use (auto-detect if None)
        """
        # Device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        # Find model
        if model_path is None:
            model_path = self._find_best_model()
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        config = checkpoint["config"]

        self._backbone_name = config["backbone"]
        self._key_mode = config["key_mode"]
        self._name = f"finetuned_{self._backbone_name}_{self._key_mode}"

        # Create model
        MultiTaskEncoder = _get_model_class()
        self.model = MultiTaskEncoder(
            backbone_name=config["backbone"],
            key_mode=config["key_mode"],
            pretrained=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self._embedding_dim = self.model.embedding_dim

        # Image transforms (validation mode - no augmentation)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Text encoder (lazy loaded - EVA-02-L for 1024-dim match)
        self._text_model = None
        self._text_preprocess = None
        self._text_tokenizer = None

    def _find_best_model(self) -> Path:
        """Find the best available trained model."""
        model_dir = Path(__file__).parents[3] / "data" / "EMB-001" / "finetuned_models"

        if not model_dir.exists():
            raise FileNotFoundError(f"No finetuned models directory: {model_dir}")

        best_model = None
        best_acc = 0.0

        for run_dir in model_dir.iterdir():
            if not run_dir.is_dir():
                continue

            checkpoint_path = run_dir / "best_model.pt"
            if not checkpoint_path.exists():
                continue

            # Load and check accuracy
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            val_acc = checkpoint.get("val_acc", 0.0)

            if val_acc > best_acc:
                best_acc = val_acc
                best_model = checkpoint_path

        if best_model is None:
            raise FileNotFoundError("No trained models found")

        print(f"Auto-selected model: {best_model.parent.name} (val_acc={best_acc:.1%})")
        return best_model

    @property
    def name(self) -> str:
        return self._name

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def backbone_name(self) -> str:
        return self._backbone_name

    @property
    def key_mode(self) -> str:
        return self._key_mode

    def _load_text_model(self):
        """Lazy load EVA-02-E text model (1024-dim to match ConvNeXt-Base)."""
        if self._text_model is None:
            print(f"[{self.name}] Loading EVA-02-E for text embeddings (1024-dim)...")
            self._text_model, self._text_preprocess, _ = open_clip.create_model_and_transforms(
                "EVA02-E-14-plus",
                pretrained="laion2b_s9b_b144k",
                device=self.device,
            )
            self._text_tokenizer = open_clip.get_tokenizer("EVA02-E-14-plus")
            self._text_model.eval()

    @torch.no_grad()
    def embed_images(self, image_paths: list[str], batch_size: int = 8) -> np.ndarray:
        """Embed images using finetuned backbone."""
        embeddings = []

        for i in tqdm(range(0, len(image_paths), batch_size), desc=f"[{self.name}] Embedding images"):
            batch_paths = image_paths[i:i + batch_size]

            images = []
            for path in batch_paths:
                img = Image.open(path).convert("RGB")
                img_tensor = self.transform(img)
                images.append(img_tensor)

            images = torch.stack(images).to(self.device)

            # Get backbone features (normalized)
            batch_emb = self.model.get_embeddings(images, normalize=True)
            embeddings.append(batch_emb.cpu().numpy())

        return np.vstack(embeddings)

    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embed texts using EVA-02-L.

        Returns 1024-dim embeddings to match image embedding dimension.
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
        Predict semantic keys for images using trained classification heads.

        This is the key capability of the finetuned model - direct label prediction
        without relying on text-image similarity.

        Returns:
            {key: [predicted_values]} for each key the model was trained on
        """
        KEY_VALUES = _get_key_values()

        predictions = {key: [] for key in self.model.all_keys}

        for i in tqdm(range(0, len(image_paths), batch_size), desc=f"[{self.name}] Predicting"):
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

    def get_trained_keys(self) -> list[str]:
        """Get list of keys this model was trained to predict."""
        return list(self.model.all_keys)


def list_available_models() -> list[dict]:
    """List all available trained finetuned models."""
    model_dir = Path(__file__).parents[3] / "data" / "EMB-001" / "finetuned_models"

    if not model_dir.exists():
        return []

    models = []
    for run_dir in model_dir.iterdir():
        if not run_dir.is_dir():
            continue

        checkpoint_path = run_dir / "best_model.pt"
        if not checkpoint_path.exists():
            continue

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {})

        models.append({
            "path": str(checkpoint_path),
            "name": run_dir.name,
            "backbone": config.get("backbone", "unknown"),
            "key_mode": config.get("key_mode", "unknown"),
            "val_acc": checkpoint.get("val_acc", 0.0),
        })

    # Sort by accuracy
    models.sort(key=lambda x: x["val_acc"], reverse=True)
    return models
