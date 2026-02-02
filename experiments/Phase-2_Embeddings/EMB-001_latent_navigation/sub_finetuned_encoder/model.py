"""
Multi-Task Classification Encoder for EXP-005 Sub-experiment

Architecture:
- Backbone: Pre-trained EfficientNet-B0 (or configurable)
- Heads: One classification head per semantic key
- Embedding: Backbone features (before heads) used for latent navigation

Configurable:
- all_keys: Train on all 14 categorical + 5 boolean keys
- top_keys: Train only on keys with >25% text alignment (from pretest)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# Key configurations based on pretest results
ALL_CATEGORICAL_KEYS = [
    "weather",
    "road_type",
    "time_of_day",
    "traffic_situation",
    "occlusion_level",
    "depth_complexity",
    "visual_degradation",
    "safety_criticality",
    "required_action",
]

ALL_BOOLEAN_KEYS = [
    "pedestrians_present",
    "cyclists_present",
    "construction_activity",
    "traffic_signals_visible",
    "similar_object_confusion",
]

# Top keys: >25% text alignment from pretest
TOP_CATEGORICAL_KEYS = [
    "weather",
    "time_of_day",
    "occlusion_level",
    "depth_complexity",
    "required_action",
    "traffic_situation",
]

TOP_BOOLEAN_KEYS = [
    "pedestrians_present",
    "traffic_signals_visible",
    "construction_activity",
]

# Value mappings for each key (for classification)
# Based on actual values in data/EXP-001/scene_classifications.json
KEY_VALUES = {
    # Categorical
    "weather": ["clear", "cloudy", "foggy", "rainy"],
    "road_type": ["urban_street", "highway", "residential", "rural", "intersection"],
    "time_of_day": ["day", "night", "dawn_dusk"],
    "traffic_situation": ["simple", "moderate", "complex", "critical"],
    "occlusion_level": ["none", "minimal", "moderate", "severe"],
    "depth_complexity": ["complex", "layered"],
    "visual_degradation": ["none", "glare", "motion_blur", "fog_haze", "low_light", "rain_artifacts", "sensor_artifact"],
    "safety_criticality": ["tier4_minor", "tier3_moderate", "tier2_severe", "tier1_catastrophic"],
    "required_action": ["none", "slow", "stop", "evade"],
    # Boolean (2 classes: False, True)
    "pedestrians_present": [False, True],
    "cyclists_present": [False, True],
    "construction_activity": [False, True],
    "traffic_signals_visible": [False, True],
    "similar_object_confusion": [False, True],
}


def get_key_config(mode: str = "all") -> tuple[list[str], list[str]]:
    """Get key configuration based on mode."""
    if mode == "all":
        return ALL_CATEGORICAL_KEYS, ALL_BOOLEAN_KEYS
    elif mode == "top":
        return TOP_CATEGORICAL_KEYS, TOP_BOOLEAN_KEYS
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'all' or 'top'.")


class MultiTaskEncoder(nn.Module):
    """
    Multi-task classification encoder.

    Uses pre-trained backbone with separate classification heads per key.
    Backbone features are used as embeddings for latent navigation.
    """

    BACKBONES = {
        "efficientnet_b0": {"dim": 1280, "params": "5.3M"},
        "efficientnet_b4": {"dim": 1792, "params": "19M"},
        "mobilenetv3_small_100": {"dim": 576, "params": "2.5M"},
        "mobilenetv3_large_100": {"dim": 960, "params": "5.4M"},
        "resnet18": {"dim": 512, "params": "11.7M"},
        "resnet50": {"dim": 2048, "params": "25.6M"},
        "convnext_small": {"dim": 768, "params": "49.5M"},
        "convnext_base": {"dim": 1024, "params": "87.6M"},
        "convnext_large": {"dim": 1536, "params": "196.2M"},
    }

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        key_mode: str = "all",  # "all" or "top"
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.key_mode = key_mode

        # Get keys to predict
        cat_keys, bool_keys = get_key_config(key_mode)
        self.categorical_keys = cat_keys
        self.boolean_keys = bool_keys
        self.all_keys = cat_keys + bool_keys

        # Load backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
        )

        # Get backbone dimension
        backbone_dim = self.BACKBONES.get(backbone_name, {}).get("dim")
        if backbone_dim is None:
            with torch.no_grad():
                dummy = torch.randn(1, 3, 224, 224)
                backbone_dim = self.backbone(dummy).shape[-1]

        self.backbone_dim = backbone_dim
        self.embedding_dim = backbone_dim  # Embeddings = backbone features

        # Optional backbone freezing
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Dropout before heads
        self.dropout = nn.Dropout(dropout)

        # Classification heads (one per key)
        self.heads = nn.ModuleDict()
        for key in self.all_keys:
            num_classes = len(KEY_VALUES[key])
            self.heads[key] = nn.Linear(backbone_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        return_embeddings: bool = False,
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Images (B, 3, H, W)
            return_embeddings: Also return backbone features

        Returns:
            logits: Dict of {key: (B, num_classes)}
            embeddings: (B, backbone_dim) if return_embeddings=True
        """
        # Backbone features
        features = self.backbone(x)  # (B, backbone_dim)
        features_dropped = self.dropout(features)

        # Classification logits per key
        logits = {}
        for key in self.all_keys:
            logits[key] = self.heads[key](features_dropped)

        if return_embeddings:
            return logits, features
        return logits

    def get_embeddings(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Get normalized embeddings for latent navigation."""
        features = self.backbone(x)
        if normalize:
            features = F.normalize(features, p=2, dim=-1)
        return features

    def get_num_params(self) -> dict:
        """Return parameter counts."""
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        heads_params = sum(p.numel() for p in self.heads.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "backbone": backbone_params,
            "heads": heads_params,
            "total": backbone_params + heads_params,
            "trainable": trainable,
        }


class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task classification.

    Supports weighted combination of per-key cross-entropy losses.
    """

    def __init__(
        self,
        keys: list[str],
        weights: dict[str, float] | None = None,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.keys = keys
        self.weights = weights or {k: 1.0 for k in keys}
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute weighted multi-task loss.

        Args:
            logits: {key: (B, num_classes)}
            labels: {key: (B,)} class indices

        Returns:
            total_loss: Weighted sum of per-key losses
            per_key_losses: {key: loss} for logging
        """
        per_key_losses = {}
        total_loss = None
        device = None

        for key in self.keys:
            if key not in logits or key not in labels:
                continue

            if device is None:
                device = logits[key].device

            # Skip samples with missing labels (-1)
            mask = labels[key] >= 0
            if mask.sum() == 0:
                continue

            loss = F.cross_entropy(
                logits[key][mask],
                labels[key][mask],
                label_smoothing=self.label_smoothing,
            )
            per_key_losses[key] = loss
            if total_loss is None:
                total_loss = self.weights.get(key, 1.0) * loss
            else:
                total_loss = total_loss + self.weights.get(key, 1.0) * loss

        # Ensure we return a tensor even if no valid labels
        if total_loss is None:
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        return total_loss, per_key_losses


def create_label_encoder(keys: list[str]) -> dict[str, dict]:
    """Create label encoders (value -> index) for each key."""
    encoders = {}
    for key in keys:
        values = KEY_VALUES[key]
        encoders[key] = {v: i for i, v in enumerate(values)}
    return encoders


def encode_labels(
    classifications: dict[str, dict],
    keys: list[str],
    encoders: dict[str, dict],
) -> dict[str, list[int]]:
    """
    Encode classification labels to indices.

    Args:
        classifications: {scene_id: {key: value}}
        keys: Keys to encode
        encoders: {key: {value: index}}

    Returns:
        {key: [indices]} for each key, -1 for missing/unknown values
    """
    encoded = {key: [] for key in keys}

    for scene_id, scene_labels in classifications.items():
        for key in keys:
            value = scene_labels.get(key)
            if value is None:
                encoded[key].append(-1)
            elif value in encoders[key]:
                encoded[key].append(encoders[key][value])
            else:
                # Unknown value
                encoded[key].append(-1)

    return encoded
