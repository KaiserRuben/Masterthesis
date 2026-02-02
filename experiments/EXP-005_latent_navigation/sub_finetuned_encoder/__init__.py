"""
Sub-experiment: Finetuned Multi-Task Encoder

Uses a small pre-trained backbone (EfficientNet-B0) with classification heads
for each semantic key. Trains on 100 labeled anchors, then uses learned
backbone features for latent navigation on the full dataset.

Key configurations:
- all: Train on all 14 categorical + 5 boolean keys
- top: Train only on keys with >25% text alignment from pretest
"""

from .model import (
    MultiTaskEncoder,
    MultiTaskLoss,
    get_key_config,
    KEY_VALUES,
    ALL_CATEGORICAL_KEYS,
    ALL_BOOLEAN_KEYS,
    TOP_CATEGORICAL_KEYS,
    TOP_BOOLEAN_KEYS,
)

from .provider import (
    FinetunedEncoderProvider,
    load_provider,
    list_available_models,
)

__all__ = [
    "MultiTaskEncoder",
    "MultiTaskLoss",
    "get_key_config",
    "KEY_VALUES",
    "ALL_CATEGORICAL_KEYS",
    "ALL_BOOLEAN_KEYS",
    "TOP_CATEGORICAL_KEYS",
    "TOP_BOOLEAN_KEYS",
    "FinetunedEncoderProvider",
    "load_provider",
    "list_available_models",
]
