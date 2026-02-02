"""
Embedding Provider Registry

Available providers:
- eva02_e: EVA02-E-14-plus (4.4B) — Best overall hypothesis
- openclip_bigg: ViT-bigG-14 (2.5B) — Best pure CLIP
- siglip2_so400m: SigLIP2 SO400M (400M) — Best text alignment hypothesis
- openai_clip_l: ViT-L-14-336 (428M) — Baseline
- eva02_l: EVA02-L-14-336 (428M) — MIM comparison
- finetuned: Custom finetuned encoder (1024-dim) — Trained on labeled anchors
"""

from .base import EmbeddingProvider, EmbeddingResult
from .eva02_e import EVA02EProvider
from .openclip_bigg import OpenCLIPBigGProvider
from .siglip2_so400m import SigLIP2Provider
from .openai_clip_l import OpenAICLIPProvider
from .eva02_l import EVA02LProvider
from .finetuned_encoder import FinetunedEncoderProvider

PROVIDERS: dict[str, type[EmbeddingProvider]] = {
    "eva02_e": EVA02EProvider,
    "openclip_bigg": OpenCLIPBigGProvider,
    "siglip2_so400m": SigLIP2Provider,
    "openai_clip_l": OpenAICLIPProvider,
    "eva02_l": EVA02LProvider,
    "finetuned": FinetunedEncoderProvider,
}


def get_provider(name: str) -> EmbeddingProvider:
    """Get provider instance by name."""
    if name not in PROVIDERS:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(f"Unknown provider: {name}. Available: {available}")
    return PROVIDERS[name]()


def list_providers() -> list[str]:
    """List available provider names."""
    return list(PROVIDERS.keys())


__all__ = [
    "EmbeddingProvider",
    "EmbeddingResult",
    "PROVIDERS",
    "get_provider",
    "list_providers",
    "EVA02EProvider",
    "OpenCLIPBigGProvider",
    "SigLIP2Provider",
    "OpenAICLIPProvider",
    "EVA02LProvider",
    "FinetunedEncoderProvider",
]
