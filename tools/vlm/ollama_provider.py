"""
Ollama VLM Provider

Functional implementation of VLM provider for Ollama.
"""

import os
from dataclasses import dataclass
from typing import Any

from ollama import Client

from .types import (
    ModelConfig,
    ModelTier,
    Message,
    CompletionResult,
    JsonSchema,
)


# =============================================================================
# DEFAULT MODEL REGISTRY
# =============================================================================

OLLAMA_MODELS: dict[str, ModelConfig] = {
    "qwen3-vl:4b": ModelConfig("qwen3-vl:4b", ModelTier.SMALL, context_window=32768),
    "qwen3-vl:8b": ModelConfig("qwen3-vl:8b", ModelTier.MEDIUM, context_window=32768),
    "qwen3-vl:30b": ModelConfig("qwen3-vl:30b", ModelTier.LARGE, context_window=32768),
}

DEFAULT_MODEL = OLLAMA_MODELS["qwen3-vl:8b"]


# =============================================================================
# PROVIDER IMPLEMENTATION
# =============================================================================

@dataclass
class OllamaProvider:
    """Ollama VLM provider."""

    host: str = "http://localhost:11434"
    default_model: ModelConfig = DEFAULT_MODEL
    _client: Client | None = None

    @property
    def client(self) -> Client:
        """Lazy client initialization."""
        if self._client is None:
            self._client = Client(host=self.host)
        return self._client

    def complete(
        self,
        messages: list[Message],
        model: ModelConfig | None = None,
        json_schema: JsonSchema | None = None,
    ) -> CompletionResult:
        """Generate a completion."""
        model = model or self.default_model

        response = self.client.chat(
            model=model.name,
            messages=[m.to_dict() for m in messages],
            format=json_schema,
            options={"num_ctx": model.context_window},
        )

        return CompletionResult(
            content=response["message"]["content"],
            model=model.name,
            prompt_tokens=response.get("prompt_eval_count"),
            completion_tokens=response.get("eval_count"),
        )

    def available_models(self) -> list[ModelConfig]:
        """List available models (intersection of registry and local)."""
        local = {m["name"] for m in self.client.list()["models"]}
        return [cfg for name, cfg in OLLAMA_MODELS.items() if name in local]


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_provider(
    host: str | None = None,
    default_model: str | ModelConfig | None = None,
) -> OllamaProvider:
    """Create an Ollama provider with configuration."""
    host = host or os.environ.get("OLLAMA_URL", "http://localhost:11434")

    if default_model is None:
        model_cfg = DEFAULT_MODEL
    elif isinstance(default_model, str):
        model_cfg = OLLAMA_MODELS.get(default_model, ModelConfig(default_model, ModelTier.MEDIUM))
    else:
        model_cfg = default_model

    return OllamaProvider(host=host, default_model=model_cfg)


def get_model(name: str) -> ModelConfig:
    """Get a model config by name, or create one if not in registry."""
    return OLLAMA_MODELS.get(name, ModelConfig(name, ModelTier.MEDIUM))


def get_model_by_tier(tier: ModelTier) -> ModelConfig:
    """Get the default model for a given tier."""
    for model in OLLAMA_MODELS.values():
        if model.tier == tier:
            return model
    raise ValueError(f"No model registered for tier: {tier}")
