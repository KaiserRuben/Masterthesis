"""
VLM Provider Types

Core types and protocols for vision-language model providers.
"""

from dataclasses import dataclass, field
from typing import Protocol, Any
from enum import Enum


class ModelTier(str, Enum):
    """Capability tiers for model selection."""
    SMALL = "small"      # Fast, simple tasks (4B)
    MEDIUM = "medium"    # Balanced (8B)
    LARGE = "large"      # Complex reasoning (30B+)


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    tier: ModelTier
    context_window: int = 32768

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Message:
    """A chat message."""
    role: str  # "system" | "user" | "assistant"
    content: str
    images: tuple[str, ...] = field(default_factory=tuple)  # base64 encoded

    def to_dict(self) -> dict[str, Any]:
        d = {"role": self.role, "content": self.content}
        if self.images:
            d["images"] = list(self.images)
        return d


@dataclass(frozen=True)
class CompletionResult:
    """Result from a model completion."""
    content: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    def parse_json(self) -> dict[str, Any]:
        """Parse content as JSON."""
        import json
        return json.loads(self.content)


class VLMProvider(Protocol):
    """Protocol for VLM providers."""

    def complete(
        self,
        messages: list[Message],
        model: ModelConfig,
        json_schema: dict[str, Any] | None = None,
    ) -> CompletionResult:
        """Generate a completion from messages."""
        ...

    def available_models(self) -> list[ModelConfig]:
        """List available models."""
        ...


# Type alias for JSON schemas (Pydantic model_json_schema output)
JsonSchema = dict[str, Any]
