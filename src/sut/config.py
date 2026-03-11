"""Configuration for the VLM system-under-test."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_CATEGORIES: tuple[str, ...] = (
    # Animals -- vivid colors and textures
    "macaw",
    "peacock",
    "flamingo",
    "monarch butterfly",
    "jellyfish",
    "chameleon",
    "toucan",
    "leopard",
    "red panda",
    "lionfish",
    # Scenes & structures
    "coral reef",
    "volcano",
    "castle",
    "mosque",
    "palace",
)

DEFAULT_PROMPT_TEMPLATE: str = (
    "What is the main subject in this image? "
    "Answer with exactly one of these options: {categories}."
)


@dataclass(frozen=True)
class VLMSUTConfig:
    """Immutable configuration for :class:`VLMSUT`.

    :param model_id: HuggingFace model identifier.
    :param device: Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
    :param categories: Labels to force-decode against.
    :param prompt_template: Prompt with a ``{categories}`` placeholder.
    :param enable_thinking: Whether to allow the model's thinking trace.
    :param max_thinking_tokens: Token budget for the thinking trace.
    :param max_pixels: Pixel cap passed to the processor (limits image
        token count).  ``None`` lets the scorer class choose its default.
    """

    model_id: str = "Qwen/Qwen3.5-9B"
    device: str = "cpu"
    categories: tuple[str, ...] = DEFAULT_CATEGORIES
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    enable_thinking: bool = False
    max_thinking_tokens: int = 2000
    max_pixels: int | None = None
