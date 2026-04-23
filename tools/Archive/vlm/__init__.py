"""
VLM Provider Package

Clean abstractions for vision-language model inference.

Usage (simple):
    from vlm import create_provider, Message, get_model

    provider = create_provider()
    result = provider.complete(
        messages=[
            Message("system", "You are a helpful assistant."),
            Message("user", "Describe this image.", images=(img_b64,)),
        ],
        model=get_model("qwen3-vl:30b"),
    )

Usage (with config and queue):
    from vlm import load_config, SyncRequestQueue, Message

    config = load_config("vlm_config.yaml")
    with SyncRequestQueue(config) as queue:
        result = queue.submit("qwen3-vl:8b", [Message("user", "Hello")])
        print(result.content)
"""

from .types import (
    ModelTier,
    ModelConfig,
    Message,
    CompletionResult,
    VLMProvider,
    JsonSchema,
)

from .ollama_provider import (
    create_provider,
    get_model,
    get_model_by_tier,
    OllamaProvider,
    OLLAMA_MODELS,
)

from .config import (
    # Config classes
    VLMConfig,
    EndpointConfig,
    ModelEndpointConfig,
    KeyModelMapping,
    # Loading
    load_config,
    # Model tiers
    DEFAULT_MODEL_TIERS,
    DEFAULT_KEY_TIERS,
    get_model_tiers,
    resolve_tier,
    # Template
    DEFAULT_CONFIG_TEMPLATE,
)

from .queue import (
    RequestQueue,
    SyncRequestQueue,
    Priority,
    EndpointStats,
    Request,
    WorkStealingQueue,
)

__all__ = [
    # Types
    "ModelTier",
    "ModelConfig",
    "Message",
    "CompletionResult",
    "VLMProvider",
    "JsonSchema",
    # Ollama
    "create_provider",
    "get_model",
    "get_model_by_tier",
    "OllamaProvider",
    "OLLAMA_MODELS",
    # Config
    "VLMConfig",
    "EndpointConfig",
    "ModelEndpointConfig",
    "KeyModelMapping",
    "load_config",
    "DEFAULT_MODEL_TIERS",
    "DEFAULT_KEY_TIERS",
    "get_model_tiers",
    "resolve_tier",
    "DEFAULT_CONFIG_TEMPLATE",
    # Queue
    "RequestQueue",
    "SyncRequestQueue",
    "Priority",
    "EndpointStats",
    "Request",
    "WorkStealingQueue",
]
