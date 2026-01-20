"""
Ollama API Proxy with model-specific context windows, output limits, and timeouts.

This module wraps the Ollama API to automatically set:
- Model-specific num_ctx (e.g., 128k for ministral, 32k for qwen3, 8k for gemma3)
- num_predict: unlimited by default (let model generate freely)
- Timeout protection (5 minutes default for chat, 1 minute for embeddings)
- Dual-host configuration: VLM models use remote host, embeddings use localhost

Host Configuration:
- VLM models (chat): http://100.114.255.41:11434 (remote)
- Embedding models: http://localhost:11434 (local)

Supported models with optimized context windows:
- ministral: 128k context (Ministral-8B, Ministral-3B)
- qwen3/qwen3-vl: 32k context
- deepseek: 32k context
- gemma3: 8k context
- others: 16k context (default)

Usage:
    Instead of:
        import ollama
        response = ollama.chat(...)

    Use:
        import ollama_proxy as ollama
        response = ollama.chat(model="qwen3-vl:8b", messages=...)  # Uses remote host
        embedding = ollama.embeddings(model="qwen3-embedding", prompt="...")  # Uses localhost
"""
from ollama import Client
from typing import Any, Dict, List, Optional
import signal
import time

# VLM_HOST = "http://100.114.255.41:11434"  # Remote host for VLM models
VLM_HOST = "http://localhost:11434"
EMBED_HOST = "http://localhost:11434"       # Local host for embedding models

# Default settings
DEFAULT_NUM_CTX = 16384
DEFAULT_NUM_PREDICT = None  # Don't limit by default - let model generate freely
DEFAULT_TIMEOUT = 300  # 5 minutes in seconds

# Model-specific context window sizes
MODEL_CTX_SIZES = {
    'ministral': 128000,  # Ministral-8B and Ministral-3B support 128k context
    'qwen3': 32768,       # Qwen3 models typically support 32k context
    'qwen3-vl': 32768,    # Qwen3-VL models support 32k context
    'deepseek': 32768,    # DeepSeek models support 32k context
    'gemma3': 8192,       # Gemma3 models support 8k context
}

_ollama = Client(host=VLM_HOST)        # VLM client (remote)
_ollama_embed = Client(host=EMBED_HOST)  # Embedding client (localhost)

def get_model_ctx_size(model: str) -> int:
    """
    Get the appropriate context window size for a model.

    Args:
        model: Model name (e.g., "ministral:8b", "qwen3-vl:30b")

    Returns:
        Context window size for the model
    """
    # Extract base model name (before colon)
    base_model = model.split(':')[0].lower()

    # Check if model matches any known pattern
    for model_prefix, ctx_size in MODEL_CTX_SIZES.items():
        if base_model.startswith(model_prefix):
            return ctx_size

    # Default fallback
    return DEFAULT_NUM_CTX


class TimeoutError(Exception):
    """Raised when an API call exceeds the timeout."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Ollama API call timed out")


def chat(
    model: str,
    messages: List[Dict[str, Any]],
    options: Optional[Dict[str, Any]] = None,
    timeout: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Proxy for ollama.chat() with model-specific context window, output limits, and timeout.

    Args:
        model: Model name (e.g., "ministral:8b", "qwen3-vl:8b")
        messages: List of message dicts with role, content, and optional images
        options: Model options dict. Defaults applied:
            - num_ctx: Model-specific (e.g., 128k for ministral, 32k for qwen3)
            - num_predict: None (unlimited by default)
        timeout: Timeout in seconds (default: 300 = 5 minutes)
        **kwargs: Additional arguments passed to ollama.chat()

    Returns:
        Response dict from Ollama API

    Raises:
        TimeoutError: If the call exceeds the timeout
    """
    # Ensure options dict exists
    if options is None:
        options = {}

    # Set defaults if not already specified - use model-specific context size
    if 'num_ctx' not in options:
        options['num_ctx'] = get_model_ctx_size(model)
    if 'num_predict' not in options and DEFAULT_NUM_PREDICT is not None:
        options['num_predict'] = DEFAULT_NUM_PREDICT

    # Set timeout
    if timeout is None:
        timeout = DEFAULT_TIMEOUT

    # Set up timeout alarm
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    try:
        # Call original ollama.chat with modified options
        response = _ollama.chat(model=model, messages=messages, options=options, **kwargs)
        signal.alarm(0)  # Cancel alarm
        return response
    except TimeoutError:
        signal.alarm(0)  # Cancel alarm
        raise
    except Exception as e:
        signal.alarm(0)  # Cancel alarm
        raise


def embeddings(
    model: str,
    prompt: str,
    options: Optional[Dict[str, Any]] = None,
    timeout: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Proxy for ollama.embeddings() with model-specific context window and timeout.

    Note: Embeddings always use localhost, while VLM chat uses remote host.

    Args:
        model: Embedding model name (e.g., "qwen3-embedding")
        prompt: Text to embed
        options: Model options dict. If num_ctx is not specified, uses model-specific default
        timeout: Timeout in seconds (default: 60 seconds for embeddings)
        **kwargs: Additional arguments passed to ollama.embeddings()

    Returns:
        Response dict with embedding vector

    Raises:
        TimeoutError: If the call exceeds the timeout
    """
    # Ensure options dict exists
    if options is None:
        options = {}

    # Set num_ctx if not already specified - use model-specific context size
    if 'num_ctx' not in options:
        options['num_ctx'] = get_model_ctx_size(model)

    # Set timeout (shorter default for embeddings)
    if timeout is None:
        timeout = 60  # 1 minute for embeddings

    # Set up timeout alarm
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    try:
        # Call embeddings on localhost client
        response = _ollama_embed.embeddings(model=model, prompt=prompt, options=options, **kwargs)
        signal.alarm(0)  # Cancel alarm
        return response
    except TimeoutError:
        signal.alarm(0)  # Cancel alarm
        raise
    except Exception as e:
        signal.alarm(0)  # Cancel alarm
        raise


def list():
    """
    Proxy for ollama.list() - no modification needed.

    Returns:
        ListResponse object with available models
    """
    return _ollama.list()


# Allow direct access to other ollama functions if needed
def __getattr__(name):
    """
    Fallback for any other ollama functions not explicitly wrapped.
    """
    return getattr(_ollama, name)
