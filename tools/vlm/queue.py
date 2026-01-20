"""
VLM Request Queue

Async request queue with:
- Per-endpoint concurrency control
- Retry logic with exponential backoff
- Request prioritization
- Batch execution
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Coroutine, TypeVar
from collections import defaultdict

from ollama import AsyncClient

from .types import Message, CompletionResult, ModelConfig, JsonSchema
from .config import VLMConfig, EndpointConfig


T = TypeVar("T")


class Priority(IntEnum):
    """Request priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2


@dataclass(order=True)
class QueuedRequest:
    """A request in the queue."""
    priority: Priority
    timestamp: float = field(compare=True)
    request_id: str = field(compare=False)
    endpoint: str = field(compare=False)
    model: str = field(compare=False)
    messages: list[Message] = field(compare=False)
    json_schema: JsonSchema | None = field(compare=False, default=None)
    future: asyncio.Future = field(compare=False, default=None)


@dataclass
class EndpointStats:
    """Statistics for an endpoint."""
    requests_completed: int = 0
    requests_failed: int = 0
    total_latency_ms: float = 0
    total_tokens: int = 0

    @property
    def avg_latency_ms(self) -> float:
        if self.requests_completed == 0:
            return 0
        return self.total_latency_ms / self.requests_completed


class RequestQueue:
    """
    Async request queue with per-endpoint concurrency control.

    Usage:
        queue = RequestQueue(config)
        await queue.start()

        # Submit requests
        result = await queue.submit(model, messages)

        # Or batch submit
        results = await queue.submit_batch([
            (model1, messages1),
            (model2, messages2),
        ])

        await queue.stop()
    """

    def __init__(self, config: VLMConfig):
        self.config = config
        self._clients: dict[str, AsyncClient] = {}
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._stats: dict[str, EndpointStats] = defaultdict(EndpointStats)
        self._request_counter = 0
        self._running = False

    async def start(self):
        """Initialize clients and semaphores."""
        for name, endpoint in self.config.endpoints.items():
            self._clients[name] = AsyncClient(host=endpoint.url)
            self._semaphores[name] = asyncio.Semaphore(endpoint.max_concurrent)
        self._running = True

    async def stop(self):
        """Cleanup resources."""
        self._running = False
        self._clients.clear()
        self._semaphores.clear()

    def _get_endpoint_name(self, model: str) -> str:
        """Get endpoint name for a model."""
        if model in self.config.models:
            return self.config.models[model].endpoint
        return "default"

    def _get_endpoint_config(self, endpoint_name: str) -> EndpointConfig:
        """Get endpoint configuration."""
        return self.config.endpoints.get(endpoint_name, self.config.endpoints["default"])

    async def _execute_with_retry(
        self,
        endpoint_name: str,
        model: str,
        messages: list[Message],
        json_schema: JsonSchema | None,
    ) -> CompletionResult:
        """Execute a request with retry logic."""
        endpoint_cfg = self._get_endpoint_config(endpoint_name)
        client = self._clients.get(endpoint_name, self._clients["default"])

        last_error: Exception | None = None

        for attempt in range(endpoint_cfg.retry_attempts):
            try:
                start_time = time.monotonic()

                # Get model config for context window
                model_cfg = self.config.get_model_config(model)

                response = await asyncio.wait_for(
                    client.chat(
                        model=model,
                        messages=[m.to_dict() for m in messages],
                        format=json_schema,
                        options={"num_ctx": model_cfg.context_window},
                    ),
                    timeout=endpoint_cfg.timeout_seconds,
                )

                latency_ms = (time.monotonic() - start_time) * 1000

                # Update stats
                stats = self._stats[endpoint_name]
                stats.requests_completed += 1
                stats.total_latency_ms += latency_ms
                if "prompt_eval_count" in response:
                    stats.total_tokens += response.get("prompt_eval_count", 0)
                    stats.total_tokens += response.get("eval_count", 0)

                return CompletionResult(
                    content=response["message"]["content"],
                    model=model,
                    prompt_tokens=response.get("prompt_eval_count"),
                    completion_tokens=response.get("eval_count"),
                )

            except asyncio.TimeoutError:
                last_error = TimeoutError(f"Request timed out after {endpoint_cfg.timeout_seconds}s")
            except Exception as e:
                last_error = e

            # Exponential backoff
            if attempt < endpoint_cfg.retry_attempts - 1:
                delay = endpoint_cfg.retry_delay_seconds * (2 ** attempt)
                await asyncio.sleep(delay)

        # All retries exhausted
        self._stats[endpoint_name].requests_failed += 1
        raise last_error or RuntimeError("Request failed")

    async def submit(
        self,
        model: str,
        messages: list[Message],
        json_schema: JsonSchema | None = None,
        priority: Priority = Priority.NORMAL,
    ) -> CompletionResult:
        """
        Submit a single request and wait for result.

        Args:
            model: Model name (e.g., "qwen3-vl:8b")
            messages: List of messages
            json_schema: Optional JSON schema for structured output
            priority: Request priority

        Returns:
            CompletionResult
        """
        endpoint_name = self._get_endpoint_name(model)
        semaphore = self._semaphores.get(endpoint_name, self._semaphores["default"])

        async with semaphore:
            return await self._execute_with_retry(
                endpoint_name, model, messages, json_schema
            )

    async def submit_batch(
        self,
        requests: list[tuple[str, list[Message], JsonSchema | None]],
        priority: Priority = Priority.NORMAL,
    ) -> list[CompletionResult]:
        """
        Submit multiple requests and wait for all results.

        Args:
            requests: List of (model, messages, json_schema) tuples

        Returns:
            List of CompletionResult in same order as requests
        """
        tasks = [
            self.submit(model, messages, schema, priority)
            for model, messages, schema in requests
        ]
        return await asyncio.gather(*tasks)

    def get_stats(self) -> dict[str, EndpointStats]:
        """Get statistics for all endpoints."""
        return dict(self._stats)

    def get_stats_summary(self) -> str:
        """Get a formatted stats summary."""
        lines = ["Endpoint Statistics:"]
        for name, stats in self._stats.items():
            lines.append(f"  {name}:")
            lines.append(f"    Completed: {stats.requests_completed}")
            lines.append(f"    Failed: {stats.requests_failed}")
            lines.append(f"    Avg latency: {stats.avg_latency_ms:.0f}ms")
            lines.append(f"    Total tokens: {stats.total_tokens}")
        return "\n".join(lines)


class SyncRequestQueue:
    """
    Synchronous wrapper around RequestQueue for non-async code.

    Usage:
        queue = SyncRequestQueue(config)

        result = queue.submit(model, messages)
        results = queue.submit_batch([...])

        queue.close()
    """

    def __init__(self, config: VLMConfig):
        self.config = config
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: RequestQueue | None = None

    def _ensure_started(self):
        """Ensure the async queue is running."""
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            self._queue = RequestQueue(self.config)
            self._loop.run_until_complete(self._queue.start())

    def submit(
        self,
        model: str,
        messages: list[Message],
        json_schema: JsonSchema | None = None,
        priority: Priority = Priority.NORMAL,
    ) -> CompletionResult:
        """Submit a single request synchronously."""
        self._ensure_started()
        return self._loop.run_until_complete(
            self._queue.submit(model, messages, json_schema, priority)
        )

    def submit_batch(
        self,
        requests: list[tuple[str, list[Message], JsonSchema | None]],
        priority: Priority = Priority.NORMAL,
    ) -> list[CompletionResult]:
        """Submit multiple requests synchronously."""
        self._ensure_started()
        return self._loop.run_until_complete(
            self._queue.submit_batch(requests, priority)
        )

    def get_stats(self) -> dict[str, EndpointStats]:
        """Get endpoint statistics."""
        if self._queue:
            return self._queue.get_stats()
        return {}

    def get_stats_summary(self) -> str:
        """Get formatted stats summary."""
        if self._queue:
            return self._queue.get_stats_summary()
        return "No requests executed"

    def close(self):
        """Cleanup resources."""
        if self._loop and self._queue:
            self._loop.run_until_complete(self._queue.stop())
            self._loop.close()
            self._loop = None
            self._queue = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
