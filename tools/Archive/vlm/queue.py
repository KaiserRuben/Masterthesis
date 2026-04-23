"""
VLM Request Queue

Async request queue with:
- Per-endpoint concurrency control
- Retry logic with exponential backoff
- Request prioritization
- Batch execution
- Work-stealing with multi-endpoint support
"""

import asyncio
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, TypeVar
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


# =============================================================================
# WORK-STEALING QUEUE
# =============================================================================

@dataclass
class Request:
    """A request in the work-stealing queue."""
    id: str
    clip_id: str
    key: str  # "stage1" or classification key
    model: str
    messages: list[Message]
    json_schema: JsonSchema | None = None
    depends_on_stage1: bool = False  # True for stage2 requests


class WorkStealingQueue:
    """
    Work-stealing queue with:
    - Multi-endpoint support
    - Dependency tracking (stage1 → stage2)
    - Lookahead for finding compatible work

    Usage:
        def on_result(clip_id: str, key: str, result: str):
            print(f"Got result for {clip_id}:{key}")

        queue = WorkStealingQueue(config, on_result=on_result)
        queue.add_requests(requests)
        queue.run()  # Blocks until all complete
    """

    def __init__(
        self,
        config: VLMConfig,
        on_result: Callable[[str, str, str, str], None],  # (clip_id, key, result, endpoint)
        lookahead: int = 10,
    ):
        self.config = config
        self.on_result = on_result
        self.lookahead = lookahead

        # State
        self._queue: list[Request] = []
        self._stage1_results: dict[str, str] = {}  # clip_id → reasoning
        self._in_flight: set[str] = set()  # request IDs currently processing
        self._lock = threading.Lock()
        self._completed_count = 0
        self._failed_count = 0
        self._errors: list[tuple[str, str, Exception]] = []  # (clip_id, key, error)

        # Per-endpoint async infrastructure (initialized in run())
        self._clients: dict[str, AsyncClient] = {}
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._stats: dict[str, EndpointStats] = defaultdict(EndpointStats)

        # Event signaled when new work may be available (stage1 completes, etc.)
        self._work_available: asyncio.Event | None = None

    @property
    def total_requests(self) -> int:
        """Total number of requests (including completed)."""
        with self._lock:
            return len(self._queue) + self._completed_count

    @property
    def pending_count(self) -> int:
        """Number of requests still pending."""
        with self._lock:
            return len(self._queue)

    @property
    def completed_count(self) -> int:
        """Number of completed requests."""
        with self._lock:
            return self._completed_count

    def add_requests(self, requests: list[Request]):
        """Add requests to queue (already interleaved by caller)."""
        with self._lock:
            self._queue.extend(requests)

    def mark_stage1_complete(self, clip_id: str, reasoning: str):
        """Record stage1 completion to unblock stage2 requests."""
        with self._lock:
            self._stage1_results[clip_id] = reasoning

    def get_stage1_result(self, clip_id: str) -> str | None:
        """Get stage1 result for a clip (for building stage2 messages)."""
        with self._lock:
            return self._stage1_results.get(clip_id)

    def _find_work(self, endpoint: str) -> Request | None:
        """
        Find next compatible request for endpoint.

        Uses two-phase search:
        1. Primary: Check within lookahead window (fast path for locality)
        2. Fallback: Check entire queue if nothing found in lookahead
        """
        supported_models = self.config.get_supported_models(endpoint)

        with self._lock:
            # Phase 1: Check within lookahead window
            for req in self._queue[:self.lookahead]:
                if self._can_execute(req, supported_models):
                    self._in_flight.add(req.id)
                    return req

            # Phase 2: Fallback to full queue scan if nothing in lookahead
            for req in self._queue[self.lookahead:]:
                if self._can_execute(req, supported_models):
                    self._in_flight.add(req.id)
                    return req

        return None  # No compatible work found

    def _can_execute(self, req: Request, supported_models: set[str]) -> bool:
        """Check if request can be executed (must be called with lock held)."""
        if req.id in self._in_flight:
            return False
        if req.model not in supported_models:
            return False
        if req.depends_on_stage1 and req.clip_id not in self._stage1_results:
            return False
        return True

    def _all_done(self) -> bool:
        """Check if all work is complete (queue empty AND nothing in-flight)."""
        with self._lock:
            return len(self._queue) == 0 and len(self._in_flight) == 0

    async def _execute_with_retry(
        self,
        endpoint_name: str,
        model: str,
        messages: list[Message],
        json_schema: JsonSchema | None,
    ) -> CompletionResult:
        """Execute a request with retry logic."""
        endpoint_cfg = self.config.endpoints.get(endpoint_name, self.config.endpoints["default"])
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

    def _handle_result(self, req: Request, result: CompletionResult, endpoint: str):
        """Process completed request."""
        is_stage1 = req.key == "stage1"
        if is_stage1:
            self.mark_stage1_complete(req.clip_id, result.content)

        # Remove from queue and in_flight
        with self._lock:
            self._queue = [r for r in self._queue if r.id != req.id]
            self._in_flight.discard(req.id)
            self._completed_count += 1

        # Callback to save result (includes endpoint for logging)
        self.on_result(req.clip_id, req.key, result.content, endpoint)

        # Signal waiting workers that new work may be available
        if is_stage1 and self._work_available:
            self._work_available.set()

    def _handle_error(self, req: Request, error: Exception):
        """Handle failed request and clean up any dependent requests."""
        with self._lock:
            self._queue = [r for r in self._queue if r.id != req.id]
            self._in_flight.discard(req.id)
            self._failed_count += 1
            self._errors.append((req.clip_id, req.key, error))

            # If this was a stage1, remove all dependent stage2 requests for this clip
            # to prevent them from blocking the queue indefinitely
            if req.key == "stage1":
                orphaned = [
                    r for r in self._queue
                    if r.clip_id == req.clip_id and r.depends_on_stage1
                ]
                for orphan in orphaned:
                    self._errors.append((orphan.clip_id, orphan.key,
                        Exception(f"Skipped: stage1 for {req.clip_id} failed")))
                    self._failed_count += 1
                self._queue = [
                    r for r in self._queue
                    if not (r.clip_id == req.clip_id and r.depends_on_stage1)
                ]

    async def _worker(self, endpoint: str):
        """Worker loop for an endpoint."""
        client = self._clients[endpoint]
        semaphore = self._semaphores[endpoint]

        while True:
            req = self._find_work(endpoint)
            if req is None:
                if self._all_done():
                    break

                # Wait for signal that new work may be available (stage1 completed)
                # Use timeout to periodically recheck in case of missed signals
                self._work_available.clear()
                try:
                    await asyncio.wait_for(self._work_available.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass  # Timeout is fine, just recheck
                continue

            async with semaphore:
                try:
                    result = await self._execute_with_retry(
                        endpoint, req.model, req.messages, req.json_schema
                    )
                    self._handle_result(req, result, endpoint)
                except Exception as e:
                    self._handle_error(req, e)

    async def _run_async(self):
        """Run all workers asynchronously."""
        # Initialize async primitives
        self._work_available = asyncio.Event()
        self._work_available.set()  # Start with event set so workers check immediately

        # Initialize clients and semaphores for each endpoint
        for name, endpoint in self.config.endpoints.items():
            self._clients[name] = AsyncClient(host=endpoint.url)
            self._semaphores[name] = asyncio.Semaphore(endpoint.max_concurrent)

        # Start a worker for each endpoint
        workers = [
            asyncio.create_task(self._worker(endpoint_name))
            for endpoint_name in self.config.endpoints.keys()
        ]

        # Wait for all workers to complete
        await asyncio.gather(*workers)

    def run(self):
        """Run the queue (blocking until all requests complete)."""
        asyncio.run(self._run_async())

    async def run_async(self):
        """Run the queue asynchronously (for use within existing event loops)."""
        await self._run_async()

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

    def get_errors(self) -> list[tuple[str, str, Exception]]:
        """Get list of (clip_id, key, error) tuples for failed requests."""
        return self._errors.copy()
