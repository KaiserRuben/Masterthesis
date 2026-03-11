"""
Test suite for WorkStealingQueue

Tests the real queue system with mocked LLM responses.
Only the AsyncClient.chat() method is mocked - everything else is real.
"""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from .queue import WorkStealingQueue, Request
from .config import VLMConfig, EndpointConfig, ModelEndpointConfig
from .types import Message, CompletionResult

# Import queue module for patching - must match how pytest imports it
# When pytest runs from tools/vlm/, it uses 'vlm.queue', not 'tools.vlm.queue'
import vlm.queue as queue_module


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def mock_async_client():
    """Fixture to patch AsyncClient before tests run."""
    with patch.object(queue_module, 'AsyncClient') as mock_cls:
        yield mock_cls


@pytest.fixture
def single_endpoint_config():
    """Config with single endpoint supporting all models."""
    config = VLMConfig()
    config.endpoints = {
        "default": EndpointConfig(
            url="http://localhost:11434",
            max_concurrent=2,
            timeout_seconds=10,
            retry_attempts=2,
            retry_delay_seconds=0.01,  # Fast retries for tests
        )
    }
    config.models = {
        "small-model": ModelEndpointConfig(
            model="small-model",
            endpoints=["default"],
        ),
        "large-model": ModelEndpointConfig(
            model="large-model",
            endpoints=["default"],
        ),
    }
    return config


@pytest.fixture
def multi_endpoint_config():
    """Config with two endpoints - one supports all models, one only small."""
    config = VLMConfig()
    config.endpoints = {
        "gpu": EndpointConfig(
            url="http://gpu:11434",
            max_concurrent=1,
            timeout_seconds=10,
            retry_attempts=2,
            retry_delay_seconds=0.01,
        ),
        "cpu": EndpointConfig(
            url="http://cpu:11434",
            max_concurrent=2,
            timeout_seconds=10,
            retry_attempts=2,
            retry_delay_seconds=0.01,
        ),
    }
    config.models = {
        "small-model": ModelEndpointConfig(
            model="small-model",
            endpoints=["gpu", "cpu"],  # Available on both
        ),
        "large-model": ModelEndpointConfig(
            model="large-model",
            endpoints=["gpu"],  # Only on GPU
        ),
    }
    return config


def make_mock_response(content: str):
    """Create a mock ollama response."""
    return {
        "message": {"content": content},
        "prompt_eval_count": 100,
        "eval_count": 50,
    }


# =============================================================================
# BASIC QUEUE TESTS
# =============================================================================

class TestBasicQueueOperations:
    """Test basic queue add/complete operations."""

    def test_add_requests(self, single_endpoint_config):
        """Test adding requests to queue."""
        results = []
        queue = WorkStealingQueue(
            config=single_endpoint_config,
            on_result=lambda cid, key, res, ep: results.append((cid, key, res)),
        )

        requests = [
            Request(id="r1", clip_id="clip_A", key="stage1", model="small-model",
                    messages=[Message("user", "test")]),
            Request(id="r2", clip_id="clip_A", key="weather", model="small-model",
                    messages=[Message("user", "test")], depends_on_stage1=True),
        ]
        queue.add_requests(requests)

        assert queue.pending_count == 2
        assert queue.completed_count == 0

    def test_mark_stage1_complete(self, single_endpoint_config):
        """Test marking stage1 as complete."""
        queue = WorkStealingQueue(
            config=single_endpoint_config,
            on_result=lambda *args: None,
        )

        queue.mark_stage1_complete("clip_A", "Scene shows a highway...")

        assert queue.get_stage1_result("clip_A") == "Scene shows a highway..."
        assert queue.get_stage1_result("clip_B") is None


# =============================================================================
# DEPENDENCY TRACKING TESTS
# =============================================================================

class TestDependencyTracking:
    """Test stage1 → stage2 dependency handling."""

    @pytest.mark.asyncio
    async def test_stage2_waits_for_stage1(self, single_endpoint_config, mock_async_client):
        """Stage2 requests should not execute until stage1 completes."""
        results = []
        execution_order = []

        async def mock_chat(**kwargs):
            model = kwargs.get("model", "")
            execution_order.append(model)
            await asyncio.sleep(0.01)
            return make_mock_response(f"Response for {model}")

        mock_client = MagicMock()
        mock_client.chat = mock_chat
        mock_async_client.return_value = mock_client

        queue = WorkStealingQueue(
            config=single_endpoint_config,
            on_result=lambda cid, key, res, ep: results.append((cid, key)),
        )

        requests = [
            Request(id="s1", clip_id="clip_A", key="stage1", model="large-model",
                    messages=[Message("user", "describe")], depends_on_stage1=False),
            Request(id="s2_weather", clip_id="clip_A", key="weather", model="small-model",
                    messages=[Message("user", "weather")], depends_on_stage1=True),
            Request(id="s2_road", clip_id="clip_A", key="road_type", model="small-model",
                    messages=[Message("user", "road")], depends_on_stage1=True),
        ]
        queue.add_requests(requests)

        await queue.run_async()

        # Stage1 should complete first
        assert results[0] == ("clip_A", "stage1")
        # Then stage2 requests (order may vary between them)
        stage2_results = results[1:]
        assert len(stage2_results) == 2
        assert all(r[0] == "clip_A" for r in stage2_results)
        assert set(r[1] for r in stage2_results) == {"weather", "road_type"}

    @pytest.mark.asyncio
    async def test_stage1_failure_removes_dependent_stage2(self, single_endpoint_config, mock_async_client):
        """When stage1 fails, dependent stage2 requests should be removed."""
        results = []

        async def mock_chat_fail_stage1(**kwargs):
            model = kwargs.get("model", "")
            if model == "large-model":
                raise Exception("GPU out of memory")
            return make_mock_response(f"Response for {model}")

        mock_client = MagicMock()
        mock_client.chat = mock_chat_fail_stage1
        mock_async_client.return_value = mock_client

        single_endpoint_config.endpoints["default"].retry_attempts = 1

        queue = WorkStealingQueue(
            config=single_endpoint_config,
            on_result=lambda cid, key, res, ep: results.append((cid, key)),
        )

        requests = [
            Request(id="s1", clip_id="clip_A", key="stage1", model="large-model",
                    messages=[Message("user", "describe")], depends_on_stage1=False),
            Request(id="s2_weather", clip_id="clip_A", key="weather", model="small-model",
                    messages=[Message("user", "weather")], depends_on_stage1=True),
            Request(id="s2_road", clip_id="clip_A", key="road_type", model="small-model",
                    messages=[Message("user", "road")], depends_on_stage1=True),
        ]
        queue.add_requests(requests)

        await queue.run_async()

        # No successful results
        assert len(results) == 0

        # All three should be in errors (stage1 + 2 orphaned stage2)
        queue_errors = queue.get_errors()
        assert len(queue_errors) == 3

        error_keys = [e[1] for e in queue_errors]
        assert "stage1" in error_keys
        assert "weather" in error_keys
        assert "road_type" in error_keys


# =============================================================================
# MULTI-ENDPOINT WORK STEALING TESTS
# =============================================================================

class TestMultiEndpointWorkStealing:
    """Test work distribution across multiple endpoints."""

    @pytest.mark.asyncio
    async def test_large_model_only_on_gpu(self, multi_endpoint_config):
        """Large model requests should only go to GPU endpoint."""
        processed_by_endpoint = {"gpu": [], "cpu": []}

        async def mock_chat(**kwargs):
            await asyncio.sleep(0.01)
            return make_mock_response("response")

        queue = WorkStealingQueue(
            config=multi_endpoint_config,
            on_result=lambda *args: None,
        )

        requests = [
            Request(id="r1", clip_id="clip_A", key="stage1", model="large-model",
                    messages=[Message("user", "test")]),
            Request(id="r2", clip_id="clip_B", key="stage1", model="large-model",
                    messages=[Message("user", "test")]),
        ]
        queue.add_requests(requests)

        # CPU endpoint should not find large-model work
        req_cpu = queue._find_work("cpu")
        assert req_cpu is None  # large-model not supported on CPU

        # GPU endpoint should find large-model work
        req_gpu = queue._find_work("gpu")
        assert req_gpu is not None
        assert req_gpu.model == "large-model"

    @pytest.mark.asyncio
    async def test_small_model_can_run_on_both(self, multi_endpoint_config):
        """Small model requests can go to either endpoint."""
        queue = WorkStealingQueue(
            config=multi_endpoint_config,
            on_result=lambda *args: None,
        )

        requests = [
            Request(id="r1", clip_id="clip_A", key="weather", model="small-model",
                    messages=[Message("user", "test")]),
        ]
        queue.add_requests(requests)

        # Both endpoints should be able to find small-model work
        # (but only one should get it due to in_flight tracking)
        req_gpu = queue._find_work("gpu")
        assert req_gpu is not None
        assert req_gpu.model == "small-model"

        # Now it's in_flight, so cpu shouldn't find it
        req_cpu = queue._find_work("cpu")
        assert req_cpu is None


# =============================================================================
# LOOKAHEAD AND WORK FINDING TESTS
# =============================================================================

class TestLookaheadBehavior:
    """Test lookahead window and fallback behavior."""

    def test_find_work_within_lookahead(self, single_endpoint_config):
        """Work within lookahead window should be found."""
        queue = WorkStealingQueue(
            config=single_endpoint_config,
            on_result=lambda *args: None,
            lookahead=5,
        )

        requests = [
            Request(id=f"r{i}", clip_id=f"clip_{i}", key="test", model="small-model",
                    messages=[Message("user", "test")])
            for i in range(10)
        ]
        queue.add_requests(requests)

        req = queue._find_work("default")
        assert req is not None
        assert req.id == "r0"

    def test_find_work_fallback_beyond_lookahead(self, single_endpoint_config):
        """Work beyond lookahead should be found via fallback."""
        queue = WorkStealingQueue(
            config=single_endpoint_config,
            on_result=lambda *args: None,
            lookahead=3,
        )

        # First 3 requests need stage1 (blocked), 4th is available
        requests = [
            Request(id="blocked_0", clip_id="clip_A", key="weather", model="small-model",
                    messages=[Message("user", "test")], depends_on_stage1=True),
            Request(id="blocked_1", clip_id="clip_B", key="weather", model="small-model",
                    messages=[Message("user", "test")], depends_on_stage1=True),
            Request(id="blocked_2", clip_id="clip_C", key="weather", model="small-model",
                    messages=[Message("user", "test")], depends_on_stage1=True),
            Request(id="available", clip_id="clip_D", key="stage1", model="small-model",
                    messages=[Message("user", "test")], depends_on_stage1=False),
        ]
        queue.add_requests(requests)

        req = queue._find_work("default")
        assert req is not None
        assert req.id == "available"

    def test_find_work_respects_model_compatibility(self, multi_endpoint_config):
        """Work finding should respect model compatibility."""
        queue = WorkStealingQueue(
            config=multi_endpoint_config,
            on_result=lambda *args: None,
            lookahead=10,
        )

        requests = [
            Request(id=f"r{i}", clip_id=f"clip_{i}", key="stage1", model="large-model",
                    messages=[Message("user", "test")])
            for i in range(5)
        ]
        queue.add_requests(requests)

        req_gpu = queue._find_work("gpu")
        assert req_gpu is not None

        # Reset in_flight to test cpu
        queue._in_flight.clear()

        req_cpu = queue._find_work("cpu")
        assert req_cpu is None  # CPU doesn't support large-model


# =============================================================================
# TERMINATION AND COMPLETION TESTS
# =============================================================================

class TestTerminationConditions:
    """Test queue completion and worker termination."""

    def test_all_done_checks_in_flight(self, single_endpoint_config):
        """_all_done should return False if requests are in-flight."""
        queue = WorkStealingQueue(
            config=single_endpoint_config,
            on_result=lambda *args: None,
        )

        requests = [
            Request(id="r1", clip_id="clip_A", key="test", model="small-model",
                    messages=[Message("user", "test")]),
        ]
        queue.add_requests(requests)

        req = queue._find_work("default")
        assert req is not None
        assert len(queue._in_flight) == 1
        assert not queue._all_done()

    def test_all_done_true_when_complete(self, single_endpoint_config):
        """_all_done should return True when queue empty and nothing in-flight."""
        queue = WorkStealingQueue(
            config=single_endpoint_config,
            on_result=lambda *args: None,
        )

        assert queue._all_done()

        requests = [
            Request(id="r1", clip_id="clip_A", key="test", model="small-model",
                    messages=[Message("user", "test")]),
        ]
        queue.add_requests(requests)
        assert not queue._all_done()

        req = queue._find_work("default")
        queue._handle_result(req, CompletionResult(content="done", model="small-model"), "default")

        assert queue._all_done()

    @pytest.mark.asyncio
    async def test_workers_exit_when_done(self, single_endpoint_config, mock_async_client):
        """Workers should exit cleanly when all work is done."""
        results = []

        async def mock_chat(**kwargs):
            await asyncio.sleep(0.01)
            return make_mock_response("response")

        mock_client = MagicMock()
        mock_client.chat = mock_chat
        mock_async_client.return_value = mock_client

        queue = WorkStealingQueue(
            config=single_endpoint_config,
            on_result=lambda cid, key, res, ep: results.append((cid, key)),
        )

        requests = [
            Request(id="r1", clip_id="clip_A", key="test", model="small-model",
                    messages=[Message("user", "test")]),
        ]
        queue.add_requests(requests)

        start = time.time()
        await queue.run_async()
        elapsed = time.time() - start

        assert len(results) == 1
        assert elapsed < 5.0


# =============================================================================
# EVENT NOTIFICATION TESTS
# =============================================================================

class TestEventNotification:
    """Test event-based worker notification."""

    @pytest.mark.asyncio
    async def test_workers_wake_on_stage1_complete(self, single_endpoint_config, mock_async_client):
        """Workers waiting on dependencies should wake when stage1 completes."""
        results = []
        stage1_complete_time = [None]
        stage2_start_times = []

        async def mock_chat(**kwargs):
            messages = kwargs.get("messages", [])
            content = str(messages)
            if "stage1" in content:
                await asyncio.sleep(0.1)
                stage1_complete_time[0] = time.time()
            else:
                stage2_start_times.append(time.time())
                await asyncio.sleep(0.01)
            return make_mock_response("response")

        mock_client = MagicMock()
        mock_client.chat = mock_chat
        mock_async_client.return_value = mock_client

        queue = WorkStealingQueue(
            config=single_endpoint_config,
            on_result=lambda cid, key, res, ep: results.append((cid, key)),
        )

        requests = [
            Request(id="s1", clip_id="clip_A", key="stage1", model="small-model",
                    messages=[Message("user", "stage1 describe")], depends_on_stage1=False),
            Request(id="s2", clip_id="clip_A", key="weather", model="small-model",
                    messages=[Message("user", "weather")], depends_on_stage1=True),
        ]
        queue.add_requests(requests)

        await queue.run_async()

        assert len(results) == 2
        if stage1_complete_time[0] and stage2_start_times:
            delay = stage2_start_times[0] - stage1_complete_time[0]
            assert delay < 0.5  # Should wake up quickly


# =============================================================================
# INTERLEAVING TESTS
# =============================================================================

class TestInterleavedProcessing:
    """Test interleaved request processing."""

    @pytest.mark.asyncio
    async def test_interleaved_clips_process_correctly(self, single_endpoint_config, mock_async_client):
        """Multiple clips with interleaved requests should all complete."""
        results = []

        async def mock_chat(**kwargs):
            await asyncio.sleep(0.01)
            return make_mock_response("response")

        mock_client = MagicMock()
        mock_client.chat = mock_chat
        mock_async_client.return_value = mock_client

        queue = WorkStealingQueue(
            config=single_endpoint_config,
            on_result=lambda cid, key, res, ep: results.append((cid, key)),
        )

        requests = []
        for clip in ["A", "B", "C"]:
            requests.append(Request(
                id=f"s1_{clip}", clip_id=f"clip_{clip}", key="stage1",
                model="small-model", messages=[Message("user", "describe")],
                depends_on_stage1=False
            ))
            requests.append(Request(
                id=f"k1_{clip}", clip_id=f"clip_{clip}", key="weather",
                model="small-model", messages=[Message("user", "weather")],
                depends_on_stage1=True
            ))
        queue.add_requests(requests)

        await queue.run_async()

        assert len(results) == 6

        for clip in ["A", "B", "C"]:
            clip_results = [(cid, key) for cid, key in results if cid == f"clip_{clip}"]
            assert len(clip_results) == 2
            assert (f"clip_{clip}", "stage1") in clip_results
            assert (f"clip_{clip}", "weather") in clip_results


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Test error handling and recovery."""

    @pytest.mark.asyncio
    async def test_retry_on_transient_failure(self, single_endpoint_config, mock_async_client):
        """Transient failures should be retried."""
        results = []
        call_count = [0]

        async def mock_chat_fail_once(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Temporary error")
            return make_mock_response("response")

        mock_client = MagicMock()
        mock_client.chat = mock_chat_fail_once
        mock_async_client.return_value = mock_client

        queue = WorkStealingQueue(
            config=single_endpoint_config,
            on_result=lambda cid, key, res, ep: results.append((cid, key)),
        )

        requests = [
            Request(id="r1", clip_id="clip_A", key="test", model="small-model",
                    messages=[Message("user", "test")]),
        ]
        queue.add_requests(requests)

        await queue.run_async()

        assert len(results) == 1
        assert call_count[0] == 2

    @pytest.mark.asyncio
    async def test_permanent_failure_recorded(self, single_endpoint_config, mock_async_client):
        """Permanent failures should be recorded in errors."""
        results = []

        async def mock_chat_always_fail(**kwargs):
            raise Exception("Permanent error")

        mock_client = MagicMock()
        mock_client.chat = mock_chat_always_fail
        mock_async_client.return_value = mock_client

        single_endpoint_config.endpoints["default"].retry_attempts = 2

        queue = WorkStealingQueue(
            config=single_endpoint_config,
            on_result=lambda cid, key, res, ep: results.append((cid, key)),
        )

        requests = [
            Request(id="r1", clip_id="clip_A", key="test", model="small-model",
                    messages=[Message("user", "test")]),
        ]
        queue.add_requests(requests)

        await queue.run_async()

        assert len(results) == 0

        errors = queue.get_errors()
        assert len(errors) == 1
        assert errors[0][0] == "clip_A"
        assert errors[0][1] == "test"


# =============================================================================
# CONCURRENCY TESTS
# =============================================================================

class TestConcurrency:
    """Test concurrent execution behavior."""

    @pytest.mark.asyncio
    async def test_respects_max_concurrent(self, single_endpoint_config, mock_async_client):
        """Should respect max_concurrent limit."""
        concurrent_count = [0]
        max_concurrent_seen = [0]

        async def mock_chat(**kwargs):
            concurrent_count[0] += 1
            max_concurrent_seen[0] = max(max_concurrent_seen[0], concurrent_count[0])
            await asyncio.sleep(0.02)
            concurrent_count[0] -= 1
            return make_mock_response("response")

        mock_client = MagicMock()
        mock_client.chat = mock_chat
        mock_async_client.return_value = mock_client

        single_endpoint_config.endpoints["default"].max_concurrent = 2

        queue = WorkStealingQueue(
            config=single_endpoint_config,
            on_result=lambda *args: None,
        )

        requests = [
            Request(id=f"r{i}", clip_id=f"clip_{i}", key="test", model="small-model",
                    messages=[Message("user", "test")])
            for i in range(10)
        ]
        queue.add_requests(requests)

        await queue.run_async()

        assert max_concurrent_seen[0] <= 2


# =============================================================================
# STATS TESTS
# =============================================================================

class TestStatistics:
    """Test statistics collection."""

    @pytest.mark.asyncio
    async def test_stats_collected(self, single_endpoint_config, mock_async_client):
        """Statistics should be collected during execution."""
        async def mock_chat(**kwargs):
            await asyncio.sleep(0.01)
            return make_mock_response("response")

        mock_client = MagicMock()
        mock_client.chat = mock_chat
        mock_async_client.return_value = mock_client

        queue = WorkStealingQueue(
            config=single_endpoint_config,
            on_result=lambda *args: None,
        )

        requests = [
            Request(id=f"r{i}", clip_id=f"clip_{i}", key="test", model="small-model",
                    messages=[Message("user", "test")])
            for i in range(5)
        ]
        queue.add_requests(requests)

        await queue.run_async()

        stats = queue.get_stats()
        assert "default" in stats
        assert stats["default"].requests_completed == 5
        assert stats["default"].requests_failed == 0
        assert stats["default"].total_latency_ms > 0


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
