"""
Batch Scene Classification

Classify scenes from the PhysicalAI-AV dataset using the vlm package.
Results are saved incrementally to allow resumption on interruption.

Features:
- Work-stealing queue with multi-endpoint support
- Partial result tracking for graceful interruption/resume
- Interleaved request processing for optimal throughput
"""

import sys
import json
import base64
import time
from io import BytesIO
from pathlib import Path
from datetime import datetime
from typing import Any

import filelock
import pandas as pd
from PIL import Image
import torch
from pydantic import BaseModel

# Add tools to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tools"))
sys.path.insert(0, str(PROJECT_ROOT / "tools" / "alpamayo" / "src"))

from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from vlm import load_config, SyncRequestQueue, Message, VLMConfig, Request, WorkStealingQueue
from scene import STAGE1_PROMPT, KEYS, get_prompt, get_schema, get_response_model

# ============================================================================
# CONFIGURATION
# ============================================================================

NUM_SCENES = 50
SEED = 42  # For reproducibility

DATA_DIR = PROJECT_ROOT / "data" / "EXP-001"
CLIP_IDS_FILE = PROJECT_ROOT / "tools" / "alpamayo" / "notebooks" / "clip_ids.parquet"


def get_run_dir(run_id: str | None = None) -> Path:
    """Get or create a run directory. If run_id is None, creates a new timestamped run."""
    if run_id is None:
        run_id = f"classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return DATA_DIR / run_id


# ============================================================================
# IMAGE PROCESSING
# ============================================================================

def create_composite_image(image_frames: torch.Tensor) -> str:
    """Create a 2x2 composite of all 4 camera views, return as base64."""
    images = []
    for cam_idx in range(4):
        img_tensor = image_frames[cam_idx, -1]
        img_np = img_tensor.permute(1, 2, 0).numpy().astype("uint8")
        images.append(Image.fromarray(img_np))

    w, h = images[0].size
    composite = Image.new("RGB", (w * 2, h * 2))
    positions = [(0, 0), (w, 0), (0, h), (w, h)]
    for img, pos in zip(images, positions):
        composite.paste(img, pos)

    composite.thumbnail((1920, 1080), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    composite.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ============================================================================
# RESULT STORAGE
# ============================================================================

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class ResultStorage:
    """Thread-safe result storage with partial tracking for resume support."""

    def __init__(self, run_dir: Path, all_keys: list[str]):
        self.run_dir = run_dir
        self.progress_file = run_dir / "progress.json"
        self.lock_file = run_dir / ".progress.lock"
        self._file_lock = filelock.FileLock(self.lock_file)
        self.all_keys = all_keys  # All classification keys (excluding stage1)

    def _load(self) -> dict:
        """Load progress from file."""
        if self.progress_file.exists():
            with open(self.progress_file) as f:
                return json.load(f)
        return {"completed": [], "failed": [], "results": [], "partial": {}}

    def _save(self, progress: dict):
        """Save progress to file."""
        with open(self.progress_file, "w") as f:
            json.dump(progress, f, indent=2, cls=NumpyEncoder)

    def save_result(self, clip_id: str, key: str, result: str):
        """
        Save a single result, moving to completed if clip is done.

        Args:
            clip_id: Clip identifier
            key: "stage1" or classification key
            result: Raw result string (JSON for stage2 keys, text for stage1)
        """
        with self._file_lock:
            progress = self._load()

            # Initialize partial entry if needed
            if clip_id not in progress.get("partial", {}):
                progress.setdefault("partial", {})[clip_id] = {}

            # Store result
            progress["partial"][clip_id][key] = result

            # Check if clip is complete
            if self._is_complete(progress["partial"][clip_id]):
                self._finalize_clip(progress, clip_id)

            self._save(progress)

    def _is_complete(self, partial_clip: dict) -> bool:
        """Check if clip has all required keys."""
        if "stage1" not in partial_clip:
            return False
        return all(key in partial_clip for key in self.all_keys)

    def _finalize_clip(self, progress: dict, clip_id: str):
        """Move clip from partial to completed."""
        partial_data = progress["partial"].pop(clip_id)

        # Build classification dict with parsed JSON for stage2 keys
        classification = {"scene_reasoning": partial_data.get("stage1", "")}
        models_used = {"stage1": "unknown"}  # Model tracking not available in queue mode

        for key in self.all_keys:
            if key in partial_data:
                try:
                    classification[key] = json.loads(partial_data[key])
                except json.JSONDecodeError:
                    classification[key] = partial_data[key]
                models_used[key] = "unknown"

        result = {
            "clip_id": clip_id,
            "classification": classification,
            "models_used": models_used,
        }

        progress["results"].append(result)
        progress["completed"].append(clip_id)

    def get_remaining_work(self, clip_ids: list[str]) -> dict[str, list[str]]:
        """
        Get work needed for resume: {clip_id: [missing_keys]}.

        Args:
            clip_ids: All clip IDs to process

        Returns:
            Dict mapping clip_id to list of missing keys (including "stage1" if needed)
        """
        with self._file_lock:
            progress = self._load()

        completed = set(progress.get("completed", []))
        partial = progress.get("partial", {})

        remaining = {}
        for clip_id in clip_ids:
            if clip_id in completed:
                continue

            if clip_id in partial:
                done_keys = set(partial[clip_id].keys())
                missing = [k for k in ["stage1"] + self.all_keys if k not in done_keys]
            else:
                missing = ["stage1"] + self.all_keys

            if missing:
                remaining[clip_id] = missing

        return remaining

    def get_partial_results(self) -> dict[str, dict[str, str]]:
        """Get partial results for building stage2 messages."""
        with self._file_lock:
            progress = self._load()
        return progress.get("partial", {})

    def get_progress(self) -> dict:
        """Get current progress state."""
        with self._file_lock:
            return self._load()

    def save_error(self, clip_id: str, key: str, error: str):
        """Record a failed request."""
        with self._file_lock:
            progress = self._load()
            progress["failed"].append({
                "clip_id": clip_id,
                "key": key,
                "error": error,
            })
            self._save(progress)


# ============================================================================
# REQUEST BUILDING
# ============================================================================

def build_interleaved_queue(
    remaining_work: dict[str, list[str]],
    clip_images: dict[str, str],
    partial_results: dict[str, dict[str, str]],
    config: VLMConfig,
    model_override: str | None = None,
    keys_per_stage1: int = 2,
) -> list[Request]:
    """
    Build interleaved request queue: [s1_A, k1_A, k2_A, s1_B, k1_B, k2_B, ...]

    Args:
        remaining_work: {clip_id: [missing_keys]} from ResultStorage.get_remaining_work()
        clip_images: {clip_id: base64_image} for stage1 requests
        partial_results: {clip_id: {key: result}} for existing stage1 results
        config: VLM configuration
        model_override: If set, use this model for all requests
        keys_per_stage1: Number of stage2 keys to add after each stage1

    Returns:
        List of Request objects in interleaved order
    """
    requests = []

    # Track which stage1s have been added and which keys remain per clip
    added_stage1 = set()
    key_iterators = {}

    for clip_id, missing_keys in remaining_work.items():
        # Separate stage1 and stage2 keys
        needs_stage1 = "stage1" in missing_keys
        stage2_keys = [k for k in missing_keys if k != "stage1"]
        key_iterators[clip_id] = iter(stage2_keys)

        if needs_stage1:
            added_stage1.add(clip_id)

    # Build interleaved queue
    pending_clips = set(remaining_work.keys())
    round_robin_order = list(pending_clips)

    while pending_clips:
        for clip_id in list(round_robin_order):
            if clip_id not in pending_clips:
                continue

            missing_keys = remaining_work[clip_id]
            needs_stage1 = "stage1" in missing_keys and clip_id in added_stage1

            # Add stage1 request if needed (and not yet added to queue)
            stage1_added_this_round = False
            if needs_stage1 and not any(r.clip_id == clip_id and r.key == "stage1" for r in requests):
                stage1_model = model_override or config.get_model_for_key("stage1")
                img_b64 = clip_images.get(clip_id)
                if img_b64:
                    requests.append(Request(
                        id=f"{clip_id}:stage1",
                        clip_id=clip_id,
                        key="stage1",
                        model=stage1_model,
                        messages=[
                            Message("system", STAGE1_PROMPT),
                            Message("user", "Describe what you see in detail.", images=(img_b64,)),
                        ],
                        depends_on_stage1=False,
                    ))
                    stage1_added_this_round = True
                    added_stage1.discard(clip_id)  # Mark as added

            # Add next N stage2 keys for this clip
            key_iter = key_iterators[clip_id]
            keys_added = 0
            for _ in range(keys_per_stage1):
                try:
                    key = next(key_iter)
                    key_model = model_override or config.get_model_for_key(key)

                    # Check if we have stage1 result (from partial or will be computed)
                    has_stage1 = (
                        clip_id in partial_results and "stage1" in partial_results[clip_id]
                    ) or any(r.clip_id == clip_id and r.key == "stage1" for r in requests)

                    requests.append(Request(
                        id=f"{clip_id}:{key}",
                        clip_id=clip_id,
                        key=key,
                        model=key_model,
                        messages=[],  # Will be built dynamically when stage1 completes
                        json_schema=get_schema(key),
                        depends_on_stage1=has_stage1 and clip_id not in partial_results,
                    ))
                    keys_added += 1
                except StopIteration:
                    pending_clips.discard(clip_id)
                    break

            # If no stage1 was added and no keys were added, remove from pending
            if not stage1_added_this_round and keys_added == 0:
                pending_clips.discard(clip_id)

    return requests


class DynamicMessageQueue(WorkStealingQueue):
    """
    WorkStealingQueue extension that builds stage2 messages dynamically.

    Stage2 requests are created with empty messages and populated just before
    execution using the stage1 result from the queue's internal state.
    """

    def __init__(
        self,
        config: VLMConfig,
        on_result,
        partial_results: dict[str, dict[str, str]],
        lookahead: int = 10,
    ):
        super().__init__(config, on_result, lookahead)
        # Pre-populate stage1 results from partial progress
        for clip_id, results in partial_results.items():
            if "stage1" in results:
                self._stage1_results[clip_id] = results["stage1"]

    def _find_work(self, endpoint: str) -> Request | None:
        """Find next compatible request, building messages if needed."""
        req = super()._find_work(endpoint)
        if req is None:
            return None

        # Build stage2 messages dynamically if empty
        if req.key != "stage1" and not req.messages:
            stage1_result = self.get_stage1_result(req.clip_id)
            if stage1_result:
                req.messages.extend([
                    Message("system", get_prompt(req.key)),
                    Message("user", f"SCENE: {stage1_result}"),
                ])

        return req


# ============================================================================
# CLASSIFICATION (Legacy - kept for backward compatibility)
# ============================================================================

def classify_scene(
    image_frames: torch.Tensor,
    queue: SyncRequestQueue,
    config: VLMConfig,
    model_override: str | None = None,
) -> tuple[dict[str, BaseModel], dict[str, str]]:
    """
    Classify a scene using two-stage approach:
    1. Generate scene reasoning (image -> text)
    2. Classify each key (text -> structured response)

    Args:
        image_frames: Tensor of camera frames
        queue: Request queue for VLM inference
        config: VLM configuration with per-key model mappings
        model_override: If set, use this model for all keys instead of per-key config

    Returns:
        Tuple of (results dict, models_used dict) where models_used maps each key
        to the model that was used for classification.
    """
    img_b64 = create_composite_image(image_frames)

    # Track which model was used for each key
    models_used: dict[str, str] = {}

    # Stage 1: Generate scene reasoning
    stage1_model = model_override or config.get_model_for_key("stage1")
    models_used["stage1"] = stage1_model

    reasoning_result = queue.submit(
        model=stage1_model,
        messages=[
            Message("system", STAGE1_PROMPT),
            Message("user", "Describe what you see in detail.", images=(img_b64,)),
        ],
    )
    reasoning = reasoning_result.content

    # Stage 2: Classify each key
    results: dict[str, BaseModel] = {"scene_reasoning": reasoning}  # type: ignore

    for key in KEYS:
        key_model = model_override or config.get_model_for_key(key)
        models_used[key] = key_model

        response_model = get_response_model(key)
        result = queue.submit(
            model=key_model,
            messages=[
                Message("system", get_prompt(key)),
                Message("user", f"SCENE: {reasoning}"),
            ],
            json_schema=get_schema(key),
        )
        # Parse directly to Pydantic model
        results[key] = response_model.model_validate_json(result.content)

    return results, models_used


def format_result_for_display(results: dict[str, Any]) -> str:
    """Format classification results for console display."""
    road = results.get("road_type")
    weather = results.get("weather")
    traffic = results.get("traffic_situation")
    vehicles = results.get("vehicle_count")

    # Extract values from Pydantic models
    road_str = road.road_type.value if hasattr(road, "road_type") else "?"
    weather_str = weather.weather.value if hasattr(weather, "weather") else "?"
    traffic_str = traffic.category.value if hasattr(traffic, "category") else "?"
    vehicles_str = vehicles.vehicle_count if hasattr(vehicles, "vehicle_count") else "?"

    return f"{road_str}, {weather_str}, {traffic_str}, vehicles={vehicles_str}"


def serialize_results(results: dict[str, Any]) -> dict[str, Any]:
    """Convert Pydantic models to JSON-serializable dicts."""
    serialized = {}
    for key, value in results.items():
        if isinstance(value, BaseModel):
            serialized[key] = value.model_dump(mode="json")
        else:
            serialized[key] = value
    return serialized


# ============================================================================
# PROGRESS MANAGEMENT
# ============================================================================

def load_run_config(run_dir: Path) -> dict | None:
    """Load run config if exists (for resuming)."""
    config_file = run_dir / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            return json.load(f)
    return None


def save_run_config(run_dir: Path, config: dict):
    """Save run config."""
    config_file = run_dir / "config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)


def load_progress(run_dir: Path) -> dict:
    """Load progress from file if exists."""
    progress_file = run_dir / "progress.json"
    if progress_file.exists():
        with open(progress_file) as f:
            return json.load(f)
    return {"completed": [], "failed": [], "results": []}


def save_progress(progress: dict, run_dir: Path):
    """Save progress to file."""
    progress_file = run_dir / "progress.json"
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)


def save_results(progress: dict, run_config: dict, run_dir: Path):
    """Save final results to output file."""
    output = {
        "metadata": {
            "model_override": run_config.get("model_override"),
            "model_tiers": run_config.get("model_tiers"),
            "endpoint": run_config["endpoint"],
            "num_scenes": len(progress["results"]),
            "num_failed": len(progress["failed"]),
            "completed_at": datetime.now().isoformat(),
        },
        "classifications": progress["results"]
    }
    output_file = run_dir / "scene_classifications.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    import random

    parser = argparse.ArgumentParser(description="Batch scene classification")
    parser.add_argument("--num-scenes", type=int, default=NUM_SCENES,
                        help=f"Number of scenes to classify (default: {NUM_SCENES})")
    parser.add_argument("--model", type=str, help="Override model (default: from config)")
    parser.add_argument("--config", type=Path, help="Path to vlm_config.yaml")
    parser.add_argument("--run-id", type=str, help="Resume existing run or specify run name")
    parser.add_argument("--use-queue", action="store_true",
                        help="Use work-stealing queue for processing (enables partial tracking)")
    parser.add_argument("--keys-per-stage1", type=int, default=2,
                        help="Number of stage2 keys to interleave per stage1 (default: 2)")
    args = parser.parse_args()

    print("=" * 80)
    print("BATCH SCENE CLASSIFICATION")
    print("=" * 80)

    # Load VLM configuration
    config = load_config(args.config)
    model_override = args.model  # None means use per-key config

    print(f"\nConfiguration:")
    if model_override:
        print(f"  Model override: {model_override} (all keys)")
    else:
        print(f"  Model tiers: {config.model_tiers}")
        print(f"  Stage1 model: {config.get_model_for_key('stage1')}")
    print(f"  Endpoint: {config.endpoints['default'].url}")

    # Setup run directory
    run_dir = get_run_dir(args.run_id)
    images_dir = run_dir / "images"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)
    print(f"  Run dir: {run_dir}")

    # Load or create run config
    run_config = load_run_config(run_dir)
    if run_config:
        # Resume: use stored clip list
        selected_clips = run_config["selected_clips"]
        # Backward compatibility: old runs stored "model" instead of "model_override"
        if "model" in run_config and "model_override" not in run_config:
            run_config["model_override"] = run_config["model"]
            run_config["model_tiers"] = config.model_tiers
        # Use stored model_override if present (to maintain consistency within a run)
        if run_config.get("model_override"):
            model_override = run_config["model_override"]
        print(f"\nResuming run with {len(selected_clips)} clips")
    else:
        # New run: sample and store clips
        random.seed(SEED)
        clip_ids_df = pd.read_parquet(CLIP_IDS_FILE)
        all_clip_ids = clip_ids_df["clip_id"].tolist()
        print(f"\nTotal clips available: {len(all_clip_ids)}")

        num_scenes = args.num_scenes
        selected_clips = random.sample(all_clip_ids, min(num_scenes, len(all_clip_ids)))

        run_config = {
            "model_override": model_override,
            "model_tiers": config.model_tiers,
            "endpoint": config.endpoints["default"].url,
            "seed": SEED,
            "num_scenes": len(selected_clips),
            "selected_clips": selected_clips,
            "created_at": datetime.now().isoformat(),
        }
        save_run_config(run_dir, run_config)
        print(f"Selected {len(selected_clips)} clips for classification")

    if args.use_queue:
        # =================================================================
        # Work-stealing queue mode (new)
        # =================================================================
        storage = ResultStorage(run_dir, list(KEYS))
        remaining_work = storage.get_remaining_work(selected_clips)

        if not remaining_work:
            print("\nAll clips already complete!")
            progress = storage.get_progress()
            save_results(progress, run_config, run_dir)
            return

        # Count total requests
        total_requests = sum(len(keys) for keys in remaining_work.values())
        print(f"\nRemaining work: {len(remaining_work)} clips, {total_requests} requests")

        # Load images for remaining clips that need stage1
        print("\nLoading images for clips needing stage1...")
        t0_us = 5_000_000
        clip_images = {}
        load_start = time.time()

        for clip_id, missing_keys in remaining_work.items():
            if "stage1" in missing_keys:
                data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
                img_b64 = create_composite_image(data["image_frames"])
                clip_images[clip_id] = img_b64

                # Save composite image
                img_path = images_dir / f"{clip_id}.jpg"
                if not img_path.exists():
                    with open(img_path, "wb") as f:
                        f.write(base64.b64decode(img_b64))

        print(f"  Loaded {len(clip_images)} images in {time.time() - load_start:.1f}s")

        # Build interleaved queue
        partial_results = storage.get_partial_results()
        requests = build_interleaved_queue(
            remaining_work=remaining_work,
            clip_images=clip_images,
            partial_results=partial_results,
            config=config,
            model_override=model_override,
            keys_per_stage1=args.keys_per_stage1,
        )

        print(f"  Built queue with {len(requests)} requests")

        # Process with work-stealing queue
        print("\n" + "-" * 80)
        print(f"Starting classification at {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 80, flush=True)

        batch_start = time.time()
        completed_count = [0]  # Use list for closure mutability

        def on_result(clip_id: str, key: str, result: str, endpoint: str):
            storage.save_result(clip_id, key, result)
            completed_count[0] += 1
            if completed_count[0] % 10 == 0 or key == "stage1":
                pct = 100 * completed_count[0] / len(requests)
                ep_tag = "L" if "local" in endpoint else "R"  # L=localhost, R=remote
                print(f"  [{completed_count[0]}/{len(requests)}] ({pct:.0f}%) [{ep_tag}] {clip_id[:12]}:{key}", flush=True)

        queue = DynamicMessageQueue(
            config=config,
            on_result=on_result,
            partial_results=partial_results,
            lookahead=args.keys_per_stage1 * len(remaining_work) + 10,
        )
        queue.add_requests(requests)

        try:
            queue.run()
        except KeyboardInterrupt:
            print("\n\nInterrupted! Progress saved to partial state.")

        # Handle errors
        for clip_id, key, error in queue.get_errors():
            storage.save_error(clip_id, key, str(error))
            print(f"  x Error [{clip_id}:{key}]: {error}")

        # Batch summary
        elapsed = time.time() - batch_start
        print("\n" + "-" * 80)
        print(f"Batch completed in {elapsed/60:.1f} minutes")
        print(f"  Completed: {queue.completed_count}")
        print(f"  Failed: {queue._failed_count}")

        # Print queue stats
        print("\n" + queue.get_stats_summary())

        # Final save
        progress = storage.get_progress()
        save_results(progress, run_config, run_dir)

    else:
        # =================================================================
        # Legacy serial mode
        # =================================================================
        progress = load_progress(run_dir)
        completed_clips = set(progress["completed"])
        print(f"Already completed: {len(completed_clips)}")

        # Filter to remaining clips
        remaining_clips = [c for c in selected_clips if c not in completed_clips]
        print(f"Remaining to classify: {len(remaining_clips)}")

        if not remaining_clips:
            print("\nAll clips already classified!")
            save_results(progress, run_config, run_dir)
            return

        # Process clips with request queue
        print("\n" + "-" * 80)
        print(f"Starting classification at {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 80, flush=True)

        batch_start = time.time()
        classify_times = []

        with SyncRequestQueue(config) as queue:
            for i, clip_id in enumerate(remaining_clips):
                n_done = len(progress["completed"])
                n_total = len(selected_clips)
                pct = 100 * n_done / n_total if n_total > 0 else 0

                # ETA calculation
                if classify_times:
                    avg_time = sum(classify_times) / len(classify_times)
                    eta_secs = avg_time * (len(remaining_clips) - i)
                    eta_str = f"ETA {eta_secs/60:.0f}m" if eta_secs > 60 else f"ETA {eta_secs:.0f}s"
                else:
                    eta_str = "ETA --"

                print(f"\n[{n_done + 1}/{n_total}] ({pct:.0f}%) {clip_id[:20]}... {eta_str}", flush=True)

                try:
                    t0_us = 5_000_000
                    clip_start = time.time()

                    # Load data
                    data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
                    load_time = time.time() - clip_start

                    # Save composite image
                    img_b64 = create_composite_image(data["image_frames"])
                    img_path = images_dir / f"{clip_id}.jpg"
                    with open(img_path, "wb") as f:
                        f.write(base64.b64decode(img_b64))

                    # Classify
                    classify_start = time.time()
                    classification, models_used = classify_scene(
                        data["image_frames"], queue, config, model_override
                    )
                    classify_time = time.time() - classify_start
                    classify_times.append(classify_time)

                    # Store result (serialize Pydantic models to dicts)
                    result = {
                        "clip_id": clip_id,
                        "t0_us": t0_us,
                        "classification": serialize_results(classification),
                        "models_used": models_used,
                        "load_time_s": round(load_time, 2),
                        "classify_time_s": round(classify_time, 2),
                    }
                    progress["results"].append(result)
                    progress["completed"].append(clip_id)

                    # Display result
                    display = format_result_for_display(classification)
                    total_time = time.time() - clip_start
                    print(f"  -> {display}", flush=True)
                    print(f"  -> {total_time:.1f}s (load {load_time:.1f}s + classify {classify_time:.1f}s)", flush=True)

                except Exception as e:
                    print(f"  x Error: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    progress["failed"].append({"clip_id": clip_id, "error": str(e)})

                # Save progress after each clip
                save_progress(progress, run_dir)

            # Batch summary
            elapsed = time.time() - batch_start
            print("\n" + "-" * 80)
            print(f"Batch completed in {elapsed/60:.1f} minutes")
            if classify_times:
                print(f"Avg classify time: {sum(classify_times)/len(classify_times):.1f}s")

            # Print queue stats
            print("\n" + queue.get_stats_summary())

        # Final save
        save_results(progress, run_config, run_dir)

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    final_progress = progress if not args.use_queue else storage.get_progress()
    print(f"Successfully classified: {len(final_progress['results'])}")
    print(f"Failed: {len(final_progress['failed'])}")
    print(f"\nResults saved to: {run_dir}")


if __name__ == "__main__":
    main()
