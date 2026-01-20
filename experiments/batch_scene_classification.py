"""
Batch Scene Classification

Classify scenes from the PhysicalAI-AV dataset using the vlm package.
Results are saved incrementally to allow resumption on interruption.
"""

import sys
import json
import base64
import time
from io import BytesIO
from pathlib import Path
from datetime import datetime
from typing import Any

import pandas as pd
from PIL import Image
import torch
from pydantic import BaseModel

# Add tools to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tools"))
sys.path.insert(0, str(PROJECT_ROOT / "tools" / "alpamayo" / "src"))

from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from vlm import load_config, SyncRequestQueue, Message
from scene import STAGE1_PROMPT, KEYS, get_prompt, get_schema, get_response_model

# ============================================================================
# CONFIGURATION
# ============================================================================

NUM_SCENES = 500
SEED = 42  # For reproducibility

SCRIPT_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = SCRIPT_DIR / "classification_results"
OUTPUT_FILE = OUTPUT_DIR / "scene_classifications.json"
PROGRESS_FILE = OUTPUT_DIR / "progress.json"
IMAGES_DIR = OUTPUT_DIR / "images"
CLIP_IDS_FILE = PROJECT_ROOT / "tools" / "alpamayo" / "notebooks" / "clip_ids.parquet"


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
# CLASSIFICATION
# ============================================================================

def classify_scene(
    image_frames: torch.Tensor,
    queue: SyncRequestQueue,
    model: str,
) -> dict[str, BaseModel]:
    """
    Classify a scene using two-stage approach:
    1. Generate scene reasoning (image -> text)
    2. Classify each key (text -> structured response)

    Returns:
        Dict mapping key names to Pydantic response models.
    """
    img_b64 = create_composite_image(image_frames)

    # Stage 1: Generate scene reasoning
    reasoning_result = queue.submit(
        model=model,
        messages=[
            Message("system", STAGE1_PROMPT),
            Message("user", "Describe what you see in detail.", images=(img_b64,)),
        ],
    )
    reasoning = reasoning_result.content

    # Stage 2: Classify each key
    results: dict[str, BaseModel] = {"scene_reasoning": reasoning}  # type: ignore

    for key in KEYS:
        response_model = get_response_model(key)
        result = queue.submit(
            model=model,
            messages=[
                Message("system", get_prompt(key)),
                Message("user", f"SCENE: {reasoning}"),
            ],
            json_schema=get_schema(key),
        )
        # Parse directly to Pydantic model
        results[key] = response_model.model_validate_json(result.content)

    return results


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

def load_progress() -> dict:
    """Load progress from file if exists."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed": [], "failed": [], "results": []}


def save_progress(progress: dict):
    """Save progress to file."""
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def save_results(progress: dict, config_info: dict):
    """Save final results to output file."""
    output = {
        "metadata": {
            "config": config_info,
            "num_scenes": len(progress["results"]),
            "num_failed": len(progress["failed"]),
            "seed": SEED,
            "timestamp": datetime.now().isoformat(),
        },
        "classifications": progress["results"]
    }
    with open(OUTPUT_FILE, "w") as f:
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
    args = parser.parse_args()

    print("=" * 80)
    print("BATCH SCENE CLASSIFICATION")
    print("=" * 80)

    # Load VLM configuration
    config = load_config(args.config)
    model = args.model or config.get_model_for_key("stage1")

    print(f"\nConfiguration:")
    print(f"  Model: {model}")
    print(f"  Endpoint: {config.endpoints['default'].url}")

    # Setup directories
    OUTPUT_DIR.mkdir(exist_ok=True)
    IMAGES_DIR.mkdir(exist_ok=True)
    random.seed(SEED)

    # Load clip IDs
    clip_ids_df = pd.read_parquet(CLIP_IDS_FILE)
    all_clip_ids = clip_ids_df["clip_id"].tolist()
    print(f"\nTotal clips available: {len(all_clip_ids)}")

    # Sample clips
    num_scenes = args.num_scenes
    selected_clips = random.sample(all_clip_ids, min(num_scenes, len(all_clip_ids)))
    print(f"Selected {len(selected_clips)} clips for classification")

    # Load progress
    progress = load_progress()
    completed_clips = set(progress["completed"])
    print(f"Already completed: {len(completed_clips)}")

    # Filter to remaining clips
    remaining_clips = [c for c in selected_clips if c not in completed_clips]
    print(f"Remaining to classify: {len(remaining_clips)}")

    if not remaining_clips:
        print("\nAll clips already classified!")
        config_info = {"model": model, "endpoint": config.endpoints["default"].url}
        save_results(progress, config_info)
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
                img_path = IMAGES_DIR / f"{clip_id}.jpg"
                with open(img_path, "wb") as f:
                    f.write(base64.b64decode(img_b64))

                # Classify
                classify_start = time.time()
                classification = classify_scene(data["image_frames"], queue, model)
                classify_time = time.time() - classify_start
                classify_times.append(classify_time)

                # Store result (serialize Pydantic models to dicts)
                result = {
                    "clip_id": clip_id,
                    "t0_us": t0_us,
                    "classification": serialize_results(classification),
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
            save_progress(progress)

        # Batch summary
        elapsed = time.time() - batch_start
        print("\n" + "-" * 80)
        print(f"Batch completed in {elapsed/60:.1f} minutes")
        if classify_times:
            print(f"Avg classify time: {sum(classify_times)/len(classify_times):.1f}s")

        # Print queue stats
        print("\n" + queue.get_stats_summary())

    # Final save
    config_info = {"model": model, "endpoint": config.endpoints["default"].url}
    save_results(progress, config_info)

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"Successfully classified: {len(progress['results'])}")
    print(f"Failed: {len(progress['failed'])}")
    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
