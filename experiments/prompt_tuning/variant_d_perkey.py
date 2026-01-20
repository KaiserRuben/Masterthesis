"""
Variant D: Per-Key Classification with Additive Scoring

Architecture:
    Image → [Stage 1: Scene Reasoning] → shared_reasoning
                                              ↓
             ┌────────────────────────────────┼────────────────────────────────┐
             ↓                                ↓                                ↓
        [small tier]                    [medium tier]                   [large tier]
        pedestrians                     weather                          traffic_situation
        cyclists                        time_of_day                      notable_elements
        construction                    vehicle_count
        signals                         road_type

Each key is classified independently using shared reasoning, allowing:
- Model selection per capability tier (configurable)
- Transparent scoring for complex keys
- Independent failure modes

Configuration:
    Uses vlm_config.yaml or environment variables.
    See: python scene_keys.py --export-env
"""

import json
import sys
import base64
from io import BytesIO
from enum import Enum
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel, Field as PydanticField
import torch

# Add tools to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tools"))
sys.path.insert(0, str(PROJECT_ROOT / "tools" / "alpamayo" / "src"))

from vlm import load_config, SyncRequestQueue, Message, VLMConfig
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

# Import from scene package (tools/scene/)
from scene import (
    STAGE1_PROMPT,
    KEYS,
    KEYS_ORIGINAL,
    get_prompt,
    get_schema,
    extract_value,
    # Enums
    RoadType,
    Weather,
    TimeOfDay,
    OcclusionLevel,
    DepthComplexity,
    VisualDegradation,
    SafetyCriticality,
    # Models
    TrafficSituationResult,
    DistanceEstimate,
    CountingResult,
)


# =============================================================================
# CLASSIFICATION RESULT MODEL (variant-specific)
# =============================================================================

class PerKeyClassification(BaseModel):
    """Complete per-key classification result."""
    scene_reasoning: str
    # Scene context
    road_type: RoadType
    weather: Weather
    time_of_day: TimeOfDay
    traffic_situation: TrafficSituationResult
    # Object detection
    pedestrians_present: bool
    cyclists_present: bool
    construction_activity: bool
    traffic_signals_visible: bool
    vehicle_count: int = PydanticField(ge=0, le=50)
    notable_elements: list[str]
    # Spatial reasoning (extended)
    occlusion_level: OcclusionLevel | None = None
    depth_complexity: DepthComplexity | None = None
    nearest_vehicle_distance: DistanceEstimate | None = None
    # Perceptual (extended)
    visual_degradation: VisualDegradation | None = None
    similar_object_confusion: bool | None = None
    # Safety (extended)
    safety_criticality: SafetyCriticality | None = None
    # Counting (extended)
    pedestrian_count: CountingResult | None = None


# =============================================================================
# IMAGE PROCESSING
# =============================================================================

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


# =============================================================================
# CLASSIFICATION FUNCTIONS
# =============================================================================

def generate_scene_reasoning(
    image_b64: str,
    queue: SyncRequestQueue,
    model: str,
) -> str:
    """Stage 1: Generate detailed scene description from image."""
    result = queue.submit(
        model=model,
        messages=[
            Message("system", STAGE1_PROMPT),
            Message("user", "Describe what you see in detail.", images=(image_b64,)),
        ],
    )
    return result.content


def classify_key(
    key: str,
    reasoning: str,
    queue: SyncRequestQueue,
    model: str,
) -> Any:
    """Stage 2: Classify a single key using the shared reasoning."""
    result = queue.submit(
        model=model,
        messages=[
            Message("system", get_prompt(key)),
            Message("user", f"SCENE: {reasoning}"),
        ],
        json_schema=get_schema(key),
    )

    parsed = result.parse_json()
    return extract_value(key, parsed)


def classify_scene_perkey(
    image_frames: torch.Tensor,
    queue: SyncRequestQueue,
    config: VLMConfig,
    model_overrides: dict[str, str] | None = None,
    extended_keys: list[str] | None = None,
    verbose: bool = False,
) -> PerKeyClassification:
    """
    Orchestrator: Generate reasoning, then classify each key.

    Args:
        image_frames: Tensor of camera images
        queue: Request queue for VLM calls
        config: VLM configuration (for key->model mapping)
        model_overrides: Optional dict of {key: model_name} overrides
        extended_keys: Optional list of extended keys to also classify
        verbose: Print progress
    """
    model_overrides = model_overrides or {}
    extended_keys = extended_keys or []

    # Stage 1: Generate reasoning
    stage1_model = model_overrides.get("stage1", config.get_model_for_key("stage1"))
    if verbose:
        print(f"Stage 1: Generating scene reasoning ({stage1_model})...")

    img_b64 = create_composite_image(image_frames)
    reasoning = generate_scene_reasoning(img_b64, queue, stage1_model)

    if verbose:
        print(f"Reasoning: {reasoning[:200]}...")
        print()

    # Stage 2: Classify each key (original keys + any requested extended keys)
    keys_to_classify = KEYS_ORIGINAL + [k for k in extended_keys if k not in KEYS_ORIGINAL]
    results: dict[str, Any] = {"scene_reasoning": reasoning}

    for key in keys_to_classify:
        model = model_overrides.get(key, config.get_model_for_key(key))
        if verbose:
            print(f"  Classifying {key} ({model})...")

        results[key] = classify_key(key, reasoning, queue, model)

    return PerKeyClassification(**results)


# =============================================================================
# ANALYSIS MODE
# =============================================================================

def run_analysis_mode(
    image_frames: torch.Tensor,
    queue: SyncRequestQueue,
    config: VLMConfig,
    models: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Run all keys on all specified models for capability comparison.

    Returns:
        Dict of {model_name: {key: result}} for analysis
    """
    from vlm import get_model_tiers
    tiers = get_model_tiers()
    models = models or [tiers["small"], tiers["medium"], tiers["large"]]

    # Generate reasoning once with the largest model
    stage1_model = config.get_model_for_key("stage1")
    print(f"Generating scene reasoning ({stage1_model})...")
    img_b64 = create_composite_image(image_frames)
    reasoning = generate_scene_reasoning(img_b64, queue, stage1_model)
    print(f"Reasoning: {reasoning[:200]}...")
    print()

    results: dict[str, dict[str, Any]] = {}

    for model_name in models:
        print(f"\n=== Testing {model_name} ===")
        results[model_name] = {"scene_reasoning": reasoning}

        for key in KEYS:
            print(f"  {key}...", end=" ", flush=True)
            try:
                value = classify_key(key, reasoning, queue, model_name)
                results[model_name][key] = value
                _print_compact_result(value)
            except Exception as e:
                print(f"ERROR: {e}")
                results[model_name][key] = {"error": str(e)}

    return results


def _print_compact_result(value: Any):
    """Print a compact representation of a result value."""
    if isinstance(value, bool):
        print("yes" if value else "no")
    elif isinstance(value, (str, int)):
        print(value)
    elif isinstance(value, Enum):
        print(value.value)
    elif isinstance(value, TrafficSituationResult):
        print(f"{value.category.value} ({value.total} pts)")
    elif isinstance(value, list):
        print(f"[{len(value)} items]")
    else:
        print("ok")


# =============================================================================
# SERIALIZATION
# =============================================================================

def serialize(obj: Any) -> Any:
    """Convert enums and pydantic models for JSON serialization."""
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize(v) for v in obj]
    return obj


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    from vlm import DEFAULT_CONFIG_TEMPLATE

    parser = argparse.ArgumentParser(description="Per-key scene classification")
    parser.add_argument("--mode", choices=["production", "analysis"], default="production",
                        help="Run mode: production (tiered) or analysis (all models)")
    parser.add_argument("--config", type=Path, help="Path to vlm_config.yaml")
    parser.add_argument("--init-config", action="store_true",
                        help="Generate a sample vlm_config.yaml and exit")
    parser.add_argument("--export-keys", action="store_true",
                        help="Export key definitions as JSON and exit")
    parser.add_argument("--clip-id", default="030c760c-ae38-49aa-9ad8-f5650a545d26",
                        help="Clip ID to classify")
    parser.add_argument("--t0-us", type=int, default=5_100_000,
                        help="Timestamp in microseconds")
    parser.add_argument("--output", type=Path, help="Output JSON file")
    args = parser.parse_args()

    # Handle --init-config
    if args.init_config:
        config_path = args.config or Path("vlm_config.yaml")
        with open(config_path, "w") as f:
            f.write(DEFAULT_CONFIG_TEMPLATE.strip())
        print(f"Created: {config_path}")
        return

    # Handle --export-keys
    if args.export_keys:
        print(json.dumps(export_json(), indent=2))
        return

    # Load configuration
    config = load_config(args.config)

    print("Variant D: Per-Key Classification")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Clip: {args.clip_id}")
    print(f"Config: {args.config or 'default'}")
    print()

    # Show endpoint configuration
    print("Endpoints:")
    for name, endpoint in config.endpoints.items():
        print(f"  {name}: {endpoint.url} (max_concurrent={endpoint.max_concurrent})")
    print()

    # Load data
    data = load_physical_aiavdataset(args.clip_id, t0_us=args.t0_us)

    # Run with request queue
    with SyncRequestQueue(config) as queue:
        if args.mode == "production":
            classification = classify_scene_perkey(
                data["image_frames"],
                queue,
                config,
                verbose=True,
            )

            result = {
                "variant": "D_perkey",
                "mode": "production",
                "clip_id": args.clip_id,
                "t0_us": args.t0_us,
                "classification": classification.model_dump(),
            }

            print("\n" + "=" * 50)
            print("Classification Result:")
            print(classification.model_dump_json(indent=2))

        else:
            analysis = run_analysis_mode(data["image_frames"], queue, config)

            result = {
                "variant": "D_perkey",
                "mode": "analysis",
                "clip_id": args.clip_id,
                "t0_us": args.t0_us,
                "analysis": serialize(analysis),
            }

            print("\n" + "=" * 50)
            print("Analysis complete. Results by model:")
            for model_name in analysis:
                print(f"\n{model_name}:")
                for key, value in analysis[model_name].items():
                    if key == "scene_reasoning":
                        continue
                    print(f"  {key}: {serialize(value)}")

        # Print queue stats
        print("\n" + queue.get_stats_summary())

    # Save output
    output_path = args.output or Path(__file__).parent / f"result_d_perkey_{args.mode}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
