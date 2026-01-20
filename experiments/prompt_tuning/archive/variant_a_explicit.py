"""
Variant A: Explicit Instructions
- More detailed prompt about edge cases
- Same schema but clearer guidance
"""

import json
import base64
from io import BytesIO
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field
from PIL import Image
import torch
from ollama import Client

from alpamayo_r1 import load_physical_aiavdataset

# ============================================================================
# SCHEMA (same as original)
# ============================================================================

class RoadType(str, Enum):
    highway = "highway"
    urban_street = "urban_street"
    residential = "residential"
    intersection = "intersection"
    parking_lot = "parking_lot"
    construction_zone = "construction_zone"
    rural = "rural"
    unknown = "unknown"

class WeatherCondition(str, Enum):
    clear = "clear"
    cloudy = "cloudy"
    rainy = "rainy"
    foggy = "foggy"
    snowy = "snowy"
    unknown = "unknown"

class TimeOfDay(str, Enum):
    day = "day"
    dawn_dusk = "dawn_dusk"
    night = "night"
    unknown = "unknown"

class TrafficDensity(str, Enum):
    empty = "empty"
    light = "light"
    moderate = "moderate"
    heavy = "heavy"
    unknown = "unknown"

class EgoAction(str, Enum):
    driving_straight = "driving_straight"
    turning_left = "turning_left"
    turning_right = "turning_right"
    lane_change_left = "lane_change_left"
    lane_change_right = "lane_change_right"
    stopping = "stopping"
    accelerating = "accelerating"
    decelerating = "decelerating"
    stationary = "stationary"
    unknown = "unknown"

class SceneClassification(BaseModel):
    road_type: RoadType
    weather: WeatherCondition
    time_of_day: TimeOfDay
    traffic_density: TrafficDensity
    ego_action: EgoAction
    pedestrians_visible: bool
    cyclists_visible: bool
    traffic_lights_visible: bool
    stop_signs_visible: bool
    num_vehicles_approx: int = Field(ge=0, le=50)
    scene_description: str
    notable_elements: list[str]

# ============================================================================
# ENHANCED PROMPT
# ============================================================================

SYSTEM_PROMPT = """You are analyzing driving scenes from an autonomous vehicle with 4 cameras.
Image layout:
- Top-left: Cross-Left 120° (left peripheral)
- Top-right: Front-Wide 120° (main forward view)
- Bottom-left: Cross-Right 120° (right peripheral)
- Bottom-right: Front-Tele 30° (narrow forward zoom)

IMPORTANT CLASSIFICATION RULES:
1. road_type: If there is active construction (excavators, workers, cones, debris), use "construction_zone" even if it's on an urban street
2. pedestrians_visible: Construction workers, people walking, anyone on foot counts as pedestrians - set TRUE if ANY people visible
3. notable_elements: ALWAYS list notable items. Examples: "excavator", "construction_workers", "traffic_cones", "debris", "emergency_vehicle", "school_zone", "crosswalk". NEVER leave empty if there's anything unusual.
4. traffic_lights_visible: Include distant traffic lights, even if small
5. cyclists_visible: Only set true if clearly identifiable bicycle/cyclist

Analyze ALL four camera views carefully before responding."""

# ============================================================================
# HELPERS
# ============================================================================

def create_composite_image(image_frames: torch.Tensor) -> str:
    images = []
    for cam_idx in range(4):
        img_tensor = image_frames[cam_idx, -1]
        img_np = img_tensor.permute(1, 2, 0).numpy().astype('uint8')
        images.append(Image.fromarray(img_np))

    w, h = images[0].size
    composite = Image.new('RGB', (w * 2, h * 2))
    positions = [(0, 0), (w, 0), (0, h), (w, h)]
    for img, pos in zip(images, positions):
        composite.paste(img, pos)

    composite.thumbnail((1920, 1080), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    composite.save(buffer, format='JPEG', quality=85)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def classify_scene(image_frames: torch.Tensor, client: Client) -> SceneClassification:
    img_b64 = create_composite_image(image_frames)
    response = client.chat(
        model="qwen3-vl:8b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Classify this driving scene.", "images": [img_b64]}
        ],
        format=SceneClassification.model_json_schema(),
        options={"num_ctx": 8192 * 4}
    )
    result = json.loads(response['message']['content'])
    return SceneClassification(**result)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
    T0_US = 5_100_000

    print("Variant A: Explicit Instructions")
    print("=" * 50)

    data = load_physical_aiavdataset(CLIP_ID, t0_us=T0_US)
    client = Client(host="http://localhost:11434")
    classification = classify_scene(data['image_frames'], client)

    result = {
        "variant": "A_explicit",
        "clip_id": CLIP_ID,
        "classification": classification.model_dump()
    }

    output_path = Path(__file__).parent / "result_a_explicit.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(classification.model_dump_json(indent=2))
    print(f"\nSaved: {output_path}")
