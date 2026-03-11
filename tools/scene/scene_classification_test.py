"""
Single Scene Classification Experiment

Test Pydantic schema + Ollama qwen3-vl:8b structured output on one scene.
Tune until satisfactory, then scale to 200 scenes.
"""

# Requires: pip install -e alpamayo/

import json
import base64
import os
from io import BytesIO
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field
from PIL import Image
import torch
from ollama import Client

from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset


# ============================================================================
# PYDANTIC SCHEMA FOR SCENE CLASSIFICATION
# ============================================================================

class RoadType(str, Enum):
    highway = "highway"
    urban_street = "urban_street"
    residential = "residential"
    intersection = "intersection"
    parking_lot = "parking_lot"
    construction_zone = "construction_zone"
    rural = "rural"


class WeatherCondition(str, Enum):
    clear = "clear"
    cloudy = "cloudy"
    overcast = "overcast"
    rainy = "rainy"
    foggy = "foggy"
    snowy = "snowy"


class TimeOfDay(str, Enum):
    day = "day"
    dawn_dusk = "dawn_dusk"
    night = "night"


class TrafficDensity(str, Enum):
    empty = "empty"
    light = "light"
    moderate = "moderate"
    heavy = "heavy"


class SceneClassification(BaseModel):
    """Classification of a driving scene from camera images."""

    # First: free-form description to ground the model
    scene_description: str = Field(
        description="Detailed description of what you observe across all four camera views"
    )

    # Structured classifications (order matches "Pay attention" in prompt)
    road_type: RoadType = Field(
        description="Type of road environment"
    )
    weather: WeatherCondition = Field(
        description="Weather conditions visible in the scene"
    )
    time_of_day: TimeOfDay = Field(
        description="Time of day based on lighting"
    )
    traffic_density: TrafficDensity = Field(
        description="Overall traffic density across all views"
    )
    has_pedestrians: bool = Field(
        description="Are any people visible (pedestrians, workers, etc.)?"
    )
    has_cyclists: bool = Field(
        description="Are cyclists or bicycles visible?"
    )
    has_traffic_signals: bool = Field(
        description="Are traffic lights or stop signs visible?"
    )
    has_construction_activity: bool = Field(
        description="Is there construction equipment or activity?"
    )
    vehicle_count: int = Field(
        ge=0, le=50,
        description="Approximate number of vehicles visible (0-50)"
    )
    notable_elements: list[str] = Field(
        default_factory=list,
        description="List of notable elements observed in the scene"
    )


# ============================================================================
# IMAGE HELPERS
# ============================================================================

def tensor_to_base64(img_tensor: torch.Tensor) -> str:
    """Convert image tensor (C, H, W) to base64 string."""
    img_np = img_tensor.permute(1, 2, 0).numpy().astype('uint8')
    img_pil = Image.fromarray(img_np)

    # Resize for faster processing (optional)
    max_size = 1024
    if max(img_pil.size) > max_size:
        img_pil.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    buffer = BytesIO()
    img_pil.save(buffer, format='JPEG', quality=85)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def create_composite_image(image_frames: torch.Tensor) -> str:
    """Create a 2x2 composite of all 4 camera views, return as base64."""
    # image_frames: (4, num_frames, 3, H, W)
    # Take the last frame (t0) from each camera

    images = []
    for cam_idx in range(4):
        img_tensor = image_frames[cam_idx, -1]  # Last frame
        img_np = img_tensor.permute(1, 2, 0).numpy().astype('uint8')
        images.append(Image.fromarray(img_np))

    # Create 2x2 composite
    w, h = images[0].size
    composite = Image.new('RGB', (w * 2, h * 2))

    # Layout: [Cross-Left, Front-Wide]
    #         [Cross-Right, Front-Tele]
    positions = [(0, 0), (w, 0), (0, h), (w, h)]
    labels = ["Cross-Left 120°", "Front-Wide 120°", "Cross-Right 120°", "Front-Tele 30°"]

    for img, pos in zip(images, positions):
        composite.paste(img, pos)

    # Resize for reasonable context length
    composite.thumbnail((1920, 1080), Image.Resampling.LANCZOS)

    buffer = BytesIO()
    composite.save(buffer, format='JPEG', quality=85)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


# ============================================================================
# CLASSIFICATION
# ============================================================================

SYSTEM_PROMPT = """Analyze this autonomous vehicle camera view (4 cameras in 2x2 grid):
- Top-left: Left peripheral 120°
- Top-right: Front wide 120°
- Bottom-left: Right peripheral 120°
- Bottom-right: Front telephoto 30°

Pay attention to:
- Road type: check lane markings, surroundings (buildings, open space), and road infrastructure
- Weather: look at the sky, visibility, and road surface conditions
- Time of day: assess lighting, shadows, and sky brightness
- Traffic density: assess how busy the road is based on vehicle presence and flow
- People: scan sidewalks, crossings, and work zones for pedestrians or workers
- Cyclists: check road edges and bike lanes, especially in the telephoto view
- Traffic signals: look for traffic lights and signs in the distance, particularly in front views
- Construction: look for equipment, cones, barriers, workers in safety vests, or debris
- Vehicles: count all visible vehicles including parked cars across all four views
- Notable elements: identify anything unusual or safety-relevant

Start by describing what you observe in scene_description, then fill in the form.
Any person on the street (including workers) counts as pedestrians."""

MODEL = "qwen3-vl:8b"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "https://eb5e2cb0eb8b.ngrok-free.app")

def classify_scene(
        image_frames: torch.Tensor,
        client: Client,
        model: str = MODEL
) -> SceneClassification:
    """Classify a scene using Ollama structured output."""

    # Create composite image
    img_b64 = create_composite_image(image_frames)

    # Call Ollama with structured output
    response = client.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": "Classify this driving scene.",
                "images": [img_b64],
            }
        ],
        format=SceneClassification.model_json_schema(),
        options={
            "num_ctx": 8192 * 4,
            "max_tokens": 8192,
        }
    )

    # Parse response
    result = json.loads(response['message']['content'])
    return SceneClassification(**result)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("SCENE CLASSIFICATION TEST - Single Scene")
    print("=" * 80)

    # Test clip
    CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
    T0_US = 5_100_000

    print(f"\nLoading scene: {CLIP_ID}")
    print(f"Timestamp: {T0_US / 1_000_000:.1f}s")

    # Load data
    script_location = str(Path(__file__).parent.resolve())
    data = load_physical_aiavdataset(CLIP_ID, t0_us=T0_US)
    print(f"✓ Data loaded: {data['image_frames'].shape}")

    # Save composite image for inspection
    img_b64 = create_composite_image(data['image_frames'])
    img_bytes = base64.b64decode(img_b64)
    with open(script_location + "/test_scene_composite.jpg", "wb") as f:
        f.write(img_bytes)
    print(f"✓ Saved composite image: {script_location}/test_scene_composite.jpg")

    # Classify
    print(f"\nClassifying with {MODEL}...")
    client = Client(host=OLLAMA_URL)

    classification = classify_scene(
        data['image_frames'],
        client,
        model=MODEL
    )

    print("\n" + "=" * 80)
    print("CLASSIFICATION RESULT")
    print("=" * 80)
    print(classification.model_dump_json(indent=2))

    # Save result
    result = {
        "clip_id": CLIP_ID,
        "t0_us": T0_US,
        "classification": classification.model_dump()
    }

    with open(script_location + "/test_scene_classification.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n✓ Saved result: {script_location}/test_scene_classification.json")
