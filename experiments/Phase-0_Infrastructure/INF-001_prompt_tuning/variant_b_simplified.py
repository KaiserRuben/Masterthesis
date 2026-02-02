"""
Variant B: Simplified Schema
- Fewer enum fields, more boolean flags
- Heavier reliance on free-form description
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
# SIMPLIFIED SCHEMA
# ============================================================================

class DrivingContext(str, Enum):
    normal = "normal"
    construction = "construction"
    heavy_traffic = "heavy_traffic"
    intersection = "intersection"
    parking = "parking"
    highway = "highway"

class SceneClassification(BaseModel):
    """Simplified driving scene classification."""

    # Primary context (single most important factor)
    driving_context: DrivingContext = Field(
        description="The primary driving context affecting behavior"
    )

    # Simple conditions
    is_daytime: bool
    is_clear_weather: bool

    # Object presence (simple booleans)
    has_pedestrians_or_workers: bool = Field(
        description="Any people visible: pedestrians, workers, cyclists"
    )
    has_traffic_signals: bool = Field(
        description="Traffic lights or stop signs visible"
    )
    has_construction_equipment: bool = Field(
        description="Excavators, cranes, construction vehicles"
    )

    # Counts
    vehicle_count: int = Field(ge=0, le=50, description="Approximate vehicle count")

    # Free-form (most important)
    full_description: str = Field(
        description="Detailed 2-3 sentence description of what you see in all four camera views"
    )

    # Key elements as simple list
    key_elements: list[str] = Field(
        description="List ALL notable objects: vehicles, people, signs, equipment, hazards"
    )

# ============================================================================
# PROMPT
# ============================================================================

SYSTEM_PROMPT = """Analyze this autonomous vehicle camera view (4 cameras in 2x2 grid).

Focus on providing an accurate, detailed description. The most important fields are:
- full_description: Describe what you see in detail
- key_elements: List every notable object (vehicles, people, equipment, signs, etc.)
- driving_context: What's the PRIMARY factor affecting driving? (construction if there's active work)
- has_pedestrians_or_workers: TRUE if ANY people visible (including construction workers)

Be thorough - list everything you observe."""

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
            {"role": "user", "content": "Describe and classify this scene.", "images": [img_b64]}
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

    print("Variant B: Simplified Schema")
    print("=" * 50)

    data = load_physical_aiavdataset(CLIP_ID, t0_us=T0_US)
    client = Client(host="http://localhost:11434")
    classification = classify_scene(data['image_frames'], client)

    result = {
        "variant": "B_simplified",
        "clip_id": CLIP_ID,
        "classification": classification.model_dump()
    }

    output_path = Path(__file__).parent.parent.parent / "data" / "EXP-004" / "result_b_simplified.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(classification.model_dump_json(indent=2))
    print(f"\nSaved: {output_path}")
