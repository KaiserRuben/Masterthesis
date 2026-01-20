"""
Variant C: Chain-of-Thought (Two-Stage)
- First: Get detailed free-form description
- Second: Extract structured classification from description
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
# SCHEMA
# ============================================================================

class RoadType(str, Enum):
    highway = "highway"
    urban_street = "urban_street"
    residential = "residential"
    intersection = "intersection"
    parking_lot = "parking_lot"
    construction_zone = "construction_zone"
    rural = "rural"

class SceneClassification(BaseModel):
    road_type: RoadType
    weather: str = Field(description="clear/cloudy/rainy/foggy/snowy")
    time_of_day: str = Field(description="day/night/dawn_dusk")
    pedestrians_visible: bool
    cyclists_visible: bool
    traffic_lights_visible: bool
    construction_activity: bool
    num_vehicles: int = Field(ge=0, le=50)
    notable_elements: list[str]
    reasoning: str = Field(description="Brief explanation of classification choices")

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

def classify_scene_cot(image_frames: torch.Tensor, client: Client) -> tuple[str, SceneClassification]:
    img_b64 = create_composite_image(image_frames)

    # Stage 1: Get detailed description
    desc_response = client.chat(
        model="qwen3-vl:8b",
        messages=[
            {
                "role": "system",
                "content": """You are analyzing a 4-camera autonomous vehicle view.
Layout: Top-left=Left peripheral, Top-right=Front wide, Bottom-left=Right peripheral, Bottom-right=Front telephoto.

Describe EACH camera view separately, noting:
- Road type and conditions
- All vehicles (count them)
- Any people (pedestrians, workers, cyclists)
- Traffic signals/signs
- Construction equipment or activity
- Weather and lighting
- Any hazards or unusual elements

Be extremely thorough."""
            },
            {
                "role": "user",
                "content": "Describe what you see in each of the four camera views.",
                "images": [img_b64]
            }
        ],
        options={"num_ctx": 8192 * 4}
    )
    description = desc_response['message']['content']

    # Stage 2: Extract structured data from description
    struct_response = client.chat(
        model="qwen3-vl:8b",
        messages=[
            {
                "role": "system",
                "content": """Extract structured classification from the scene description.

RULES:
- road_type: Use "construction_zone" if ANY construction activity/equipment mentioned
- pedestrians_visible: TRUE if workers, people walking, anyone on foot mentioned
- cyclists_visible: TRUE only if bicycles/cyclists explicitly mentioned
- construction_activity: TRUE if excavators, workers in vests, cones, debris mentioned
- notable_elements: List ALL unusual items mentioned (construction equipment, workers, signs, etc.)
- reasoning: Explain your classification choices briefly"""
            },
            {
                "role": "user",
                "content": f"Based on this scene description, extract the structured classification:\n\n{description}"
            }
        ],
        format=SceneClassification.model_json_schema(),
        options={"num_ctx": 8192 * 4}
    )

    result = json.loads(struct_response['message']['content'])
    return description, SceneClassification(**result)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
    T0_US = 5_100_000

    print("Variant C: Chain-of-Thought (Two-Stage)")
    print("=" * 50)

    data = load_physical_aiavdataset(CLIP_ID, t0_us=T0_US)
    client = Client(host="http://localhost:11434")

    description, classification = classify_scene_cot(data['image_frames'], client)

    result = {
        "variant": "C_cot",
        "clip_id": CLIP_ID,
        "raw_description": description,
        "classification": classification.model_dump()
    }

    output_path = Path(__file__).parent / "result_c_cot.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print("Raw Description:")
    print("-" * 40)
    print(description)
    print("\nClassification:")
    print("-" * 40)
    print(classification.model_dump_json(indent=2))
    print(f"\nSaved: {output_path}")
