"""
Phase 1: Compute Embeddings for Classification Results

Embeds classification key-value pairs using Qwen3-Embedding via Ollama.
Each key-value is formatted as "{key}: {value}" and embedded to R^4096.

Input:  data/EXP-003/progress.json
Output: data/EXP-003/embeddings.npz
"""

import json
import sys
from pathlib import Path

import httpx
import numpy as np
from tqdm import tqdm

# Configuration
RUN_DIR = Path(__file__).parent.parent.parent / "data" / "EXP-003"
OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "qwen3-embedding:latest"

# Keys to embed (exclude scene_reasoning - it's raw text, not structured)
EMBED_KEYS = [
    "weather",
    "time_of_day",
    "road_type",
    "traffic_situation",
    "pedestrians_present",
    "cyclists_present",
    "construction_activity",
    "traffic_signals_visible",
    "vehicle_count",
    "occlusion_level",
    "depth_complexity",
    "nearest_vehicle_distance",
    "visual_degradation",
    "similar_object_confusion",
    "safety_criticality",
    "vulnerable_road_users",
    "required_action",
    "pedestrian_count",
    "vehicle_count_by_type",
]


def extract_value(classification: dict, key: str) -> str:
    """Extract the primary value from a classification result."""
    val = classification.get(key)
    if val is None:
        return "unknown"

    if isinstance(val, dict):
        # Most keys have a field with the same name as the key
        if key in val:
            v = val[key]
            if isinstance(v, bool):
                return str(v).lower()
            return str(v)
        # Special cases
        if key == "traffic_situation" and "category" in val:
            return val["category"]
        if key == "nearest_vehicle_distance" and "estimated_meters" in val:
            return f"{val['estimated_meters']}m"
        if key == "vulnerable_road_users" and "total_count" in val:
            return f"count_{val['total_count']}"
        if key == "pedestrian_count" and "count" in val:
            return f"count_{val['count']}"
        if key == "vehicle_count_by_type":
            # Summarize vehicle types
            total = sum(val.get(k, 0) for k in ["cars", "suvs_trucks", "commercial", "motorcycles", "other"])
            return f"total_{total}"
        # Fallback: return first non-reasoning value
        for k, v in val.items():
            if k != "reasoning" and not isinstance(v, (list, dict)):
                return str(v)

    return str(val)


def format_key_value(key: str, value: str) -> str:
    """Format key-value pair for embedding."""
    # Clean up the key name for readability
    key_clean = key.replace("_", " ")
    return f"{key_clean}: {value}"


def embed_text(text: str, client: httpx.Client) -> np.ndarray:
    """Embed text using Ollama API."""
    response = client.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": text},
        timeout=60.0,
    )
    response.raise_for_status()
    data = response.json()
    return np.array(data["embedding"], dtype=np.float32)


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length."""
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v
    return v / norm


def main():
    print("=" * 60)
    print("PHASE 1: COMPUTE EMBEDDINGS")
    print("=" * 60)

    # Load classification results
    progress_file = RUN_DIR / "progress.json"
    if not progress_file.exists():
        print(f"Error: {progress_file} not found")
        sys.exit(1)

    with open(progress_file) as f:
        progress = json.load(f)

    results = progress["results"]
    n_scenes = len(results)
    n_keys = len(EMBED_KEYS)

    print(f"\nInput: {progress_file}")
    print(f"Scenes: {n_scenes}")
    print(f"Keys: {n_keys}")
    print(f"Model: {MODEL}")

    # Prepare storage
    clip_ids = []
    all_embeddings = []
    all_texts = []  # Store formatted texts for reference

    # Process each scene
    print(f"\nEmbedding {n_scenes} scenes x {n_keys} keys...")

    with httpx.Client() as client:
        for result in tqdm(results, desc="Scenes"):
            clip_id = result["clip_id"]
            classification = result["classification"]
            clip_ids.append(clip_id)

            scene_embeddings = []
            scene_texts = []

            for key in EMBED_KEYS:
                value = extract_value(classification, key)
                text = format_key_value(key, value)
                scene_texts.append(text)

                # Embed and normalize
                embedding = embed_text(text, client)
                embedding = normalize(embedding)
                scene_embeddings.append(embedding)

            all_embeddings.append(scene_embeddings)
            all_texts.append(scene_texts)

    # Convert to numpy arrays
    embeddings = np.array(all_embeddings, dtype=np.float32)  # (n_scenes, n_keys, dim)

    print(f"\nEmbedding shape: {embeddings.shape}")
    print(f"Embedding dim: {embeddings.shape[-1]}")

    # Verify normalization
    norms = np.linalg.norm(embeddings, axis=-1)
    print(f"Norm range: [{norms.min():.4f}, {norms.max():.4f}]")

    # Save
    output_file = RUN_DIR / "embeddings.npz"
    np.savez(
        output_file,
        clip_ids=np.array(clip_ids),
        keys=np.array(EMBED_KEYS),
        embeddings=embeddings,
    )
    print(f"\nSaved: {output_file}")

    # Also save texts for reference
    texts_file = RUN_DIR / "embedding_texts.json"
    with open(texts_file, "w") as f:
        json.dump({
            "clip_ids": clip_ids,
            "keys": EMBED_KEYS,
            "texts": all_texts,
        }, f, indent=2)
    print(f"Saved: {texts_file}")

    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
