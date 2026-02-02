"""
Phase 2: Compute Class Centroids from Embeddings

Groups scenes by key-value and computes normalized centroids in embedding space.

Input:  data/EXP-003/embeddings.npz
        data/EXP-003/embedding_texts.json
Output: data/EXP-003/centroids.json
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np

# Configuration
RUN_DIR = Path(__file__).parent.parent.parent / "data" / "EXP-003"


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length."""
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v
    return v / norm


def compute_centroid(embeddings: list[np.ndarray]) -> np.ndarray:
    """Compute normalized centroid from list of embeddings."""
    if not embeddings:
        raise ValueError("Cannot compute centroid of empty list")
    mean = np.mean(embeddings, axis=0)
    return normalize(mean)


def main():
    print("=" * 60)
    print("PHASE 2: COMPUTE CENTROIDS")
    print("=" * 60)

    # Load embeddings
    embeddings_file = RUN_DIR / "embeddings.npz"
    texts_file = RUN_DIR / "embedding_texts.json"

    data = np.load(embeddings_file)
    clip_ids = data["clip_ids"].tolist()
    keys = data["keys"].tolist()
    embeddings = data["embeddings"]  # (n_scenes, n_keys, dim)

    with open(texts_file) as f:
        texts_data = json.load(f)
    all_texts = texts_data["texts"]  # (n_scenes, n_keys)

    n_scenes, n_keys, dim = embeddings.shape
    print(f"\nInput: {embeddings_file}")
    print(f"Scenes: {n_scenes}, Keys: {n_keys}, Dim: {dim}")

    # Group by key-value
    centroids = {}

    for key_idx, key in enumerate(keys):
        print(f"\nProcessing: {key}")

        # Collect values and their embeddings
        value_embeddings = defaultdict(list)
        value_clip_ids = defaultdict(list)

        for scene_idx in range(n_scenes):
            # Extract value from text (format: "key: value")
            text = all_texts[scene_idx][key_idx]
            value = text.split(": ", 1)[1] if ": " in text else text

            emb = embeddings[scene_idx, key_idx]
            value_embeddings[value].append(emb)
            value_clip_ids[value].append(clip_ids[scene_idx])

        # Compute centroids
        key_centroids = {}
        for value, embs in value_embeddings.items():
            centroid = compute_centroid(embs)

            # Verify normalization
            norm = np.linalg.norm(centroid)
            assert abs(norm - 1.0) < 1e-6, f"Centroid not normalized: {norm}"

            key_centroids[value] = {
                "centroid": centroid.tolist(),
                "count": len(embs),
                "clip_ids": value_clip_ids[value],
            }
            print(f"  {value}: n={len(embs)}")

        centroids[key] = key_centroids

    # Summary statistics
    print("\n" + "-" * 60)
    print("SUMMARY")
    print("-" * 60)

    total_centroids = 0
    for key, values in centroids.items():
        n_values = len(values)
        total_centroids += n_values
        counts = [v["count"] for v in values.values()]
        print(f"{key}: {n_values} values, counts: {counts}")

    print(f"\nTotal centroids: {total_centroids}")

    # Save
    output_file = RUN_DIR / "centroids.json"

    # Convert numpy types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return obj

    with open(output_file, "w") as f:
        json.dump(centroids, f, indent=2, default=convert_to_serializable)

    print(f"\nSaved: {output_file}")

    print("\n" + "=" * 60)
    print("PHASE 2 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
