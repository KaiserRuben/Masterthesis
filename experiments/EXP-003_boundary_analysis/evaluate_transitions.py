"""
Phase 4: Evaluate VLM Response Along Interpolation Paths

For each interpolation path, finds nearest neighbors in the embedding space
and computes divergence metrics to identify transition boundaries.

Since we cannot directly "decode" embeddings back to text/images, we use
a nearest-neighbor approach: for each point on the path, find the closest
original scene embedding and use its classification as a proxy.

Input:  data/EXP-003/interpolations.npz
        data/EXP-003/interpolation_metadata.json
        data/EXP-003/embeddings.npz
        data/EXP-003/progress.json
Output: data/EXP-003/transitions.json
"""

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Configuration
RUN_DIR = Path(__file__).parent.parent.parent / "data" / "EXP-003"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance (1 - similarity)."""
    return 1.0 - cosine_similarity(a, b)


def find_nearest_neighbor(
    query: np.ndarray,
    embeddings: np.ndarray,
) -> tuple[int, float]:
    """Find nearest neighbor in embedding space."""
    # Compute cosine similarities
    similarities = np.dot(embeddings, query)  # Already normalized
    best_idx = int(np.argmax(similarities))
    best_sim = float(similarities[best_idx])
    return best_idx, best_sim


def compute_divergence_curve(
    path: np.ndarray,
    key_idx: int,
    embeddings: np.ndarray,
    all_texts: list[list[str]],
    keys: list[str],
) -> dict:
    """
    Compute divergence curve along an interpolation path.

    For each point on the path:
    1. Find nearest neighbor scene embedding for this key
    2. Get the classification value at that scene
    3. Track when the value changes (transition)

    Returns dict with:
    - t_values: interpolation parameters
    - nn_indices: nearest neighbor indices at each t
    - nn_values: classification values at each t
    - divergence: cumulative distance from start
    - transitions: list of t values where classification changed
    """
    n_steps = path.shape[0]
    t_values = np.linspace(0, 1, n_steps)

    # Extract embeddings for this key from all scenes
    key_embeddings = embeddings[:, key_idx, :]  # (n_scenes, dim)

    nn_indices = []
    nn_values = []
    nn_similarities = []

    for t_idx in range(n_steps):
        point = path[t_idx]
        idx, sim = find_nearest_neighbor(point, key_embeddings)
        nn_indices.append(idx)
        nn_similarities.append(sim)

        # Get value from text (format: "key: value")
        text = all_texts[idx][key_idx]
        value = text.split(": ", 1)[1] if ": " in text else text
        nn_values.append(value)

    # Compute divergence curve: distance from start point
    divergence = []
    start_point = path[0]
    for t_idx in range(n_steps):
        dist = cosine_distance(path[t_idx], start_point)
        divergence.append(dist)

    # Find transitions (where nn_value changes)
    transitions = []
    for t_idx in range(1, n_steps):
        if nn_values[t_idx] != nn_values[t_idx - 1]:
            transitions.append({
                "t": float(t_values[t_idx]),
                "from_value": nn_values[t_idx - 1],
                "to_value": nn_values[t_idx],
            })

    # Compute gradient of divergence
    divergence = np.array(divergence)
    dt = t_values[1] - t_values[0]
    gradient = np.gradient(divergence, dt)

    # Find t* (maximum gradient = sharpest transition)
    t_star_idx = int(np.argmax(np.abs(gradient)))
    t_star = float(t_values[t_star_idx])
    sharpness = float(np.max(np.abs(gradient)))
    total_divergence = float(divergence[-1])

    return {
        "t_values": t_values.tolist(),
        "nn_indices": nn_indices,
        "nn_values": nn_values,
        "nn_similarities": nn_similarities,
        "divergence": divergence.tolist(),
        "gradient": gradient.tolist(),
        "transitions": transitions,
        "t_star": t_star,
        "sharpness": sharpness,
        "total_divergence": total_divergence,
    }


def main():
    print("=" * 60)
    print("PHASE 4: EVALUATE TRANSITIONS")
    print("=" * 60)

    # Load data
    interpolations = np.load(RUN_DIR / "interpolations.npz")
    paths = interpolations["paths"]  # (n_paths, n_steps, dim)
    t_values = interpolations["t_values"]

    with open(RUN_DIR / "interpolation_metadata.json") as f:
        path_metadata = json.load(f)

    embeddings_data = np.load(RUN_DIR / "embeddings.npz")
    embeddings = embeddings_data["embeddings"]  # (n_scenes, n_keys, dim)
    keys = embeddings_data["keys"].tolist()

    with open(RUN_DIR / "embedding_texts.json") as f:
        texts_data = json.load(f)
    all_texts = texts_data["texts"]

    n_paths = paths.shape[0]
    print(f"\nPaths: {n_paths}")
    print(f"Steps per path: {paths.shape[1]}")

    # Process each path
    results = []

    for path_idx in tqdm(range(n_paths), desc="Evaluating paths"):
        meta = path_metadata[path_idx]
        key = meta["key"]
        key_idx = keys.index(key)

        path = paths[path_idx]
        divergence_data = compute_divergence_curve(
            path, key_idx, embeddings, all_texts, keys
        )

        result = {
            **meta,
            **divergence_data,
        }
        results.append(result)

    # Aggregate by key
    print("\n" + "-" * 60)
    print("TRANSITION SUMMARY BY KEY")
    print("-" * 60)

    key_stats = {}
    for key in set(m["key"] for m in path_metadata):
        key_results = [r for r in results if r["key"] == key]

        n_with_transitions = sum(1 for r in key_results if r["transitions"])
        mean_sharpness = np.mean([r["sharpness"] for r in key_results])
        mean_divergence = np.mean([r["total_divergence"] for r in key_results])

        key_stats[key] = {
            "n_paths": len(key_results),
            "n_with_transitions": n_with_transitions,
            "transition_rate": n_with_transitions / len(key_results),
            "mean_sharpness": float(mean_sharpness),
            "mean_divergence": float(mean_divergence),
        }

        print(f"{key}:")
        print(f"  Paths: {len(key_results)}, Transitions: {n_with_transitions} ({100*n_with_transitions/len(key_results):.0f}%)")
        print(f"  Mean sharpness: {mean_sharpness:.4f}, Mean divergence: {mean_divergence:.4f}")

    # Save results
    output = {
        "paths": results,
        "key_stats": key_stats,
        "summary": {
            "total_paths": n_paths,
            "total_with_transitions": sum(1 for r in results if r["transitions"]),
            "mean_sharpness": float(np.mean([r["sharpness"] for r in results])),
            "mean_divergence": float(np.mean([r["total_divergence"] for r in results])),
        },
    }

    output_file = RUN_DIR / "transitions.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved: {output_file}")

    print("\n" + "=" * 60)
    print("PHASE 4 COMPLETE")
    print("=" * 60)
    print(f"Total paths with transitions: {output['summary']['total_with_transitions']}/{n_paths}")


if __name__ == "__main__":
    main()
