"""
Phase 3: Compute SLERP Interpolation Paths Between Centroids

For each key, generates interpolation paths between all centroid pairs.
Uses Spherical Linear Interpolation (SLERP) on the unit sphere.

Input:  data/EXP-003/centroids.json
Output: data/EXP-003/interpolations.npz
"""

import json
from pathlib import Path
from itertools import combinations

import numpy as np

# Configuration
RUN_DIR = Path(__file__).parent.parent.parent / "data" / "EXP-003"
N_STEPS = 21  # t in {0.0, 0.05, ..., 1.0}

# Keys to interpolate (exclude high-cardinality keys like vehicle_count)
INTERPOLATE_KEYS = [
    "weather",
    "time_of_day",
    "road_type",
    "traffic_situation",
    "pedestrians_present",
    "construction_activity",
    "traffic_signals_visible",
    "occlusion_level",
    "depth_complexity",
    "visual_degradation",
    "similar_object_confusion",
    "safety_criticality",
    "required_action",
]


def slerp(v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical Linear Interpolation between two unit vectors.

    γ(t) = sin((1-t)θ)/sin(θ) · v0 + sin(tθ)/sin(θ) · v1

    Args:
        v0: Start vector (unit norm)
        v1: End vector (unit norm)
        t: Interpolation parameter in [0, 1]

    Returns:
        Interpolated unit vector
    """
    # Compute angle between vectors
    dot = np.clip(np.dot(v0, v1), -1.0, 1.0)
    theta = np.arccos(dot)

    # Handle edge cases
    if abs(theta) < 1e-6:
        # Vectors are nearly identical
        return v0
    if abs(theta - np.pi) < 1e-6:
        # Vectors are nearly opposite - use linear interpolation
        result = (1 - t) * v0 + t * v1
        return result / np.linalg.norm(result)

    # SLERP formula
    sin_theta = np.sin(theta)
    w0 = np.sin((1 - t) * theta) / sin_theta
    w1 = np.sin(t * theta) / sin_theta

    return w0 * v0 + w1 * v1


def compute_path(v0: np.ndarray, v1: np.ndarray, n_steps: int = N_STEPS) -> np.ndarray:
    """
    Compute full interpolation path between two centroids.

    Returns:
        Array of shape (n_steps, dim) with interpolated vectors
    """
    t_values = np.linspace(0, 1, n_steps)
    path = np.array([slerp(v0, v1, t) for t in t_values])
    return path


def main():
    print("=" * 60)
    print("PHASE 3: COMPUTE INTERPOLATION PATHS")
    print("=" * 60)

    # Load centroids
    centroids_file = RUN_DIR / "centroids.json"
    with open(centroids_file) as f:
        centroids = json.load(f)

    print(f"\nInput: {centroids_file}")
    print(f"Interpolation steps: {N_STEPS}")
    print(f"Keys to process: {len(INTERPOLATE_KEYS)}")

    # Storage for paths
    all_paths = {}
    path_metadata = []

    total_paths = 0

    for key in INTERPOLATE_KEYS:
        if key not in centroids:
            print(f"\nSkipping {key}: not in centroids")
            continue

        key_centroids = centroids[key]
        values = list(key_centroids.keys())
        n_values = len(values)

        if n_values < 2:
            print(f"\nSkipping {key}: only {n_values} value(s)")
            continue

        print(f"\n{key}: {n_values} values -> {n_values * (n_values - 1)} directed paths")

        key_paths = []

        # Generate all directed pairs (A→B and B→A are different)
        for v_a, v_b in [(a, b) for a in values for b in values if a != b]:
            centroid_a = np.array(key_centroids[v_a]["centroid"])
            centroid_b = np.array(key_centroids[v_b]["centroid"])

            # Compute SLERP path
            path = compute_path(centroid_a, centroid_b)

            # Verify endpoints
            assert np.allclose(path[0], centroid_a, atol=1e-6), "Path start mismatch"
            assert np.allclose(path[-1], centroid_b, atol=1e-6), "Path end mismatch"

            # Verify all points on unit sphere
            norms = np.linalg.norm(path, axis=1)
            assert np.allclose(norms, 1.0, atol=1e-6), f"Points not on sphere: {norms}"

            # Compute angle between centroids
            dot = np.clip(np.dot(centroid_a, centroid_b), -1.0, 1.0)
            theta = np.arccos(dot)

            key_paths.append(path)
            path_metadata.append({
                "key": key,
                "value_a": v_a,
                "value_b": v_b,
                "count_a": key_centroids[v_a]["count"],
                "count_b": key_centroids[v_b]["count"],
                "theta": float(theta),
                "theta_deg": float(np.degrees(theta)),
            })

            total_paths += 1

        # Stack paths for this key
        all_paths[key] = np.array(key_paths)  # (n_pairs, n_steps, dim)

    # Summary
    print("\n" + "-" * 60)
    print("SUMMARY")
    print("-" * 60)
    print(f"Total paths: {total_paths}")

    # Angle statistics
    thetas = [m["theta_deg"] for m in path_metadata]
    print(f"Angle range: {min(thetas):.1f}° - {max(thetas):.1f}°")
    print(f"Mean angle: {np.mean(thetas):.1f}°")

    # Save paths as npz
    output_file = RUN_DIR / "interpolations.npz"

    # Flatten all paths into single array with index
    flat_paths = []
    path_indices = []
    idx = 0
    for key in INTERPOLATE_KEYS:
        if key in all_paths:
            for path in all_paths[key]:
                flat_paths.append(path)
                path_indices.append(idx)
                idx += 1

    flat_paths = np.array(flat_paths)  # (total_paths, n_steps, dim)

    np.savez(
        output_file,
        paths=flat_paths,
        t_values=np.linspace(0, 1, N_STEPS),
    )
    print(f"\nSaved paths: {output_file}")
    print(f"  Shape: {flat_paths.shape}")

    # Save metadata as JSON
    metadata_file = RUN_DIR / "interpolation_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(path_metadata, f, indent=2)
    print(f"Saved metadata: {metadata_file}")

    print("\n" + "=" * 60)
    print("PHASE 3 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
