"""
Run Sub-experiment: Finetuned Encoder vs CLIP

This script:
1. Trains two models (all keys vs top keys)
2. Runs the main experiment pipeline with each
3. Compares results with CLIP baselines

Usage:
    # Train and evaluate both configurations
    python run_subexperiment.py

    # Only evaluate existing models
    python run_subexperiment.py --eval-only

    # Train specific configuration
    python run_subexperiment.py --key-mode top --train-only
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from sub_finetuned_encoder.train import train
from sub_finetuned_encoder.provider import load_provider, list_available_models
from sub_finetuned_encoder.model import KEY_VALUES
from data_loader import load_experiment_data, get_anchor_image_paths
from analysis import (
    compute_pca,
    compute_umap_3d,
    compute_clusters,
    compute_cluster_metrics,
    compute_alignment_metrics,
    build_navigation_graph,
    compute_navigation_metrics,
)


OUTPUT_DIR = Path(__file__).parents[3] / "data" / "EXP-005"


def run_evaluation(
    provider,
    superset_size: int = 0,
    k_neighbors: int = 10,
) -> dict:
    """
    Run evaluation pipeline for a provider.

    Returns metrics dict.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {provider.name}")
    print(f"{'='*60}")

    # Load data
    scene_ids, images, classifications = load_experiment_data(
        superset_size=superset_size,
        show_progress=True,
    )

    # Convert images to paths (anchors are already paths)
    image_paths = []
    anchor_paths = get_anchor_image_paths()
    temp_dir = OUTPUT_DIR / "temp_composites"
    temp_dir.mkdir(exist_ok=True)

    for sid, img in zip(scene_ids, images):
        if isinstance(img, Path):
            image_paths.append(str(img))
        else:
            # Save PIL image temporarily
            temp_path = temp_dir / f"{sid}.jpg"
            if not temp_path.exists():
                img.save(temp_path, "JPEG", quality=85)
            image_paths.append(str(temp_path))

    # Embed images
    print(f"Embedding {len(image_paths)} images...")
    embeddings = provider.embed_images(image_paths, batch_size=8)
    print(f"Embeddings shape: {embeddings.shape}")

    # Identify anchor indices
    anchor_mask = np.array([sid in classifications for sid in scene_ids])
    anchor_indices = np.where(anchor_mask)[0]
    print(f"Anchors: {len(anchor_indices)}")

    # Dimensionality reduction
    print("Computing PCA...")
    pca_embeddings, n_components, pca_model = compute_pca(embeddings)

    print("Computing UMAP...")
    umap_coords = compute_umap_3d(embeddings)

    # Clustering
    print("Computing clusters...")
    cluster_labels, _ = compute_clusters(embeddings)
    cluster_metrics = compute_cluster_metrics(embeddings, cluster_labels)

    # Alignment metrics (anchors only)
    print("Computing alignment metrics...")
    anchor_embeddings = embeddings[anchor_indices]
    anchor_scene_ids = [scene_ids[i] for i in anchor_indices]
    anchor_classifications = {sid: classifications[sid] for sid in anchor_scene_ids}
    anchor_cluster_labels = cluster_labels[anchor_indices]

    # Keys to evaluate
    keys_to_evaluate = list(provider.model.all_keys) if hasattr(provider, 'model') else [
        "weather", "road_type", "time_of_day", "traffic_situation",
        "occlusion_level", "depth_complexity"
    ]

    alignment_metrics = compute_alignment_metrics(
        anchor_embeddings,
        anchor_classifications,
        anchor_scene_ids,
        anchor_cluster_labels,
        keys_to_evaluate,
    )

    # Navigation graph
    print("Building navigation graph...")
    nav_graph = build_navigation_graph(
        embeddings,
        scene_ids,
        classifications,
        keys_to_evaluate,
        k=k_neighbors,
    )
    nav_metrics = compute_navigation_metrics(nav_graph, scene_ids, classifications, keys_to_evaluate)

    # Prediction accuracy (for finetuned model)
    pred_accuracy = {}
    if hasattr(provider, "predict_keys"):
        print("Computing prediction accuracy on anchors...")
        anchor_paths = [image_paths[i] for i in anchor_indices]
        predictions = provider.predict_keys(anchor_paths)

        for key in predictions:
            if key not in classifications[anchor_scene_ids[0]]:
                continue

            correct = 0
            total = 0
            for i, sid in enumerate(anchor_scene_ids):
                raw_val = classifications[sid].get(key)
                # Extract value from nested dict
                if isinstance(raw_val, dict):
                    true_val = raw_val.get(key) or raw_val.get('category')
                else:
                    true_val = raw_val

                pred_val = predictions[key][i]
                if true_val is not None:
                    total += 1
                    if true_val == pred_val:
                        correct += 1

            if total > 0:
                pred_accuracy[key] = correct / total

        if pred_accuracy:
            print(f"Mean prediction accuracy: {np.mean(list(pred_accuracy.values())):.3f}")

    # Compile results
    results = {
        "model": provider.name,
        "embedding_dim": provider.embedding_dim,
        "n_scenes": len(scene_ids),
        "n_anchors": len(anchor_indices),
        "n_superset": len(scene_ids) - len(anchor_indices),
        "timestamp": datetime.now().isoformat(),
        "pca": {
            "n_components": n_components,
            "variance_explained": float(sum(pca_model.explained_variance_ratio_)),
        },
        "cluster_metrics": cluster_metrics,
        "navigation_metrics": nav_metrics,
        "alignment_metrics": alignment_metrics,
        "prediction_accuracy": pred_accuracy,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Run finetuned encoder sub-experiment")
    parser.add_argument("--key-mode", choices=["all", "top", "both"], default="both",
                        help="Key configuration to train/evaluate")
    parser.add_argument("--train-only", action="store_true", help="Only train, don't evaluate")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate existing models")
    parser.add_argument("--superset-size", type=int, default=0, help="Superset size for evaluation")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--backbone", default="efficientnet_b0", help="Backbone model")

    args = parser.parse_args()

    # Determine which configs to run
    if args.key_mode == "both":
        key_modes = ["all", "top"]
    else:
        key_modes = [args.key_mode]

    trained_models = {}
    results = {}

    # Training phase
    if not args.eval_only:
        for key_mode in key_modes:
            print(f"\n{'#'*60}")
            print(f"Training: {args.backbone} with {key_mode} keys")
            print(f"{'#'*60}")

            model_dir = train(
                backbone=args.backbone,
                key_mode=key_mode,
                epochs=args.epochs,
            )
            trained_models[key_mode] = model_dir

    # Evaluation phase
    if not args.train_only:
        # Find models to evaluate
        if args.eval_only:
            available = list_available_models()
            for key_mode in key_modes:
                # Find latest model with this key_mode
                matching = [m for m in available if m["key_mode"] == key_mode]
                if matching:
                    # Sort by name (timestamp) descending
                    matching.sort(key=lambda x: x["name"], reverse=True)
                    trained_models[key_mode] = Path(matching[0]["path"])
                    print(f"Found existing model for {key_mode}: {matching[0]['name']}")
                else:
                    print(f"No existing model found for {key_mode}")

        # Evaluate each model
        for key_mode, model_dir in trained_models.items():
            provider = load_provider(model_dir)
            results[key_mode] = run_evaluation(
                provider,
                superset_size=args.superset_size,
            )

        # Save combined results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = OUTPUT_DIR / f"finetuned_comparison_{timestamp}.json"

        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(results_path, "w") as f:
            json.dump(convert_numpy(results), f, indent=2)
        print(f"\nResults saved to: {results_path}")

        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")

        for key_mode, res in results.items():
            print(f"\n{key_mode.upper()} KEYS:")
            print(f"  Clusters: {res['cluster_metrics']['n_clusters']}")
            print(f"  Silhouette: {res['cluster_metrics']['silhouette_score']:.3f}")
            print(f"  Single-key-diff ratio: {res['navigation_metrics']['single_key_diff_ratio']:.3f}")
            if res['prediction_accuracy']:
                mean_acc = np.mean(list(res['prediction_accuracy'].values()))
                print(f"  Mean prediction accuracy: {mean_acc:.3f}")


if __name__ == "__main__":
    main()
