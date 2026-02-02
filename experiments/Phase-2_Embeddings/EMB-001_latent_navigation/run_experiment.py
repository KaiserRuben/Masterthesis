"""
EMB-001: Run Experiment Pipeline

Runs full experiment pipeline for one or more embedding models.

Usage:
    python run_experiment.py --provider eva02_e
    python run_experiment.py --provider siglip2_so400m --superset-size 1000
    python run_experiment.py --provider all --superset-size 10000

Steps:
    1. Load scenes + classifications (anchors + superset)
    2. Embed all images
    3. Embed text anchors (semantic key values as text)
    4. Run structure analysis (PCA, UMAP, HDBSCAN)
    5. Build navigation graph (k-NN)
    6. Compute alignment metrics (on anchors only)
    7. Generate visualization
    8. Save all outputs
"""

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from PIL import Image

# Add parent to path for tools access
sys.path.insert(0, str(Path(__file__).parents[2]))

from tools.scene.enums import (
    Weather, RoadType, TimeOfDay, TrafficSituation, OcclusionLevel,
    DepthComplexity, VisualDegradation, SafetyCriticality, RequiredAction,
)

from providers import get_provider, list_providers
from analysis import (
    compute_pca,
    compute_umap_3d,
    compute_clusters,
    compute_cluster_metrics,
    compute_alignment_metrics,
    build_navigation_graph,
    compute_navigation_metrics,
    compute_text_anchor_alignment,
)
from visualize import create_visualization
from data_loader import load_experiment_data


# =============================================================================
# CONFIG
# =============================================================================

DATA_DIR = Path(__file__).parents[2] / "data" / "CLS-001"
OUTPUT_BASE = Path(__file__).parents[2] / "data" / "EMB-001"

# All categorical keys with their enum values
CATEGORICAL_KEYS_ALL = {
    "weather": [e.value for e in Weather],
    "road_type": [e.value for e in RoadType],
    "time_of_day": [e.value for e in TimeOfDay],
    "traffic_situation": [e.value for e in TrafficSituation],
    "occlusion_level": [e.value for e in OcclusionLevel],
    "depth_complexity": [e.value for e in DepthComplexity],
    "visual_degradation": [e.value for e in VisualDegradation],
    "safety_criticality": [e.value for e in SafetyCriticality],
    "required_action": [e.value for e in RequiredAction],
}

# Top categorical keys (>25% text alignment in validation)
# Excluded: traffic_situation (19%), safety_criticality (19%), visual_degradation (9%)
CATEGORICAL_KEYS_TOP = {
    "weather": [e.value for e in Weather],
    "road_type": [e.value for e in RoadType],
    "time_of_day": [e.value for e in TimeOfDay],
    "occlusion_level": [e.value for e in OcclusionLevel],
    "depth_complexity": [e.value for e in DepthComplexity],
    "required_action": [e.value for e in RequiredAction],
}

# Boolean keys
BOOLEAN_KEYS_ALL = [
    "pedestrians_present",
    "cyclists_present",
    "construction_activity",
    "traffic_signals_visible",
    "similar_object_confusion",
]

# Top boolean keys (visually obvious)
BOOLEAN_KEYS_TOP = [
    "pedestrians_present",
    "cyclists_present",
    "traffic_signals_visible",
]

# Key sets for selection
KEY_SETS = {
    "all": {
        "categorical": CATEGORICAL_KEYS_ALL,
        "boolean": BOOLEAN_KEYS_ALL,
    },
    "top": {
        "categorical": CATEGORICAL_KEYS_TOP,
        "boolean": BOOLEAN_KEYS_TOP,
    },
}

# Default: all keys
CATEGORICAL_KEYS = CATEGORICAL_KEYS_ALL
BOOLEAN_KEYS = BOOLEAN_KEYS_ALL
ALL_EVAL_KEYS = list(CATEGORICAL_KEYS.keys()) + BOOLEAN_KEYS

# Text anchor template (from pre-test: photo_prompt performed best)
TEXT_TEMPLATE = "A photo of a driving scene with {value_clean}"

# Human-readable value transforms
VALUE_TRANSFORMS = {
    # weather
    "clear": "clear weather",
    "cloudy": "cloudy weather",
    "rainy": "rainy weather",
    "foggy": "foggy weather",
    "snowy": "snowy weather",
    # road_type
    "highway": "a highway",
    "urban_street": "an urban street",
    "residential": "a residential area",
    "intersection": "an intersection",
    "parking_lot": "a parking lot",
    "construction_zone": "a construction zone",
    "rural": "a rural road",
    # time_of_day
    "day": "daytime lighting",
    "dawn_dusk": "dawn or dusk lighting",
    "night": "nighttime lighting",
    # traffic_situation
    "simple": "simple traffic",
    "moderate": "moderate traffic",
    "complex": "complex traffic",
    "critical": "critical traffic",
    # occlusion_level
    "none": "no occlusion",
    "minimal": "minimal occlusion",
    "moderate": "moderate occlusion",
    "severe": "severe occlusion",
    # depth_complexity
    "flat": "flat depth",
    "layered": "layered depth",
    "complex": "complex depth",
    # visual_degradation
    "glare": "glare",
    "low_light": "low light",
    "motion_blur": "motion blur",
    "rain_artifacts": "rain artifacts",
    "fog_haze": "fog or haze",
    "sensor_artifact": "sensor artifacts",
    # safety_criticality
    "tier1_catastrophic": "catastrophic safety risk",
    "tier2_severe": "severe safety risk",
    "tier3_moderate": "moderate safety risk",
    "tier4_minor": "minor safety risk",
    # required_action
    "slow": "slowing required",
    "stop": "stopping required",
    "evade": "evasion required",
    # booleans
    "true": "pedestrians visible",  # Will be key-specific
    "false": "no pedestrians visible",
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_classifications() -> tuple[dict[str, dict], list[dict]]:
    """
    Load scene classifications.

    Returns:
        (classifications_by_id, raw_list)
        classifications_by_id: {scene_id: {key: value}}
    """
    with open(DATA_DIR / "scene_classifications.json") as f:
        data = json.load(f)

    classifications_list = data.get("classifications", [])

    result = {}
    for item in classifications_list:
        clip_id = item["clip_id"]
        cls = item["classification"]

        # Extract simple values from complex responses
        extracted = {}
        for key in ALL_EVAL_KEYS:
            if key in cls:
                val = cls[key]
                if isinstance(val, dict):
                    if key in val:
                        extracted[key] = val[key]
                    elif "category" in val:
                        extracted[key] = val["category"]
                    else:
                        for k, v in val.items():
                            if k not in ["reasoning", "points", "total", "occluded_objects"]:
                                if isinstance(v, (str, bool, int, float)):
                                    extracted[key] = v
                                    break
                elif isinstance(val, (str, bool, int, float)):
                    extracted[key] = val

        result[clip_id] = extracted

    return result, classifications_list


def get_image_paths(classifications: dict[str, dict]) -> list[tuple[str, Path]]:
    """Get (scene_id, path) pairs for all available images."""
    image_dir = DATA_DIR / "images"
    pairs = []
    for clip_id in classifications:
        img_path = image_dir / f"{clip_id}.jpg"
        if img_path.exists():
            pairs.append((clip_id, img_path))
    return pairs


# =============================================================================
# TEXT ANCHORS
# =============================================================================

def generate_text_anchors(keys: dict[str, list[str]]) -> dict[str, dict[str, str]]:
    """
    Generate text anchors for categorical keys.

    Returns:
        {key: {value: text_anchor}}
    """
    anchors = {}
    for key, values in keys.items():
        anchors[key] = {}
        for v in values:
            value_clean = VALUE_TRANSFORMS.get(v, v.replace("_", " "))
            anchors[key][v] = TEXT_TEMPLATE.format(value_clean=value_clean)
    return anchors


def generate_boolean_anchors(keys: list[str]) -> dict[str, dict[str, str]]:
    """
    Generate text anchors for boolean keys.

    Returns:
        {key: {True: text, False: text}}
    """
    boolean_phrases = {
        "pedestrians_present": ("pedestrians visible", "no pedestrians"),
        "cyclists_present": ("cyclists visible", "no cyclists"),
        "construction_activity": ("construction activity", "no construction"),
        "traffic_signals_visible": ("traffic signals visible", "no traffic signals"),
        "similar_object_confusion": ("similar confusable objects", "distinct objects"),
    }

    anchors = {}
    for key in keys:
        true_phrase, false_phrase = boolean_phrases.get(key, (key.replace("_", " "), f"no {key.replace('_', ' ')}"))
        anchors[key] = {
            True: TEXT_TEMPLATE.format(value_clean=true_phrase),
            False: TEXT_TEMPLATE.format(value_clean=false_phrase),
        }
    return anchors


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_experiment(
    provider_name: str,
    output_dir: Path,
    batch_size: int = 8,
    k_neighbors: int = 10,
    superset_size: int = 0,
    num_workers: int = 5,
    key_set: str = "all",
) -> dict:
    """
    Run full experiment pipeline for one model.

    Args:
        provider_name: Name of the embedding provider
        output_dir: Directory to save outputs
        batch_size: Batch size for embedding
        k_neighbors: k for k-NN graph
        superset_size: Number of additional unlabeled scenes (0 = anchors only)
        key_set: Which keys to evaluate ("all" or "top")

    Returns:
        Dictionary of all metrics
    """
    # Select key set
    selected_keys = KEY_SETS[key_set]
    categorical_keys = selected_keys["categorical"]
    boolean_keys = selected_keys["boolean"]
    all_eval_keys = list(categorical_keys.keys()) + boolean_keys

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Running experiment: {provider_name}")
    print(f"Output: {output_dir}")
    print(f"Superset size: {superset_size}")
    print(f"Key set: {key_set} ({len(categorical_keys)} categorical + {len(boolean_keys)} boolean)")
    print(f"{'='*60}\n")

    # --- 1. Load data ---
    print("Step 1: Loading data...")
    scene_ids, images, raw_classifications = load_experiment_data(
        superset_size=superset_size,
        show_progress=True,
        num_workers=num_workers,
    )

    # Extract simple values from complex classification responses
    classifications = {}
    for clip_id, cls in raw_classifications.items():
        extracted = {}
        for key in all_eval_keys:
            if key in cls:
                val = cls[key]
                if isinstance(val, dict):
                    # Try key name first (e.g., {"weather": "clear"})
                    if key in val:
                        extracted[key] = val[key]
                    elif "category" in val:
                        extracted[key] = val["category"]
                    else:
                        # Fallback: find first simple value
                        for k, v in val.items():
                            if k not in ["reasoning", "points", "total", "occluded_objects"]:
                                if isinstance(v, (str, bool, int, float)):
                                    extracted[key] = v
                                    break
                elif isinstance(val, (str, bool, int, float)):
                    extracted[key] = val
        classifications[clip_id] = extracted

    # Convert images to paths (save PIL Images to temp files if needed)
    image_paths = []
    temp_files = []
    for img in images:
        if isinstance(img, Path):
            image_paths.append(str(img))
        elif isinstance(img, Image.Image):
            # Save PIL Image to temp file
            tmp = NamedTemporaryFile(suffix=".jpg", delete=False)
            img.save(tmp.name, "JPEG", quality=85)
            image_paths.append(tmp.name)
            temp_files.append(tmp.name)
        else:
            raise TypeError(f"Unexpected image type: {type(img)}")

    n_anchors = len([s for s in scene_ids if s in classifications])
    print(f"  Total scenes: {len(scene_ids)} ({n_anchors} anchors, {len(scene_ids) - n_anchors} superset)")

    # --- 2. Load provider and embed images ---
    print("\nStep 2: Embedding images...")
    provider = get_provider(provider_name)
    image_embeddings = provider.embed_images(image_paths, batch_size=batch_size)
    print(f"  Image embeddings: {image_embeddings.shape}")

    # Save embeddings
    np.savez(
        output_dir / "embeddings.npz",
        embeddings=image_embeddings,
        scene_ids=scene_ids,
        model_name=provider.name,
        embedding_dim=provider.embedding_dim,
    )

    # --- 3. Embed text anchors ---
    print("\nStep 3: Embedding text anchors...")
    cat_anchors = generate_text_anchors(categorical_keys)
    bool_anchors = generate_boolean_anchors(boolean_keys)

    text_anchor_embeddings = {}

    # Categorical
    for key, value_texts in cat_anchors.items():
        texts = list(value_texts.values())
        values = list(value_texts.keys())
        embs = provider.embed_texts(texts)
        text_anchor_embeddings[key] = {v: embs[i] for i, v in enumerate(values)}
        print(f"  {key}: {len(values)} anchors")

    # Boolean
    for key, value_texts in bool_anchors.items():
        texts = [value_texts[True], value_texts[False]]
        embs = provider.embed_texts(texts)
        text_anchor_embeddings[key] = {True: embs[0], False: embs[1]}
        print(f"  {key}: 2 anchors (bool)")

    # Save text anchors
    np.savez(
        output_dir / "text_anchors.npz",
        **{f"{k}_{v}": e for k, ve in text_anchor_embeddings.items() for v, e in ve.items()},
    )

    # --- 4. Structure analysis ---
    print("\nStep 4: Structure analysis...")

    # PCA
    print("  Computing PCA...")
    pca_embeddings, n_components, pca_model = compute_pca(image_embeddings)
    print(f"  PCA: {image_embeddings.shape[1]} -> {n_components} dims (95% variance)")

    # UMAP
    print("  Computing UMAP (3D)...")
    umap_coords = compute_umap_3d(image_embeddings)
    np.savez(output_dir / "umap_coords.npz", coords=umap_coords, scene_ids=scene_ids)
    print(f"  UMAP coords: {umap_coords.shape}")

    # HDBSCAN
    print("  Computing HDBSCAN clusters...")
    cluster_labels, clusterer = compute_clusters(pca_embeddings)
    cluster_metrics = compute_cluster_metrics(pca_embeddings, cluster_labels)
    print(f"  Clusters: {cluster_metrics['n_clusters']}, noise: {cluster_metrics['noise_ratio']:.1%}")

    # --- 5. Navigation graph ---
    print("\nStep 5: Building navigation graph...")
    graph = build_navigation_graph(
        image_embeddings,
        scene_ids,
        classifications,
        keys_to_compare=list(categorical_keys.keys()) + boolean_keys,
        k=k_neighbors,
    )
    nav_metrics = compute_navigation_metrics(
        graph, scene_ids, classifications, list(categorical_keys.keys())
    )
    print(f"  Edges: {nav_metrics['n_edges']}, single-key-diff: {nav_metrics['n_single_key_diff']}")
    print(f"  Mean coverage: {nav_metrics['mean_coverage']:.1%}")

    # Save graph
    with open(output_dir / "navigation_graph.pkl", "wb") as f:
        pickle.dump(graph, f)

    # --- 6. Alignment metrics ---
    print("\nStep 6: Computing alignment metrics...")

    # Cluster alignment
    alignment_metrics = compute_alignment_metrics(
        image_embeddings,
        classifications,
        scene_ids,
        cluster_labels,
        keys_to_evaluate=list(categorical_keys.keys()),
    )

    # Text anchor alignment
    text_alignment = {}
    for key, values in categorical_keys.items():
        if key in text_anchor_embeddings:
            text_alignment[key] = compute_text_anchor_alignment(
                image_embeddings,
                text_anchor_embeddings[key],
                scene_ids,
                classifications,
                key,
                values,
            )
            print(f"  {key}: text_acc={text_alignment[key]['accuracy']:.3f}")

    # Prediction-based evaluation (for finetuned models)
    prediction_metrics = {}
    if hasattr(provider, 'predict_keys'):
        print("\n  [Finetuned model] Running direct key predictions...")

        # Get anchor image paths only
        anchor_indices = [i for i, sid in enumerate(scene_ids) if sid in classifications]
        anchor_paths = [image_paths[i] for i in anchor_indices]
        anchor_ids = [scene_ids[i] for i in anchor_indices]

        # Predict
        predictions = provider.predict_keys(anchor_paths, batch_size=batch_size)

        # Evaluate
        trained_keys = provider.get_trained_keys()
        for key in trained_keys:
            if key not in predictions:
                continue

            # Get ground truth
            gt_values = []
            pred_values = predictions[key]
            valid_indices = []

            for i, sid in enumerate(anchor_ids):
                if key in classifications.get(sid, {}):
                    gt_val = classifications[sid][key]
                    # Handle boolean keys
                    if isinstance(gt_val, bool):
                        gt_values.append(gt_val)
                    else:
                        gt_values.append(str(gt_val).lower())
                    valid_indices.append(i)

            # Filter predictions to valid indices
            pred_filtered = [pred_values[i] for i in valid_indices]

            # Compute accuracy
            if gt_values:
                correct = sum(
                    1 for gt, pred in zip(gt_values, pred_filtered)
                    if str(gt).lower() == str(pred).lower()
                )
                accuracy = correct / len(gt_values)
                prediction_metrics[key] = {
                    "accuracy": accuracy,
                    "n_samples": len(gt_values),
                    "correct": correct,
                }
                print(f"  {key}: pred_acc={accuracy:.3f} ({correct}/{len(gt_values)})")

    # --- 7. Compile and save results ---
    print("\nStep 7: Saving results...")

    n_anchors = len([s for s in scene_ids if s in classifications])
    results = {
        "model": provider_name,
        "embedding_dim": provider.embedding_dim,
        "n_scenes": len(scene_ids),
        "n_anchors": n_anchors,
        "n_superset": len(scene_ids) - n_anchors,
        "key_set": key_set,
        "n_categorical_keys": len(categorical_keys),
        "n_boolean_keys": len(boolean_keys),
        "timestamp": datetime.now().isoformat(),
        "cluster_metrics": cluster_metrics,
        "navigation_metrics": nav_metrics,
        "alignment_metrics": alignment_metrics,
        "text_alignment": text_alignment,
        "prediction_metrics": prediction_metrics,  # Finetuned model only
    }

    with open(output_dir / "analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # --- 8. Visualization ---
    print("\nStep 8: Generating visualization...")

    # Project text anchors to UMAP space (approximate via nearest neighbor)
    # For simplicity, skip text anchor coords in viz for now

    create_visualization(
        coords_3d=umap_coords,
        scene_ids=scene_ids,
        classifications=classifications,
        cluster_labels=cluster_labels,
        graph=graph,
        text_anchor_coords=None,  # TODO: project text anchors
        keys_for_coloring=list(categorical_keys.keys())[:5],  # Top 5 keys
        output_path=output_dir / "visualization.html",
        title=f"EMB-001: {provider_name}",
    )

    # Clean up temp files
    import os
    for tmp_path in temp_files:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    print(f"\nExperiment complete: {provider_name}")
    print(f"Results saved to: {output_dir}")

    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run EMB-001 experiment pipeline")
    parser.add_argument(
        "--provider",
        type=str,
        required=True,
        help=f"Provider name or 'all'. Available: {', '.join(list_providers())}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for embedding (default: 8)",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=10,
        help="k for k-NN graph (default: 10)",
    )
    parser.add_argument(
        "--superset-size",
        type=int,
        default=0,
        help="Number of additional unlabeled scenes (default: 0 = anchors only)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=5,
        help="Number of parallel workers for data loading (default: 5)",
    )
    parser.add_argument(
        "--key-set",
        type=str,
        choices=["all", "top"],
        default="all",
        help="Key set to evaluate: 'all' (9 categorical + 5 boolean) or 'top' (6 categorical + 3 boolean with >25%% text alignment)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Tag for this run (creates subdirectory: data/EMB-001/{tag}/...)",
    )
    args = parser.parse_args()

    # Timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.provider == "all":
        providers = list_providers()
    else:
        providers = [args.provider]

    # Determine output base (with optional tag subdirectory)
    if args.tag:
        run_output_base = OUTPUT_BASE / args.tag
    else:
        run_output_base = OUTPUT_BASE

    all_results = {}

    for provider_name in providers:
        output_dir = run_output_base / f"{provider_name}_{args.key_set}_{timestamp}"
        try:
            results = run_experiment(
                provider_name=provider_name,
                output_dir=output_dir,
                batch_size=args.batch_size,
                k_neighbors=args.k_neighbors,
                superset_size=args.superset_size,
                num_workers=args.num_workers,
                key_set=args.key_set,
            )
            all_results[provider_name] = results
        except Exception as e:
            print(f"ERROR running {provider_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[provider_name] = {"error": str(e)}

    # Save comparison summary if multiple models
    if len(providers) > 1:
        summary_path = run_output_base / f"comparison_summary_{args.key_set}_{timestamp}.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nComparison summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
