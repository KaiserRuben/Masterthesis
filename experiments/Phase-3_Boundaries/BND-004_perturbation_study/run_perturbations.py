#!/usr/bin/env python3
"""
BND-004: Run Perturbation Inference

Systematically varies text descriptions while keeping images fixed,
measuring how the trajectory prediction model responds to text-image
alignment vs misalignment.

Usage:
    python run_perturbations.py                    # Run with defaults
    python run_perturbations.py --n-scenes 10     # Pilot study
    python run_perturbations.py --resume          # Resume from checkpoint
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Project root
PROJECT_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools" / "alpamayo" / "src"))

from pipeline.lib.schema import load_scenes
from pipeline.step_1_embed import TEXT_VOCABULARY


# =============================================================================
# CONSTANTS (derived from source definitions)
# =============================================================================

# Keys to perturb - all keys with vocabulary definitions
PERTURBATION_KEYS = list(TEXT_VOCABULARY.keys())

# Model settings
MODEL_ID = "nvidia/Alpamayo-R1-10B"
T0_US = 5_000_000  # 5 seconds into clip

# Paths
DATA_DIR = PROJECT_ROOT / "data"
PIPELINE_DIR = DATA_DIR / "pipeline"
ANCHOR_FILE = DATA_DIR / "CLS-001" / "scene_classifications.json"
OUTPUT_DIR = DATA_DIR / "BND-004"


def load_anchor_labels() -> dict[str, dict[str, Any]]:
    """
    Load full anchor labels from CLS-001 (26 keys per scene).

    Returns:
        {clip_id: {key: value, ...}}
    """
    import json

    with open(ANCHOR_FILE) as f:
        data = json.load(f)

    labels = {}
    for item in data.get("classifications", []):
        clip_id = item["clip_id"]
        cls = item.get("classification", {})

        # Extract simple values from classification (some have nested structure)
        simple_labels = {}
        for key, val in cls.items():
            if isinstance(val, dict):
                # Extract the actual value from nested dict
                # e.g., {"weather": "clear", "reasoning": "..."} -> "clear"
                for k, v in val.items():
                    if k == key or k not in ["reasoning", "confidence"]:
                        if not isinstance(v, (dict, list)):
                            simple_labels[key] = v
                            break
            elif isinstance(val, (str, bool, int, float)):
                simple_labels[key] = val

        labels[clip_id] = simple_labels

    return labels


# =============================================================================
# TEXT CONTEXT GENERATION
# =============================================================================

def create_scene_description(labels: dict[str, str]) -> str:
    """
    Create a natural language scene description from labels.

    Args:
        labels: Dict of {key: value} for scene attributes

    Returns:
        Natural language description string
    """
    parts = []

    # Weather and time
    if "weather" in labels:
        parts.append(f"{labels['weather']} weather")
    if "time_of_day" in labels:
        parts.append(f"during {labels['time_of_day'].replace('_', ' ')}")

    # Road context
    if "road_type" in labels:
        road = labels["road_type"].replace("_", " ")
        parts.append(f"on a {road}")

    # Complexity
    if "depth_complexity" in labels:
        parts.append(f"with {labels['depth_complexity']} depth complexity")
    if "occlusion_level" in labels:
        parts.append(f"and {labels['occlusion_level']} occlusion")

    # Action
    if "required_action" in labels:
        action = labels["required_action"]
        if action != "none":
            parts.append(f"requiring {action} action")

    if not parts:
        return ""

    return "Scene context: " + ", ".join(parts) + "."


def create_message_with_context(frames: torch.Tensor, context: str = ""):
    """
    Construct message with optional scene context injected.

    Modified from alpamayo_r1.helper.create_message to accept text context.
    """
    assert frames.ndim == 4, f"{frames.ndim=}, expected (N, C, H, W)"

    num_traj_token = 48
    hist_traj_placeholder = (
        f"<|traj_history_start|>{'<|traj_history|>' * num_traj_token}<|traj_history_end|>"
    )

    # Build the prompt with optional context
    if context:
        prompt_text = f"{context}\n\n{hist_traj_placeholder}output the chain-of-thought reasoning of the driving process, then output the future trajectory."
    else:
        prompt_text = f"{hist_traj_placeholder}output the chain-of-thought reasoning of the driving process, then output the future trajectory."

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a driving assistant that generates safe and accurate actions.",
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "image", "image": frame} for frame in frames]
            + [
                {
                    "type": "text",
                    "text": prompt_text,
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "<|cot_start|>",
                }
            ],
        },
    ]


# =============================================================================
# INFERENCE
# =============================================================================

# Scene data cache to avoid reloading for each perturbation
_scene_data_cache: dict[str, dict] = {}


def get_scene_data(clip_id: str, t0_us: int = 5_000_000) -> dict:
    """
    Load scene data with caching.

    Each scene is loaded once and reused for all 58 perturbations.
    """
    cache_key = f"{clip_id}_{t0_us}"

    if cache_key not in _scene_data_cache:
        from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
        _scene_data_cache[cache_key] = load_physical_aiavdataset(clip_id, t0_us=t0_us)

    return _scene_data_cache[cache_key]


def clear_scene_cache():
    """Clear scene cache to free memory."""
    global _scene_data_cache
    _scene_data_cache.clear()


def setup_environment():
    """Setup environment for inference."""
    # Monkey-patch for scipy compatibility
    pd.options.mode.copy_on_write = False

    from physical_ai_av import egomotion
    import scipy.spatial.transform as spt

    @classmethod
    def patched_from_egomotion_df(cls, egomotion_df):
        return cls(
            pose=spt.RigidTransform.from_components(
                rotation=spt.Rotation.from_quat(
                    egomotion_df[["qx", "qy", "qz", "qw"]].to_numpy().copy()
                ),
                translation=egomotion_df[["x", "y", "z"]].to_numpy().copy(),
            ),
            velocity=egomotion_df[["vx", "vy", "vz"]].to_numpy().copy(),
            acceleration=egomotion_df[["ax", "ay", "az"]].to_numpy().copy(),
            curvature=egomotion_df[["curvature"]].to_numpy().copy(),
        )

    egomotion.EgomotionState.from_egomotion_df = patched_from_egomotion_df
    print("Applied scipy compatibility patch")

    # HuggingFace caching configuration
    # Use local cache to avoid repeated downloads and rate limits
    os.environ.setdefault("HF_HUB_CACHE", str(Path.home() / ".cache" / "huggingface" / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(Path.home() / ".cache" / "huggingface" / "transformers"))

    # If files already downloaded, prefer offline mode to avoid API calls
    # Set HF_HUB_OFFLINE=1 manually if you want to force offline mode
    print(f"HuggingFace cache: {os.environ.get('HF_HUB_CACHE', 'default')}")

    # HuggingFace auth
    token = os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        print("HuggingFace: Authenticated")


def load_model():
    """Load Alpamayo model and processor."""
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1.helper import get_processor

    print(f"Loading model: {MODEL_ID}")
    model = AlpamayoR1.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    processor = get_processor(model.tokenizer)
    print("Model loaded")

    return model, processor


def run_single_inference(
    model,
    processor,
    clip_id: str,
    context: str,
    t0_us: int = 5_000_000,
) -> dict[str, Any]:
    """
    Run inference with a specific text context.

    Args:
        model: Alpamayo model
        processor: Model processor
        clip_id: Scene clip ID
        context: Text context to inject
        t0_us: Timestamp in microseconds

    Returns:
        Dict with ADE and metadata
    """
    from alpamayo_r1.helper import to_device

    # Load scene data (cached - loads once per scene, reused for all 58 perturbations)
    data = get_scene_data(clip_id, t0_us)

    # Create message with context
    messages = create_message_with_context(
        data["image_frames"].flatten(0, 1),
        context=context,
    )

    # Process inputs
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )

    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = to_device(model_inputs, "cuda")

    # Run inference
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=256,
            return_extra=True,
        )

    # Compute ADE
    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    min_ade = float(diff.min())

    # Extract CoT
    cot_raw = extra.get("cot", [])
    if cot_raw:
        cot_text = cot_raw[0]
        if isinstance(cot_text, list):
            cot_text = cot_text[0] if cot_text else ""
    else:
        cot_text = ""

    return {
        "ade": min_ade,
        "cot": str(cot_text),
    }


# =============================================================================
# PERTURBATION RUNNER
# =============================================================================

def get_perturbation_combinations(
    original_labels: dict[str, str],
    keys_to_perturb: list[str],
) -> list[dict]:
    """
    Generate all perturbation combinations.

    Args:
        original_labels: Original scene labels
        keys_to_perturb: Which keys to vary

    Returns:
        List of perturbation specs
    """
    combinations = []

    for key in keys_to_perturb:
        if key not in TEXT_VOCABULARY:
            continue

        original_value = original_labels.get(key)
        if pd.isna(original_value):
            continue

        # Normalize original_value to string for comparison (bool -> "true"/"false")
        original_value_str = str(original_value).lower() if isinstance(original_value, bool) else str(original_value)

        # Generate perturbations for each possible value
        for perturbed_value in TEXT_VOCABULARY[key].keys():
            combinations.append({
                "key": key,
                "original_value": original_value,  # Keep original for storage
                "perturbed_value": perturbed_value,
                "is_aligned": perturbed_value == original_value_str,  # Compare normalized
            })

    return combinations


def run_perturbation_study(
    model,
    processor,
    scenes_df: pd.DataFrame,
    anchor_labels: dict[str, dict[str, Any]],
    output_dir: Path,
    resume: bool = False,
    checkpoint_interval: int = 10,
) -> pd.DataFrame:
    """
    Run full perturbation study across selected scenes.
    """
    results_file = output_dir / "perturbation_results.parquet"
    keys_to_perturb = PERTURBATION_KEYS

    # Load existing results if resuming
    if resume and results_file.exists():
        existing_results = pd.read_parquet(results_file)
        completed_combos = set(zip(
            existing_results["clip_id"],
            existing_results["key"],
            existing_results["perturbed_value"],
        ))
        print(f"Resuming: {len(existing_results)} existing results")
    else:
        existing_results = pd.DataFrame()
        completed_combos = set()

    results = []

    # Process each scene
    for scene_idx, (_, scene_row) in enumerate(scenes_df.iterrows()):
        clip_id = scene_row["clip_id"]
        print(f"\n[{scene_idx + 1}/{len(scenes_df)}] Scene: {clip_id}")

        # Get original labels from anchor data (full 26 keys)
        if clip_id not in anchor_labels:
            print(f"  Skipping: no anchor labels")
            continue

        original_labels = {
            key: anchor_labels[clip_id].get(key)
            for key in keys_to_perturb
            if anchor_labels[clip_id].get(key) is not None
        }

        if not original_labels:
            print(f"  Skipping: no labels available")
            continue

        # Get perturbation combinations
        combinations = get_perturbation_combinations(original_labels, keys_to_perturb)
        print(f"  Combinations: {len(combinations)}")

        # Run each perturbation
        for combo in tqdm(combinations, desc="  Perturbations", leave=False):
            # Check if already completed
            combo_key = (clip_id, combo["key"], combo["perturbed_value"])
            if combo_key in completed_combos:
                continue

            # Create perturbed labels
            perturbed_labels = original_labels.copy()
            perturbed_labels[combo["key"]] = combo["perturbed_value"]

            # Generate context string
            context = create_scene_description(perturbed_labels)

            # Run inference
            try:
                inference_result = run_single_inference(
                    model, processor, clip_id, context, T0_US
                )

                result = {
                    "clip_id": clip_id,
                    "key": combo["key"],
                    # Convert to string for consistent parquet types (booleans -> "true"/"false")
                    "original_value": str(combo["original_value"]).lower() if isinstance(combo["original_value"], bool) else str(combo["original_value"]),
                    "perturbed_value": str(combo["perturbed_value"]),
                    "is_aligned": combo["is_aligned"],
                    "context_text": context,
                    "ade": inference_result["ade"],
                    "cot": inference_result["cot"],
                    "timestamp": datetime.now().isoformat(),
                }
                results.append(result)

            except Exception as e:
                print(f"  Error: {e}")
                continue

        # Clear scene cache after processing all perturbations for this scene
        clear_scene_cache()

        # Checkpoint
        if (scene_idx + 1) % checkpoint_interval == 0 and results:
            checkpoint_df = pd.concat([
                existing_results,
                pd.DataFrame(results),
            ], ignore_index=True)
            checkpoint_df.to_parquet(results_file, index=False)
            print(f"  Checkpoint: {len(checkpoint_df)} total results")

    # Final save
    final_df = pd.concat([
        existing_results,
        pd.DataFrame(results),
    ], ignore_index=True)

    final_df.to_parquet(results_file, index=False)
    print(f"\nSaved {len(final_df)} results to {results_file}")

    return final_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="BND-004: Run perturbation inference study",
    )
    parser.add_argument(
        "--n-scenes", type=int, default=50,
        help="Number of scenes to process"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=10,
        help="Save checkpoint every N scenes"
    )

    args = parser.parse_args()

    # Setup paths
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BND-004: PERTURBATION STUDY")
    print("=" * 60)
    print(f"Scenes: {args.n_scenes}")
    print(f"Keys: {PERTURBATION_KEYS}")
    print(f"Output: {OUTPUT_DIR}")

    # Check CUDA
    if not torch.cuda.is_available():
        print("\nError: CUDA not available. This experiment requires a GPU.")
        return 1

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # Load scenes
    scenes_file = PIPELINE_DIR / "scenes.parquet"
    if not scenes_file.exists():
        print("\nError: scenes.parquet not found. Run pipeline first.")
        return 1

    df = load_scenes(scenes_file)
    print(f"\nTotal scenes: {len(df)}")

    # Load anchor labels (full 26 keys)
    print("Loading anchor labels from CLS-001...")
    anchor_labels = load_anchor_labels()
    print(f"Anchor scenes with full labels: {len(anchor_labels)}")

    # Filter to anchor scenes only (they have all labels)
    df = df[df["clip_id"].isin(anchor_labels.keys())]
    print(f"Anchor scenes in pipeline: {len(df)}")

    # Filter: need baseline ADE
    df = df[df["has_ade"] == True]
    print(f"With baseline ADE: {len(df)}")

    # Prioritize anchors, then by confidence
    df = df.sort_values(
        by=["is_anchor", "label_confidence"],
        ascending=[False, False],
    )

    # Sample
    if len(df) > args.n_scenes:
        df = df.head(args.n_scenes)
    print(f"Selected: {len(df)} scenes")

    # Estimate total inferences
    avg_values_per_key = np.mean([len(TEXT_VOCABULARY[k]) for k in PERTURBATION_KEYS])
    total_inferences = int(len(df) * len(PERTURBATION_KEYS) * avg_values_per_key)
    print(f"Estimated inferences: ~{total_inferences}")

    # Setup environment
    setup_environment()

    # Load model
    model, processor = load_model()

    # Run study
    print("\n" + "=" * 60)
    print("RUNNING PERTURBATION STUDY")
    print("=" * 60)

    results_df = run_perturbation_study(
        model, processor, df, anchor_labels, OUTPUT_DIR,
        resume=args.resume,
        checkpoint_interval=args.checkpoint_interval,
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total results: {len(results_df)}")
    print(f"Unique scenes: {results_df['clip_id'].nunique()}")
    print(f"Keys perturbed: {results_df['key'].nunique()}")

    # Quick analysis
    aligned = results_df[results_df["is_aligned"] == True]["ade"]
    misaligned = results_df[results_df["is_aligned"] == False]["ade"]
    print(f"\nADE (aligned):    {aligned.mean():.3f} ± {aligned.std():.3f}")
    print(f"ADE (misaligned): {misaligned.mean():.3f} ± {misaligned.std():.3f}")

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("Run analyze_results.py for detailed analysis")

    return 0


if __name__ == "__main__":
    sys.exit(main())
