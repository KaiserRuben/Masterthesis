#!/usr/bin/env python3
"""
Step 3: Run Inference

Runs Alpamayo-R1-10B inference on scenes.
Computes ADE and trajectory classes.

Usage:
    python pipeline/step_3_infer.py
    python pipeline/step_3_infer.py --max-scenes 100
    python pipeline/step_3_infer.py --resume
"""

import argparse
import subprocess
import sys
import time
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent))

from lib.schema import load_scenes, save_scenes
from lib.trajectory import classify_trajectory
from lib.io import load_config, get_repo_root
from lib.models import InferenceResult


# Monkey-patch for scipy compatibility
# Must be called BEFORE importing alpamayo
def patch_physical_ai_av():
    """Monkey-patch physical_ai_av.egomotion to fix read-only array issue with scipy>=1.16."""
    # Fix for pandas 2.x + scipy 1.16+ compatibility
    pd.options.mode.copy_on_write = False

    from physical_ai_av import egomotion
    import scipy.spatial.transform as spt

    @classmethod
    def patched_from_egomotion_df(cls, egomotion_df):
        # Same as original but with .copy() to make arrays writable for scipy
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
    print("Applied scipy>=1.16 compatibility patch")


def setup_alpamayo():
    """Clone and setup alpamayo if not present."""
    try:
        from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
        return
    except ImportError:
        pass

    print("Cloning Alpamayo-R1...")
    if not Path("alpamayo").exists():
        subprocess.check_call([
            "git", "clone", "--depth", "1",
            "https://github.com/NVlabs/Alpamayo.git", "alpamayo"
        ])

    alpamayo_src = Path("alpamayo/src").resolve()
    if str(alpamayo_src) not in sys.path:
        sys.path.insert(0, str(alpamayo_src))


def setup_huggingface():
    """Setup HuggingFace authentication."""
    token = os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        print("HuggingFace: Authenticated")
    else:
        print("WARNING: HF_TOKEN not set. Model download may fail.")


def run_inference(model, processor, clip_id: str, t0_us: int = 5_000_000) -> InferenceResult:
    """Run inference on a single scene."""
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo_r1 import helper

    # Load data
    load_start = time.time()
    data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
    load_time = time.time() - load_start

    # Prepare inputs
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
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
    model_inputs = helper.to_device(model_inputs, "cuda")

    # Run inference
    infer_start = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=256,
            return_extra=True,
        )
    infer_time = time.time() - infer_start

    # Compute metrics
    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    min_ade = float(diff.min())

    # Get best trajectory
    best_idx = diff.argmin()
    best_pred_xy = pred_xy[best_idx]

    # Classify trajectory (returns TrajectoryClassification model)
    traj_class = classify_trajectory(best_pred_xy, gt_xy)

    # Extract CoC reasoning (may be nested list like [['text']])
    coc_raw = extra.get("cot", [])
    if coc_raw:
        coc_text = coc_raw[0]
        # Flatten if still a list
        if isinstance(coc_text, list):
            coc_text = coc_text[0] if coc_text else ""
    else:
        coc_text = ""

    return InferenceResult(
        clip_id=clip_id,
        t0_us=t0_us,
        ade=min_ade,
        coc_reasoning=str(coc_text),
        traj_direction=traj_class.direction,
        traj_speed=traj_class.speed,
        traj_lateral=traj_class.lateral,
        load_time_s=round(load_time, 2),
        inference_time_s=round(infer_time, 2),
        inference_timestamp=datetime.now().isoformat(),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Step 3: Run Alpamayo-R1 inference",
    )
    parser.add_argument(
        "--max-scenes", type=int, default=None,
        help="Maximum number of scenes to process"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint (skips already-processed scenes)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config file"
    )

    args = parser.parse_args()

    # Load config
    config_path = args.config or (Path(__file__).parent / "config.yaml")
    config = load_config(config_path)

    # Resolve parameters
    t0_us = config.dataset.t0_us
    checkpoint_interval = config.inference.checkpoint_interval
    model_id = config.inference.model_id

    # Resolve paths
    repo_root = get_repo_root()
    scenes_file = repo_root / config.paths.scenes_file

    print("=" * 60)
    print("STEP 3: INFERENCE")
    print("=" * 60)
    print(f"Model: {model_id}")
    print(f"Max scenes: {args.max_scenes or 'all'}")

    # Check CUDA
    if not torch.cuda.is_available():
        print("\nError: CUDA not available. This step requires a GPU.")
        return 1

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # Load scenes
    if not scenes_file.exists():
        print("\nError: scenes.parquet not found. Run step_0_sample.py first.")
        return 1

    df = load_scenes(scenes_file)
    print(f"Total scenes: {len(df)}")

    # Filter scenes without ADE
    to_process = df[df["has_ade"] != True].copy()
    print(f"To process: {len(to_process)}")

    if len(to_process) == 0:
        print("\nAll scenes already have ADE. Nothing to do.")
        return 0

    # Prioritize: anchors first, then by label_confidence descending
    to_process = to_process.sort_values(
        by=["is_anchor", "label_confidence"],
        ascending=[False, False],
    )

    # Apply max-scenes limit
    if args.max_scenes and args.max_scenes < len(to_process):
        to_process = to_process.head(args.max_scenes)
        print(f"Limited to {len(to_process)} scenes")

    # Apply scipy patch
    patch_physical_ai_av()

    # Setup alpamayo
    print("\nSetting up Alpamayo...")
    setup_alpamayo()
    setup_huggingface()

    # Import alpamayo
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1 import helper

    # Load model
    print(f"\nLoading model: {model_id}...")
    model_load_start = time.time()
    model = AlpamayoR1.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda")
    processor = helper.get_processor(model.tokenizer)
    model_load_time = time.time() - model_load_start
    print(f"Model loaded in {model_load_time:.1f}s")

    # Process scenes
    clip_ids = to_process["clip_id"].tolist()
    results = []
    failed = []

    print(f"\nProcessing {len(clip_ids)} scenes...")
    print("-" * 60)

    for i, clip_id in enumerate(clip_ids):
        print(f"\n[{i+1}/{len(clip_ids)}] {clip_id}")

        try:
            result = run_inference(model, processor, clip_id, t0_us)
            results.append(result)
            print(f"  ADE: {result.ade:.3f}m | {result.traj_direction}/{result.traj_speed}/{result.traj_lateral} | {result.inference_time_s:.1f}s")

            # Update DataFrame
            idx = df[df["clip_id"] == clip_id].index[0]
            df.loc[idx, "ade"] = result.ade
            df.loc[idx, "coc_reasoning"] = result.coc_reasoning
            df.loc[idx, "has_ade"] = True
            df.loc[idx, "inference_timestamp"] = result.inference_timestamp
            df.loc[idx, "traj_direction"] = result.traj_direction
            df.loc[idx, "traj_speed"] = result.traj_speed
            df.loc[idx, "traj_lateral"] = result.traj_lateral

            # Checkpoint
            if (i + 1) % checkpoint_interval == 0:
                save_scenes(df, scenes_file)
                print(f"  [Checkpoint saved: {i+1} scenes]")

        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            failed.append((clip_id, str(e)))

    # Final save
    save_scenes(df, scenes_file)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    successful = len(results)
    print(f"Successful: {successful}/{len(clip_ids)}")

    if results:
        ades = [r.ade for r in results]
        print(f"ADE: mean={np.mean(ades):.3f}m, min={np.min(ades):.3f}m, max={np.max(ades):.3f}m")

        # Trajectory class distribution
        print("\nTrajectory classes:")
        for dim in ["direction", "speed", "lateral"]:
            values = [getattr(r, f"traj_{dim}") for r in results]
            unique, counts = np.unique(values, return_counts=True)
            print(f"  {dim}: {dict(zip(unique, counts))}")

    if failed:
        print(f"\nFailed: {len(failed)}")
        for clip_id, error in failed[:5]:
            print(f"  {clip_id}: {error}")

    print(f"\nTotal with ADE: {df['has_ade'].sum()}/{len(df)}")
    print(f"Output: {scenes_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
