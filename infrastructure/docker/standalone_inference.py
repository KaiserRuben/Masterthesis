#!/usr/bin/env python3
"""
Standalone Alpamayo-R1 Inference Script

Mirrors the workstation basic_inference_test.py for cloud GPU providers.

Usage:
    pip install -r requirements.txt
    export HF_TOKEN="your_token"
    python standalone_inference.py -o output.json

Options:
    --ids FILE  Clip IDs file (default: inference_target_ids.txt)
    -o FILE     Output JSON file (default: inference_output.json)
    --t0 INT    Timestamp in microseconds (default: 5000000)
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Import torch/numpy BEFORE alpamayo (order matters!)
import torch
import numpy as np

# Fix for pandas 2.x + scipy 1.16+ compatibility issue
# DataFrame.to_numpy() returns read-only arrays, scipy.Rotation expects writable
import pandas as pd
pd.options.mode.copy_on_write = False


def patch_physical_ai_av():
    """Monkey-patch physical_ai_av.egomotion to fix read-only array issue with scipy>=1.16."""
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


patch_physical_ai_av()


def patch_physical_ai_av():
    """Monkey-patch physical_ai_av to fix read-only array issue."""
    try:
        from physical_ai_av import egomotion
        import scipy.spatial.transform as spt

        original_from_egomotion_df = egomotion.EgomotionState.from_egomotion_df

        @classmethod
        def patched_from_egomotion_df(cls, egomotion_df):
            """Patched version that copies arrays to make them writable."""
            return cls(
                timestamp_us=egomotion_df["timestamp_us"].to_numpy().copy(),
                position=egomotion_df[["x", "y", "z"]].to_numpy().copy(),
                rotation=spt.Rotation.from_quat(
                    egomotion_df[["qx", "qy", "qz", "qw"]].to_numpy().copy()
                ),
            )

        egomotion.EgomotionState.from_egomotion_df = patched_from_egomotion_df
        print("Patched physical_ai_av for read-only array fix")
    except Exception as e:
        print(f"Warning: Could not patch physical_ai_av: {e}")


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return super().default(obj)


# =============================================================================
# CLIP IDS TO PROCESS
# =============================================================================
def load_clip_ids(ids_file: str | None = None) -> list[str]:
    """Load clip IDs from file or use default."""
    if ids_file and Path(ids_file).exists():
        with open(ids_file) as f:
            return [line.strip() for line in f if line.strip()]

    # Default: empty list (must provide --ids file)
    return []


CLIP_IDS = []  # Populated from --ids argument in main()

MODEL_ID = "nvidia/Alpamayo-R1-10B"


# =============================================================================
# SETUP
# =============================================================================
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
    import os
    token = os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        print("HuggingFace: Authenticated")
    else:
        print("WARNING: HF_TOKEN not set")


# =============================================================================
# INFERENCE (exact copy from workstation/basic_inference_test.py)
# =============================================================================
def run_inference(model, processor, clip_id: str, t0_us: int = 5_000_000) -> dict:
    """Run inference on a single scene and return all outputs."""
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

    # Compute metrics (exact same as workstation version)
    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    min_ade = float(diff.min())

    # Extract CoC reasoning
    coc_text = extra["cot"][0] if extra.get("cot") else ""

    return {
        "clip_id": clip_id,
        "t0_us": t0_us,
        "coc_reasoning": coc_text,
        "min_ade": min_ade,
        "predicted_trajectory": pred_xy[0].tolist(),
        "ground_truth_trajectory": gt_xy.tolist(),
        "load_time_s": round(load_time, 2),
        "inference_time_s": round(infer_time, 2),
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Alpamayo-R1 Inference")
    parser.add_argument("-o", "--output", default="inference_output.json")
    parser.add_argument("--ids", type=str, default="inference_target_ids.txt", help="Path to file with clip IDs (one per line)")
    parser.add_argument("--t0", type=int, default=5_000_000)
    args = parser.parse_args()

    # Load clip IDs from file
    global CLIP_IDS
    CLIP_IDS = load_clip_ids(args.ids)
    if not CLIP_IDS:
        print(f"ERROR: No clip IDs found in {args.ids}")
        sys.exit(1)

    print("=" * 70)
    print("ALPAMAYO-R1 STANDALONE INFERENCE")
    print("=" * 70)

    # Setup
    setup_alpamayo()
    setup_huggingface()

    # Now import alpamayo (after setup)
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1 import helper

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"Clips to process: {len(CLIP_IDS)}")

    # Load model
    print(f"\nLoading model: {MODEL_ID}...")
    model_load_start = time.time()
    model = AlpamayoR1.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to("cuda")
    processor = helper.get_processor(model.tokenizer)
    model_load_time = time.time() - model_load_start
    print(f"Model loaded in {model_load_time:.1f}s")

    # Run inference
    results = []
    print("\n" + "-" * 70)

    for i, clip_id in enumerate(CLIP_IDS):
        print(f"\n[{i+1}/{len(CLIP_IDS)}] {clip_id}")

        try:
            result = run_inference(model, processor, clip_id, args.t0)
            results.append(result)
            print(f"  minADE: {result['min_ade']:.3f}m | infer: {result['inference_time_s']:.1f}s")

            # Checkpoint every 5 clips
            if (i + 1) % 5 == 0:
                save_results(args.output, results, gpu_name, model_load_time, args.t0)
                print(f"  [Checkpoint saved]")

        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            print("  FULL TRACEBACK:")
            traceback.print_exc()
            results.append({"clip_id": clip_id, "error": str(e)})

    # Final save
    save_results(args.output, results, gpu_name, model_load_time, args.t0)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    successful = [r for r in results if "min_ade" in r]
    if successful:
        ades = [r["min_ade"] for r in successful]
        print(f"Successful: {len(successful)}/{len(CLIP_IDS)}")
        print(f"minADE: mean={np.mean(ades):.3f}m, range=[{min(ades):.3f}, {max(ades):.3f}]m")

    print(f"\nOutput: {args.output}")


def save_results(output_path, results, gpu_name, model_load_time, t0_us):
    output = {
        "metadata": {
            "model_id": MODEL_ID,
            "num_samples": len(results),
            "t0_us": t0_us,
            "gpu": gpu_name,
            "model_load_time_s": round(model_load_time, 2),
            "timestamp": datetime.now().isoformat(),
        },
        "results": results,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)


if __name__ == "__main__":
    main()
