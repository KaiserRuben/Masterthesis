#!/usr/bin/env python3
"""
Alpamayo-R1 Inference Script (Docker-compatible)

Runs trajectory prediction on specified clip IDs.
Designed to run in Docker container on cloud GPU providers.

Usage:
    python run_inference.py --clip-ids clips.txt -o output.json
    python run_inference.py --clip-id abc123 -o output.json

Environment Variables:
    HF_TOKEN: HuggingFace API token (required for dataset access)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return super().default(obj)


def setup_huggingface():
    """Setup HuggingFace authentication from environment."""
    token = os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        print("HuggingFace: Authenticated via HF_TOKEN")
    else:
        print("WARNING: HF_TOKEN not set. Dataset access may fail.")


def run_inference(model, processor, clip_id: str, t0_us: int = 5_000_000) -> dict:
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


def main():
    parser = argparse.ArgumentParser(description="Run Alpamayo-R1 trajectory inference")
    parser.add_argument("--clip-id", type=str, help="Single clip ID to process")
    parser.add_argument("--clip-ids", type=str, help="File with clip IDs (one per line)")
    parser.add_argument("--t0", type=int, default=5_000_000, help="Timestamp in microseconds")
    parser.add_argument("-o", "--output", type=str, default='/tmp', help="Output JSON file")
    args = parser.parse_args()

    print("=" * 80)
    print("ALPAMAYO-R1 DOCKER INFERENCE")
    print("=" * 80)

    # Setup HuggingFace
    setup_huggingface()

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # Get clip IDs
    if args.clip_id:
        clip_ids = [args.clip_id]
    elif args.clip_ids:
        with open(args.clip_ids) as f:
            clip_ids = [line.strip() for line in f if line.strip()]
    else:
        print("ERROR: Must specify --clip-id or --clip-ids")
        sys.exit(1)

    print(f"Clips to process: {len(clip_ids)}")

    # Load model
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1 import helper

    MODEL_ID = "nvidia/Alpamayo-R1-10B"
    print(f"\nLoading model: {MODEL_ID}...")
    model_load_start = time.time()
    model = AlpamayoR1.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to("cuda")
    processor = helper.get_processor(model.tokenizer)
    model_load_time = time.time() - model_load_start
    print(f"Model loaded in {model_load_time:.1f}s")

    # Run inference
    results = []
    print("\n" + "-" * 80)

    for i, clip_id in enumerate(clip_ids):
        print(f"\n[{i+1}/{len(clip_ids)}] Clip: {clip_id[:16]}...")

        try:
            result = run_inference(model, processor, clip_id, args.t0)
            results.append(result)
            print(f"    minADE: {result['min_ade']:.3f} m")
            print(f"    Time: load={result['load_time_s']:.1f}s, infer={result['inference_time_s']:.1f}s")

            # Save intermediate results (crash recovery)
            if (i + 1) % 5 == 0:
                _save_results(args.output, results, MODEL_ID, gpu_name, model_load_time, args.t0)
                print(f"    [Checkpoint saved: {len(results)} results]")

        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({"clip_id": clip_id, "error": str(e)})

    # Final save
    _save_results(args.output, results, MODEL_ID, gpu_name, model_load_time, args.t0)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    successful = [r for r in results if "min_ade" in r]
    if successful:
        ades = [r["min_ade"] for r in successful]
        print(f"Successful: {len(successful)}/{len(clip_ids)}")
        print(f"minADE: mean={np.mean(ades):.3f}m, range=[{min(ades):.3f}, {max(ades):.3f}]m")

    print(f"\nResults saved to: {args.output}")


def _save_results(output_path, results, model_id, gpu_name, model_load_time, t0_us):
    """Save results to JSON file."""
    output = {
        "metadata": {
            "model_id": model_id,
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
