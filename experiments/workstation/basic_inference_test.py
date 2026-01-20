"""
Basic Inference Test for Workstation

Runs Alpamayo-R1 inference on sample scenes.
Saves all outputs (CoC reasoning, trajectories, metrics) for analysis.

Usage:
    python basic_inference_test.py                    # Run 5 random scenes
    python basic_inference_test.py -n 10              # Run 10 random scenes
    python basic_inference_test.py --clip-id abc123   # Run specific clip
    python basic_inference_test.py --list-clips       # List available clips

Requirements:
- NVIDIA GPU with 24GB+ VRAM
- pip install -e alpamayo/
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import pandas as pd

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_NUM_SAMPLES = 5
DEFAULT_SEED = 42
MODEL_ID = "nvidia/Alpamayo-R1-10B"

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = SCRIPT_DIR / "inference_results"
CLIP_IDS_FILE = PROJECT_ROOT / "alpamayo" / "notebooks" / "clip_ids.parquet"


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Alpamayo-R1 inference on driving scenes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                         # 5 random scenes
  %(prog)s -n 10                   # 10 random scenes
  %(prog)s --clip-id abc123        # Specific clip
  %(prog)s --clip-ids clips.txt    # Clips from file (one per line)
  %(prog)s --list-clips            # Show available clips
  %(prog)s -o results.json         # Custom output file
        """
    )
    parser.add_argument("-n", "--num-samples", type=int, default=DEFAULT_NUM_SAMPLES,
                        help=f"Number of random scenes to test (default: {DEFAULT_NUM_SAMPLES})")
    parser.add_argument("--clip-id", type=str,
                        help="Test a specific clip ID")
    parser.add_argument("--clip-ids", type=str,
                        help="File with clip IDs (one per line)")
    parser.add_argument("--t0", type=int, default=5_000_000,
                        help="Timestamp in microseconds (default: 5000000 = 5s)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed (default: {DEFAULT_SEED})")
    parser.add_argument("-o", "--output", type=str,
                        help="Output JSON file (default: auto-generated in inference_results/)")
    parser.add_argument("--list-clips", action="store_true",
                        help="List available clip IDs and exit")
    return parser.parse_args()


# ============================================================================
# INFERENCE
# ============================================================================

def run_inference(model, processor, clip_id: str, t0_us: int = 5_000_000) -> dict:
    """Run inference on a single scene and return all outputs."""

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


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()

    # Load all clip IDs
    clip_ids_df = pd.read_parquet(CLIP_IDS_FILE)
    all_clip_ids = clip_ids_df["clip_id"].tolist()

    # Handle --list-clips
    if args.list_clips:
        print(f"Available clips ({len(all_clip_ids)}):")
        for cid in all_clip_ids[:20]:
            print(f"  {cid}")
        if len(all_clip_ids) > 20:
            print(f"  ... and {len(all_clip_ids) - 20} more")
        print(f"\nTotal: {len(all_clip_ids)} clips")
        sys.exit(0)

    print("=" * 80)
    print("ALPAMAYO-R1 BASIC INFERENCE TEST")
    print("=" * 80)

    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires an NVIDIA GPU.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # Determine which clips to test
    if args.clip_id:
        selected_clips = [args.clip_id]
        print(f"\nTesting specific clip: {args.clip_id}")
    elif args.clip_ids:
        with open(args.clip_ids) as f:
            selected_clips = [line.strip() for line in f if line.strip()]
        print(f"\nLoaded {len(selected_clips)} clips from {args.clip_ids}")
    else:
        selected_clips = random.sample(all_clip_ids, min(args.num_samples, len(all_clip_ids)))
        print(f"\nSelected {len(selected_clips)} random clips (seed={args.seed})")

    # Load model
    print(f"\nLoading model: {MODEL_ID}...")
    model_load_start = time.time()
    model = AlpamayoR1.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to("cuda")
    processor = helper.get_processor(model.tokenizer)
    model_load_time = time.time() - model_load_start
    print(f"Model loaded in {model_load_time:.1f}s")

    # Run inference on each clip
    results = []
    print("\n" + "-" * 80)

    for i, clip_id in enumerate(selected_clips):
        print(f"\n[{i+1}/{len(selected_clips)}] Clip: {clip_id[:16]}...")

        try:
            result = run_inference(model, processor, clip_id, args.t0)
            results.append(result)

            print(f"    minADE: {result['min_ade']:.3f} m")
            coc_preview = result['coc_reasoning'][:100] + "..." if len(result['coc_reasoning']) > 100 else result['coc_reasoning']
            print(f"    CoC: {coc_preview}")
            print(f"    Time: load={result['load_time_s']:.1f}s, infer={result['inference_time_s']:.1f}s")

        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({
                "clip_id": clip_id,
                "error": str(e),
            })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    successful = [r for r in results if "min_ade" in r]
    if successful:
        ades = [r["min_ade"] for r in successful]
        print(f"Successful: {len(successful)}/{len(selected_clips)}")
        print(f"minADE: mean={np.mean(ades):.3f}m, std={np.std(ades):.3f}m, range=[{min(ades):.3f}, {max(ades):.3f}]m")

        infer_times = [r["inference_time_s"] for r in successful]
        print(f"Inference time: mean={np.mean(infer_times):.1f}s")

    # Save results
    output = {
        "metadata": {
            "model_id": MODEL_ID,
            "num_samples": len(selected_clips),
            "seed": args.seed,
            "t0_us": args.t0,
            "gpu": gpu_name,
            "model_load_time_s": round(model_load_time, 2),
            "timestamp": datetime.now().isoformat(),
        },
        "results": results,
    }

    # Determine output file
    if args.output:
        output_file = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = OUTPUT_DIR / f"inference_{timestamp}.json"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
