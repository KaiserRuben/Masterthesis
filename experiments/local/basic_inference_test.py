"""
Basic Inference Test for Local (Mac/MPS)

Runs Alpamayo-R1 inference on sample scenes using Apple Silicon MPS backend.
Saves all outputs (CoC reasoning, trajectories, metrics) for analysis.

Usage:
    python basic_inference_test.py                    # Run 5 random scenes
    python basic_inference_test.py -n 10              # Run 10 random scenes
    python basic_inference_test.py --clip-id abc123   # Run specific clip
    python basic_inference_test.py --list-clips       # List available clips
    python basic_inference_test.py --cpu              # Force CPU (if MPS fails)

Requirements:
- Apple Silicon Mac (M1/M2/M3) or CPU fallback
- pip install -e tools/alpamayo/

NOTE: This script uses 'eager' attention instead of flash-attn.
      Inference will be slower than on NVIDIA GPUs.
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


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return super().default(obj)


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_NUM_SAMPLES = 5
DEFAULT_SEED = 42
MODEL_ID = "nvidia/Alpamayo-R1-10B"

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "alpamayo_inference"
CLIP_IDS_FILE = PROJECT_ROOT / "tools" / "alpamayo" / "notebooks" / "clip_ids.parquet"


# ============================================================================
# DEVICE DETECTION
# ============================================================================

def get_device(force_cpu: bool = False) -> tuple[str, str]:
    """Detect best available device (MPS, CPU)."""
    if force_cpu:
        return "cpu", "CPU (forced)"

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # Test MPS actually works
        try:
            x = torch.tensor([1.0], device="mps")
            _ = x * 2
            return "mps", "Apple MPS"
        except Exception as e:
            print(f"MPS available but failed test: {e}")
            print("Falling back to CPU...")

    return "cpu", "CPU"


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Alpamayo-R1 inference on driving scenes (Mac/MPS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                         # 5 random scenes
  %(prog)s -n 10                   # 10 random scenes
  %(prog)s --clip-id abc123        # Specific clip
  %(prog)s --clip-ids clips.txt    # Clips from file (one per line)
  %(prog)s --list-clips            # Show available clips
  %(prog)s --cpu                   # Force CPU backend
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
                        help="Output JSON file (default: auto-generated)")
    parser.add_argument("--list-clips", action="store_true",
                        help="List available clip IDs and exit")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU backend (slower but more compatible)")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32",
                        help="Model dtype (default: float32 for MPS compatibility)")
    parser.add_argument("--low-memory", action="store_true",
                        help="Enable low memory mode (offload to CPU when possible)")
    return parser.parse_args()


# ============================================================================
# INFERENCE
# ============================================================================

def run_inference(model, processor, clip_id: str, device: str, t0_us: int = 5_000_000) -> dict:
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
    model_inputs = helper.to_device(model_inputs, device)

    # Run inference
    infer_start = time.time()

    # MPS doesn't support bfloat16 autocast well, use no autocast or float32
    if device == "mps":
        # Run without autocast on MPS
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=256,
            return_extra=True,
        )
    else:
        # CPU - also run without autocast
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

def get_clip_ids() -> list[str]:
    """Load clip IDs from cache or fetch from HuggingFace dataset."""
    if CLIP_IDS_FILE.exists():
        df = pd.read_parquet(CLIP_IDS_FILE)
        return df["clip_id"].tolist()

    # Fetch from HuggingFace dataset
    print("Fetching clip IDs from HuggingFace dataset...")
    from datasets import load_dataset
    ds = load_dataset("nvidia/PhysicalAI-Autonomous-Vehicles", split="train", streaming=True)
    clip_ids = []
    for i, sample in enumerate(ds):
        clip_ids.append(sample["clip_id"])
        if (i + 1) % 100 == 0:
            print(f"  Found {i + 1} clips...")
    print(f"Total: {len(clip_ids)} clips")

    # Cache for future runs
    CLIP_IDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"clip_id": clip_ids}).to_parquet(CLIP_IDS_FILE)
    print(f"Cached to {CLIP_IDS_FILE}")

    return clip_ids


def main():
    args = parse_args()

    # Lazy imports (after arg parsing for --list-clips)
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1 import helper

    # Load all clip IDs
    all_clip_ids = get_clip_ids()

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
    print("ALPAMAYO-R1 LOCAL INFERENCE TEST (Mac/MPS)")
    print("=" * 80)

    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    # Detect device
    device, device_name = get_device(args.cpu)
    print(f"Device: {device_name}")

    if device == "mps":
        torch.mps.manual_seed(args.seed)

    # Determine dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    model_dtype = dtype_map[args.dtype]
    print(f"Model dtype: {args.dtype}")

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

    # Load model with eager attention (no flash-attn on Mac)
    print(f"\nLoading model: {MODEL_ID}...")
    print("  Using 'eager' attention (flash-attn not available on Mac)")
    model_load_start = time.time()

    # Load with eager attention to avoid flash-attn requirement
    model = AlpamayoR1.from_pretrained(
        MODEL_ID,
        dtype=model_dtype,
        attn_implementation="eager",  # Use eager attention (no flash-attn)
    ).to(device)

    processor = helper.get_processor(model.tokenizer)
    model_load_time = time.time() - model_load_start
    print(f"Model loaded in {model_load_time:.1f}s")

    # Run inference on each clip
    results = []
    print("\n" + "-" * 80)

    for i, clip_id in enumerate(selected_clips):
        print(f"\n[{i+1}/{len(selected_clips)}] Clip: {clip_id[:16]}...")

        try:
            result = run_inference(model, processor, clip_id, device, args.t0)
            results.append(result)

            print(f"    minADE: {result['min_ade']:.3f} m")
            coc_preview = result['coc_reasoning'][:100] + "..." if len(result['coc_reasoning']) > 100 else result['coc_reasoning']
            print(f"    CoC: {coc_preview}")
            print(f"    Time: load={result['load_time_s']:.1f}s, infer={result['inference_time_s']:.1f}s")

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
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
            "device": device_name,
            "dtype": args.dtype,
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
        output_file = OUTPUT_DIR / f"inference_local_{timestamp}.json"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
