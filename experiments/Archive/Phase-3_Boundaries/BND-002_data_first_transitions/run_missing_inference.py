#!/usr/bin/env python3
"""
BND-002: Run Alpamayo inference on missing CLS scenes.

Extracts the 47 clip_ids that have classifications but no trajectories,
and runs Alpamayo inference on them.

Usage:
    # Just create the clip_ids file (no inference)
    python run_missing_inference.py --prepare-only

    # Run inference locally (Mac/MPS)
    python run_missing_inference.py --local

    # Run inference on workstation (CUDA)
    python run_missing_inference.py --workstation
"""

import json
import sys
import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path("/")
DATA_GAPS_PATH = PROJECT_ROOT / "data/BND-002/data_gaps.json"
MISSING_CLIPS_PATH = PROJECT_ROOT / "data/BND-002/missing_clips.txt"
OUTPUT_DIR = PROJECT_ROOT / "data/alpamayo_outputs"


def extract_missing_clip_ids():
    """Extract the 47 clip_ids that need trajectory inference."""
    print("=== Extracting Missing Clip IDs ===")

    with open(DATA_GAPS_PATH) as f:
        data = json.load(f)

    # Get the clip_ids from the first recommended action (run Alpamayo on CLS scenes)
    missing_clips = data["recommended_actions"][0]["scenes"]

    print(f"  Found {len(missing_clips)} scenes needing trajectory inference")

    # Write to file for inference script
    with open(MISSING_CLIPS_PATH, "w") as f:
        for clip_id in missing_clips:
            f.write(f"{clip_id}\n")

    print(f"  Saved to: {MISSING_CLIPS_PATH}")

    return missing_clips


def run_inference_local():
    """Run inference using Mac/MPS backend."""
    print("\n=== Running Local Inference (MPS) ===")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"gap_fill_local_{timestamp}.json"

    script_path = PROJECT_ROOT / "infrastructure/local/basic_inference_test.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--clip-ids", str(MISSING_CLIPS_PATH),
        "-o", str(output_path),
    ]

    print(f"  Command: {' '.join(cmd)}")
    print(f"  Output: {output_path}")
    print("\n  Starting inference (this will take several hours)...\n")

    subprocess.run(cmd, cwd=PROJECT_ROOT)

    return output_path


def run_inference_workstation():
    """Run inference using CUDA backend on workstation."""
    print("\n=== Running Workstation Inference (CUDA) ===")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"gap_fill_workstation_{timestamp}.json"

    script_path = PROJECT_ROOT / "infrastructure/workstation/basic_inference_test.py"

    cmd = [
        "python",  # Use system python on workstation
        str(script_path),
        "--clip-ids", str(MISSING_CLIPS_PATH),
        "-o", str(output_path),
    ]

    print(f"  Command: {' '.join(cmd)}")
    print(f"  Output: {output_path}")
    print("\n  Note: Run this command on the workstation with CUDA GPU.\n")

    # Print the command for manual execution on workstation
    print("=" * 60)
    print("COPY THIS COMMAND TO RUN ON WORKSTATION:")
    print("=" * 60)
    print(f"cd {PROJECT_ROOT}")
    print(f"python infrastructure/workstation/basic_inference_test.py \\")
    print(f"  --clip-ids data/BND-002/missing_clips.txt \\")
    print(f"  -o {output_path}")
    print("=" * 60)

    return output_path


def merge_results(new_output_path: Path):
    """Merge new inference results with existing data."""
    print("\n=== Merging Results ===")

    # Load existing trajectories
    existing_path = OUTPUT_DIR / "workstation/inference_20260120_154727.json"
    if not existing_path.exists():
        existing_path = OUTPUT_DIR / "inference_20260120_154727.json"

    with open(existing_path) as f:
        existing_data = json.load(f)

    # Load new results
    with open(new_output_path) as f:
        new_data = json.load(f)

    # Merge
    existing_ids = {r["clip_id"] for r in existing_data["results"]}

    added = 0
    for result in new_data["results"]:
        if result["clip_id"] not in existing_ids:
            existing_data["results"].append(result)
            added += 1

    # Update metadata
    existing_data["metadata"]["num_samples"] = len(existing_data["results"])
    existing_data["metadata"]["merged_at"] = datetime.now().isoformat()

    # Save merged file
    merged_path = OUTPUT_DIR / "merged_inference.json"
    with open(merged_path, "w") as f:
        json.dump(existing_data, f, indent=2)

    print(f"  Added {added} new results")
    print(f"  Total results: {len(existing_data['results'])}")
    print(f"  Saved to: {merged_path}")

    return merged_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Alpamayo inference on missing scenes")
    parser.add_argument("--prepare-only", action="store_true", help="Only create clip_ids file")
    parser.add_argument("--local", action="store_true", help="Run inference locally (MPS)")
    parser.add_argument("--workstation", action="store_true", help="Print workstation command")
    parser.add_argument("--merge", type=str, help="Merge results from given file")

    args = parser.parse_args()

    # Always extract clip_ids first
    missing_clips = extract_missing_clip_ids()

    if args.prepare_only:
        print("\n✓ Clip IDs file created. Ready for inference.")
        return

    if args.merge:
        merge_results(Path(args.merge))
        return

    if args.local:
        output_path = run_inference_local()
        print(f"\n✓ Inference complete. Merge with:")
        print(f"  python {__file__} --merge {output_path}")
    elif args.workstation:
        run_inference_workstation()
    else:
        # Default: just show info
        print("\n" + "=" * 60)
        print("NEXT STEPS:")
        print("=" * 60)
        print(f"1. Missing clips file: {MISSING_CLIPS_PATH}")
        print(f"2. Run inference:")
        print(f"   - Local (Mac):      python {__file__} --local")
        print(f"   - Workstation:      python {__file__} --workstation")
        print(f"3. After inference:    python {__file__} --merge <output.json>")


if __name__ == "__main__":
    main()
