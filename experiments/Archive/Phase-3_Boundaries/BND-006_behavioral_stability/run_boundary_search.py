#!/usr/bin/env python3
"""
BND-006: Run Pareto Boundary Search on Anchor Clips

3-objective NSGA-II optimization for behavioral boundaries in
Alpamayo-R1 under visual manipulations (brightness, fog, contrast).

Objectives:
  1. Minimize LPIPS  (perceptual distance to original)
  2. Maximize ADE    (trajectory deviation from GT)
  3. Maximize Var(ADE) (output variance across k stochastic runs)

Loads diverse anchor clips (by weather), runs BoundaryFinder with k-sample
inference on each, and saves full archives (parquet) + Pareto summary (JSON).

Usage:
    python run_boundary_search.py                      # 4 diverse anchors
    python run_boundary_search.py --clip-ids A B C     # specific clips
    python run_boundary_search.py --k-samples 5        # fewer stochastic runs
    python run_boundary_search.py --resume             # skip completed clips
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Project root
PROJECT_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools" / "smoo"))
sys.path.insert(0, str(PROJECT_ROOT / "tools" / "alpamayo" / "src"))

from archive.pipeline.lib.io import get_git_hash

from boundary_finder import (
    AlpamayoSUT,
    BoundaryFinder,
    BoundaryResult,
    DEFAULT_MANIPULATIONS,
    LPIPSCriterion,
    OllamaSUT,
    OutputVarianceCriterion,
    PixelManipulator,
    TrajectoryDeviationCriterion,
    _detect_device,
)
from pymoo.algorithms.moo.nsga2 import NSGA2
from src.objectives import CriterionCollection
from src.optimizer import PymooOptimizer

# =============================================================================
# CONSTANTS
# =============================================================================

MODEL_ID = "nvidia/Alpamayo-R1-10B"
T0_US = 5_000_000

DATA_DIR = PROJECT_ROOT / "data"
ANCHOR_FILE = DATA_DIR / "CLS-001" / "scene_classifications.json"
OUTPUT_DIR = DATA_DIR / "BND-006" / "boundary_search"

# Priority order for diverse clip selection by weather
WEATHER_PRIORITY = ["clear", "cloudy", "foggy", "rainy"]


# =============================================================================
# ENVIRONMENT SETUP (reused from BND-004)
# =============================================================================

def setup_environment():
    """Setup environment for inference (scipy patch + HF cache)."""
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

    token = os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        print("HuggingFace: Authenticated")


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(device: str, dtype: torch.dtype):
    """Load Alpamayo-R1 model and processor."""
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1.helper import get_processor

    print(f"Loading model: {MODEL_ID} on {device} ({dtype})")
    if device == "cuda":
        model = AlpamayoR1.from_pretrained(MODEL_ID, torch_dtype=dtype, device_map="cuda")
    else:
        model = AlpamayoR1.from_pretrained(MODEL_ID, dtype=dtype).to(device)
    model.eval()

    processor = get_processor(model.tokenizer)
    print("Model loaded")
    return model, processor


# =============================================================================
# CLIP SELECTION
# =============================================================================

def load_anchor_labels() -> dict[str, dict[str, str]]:
    """Load anchor labels from CLS-001 (same as BND-004)."""
    with open(ANCHOR_FILE) as f:
        data = json.load(f)

    labels = {}
    for item in data.get("classifications", []):
        clip_id = item["clip_id"]
        cls = item.get("classification", {})

        simple_labels = {}
        for key, val in cls.items():
            if isinstance(val, dict):
                for k, v in val.items():
                    if k == key or k not in ["reasoning", "confidence"]:
                        if not isinstance(v, (dict, list)):
                            simple_labels[key] = v
                            break
            elif isinstance(val, (str, bool, int, float)):
                simple_labels[key] = val

        labels[clip_id] = simple_labels

    return labels


def select_clips(n: int = 4, clip_ids: list[str] | None = None) -> list[str]:
    """Pick n diverse anchors from CLS-001 by weather condition.

    Prioritizes one clip from each weather category (clear/cloudy/foggy/rainy)
    to test boundary search across different visual conditions.
    """
    if clip_ids:
        return list(clip_ids)

    labels = load_anchor_labels()

    # Group clips by weather
    by_weather: dict[str, list[str]] = {}
    for clip_id, lab in labels.items():
        weather = str(lab.get("weather", "unknown"))
        by_weather.setdefault(weather, []).append(clip_id)

    print(f"Clips by weather: {
        {k: len(v) for k, v in sorted(by_weather.items())}
    }")

    # Pick one from each priority weather category
    selected: list[str] = []
    for weather in WEATHER_PRIORITY:
        if weather in by_weather and len(selected) < n:
            selected.append(by_weather[weather][0])

    return selected[:n]


# =============================================================================
# BOUNDARY SEARCH
# =============================================================================

def collect_warm_seeds(output_dir: Path, n_manipulations: int) -> np.ndarray | None:
    """Collect warm-start seeds from prior boundary search archives.

    Samples manipulation intensity vectors from existing archives.
    """
    archives = sorted(output_dir.glob("*/archive.parquet"))
    if not archives:
        return None

    all_seeds: list[np.ndarray] = []
    for archive_path in archives:
        df = pd.read_parquet(
            archive_path, columns=["brightness", "fog", "contrast"],
        )
        all_seeds.append(df[["brightness", "fog", "contrast"]].to_numpy())

    if not all_seeds:
        return None

    seeds = np.unique(np.vstack(all_seeds), axis=0)
    if len(seeds) > 20:
        indices = np.random.choice(len(seeds), 20, replace=False)
        seeds = seeds[indices]
    return seeds


def create_optimizer(
    pop_size: int, warm_seeds: np.ndarray | None = None,
) -> PymooOptimizer:
    """Create a fresh NSGA-II optimizer with seeded initial population.

    Population layout: [zeros, ones, eye, warm_seeds, random_fill].
    """
    n = len(DEFAULT_MANIPULATIONS)
    n_fixed = 2 + n  # zeros + ones + eye
    parts: list[np.ndarray] = [np.zeros(n), np.ones(n), np.eye(n)]

    if warm_seeds is not None and len(warm_seeds) > 0:
        n_warm = min(len(warm_seeds), pop_size - n_fixed)
        if n_warm > 0:
            parts.append(warm_seeds[:n_warm])
            n_fixed += n_warm

    n_random = pop_size - n_fixed
    if n_random > 0:
        parts.append(np.random.uniform(0, 1, (n_random, n)))

    seeds = np.vstack(parts)
    return PymooOptimizer(
        bounds=(0, 1),
        algorithm=NSGA2,
        algo_params={"pop_size": pop_size, "sampling": seeds},
        num_objectives=3,
        solution_shape=(n,),
    )


# Camera names in index order (matching load_physical_aiavdataset sort)
CAMERA_NAMES = ["left_120", "front_wide", "right_120", "front_tele"]
N_CAMERAS = 4


def make_composite(frames: torch.Tensor, label: str = "") -> Image.Image:
    """Create a composite image from (N_cameras, N_timesteps, C, H, W) frames.

    Layout: cameras as rows, timesteps as columns.
    """
    n_cams, n_times = frames.shape[0], frames.shape[1]
    h, w = frames.shape[3], frames.shape[4]

    composite = Image.new("RGB", (n_times * w, n_cams * h))
    for cam in range(n_cams):
        for t in range(n_times):
            img = Image.fromarray(frames[cam, t].permute(1, 2, 0).cpu().numpy())
            composite.paste(img, (t * w, cam * h))

    return composite


def annotate_composite(composite: Image.Image, n_cams: int, n_times: int) -> Image.Image:
    """Add camera row labels and timestep column headers to a composite image."""
    w_cell = composite.width // n_times
    h_cell = composite.height // n_cams

    font_size = max(24, h_cell // 20)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except OSError:
        font = ImageFont.load_default()

    margin_left = font_size * 7
    margin_top = int(font_size * 2.5)
    annotated = Image.new(
        "RGB",
        (composite.width + margin_left, composite.height + margin_top),
        (255, 255, 255),
    )
    annotated.paste(composite, (margin_left, margin_top))
    draw = ImageDraw.Draw(annotated)

    cam_labels = CAMERA_NAMES[:n_cams]
    for i, name in enumerate(cam_labels):
        y = margin_top + i * h_cell + h_cell // 2
        draw.text((margin_left // 2, y), name, fill=(0, 0, 0), font=font, anchor="mm")

    for t in range(n_times):
        x = margin_left + t * w_cell + w_cell // 2
        draw.text((x, margin_top // 2), f"t{t}", fill=(0, 0, 0), font=font, anchor="mm")

    return annotated


def save_boundary_images(
    scene_data: dict,
    result: BoundaryResult,
    manipulator: PixelManipulator,
    clip_dir: Path,
) -> None:
    """Save composites for original scene + Pareto front solutions."""
    frames = scene_data["image_frames"]  # (N_cameras, N_timesteps, C, H, W)
    n_cams, n_times = frames.shape[0], frames.shape[1]

    # --- Per-camera raw frames ---
    raw_dir = clip_dir / "frames"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for cam in range(n_cams):
        cam_name = CAMERA_NAMES[cam] if cam < len(CAMERA_NAMES) else f"cam{cam}"
        for t in range(n_times):
            img = Image.fromarray(frames[cam, t].permute(1, 2, 0).cpu().numpy())
            img.save(raw_dir / f"{cam_name}_t{t}.png")

    # --- Original composites ---
    comp_orig = make_composite(frames)
    comp_orig.save(clip_dir / "composite_original.png")
    annotate_composite(comp_orig, n_cams, n_times).save(
        clip_dir / "composite_original_annotated.png"
    )

    # --- Top Pareto front composites (by highest mean_ade) ---
    flat = frames.flatten(0, 1)
    shape = frames.shape
    pareto_sorted = sorted(result.pareto_front, key=lambda p: p["mean_ade"], reverse=True)

    for rank, point in enumerate(pareto_sorted[:3]):
        intensities = {k: point[k] for k in ["brightness", "fog", "contrast"]}
        mod = manipulator.apply(flat, intensities).reshape(shape)
        comp = make_composite(mod)
        comp.save(clip_dir / f"composite_pareto_{rank}.png")
        annotate_composite(comp, n_cams, n_times).save(
            clip_dir / f"composite_pareto_{rank}_annotated.png"
        )

    n_saved = min(3, len(pareto_sorted))
    print(f"  Saved composites: original + {n_saved} Pareto solutions (plain + annotated)")


def save_archive(result: BoundaryResult, output_file: Path) -> None:
    """Save full search archive as parquet.

    Each row is one unique individual evaluated. Columns:
    - clip_id:      scene identifier
    - query_index:  monotonic evaluation order (0..N-1)
    - generation:   NSGA-II generation this individual belonged to
    - brightness/fog/contrast: manipulation intensities in [0, 1]
    - lpips:        perceptual distance to original (lower = closer)
    - mean_ade:     mean trajectory ADE across k runs (higher = more deviation)
    - ade_variance: variance of ADE across k runs (higher = more unstable)
    """
    n = len(result.archive_intensities)
    df = pd.DataFrame({
        "clip_id": result.clip_id,
        "query_index": range(n),
        "generation": result.archive_generations,
        "brightness": [float(x[0]) for x in result.archive_intensities],
        "fog": [float(x[1]) for x in result.archive_intensities],
        "contrast": [float(x[2]) for x in result.archive_intensities],
        "lpips": result.archive_lpips,
        "mean_ade": result.archive_mean_ade,
        "ade_variance": result.archive_ade_var,
        "metadata": [json.dumps(m) for m in result.archive_metadata],
    })
    df.to_parquet(output_file, index=False)
    print(f"  Saved archive: {output_file} ({len(df)} rows)")


def run_boundary_search(
    sut,
    clip_ids: list[str],
    output_dir: Path,
    *,
    pop_size: int = 30,
    max_generations: int = 50,
    stagnation_limit: int = 10,
    k_samples: int = 10,
    resume: bool = False,
    warm_seeds: np.ndarray | None = None,
) -> list[BoundaryResult]:
    """Run BoundaryFinder.test() for each clip, save archives."""
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

    manipulator = PixelManipulator(DEFAULT_MANIPULATIONS)
    objectives = CriterionCollection(
        LPIPSCriterion(),
        TrajectoryDeviationCriterion(),
        OutputVarianceCriterion(),
    )
    results: list[BoundaryResult] = []

    clip_bar = tqdm(clip_ids, desc="Clips", unit="clip")
    for clip_id in clip_bar:
        clip_bar.set_description(f"Clip {clip_id}")
        clip_dir = output_dir / clip_id
        archive_file = clip_dir / "archive.parquet"

        if resume and archive_file.exists():
            tqdm.write(f"  Skipping {clip_id} (archive exists)")
            continue

        clip_dir.mkdir(parents=True, exist_ok=True)

        tqdm.write(f"  Loading scene {clip_id} (t0={T0_US})...")
        data = load_physical_aiavdataset(clip_id, t0_us=T0_US)

        optimizer = create_optimizer(pop_size, warm_seeds=warm_seeds)
        finder = BoundaryFinder(
            sut=sut,
            manipulator=manipulator,
            optimizer=optimizer,
            objectives=objectives,
            k_samples=k_samples,
        )

        result = finder.test(
            clip_id, data,
            max_generations=max_generations,
            stagnation_limit=stagnation_limit,
        )
        results.append(result)

        # Save archive + images
        save_archive(result, archive_file)
        save_boundary_images(data, result, manipulator, clip_dir)

        n_pareto = len(result.pareto_front)
        tqdm.write(f"  Pareto front: {n_pareto} solutions  "
                    f"queries={result.query_count}")
        clip_bar.set_postfix(done=len(results), pareto=n_pareto)

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="BND-006: Run Pareto boundary search on anchor clips",
    )
    parser.add_argument(
        "--clip-ids", nargs="+", default=None,
        help="Specific clip IDs to search (default: diverse anchors)",
    )
    parser.add_argument(
        "--n-clips", type=int, default=4,
        help="Number of diverse clips to select (ignored if --clip-ids given)",
    )
    parser.add_argument(
        "--pop-size", type=int, default=20,
        help="NSGA-II population size",
    )
    parser.add_argument(
        "--max-generations", type=int, default=50,
        help="Maximum generations per clip",
    )
    parser.add_argument(
        "--stagnation-limit", type=int, default=10,
        help="Stop after N generations without Pareto front growth",
    )
    parser.add_argument(
        "--k-samples", type=int, default=10,
        help="Number of stochastic forward passes per individual (default: 10)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip clips with existing archive files",
    )
    parser.add_argument(
        "--sut", choices=["alpamayo", "ollama"], default="alpamayo",
        help="SUT to use (default: alpamayo)",
    )
    parser.add_argument(
        "--ollama-model", type=str, default="qwen3-vl:8b",
        help="Ollama model name (default: qwen3-vl:8b)",
    )
    parser.add_argument(
        "--no-warm-start", action="store_true",
        help="Disable warm-start seeding from prior archives",
    )

    args = parser.parse_args()

    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BND-006: PARETO BOUNDARY SEARCH")
    print("=" * 60)

    device, dtype = _detect_device()
    print(f"Device: {device} ({dtype})")
    print(f"SUT: {args.sut}" + (f" ({args.ollama_model})" if args.sut == "ollama" else ""))

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # Select clips
    clip_ids = select_clips(n=args.n_clips, clip_ids=args.clip_ids)
    print(f"Clips: {clip_ids}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Params: pop_size={args.pop_size}, max_gen={args.max_generations}, "
          f"stagnation={args.stagnation_limit}, k_samples={args.k_samples}")

    # Create SUT
    if args.sut == "ollama":
        sut = OllamaSUT(model=args.ollama_model)
    else:
        setup_environment()
        model, processor = load_model(device, dtype)
        sut = AlpamayoSUT(model, processor, device=device, dtype=dtype)

    # Warm-start seeds
    warm_seeds = None
    if not args.no_warm_start:
        warm_seeds = collect_warm_seeds(OUTPUT_DIR, len(DEFAULT_MANIPULATIONS))
        if warm_seeds is not None:
            print(f"Warm-start: {len(warm_seeds)} seeds from prior archives")

    # Run search
    print("\n" + "=" * 60)
    print("RUNNING PARETO BOUNDARY SEARCH")
    print("=" * 60)

    results = run_boundary_search(
        sut, clip_ids, OUTPUT_DIR,
        pop_size=args.pop_size,
        max_generations=args.max_generations,
        stagnation_limit=args.stagnation_limit,
        k_samples=args.k_samples,
        resume=args.resume,
        warm_seeds=warm_seeds,
    )

    # Save summary JSON
    summary = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "git_hash": get_git_hash(),
            "params": {
                "pop_size": args.pop_size,
                "max_generations": args.max_generations,
                "stagnation_limit": args.stagnation_limit,
                "k_samples": args.k_samples,
                "device": device,
                "sut": args.sut,
            },
            "clip_ids": clip_ids,
        },
        "results": [],
    }

    for result in results:
        entry = {
            "clip_id": result.clip_id,
            "archive_file": f"{result.clip_id}/archive.parquet",
            "generations": result.generations,
            "query_count": result.query_count,
            "n_archive": len(result.archive_intensities),
            "n_pareto": len(result.pareto_front),
            "pareto_front": result.pareto_front,
        }
        summary["results"].append(entry)

    summary_file = OUTPUT_DIR / "pareto_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {summary_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for result in results:
        print(f"  {result.clip_id}: {len(result.pareto_front)} Pareto solutions "
              f"(queries={result.query_count}, gens={result.generations})")
        if result.pareto_front:
            best_ade = max(p["mean_ade"] for p in result.pareto_front)
            best_lpips = min(p["lpips"] for p in result.pareto_front)
            print(f"    best ADE={best_ade:.4f}  best LPIPS={best_lpips:.4f}")

    total_pareto = sum(len(r.pareto_front) for r in results)
    print(f"\nTotal Pareto solutions: {total_pareto} across {len(results)} clips")
    print(f"Output: {OUTPUT_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
