"""
Data Loader for EXP-005

Handles two types of data:
1. ANCHORS (100 scenes): Labeled scenes from data/EXP-001/
2. SUPERSET (N scenes): Unlabeled scenes from physical_ai_av

The superset provides dense embedding space for better clustering,
while anchors provide ground truth for alignment metrics.
"""

import json
import random
from base64 import b64decode
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import physical_ai_av
import torch
from PIL import Image
from tqdm import tqdm


# =============================================================================
# CONFIG
# =============================================================================

DATA_DIR = Path(__file__).parents[2] / "data"
ANCHOR_DIR = DATA_DIR / "EXP-001"
CACHE_DIR = DATA_DIR / "EXP-005" / "image_cache"

# Camera order for composite (matches EXP-001 generation)
CAMERA_FEATURES = [
    "CAMERA_CROSS_LEFT_120FOV",    # Index 0 → top-left
    "CAMERA_FRONT_WIDE_120FOV",    # Index 1 → top-right
    "CAMERA_CROSS_RIGHT_120FOV",   # Index 2 → bottom-left
    "CAMERA_FRONT_TELE_30FOV",     # Index 3 → bottom-right
]

# Default timestamp (5 seconds into clip)
DEFAULT_T0_US = 5_000_000


# =============================================================================
# ANCHOR DATA
# =============================================================================

def load_anchor_classifications() -> dict[str, dict]:
    """Load the 100 labeled anchor scenes."""
    with open(ANCHOR_DIR / "scene_classifications.json") as f:
        data = json.load(f)

    classifications_list = data.get("classifications", [])

    result = {}
    for item in classifications_list:
        clip_id = item["clip_id"]
        result[clip_id] = item["classification"]

    return result


def get_anchor_image_paths() -> dict[str, Path]:
    """Get paths to pre-generated anchor composite images."""
    image_dir = ANCHOR_DIR / "images"
    return {p.stem: p for p in image_dir.glob("*.jpg")}


def get_anchor_clip_ids() -> set[str]:
    """Get set of anchor clip IDs (to exclude from superset)."""
    classifications = load_anchor_classifications()
    return set(classifications.keys())


# =============================================================================
# SUPERSET DATA
# =============================================================================

def get_available_clip_ids(
    split: str | None = None,
    exclude_anchors: bool = True,
) -> list[str]:
    """
    Get list of available clip IDs from physical_ai_av.

    Args:
        split: Optional filter by split ('train', 'val', 'test')
        exclude_anchors: Whether to exclude the 100 anchor clips

    Returns:
        List of clip IDs
    """
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    clip_index = avdi.clip_index

    # Filter valid clips
    valid = clip_index[clip_index["clip_is_valid"] == True]

    # Filter by split if specified
    if split:
        valid = valid[valid["split"] == split]

    clip_ids = valid.index.tolist()

    # Exclude anchors
    if exclude_anchors:
        anchor_ids = get_anchor_clip_ids()
        clip_ids = [c for c in clip_ids if c not in anchor_ids]

    return clip_ids


def sample_superset_clips(
    n: int,
    split: str | None = "train",
    seed: int = 42,
) -> list[str]:
    """
    Sample N clips for the superset.

    Args:
        n: Number of clips to sample
        split: Split to sample from (default: train)
        seed: Random seed for reproducibility

    Returns:
        List of sampled clip IDs
    """
    available = get_available_clip_ids(split=split, exclude_anchors=True)

    if n > len(available):
        print(f"Warning: Requested {n} clips but only {len(available)} available. Using all.")
        return available

    random.seed(seed)
    return random.sample(available, n)


# =============================================================================
# COMPOSITE IMAGE GENERATION
# =============================================================================

def create_composite_image(image_frames: torch.Tensor) -> Image.Image:
    """
    Create 2x2 composite from 4 camera views.

    Args:
        image_frames: Tensor of shape (4, num_frames, 3, H, W)

    Returns:
        PIL Image of composite
    """
    images = []
    for cam_idx in range(4):
        # Take the last frame (most recent)
        img_tensor = image_frames[cam_idx, -1]  # (3, H, W)
        img_np = img_tensor.permute(1, 2, 0).numpy().astype("uint8")
        images.append(Image.fromarray(img_np))

    w, h = images[0].size
    composite = Image.new("RGB", (w * 2, h * 2))
    positions = [(0, 0), (w, 0), (0, h), (w, h)]

    for img, pos in zip(images, positions):
        composite.paste(img, pos)

    # Resize to max 1920x1080
    composite.thumbnail((1920, 1080), Image.Resampling.LANCZOS)

    return composite


def load_clip_composite(
    clip_id: str,
    t0_us: int = DEFAULT_T0_US,
    avdi: physical_ai_av.PhysicalAIAVDatasetInterface | None = None,
    cache_dir: Path | None = None,
) -> Image.Image:
    """
    Load or generate composite image for a clip.

    Args:
        clip_id: Clip ID
        t0_us: Timestamp in microseconds
        avdi: Optional pre-initialized dataset interface
        cache_dir: Optional cache directory for generated composites

    Returns:
        PIL Image of composite
    """
    # Check cache first
    if cache_dir:
        cache_path = cache_dir / f"{clip_id}.jpg"
        if cache_path.exists():
            return Image.open(cache_path)

    # Initialize interface if needed
    if avdi is None:
        avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    # Get camera features
    camera_features = [
        getattr(avdi.features.CAMERA, name) for name in CAMERA_FEATURES
    ]

    # Compute image timestamps (4 frames ending at t0)
    num_frames = 4
    time_step_us = 100_000  # 0.1s
    image_timestamps = np.array([
        t0_us - (num_frames - 1 - i) * time_step_us
        for i in range(num_frames)
    ], dtype=np.int64)

    # Load frames from each camera
    frames_list = []
    for cam_feature in camera_features:
        camera = avdi.get_clip_feature(clip_id, cam_feature, maybe_stream=True)
        frames, _ = camera.decode_images_from_timestamps(image_timestamps)
        # frames: (num_frames, H, W, 3) -> (num_frames, 3, H, W)
        frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)
        frames_list.append(frames_tensor)

    # Stack: (4, num_frames, 3, H, W)
    image_frames = torch.stack(frames_list, dim=0)

    # Create composite
    composite = create_composite_image(image_frames)

    # Cache if directory provided
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        composite.save(cache_path, "JPEG", quality=85)

    return composite


# =============================================================================
# COMBINED DATA LOADING
# =============================================================================

def _load_single_clip(
    clip_id: str,
    cache_dir: Path | None,
) -> tuple[str, Image.Image | None, str | None]:
    """Load a single clip composite. Thread-safe (creates own avdi instance)."""
    try:
        img = load_clip_composite(clip_id, avdi=None, cache_dir=cache_dir)
        return (clip_id, img, None)
    except Exception as e:
        return (clip_id, None, str(e))


def load_experiment_data(
    superset_size: int = 0,
    cache_dir: Path | None = CACHE_DIR,
    show_progress: bool = True,
    num_workers: int = 5,
) -> tuple[list[str], list[Path | Image.Image], dict[str, dict]]:
    """
    Load combined anchor + superset data.

    Args:
        superset_size: Number of additional unlabeled scenes (0 = anchors only)
        cache_dir: Cache directory for superset images
        show_progress: Show progress bar
        num_workers: Number of parallel workers for superset loading

    Returns:
        (scene_ids, images, classifications)
        - scene_ids: List of all scene IDs
        - images: List of image paths (anchors) or PIL Images (superset)
        - classifications: Dict of {scene_id: {key: value}} (anchors only)
    """
    # Load anchors
    classifications = load_anchor_classifications()
    anchor_paths = get_anchor_image_paths()

    scene_ids = []
    images = []

    # Add anchors
    for clip_id in classifications:
        if clip_id in anchor_paths:
            scene_ids.append(clip_id)
            images.append(anchor_paths[clip_id])

    print(f"Loaded {len(scene_ids)} anchor scenes")

    # Add superset if requested
    if superset_size > 0:
        superset_clips = sample_superset_clips(superset_size)
        print(f"Sampling {len(superset_clips)} superset scenes...")

        # Parallel loading with thread pool
        superset_results = {}
        failed = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_load_single_clip, clip_id, cache_dir): clip_id
                for clip_id in superset_clips
            }

            pbar = tqdm(total=len(superset_clips), desc=f"Loading superset ({num_workers} workers)") if show_progress else None

            for future in as_completed(futures):
                clip_id, img, error = future.result()
                if error:
                    failed.append((clip_id, error))
                else:
                    superset_results[clip_id] = img

                if pbar:
                    pbar.update(1)

            if pbar:
                pbar.close()

        # Add results in original order (preserves reproducibility)
        for clip_id in superset_clips:
            if clip_id in superset_results:
                scene_ids.append(clip_id)
                images.append(superset_results[clip_id])

        if failed:
            print(f"Warning: Failed to load {len(failed)} clips")
            for clip_id, error in failed[:5]:
                print(f"  {clip_id}: {error}")
            if len(failed) > 5:
                print(f"  ... and {len(failed) - 5} more")

    print(f"Total scenes: {len(scene_ids)} ({len(classifications)} anchors + {len(scene_ids) - len(classifications)} superset)")

    return scene_ids, images, classifications


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data loader utilities")
    parser.add_argument("--check", action="store_true", help="Check available clips")
    parser.add_argument("--sample", type=int, help="Sample N clips and show IDs")
    args = parser.parse_args()

    if args.check:
        print("Checking available clips...")
        for split in ["train", "val", "test"]:
            clips = get_available_clip_ids(split=split)
            print(f"  {split}: {len(clips)} clips")

        anchors = get_anchor_clip_ids()
        print(f"  anchors: {len(anchors)} clips (excluded)")

    if args.sample:
        clips = sample_superset_clips(args.sample)
        print(f"Sampled {len(clips)} clips:")
        for c in clips[:10]:
            print(f"  {c}")
        if len(clips) > 10:
            print(f"  ... and {len(clips) - 10} more")
