"""
Composite Image Generation

Creates 2x2 composite images from 4 camera views.
Adapted from experiments/Phase-2_Embeddings/EMB-001_latent_navigation/data_loader.py
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Lazy import to avoid loading physical_ai_av until needed
_avdi = None


# Camera order for composite (matches CLS-001 generation)
CAMERA_FEATURES = [
    "CAMERA_CROSS_LEFT_120FOV",    # Index 0: top-left
    "CAMERA_FRONT_WIDE_120FOV",    # Index 1: top-right
    "CAMERA_CROSS_RIGHT_120FOV",   # Index 2: bottom-left
    "CAMERA_FRONT_TELE_30FOV",     # Index 3: bottom-right
]

# Default timestamp (5 seconds into clip)
DEFAULT_T0_US = 5_000_000


def _get_avdi():
    """Get or create PhysicalAIAVDatasetInterface (singleton)."""
    global _avdi
    if _avdi is None:
        import physical_ai_av
        _avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    return _avdi


def _create_composite_from_frames(image_frames: torch.Tensor) -> Image.Image:
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


def create_composite(
    clip_id: str,
    t0_us: int = DEFAULT_T0_US,
    cache_dir: Path | str | None = None,
) -> Image.Image:
    """
    Create or load cached composite image for a clip.

    Args:
        clip_id: Clip ID from PhysicalAI-AV dataset
        t0_us: Timestamp in microseconds (default: 5 seconds)
        cache_dir: Optional cache directory for generated composites

    Returns:
        PIL Image of 2x2 composite
    """
    # Check cache first
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_path = cache_dir / f"{clip_id}.jpg"
        if cache_path.exists():
            return Image.open(cache_path)

    # Get dataset interface
    import physical_ai_av
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
    composite = _create_composite_from_frames(image_frames)

    # Cache if directory provided
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{clip_id}.jpg"
        composite.save(cache_path, "JPEG", quality=85)

    return composite


def _load_single_composite(
    clip_id: str,
    t0_us: int,
    cache_dir: Path | None,
) -> tuple[str, Image.Image | None, str | None]:
    """Load a single clip composite. Thread-safe (creates own avdi instance)."""
    try:
        img = create_composite(clip_id, t0_us=t0_us, cache_dir=cache_dir)
        return (clip_id, img, None)
    except Exception as e:
        return (clip_id, None, str(e))


def ensure_composites(
    clip_ids: list[str],
    cache_dir: Path | str,
    t0_us: int = DEFAULT_T0_US,
    num_workers: int = 4,
    show_progress: bool = True,
) -> dict[str, Path]:
    """
    Ensure composite images exist for all clip IDs.

    Creates missing composites in parallel, returns paths to all.

    Args:
        clip_ids: List of clip IDs
        cache_dir: Cache directory for composites
        t0_us: Timestamp in microseconds
        num_workers: Number of parallel workers
        show_progress: Show progress bar

    Returns:
        Dict mapping clip_id -> Path to composite image
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    result = {}
    to_generate = []

    # Check which composites already exist
    for clip_id in clip_ids:
        cache_path = cache_dir / f"{clip_id}.jpg"
        if cache_path.exists():
            result[clip_id] = cache_path
        else:
            to_generate.append(clip_id)

    if not to_generate:
        return result

    # Generate missing composites in parallel
    failed = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_load_single_composite, clip_id, t0_us, cache_dir): clip_id
            for clip_id in to_generate
        }

        pbar = None
        if show_progress:
            pbar = tqdm(total=len(to_generate), desc="Generating composites")

        for future in as_completed(futures):
            clip_id, img, error = future.result()
            if error:
                failed.append((clip_id, error))
            else:
                result[clip_id] = cache_dir / f"{clip_id}.jpg"

            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

    if failed:
        print(f"Warning: Failed to generate {len(failed)} composites")
        for clip_id, error in failed[:5]:
            print(f"  {clip_id}: {error}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")

    return result
