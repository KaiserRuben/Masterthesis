"""
RefCOCO dataset loader for lmms-lab/RefCOCO format.

Dataset structure (lmms-eval format):
{
    'question_id': str,
    'image': PIL.Image,
    'question': str (generic prompt),
    'answer': list[str] (referring expressions),
    'segmentation': list[float] (polygon points),
    'bbox': [x, y, width, height] (pixels),
    'iscrowd': int,
    'file_name': str
}
"""

from datasets import load_dataset
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional


def load_refcoco(split='val', num_samples: Optional[int] = None):
    """
    Load RefCOCO dataset from lmms-lab.

    Args:
        split: One of ['val', 'test', 'testA', 'testB']
        num_samples: If specified, load only first N samples

    Returns:
        HuggingFace Dataset object
    """
    split_str = f"{split}[:{num_samples}]" if num_samples else split
    return load_dataset("lmms-lab/RefCOCO", split=split_str)


def normalize_bbox(bbox: List[float], image_size: Tuple[int, int]) -> np.ndarray:
    """
    Convert bbox from [x, y, w, h] pixels to [x1, y1, x2, y2] normalized [0,1].

    Args:
        bbox: [x, y, width, height] in pixels
        image_size: (width, height) of image

    Returns:
        np.ndarray: [x1, y1, x2, y2] normalized to [0, 1]
    """
    x, y, w, h = bbox
    img_width, img_height = image_size

    # Convert to [x1, y1, x2, y2] and normalize
    return np.array([
        x / img_width,
        y / img_height,
        (x + w) / img_width,
        (y + h) / img_height
    ])


def denormalize_bbox(bbox_norm: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    """
    Convert normalized bbox [x1, y1, x2, y2] back to [x, y, w, h] pixels.

    Args:
        bbox_norm: [x1, y1, x2, y2] normalized [0, 1]
        image_size: (width, height) of image

    Returns:
        np.ndarray: [x, y, width, height] in pixels
    """
    x1, y1, x2, y2 = bbox_norm
    img_width, img_height = image_size

    return np.array([
        x1 * img_width,
        y1 * img_height,
        (x2 - x1) * img_width,
        (y2 - y1) * img_height
    ])


def get_sample_info(sample: Dict) -> Dict:
    """
    Extract key information from a RefCOCO sample.

    Returns dict with:
    - image: PIL.Image
    - expressions: list of referring expressions
    - bbox_pixels: [x, y, w, h]
    - bbox_normalized: [x1, y1, x2, y2]
    - image_size: (width, height)
    - file_name: str
    """
    image = sample['image']
    img_size = image.size

    bbox_pixels = sample['bbox']
    bbox_normalized = normalize_bbox(bbox_pixels, img_size)

    return {
        'image': image,
        'expressions': sample['answer'],  # Multiple referring expressions
        'bbox_pixels': bbox_pixels,
        'bbox_normalized': bbox_normalized,
        'image_size': img_size,
        'file_name': sample['file_name']
    }


def compute_bbox_difficulty(bbox_pixels: List[float], image_size: Tuple[int, int]) -> str:
    """
    Classify bbox difficulty based on size.

    Args:
        bbox_pixels: [x, y, w, h]
        image_size: (width, height)

    Returns:
        'easy', 'medium', or 'hard'
    """
    x, y, w, h = bbox_pixels
    img_width, img_height = image_size

    # Normalized area
    area_norm = (w * h) / (img_width * img_height)

    if area_norm < 0.05:
        return 'hard'  # Small object
    elif area_norm < 0.25:
        return 'medium'
    else:
        return 'easy'  # Large object


def get_balanced_subset(dataset, n_per_difficulty: int = 10) -> List[int]:
    """
    Create balanced subset with equal samples from each difficulty level.

    Args:
        dataset: RefCOCO dataset
        n_per_difficulty: Number of samples per difficulty level

    Returns:
        List of sample indices
    """
    difficulty_indices = {'easy': [], 'medium': [], 'hard': []}

    for idx, sample in enumerate(dataset):
        bbox = sample['bbox']
        img_size = sample['image'].size
        diff = compute_bbox_difficulty(bbox, img_size)
        difficulty_indices[diff].append(idx)

    # Sample from each difficulty
    np.random.seed(42)
    selected_indices = []

    for diff, indices in difficulty_indices.items():
        if len(indices) >= n_per_difficulty:
            selected = np.random.choice(indices, n_per_difficulty, replace=False)
            selected_indices.extend(selected.tolist())

    return sorted(selected_indices)


if __name__ == "__main__":
    # Quick test
    print("Loading RefCOCO validation set...")
    ds = load_refcoco('val', num_samples=10)

    print(f"Loaded {len(ds)} samples")
    print(f"\nSample 0:")
    info = get_sample_info(ds[0])
    print(f"  Image size: {info['image_size']}")
    print(f"  Bbox (pixels): {info['bbox_pixels']}")
    print(f"  Bbox (normalized): {info['bbox_normalized']}")
    print(f"  Expressions ({len(info['expressions'])}):")
    for expr in info['expressions']:
        print(f"    - {expr}")
    print(f"  Difficulty: {compute_bbox_difficulty(info['bbox_pixels'], info['image_size'])}")
