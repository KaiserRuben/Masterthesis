"""
Perturbation Screening Experiment

Quick test to identify which perturbation categories show structure.
Tests 3 categories (Geometric, Occlusion, Quality) on 5 samples.

Total: 45 inferences, ~5-15 minutes runtime.
"""

import sys
sys.path.insert(0, 'scripts')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
from tqdm import tqdm
from collections import defaultdict
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist

# Import existing utilities
import scripts.ollama_proxy as ollama
from scripts.refcoco_loader import load_refcoco, get_sample_info, compute_bbox_difficulty

sns.set_style('whitegrid')


# ============================================================================
# PERTURBATION IMPLEMENTATIONS
# ============================================================================

def apply_geometric_perturbation(image: Image.Image, rotation=0, scale=1.0):
    """Apply geometric transformations."""
    img = image.copy()

    # Rotation
    if rotation != 0:
        img = img.rotate(rotation, expand=False, fillcolor=(128, 128, 128))

    # Scale (zoom)
    if scale != 1.0:
        w, h = img.size
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Crop/pad to original size
        if scale > 1.0:  # Zoom in - crop
            left = (new_w - w) // 2
            top = (new_h - h) // 2
            img = img.crop((left, top, left + w, top + h))
        else:  # Zoom out - pad
            padded = Image.new('RGB', (w, h), (128, 128, 128))
            offset = ((w - new_w) // 2, (h - new_h) // 2)
            padded.paste(img, offset)
            img = padded

    return img


def apply_occlusion_perturbation(image: Image.Image, coverage=0.0, seed=42):
    """Apply random occlusion masks."""
    if coverage == 0.0:
        return image.copy()

    img = image.copy()
    w, h = img.size

    # Create random mask
    np.random.seed(seed)
    mask = Image.new('L', (w, h), 255)
    draw = ImageDraw.Draw(mask)

    # Random rectangles until coverage reached
    total_pixels = w * h
    target_pixels = int(total_pixels * coverage)
    covered = 0

    max_attempts = 20
    for _ in range(max_attempts):
        if covered >= target_pixels:
            break

        # Random rectangle
        x1 = np.random.randint(0, w)
        y1 = np.random.randint(0, h)
        rect_w = np.random.randint(w // 10, w // 3)
        rect_h = np.random.randint(h // 10, h // 3)
        x2 = min(x1 + rect_w, w)
        y2 = min(y1 + rect_h, h)

        draw.rectangle([x1, y1, x2, y2], fill=0)
        covered += (x2 - x1) * (y2 - y1)

    # Apply mask (gray out occluded regions)
    gray = Image.new('RGB', (w, h), (128, 128, 128))
    img = Image.composite(img, gray, mask)

    return img


def apply_quality_perturbation(image: Image.Image, blur=0, noise=0.0):
    """Apply quality degradation."""
    img = image.copy()

    # Blur
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))

    # Noise
    if noise > 0:
        arr = np.array(img).astype(np.float32)
        noise_arr = np.random.normal(0, noise * 255, arr.shape)
        arr = np.clip(arr + noise_arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    return img


# ============================================================================
# VLM PREDICTOR (from notebook)
# ============================================================================

class VLMPredictor:
    """VLM predictor for bbox prediction."""

    def __init__(self, model_name="qwen3-vl:8b"):
        self.model_name = model_name
        self.parse_failures = 0
        self.total_calls = 0

    def predict_bbox(self, image: Image.Image, expression: str) -> np.ndarray:
        """Predict bbox for referring expression."""
        import io
        import base64
        import re

        self.total_calls += 1

        # Convert image to base64
        if image.mode != 'RGB':
            image = image.convert('RGB')
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode()

        # Qwen3-VL prompt
        prompt = f'Where is "{expression}" in this image? Output the bounding box in format: {{"bbox_2d": [x_min, y_min, x_max, y_max]}} using coordinates 0-1000.'

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [img_b64]
                }]
            )

            content = response['message']['content']

            # Parse bbox
            json_match = re.search(r'\{"bbox_2d"\s*:\s*\[([^\]]+)\]\}', content)
            if json_match:
                coords_str = json_match.group(1)
                numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+', coords_str)
            else:
                numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+', content)

            if len(numbers) >= 4:
                bbox_1000 = np.array([float(n) for n in numbers[:4]])
                bbox = bbox_1000 / 1000.0
                bbox = np.clip(bbox, 0, 1)

                if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                    self.parse_failures += 1
                    return np.array([0.0, 0.0, 0.0, 0.0])

                return bbox
            else:
                self.parse_failures += 1
                return np.array([0.0, 0.0, 0.0, 0.0])

        except Exception as e:
            self.parse_failures += 1
            return np.array([0.0, 0.0, 0.0, 0.0])


def compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """Compute IoU between two bboxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def iou_to_class(iou: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Discretize IoU into classes."""
    classes = np.zeros_like(iou, dtype=int)
    for i, t in enumerate(sorted(thresholds)):
        classes[iou >= t] = i + 1
    return classes


# ============================================================================
# METRIC COMPUTATION
# ============================================================================

def compute_metrics(sample_results, category_name, thresholds):
    """Compute 4 key metrics for a perturbation category."""

    # Extract data
    all_boundaries = []
    all_ious = []

    for sample in sample_results:
        iou_orig = sample['iou_original']
        ious_pert = sample['ious_perturbed']
        pert_coords = sample['pert_coords']  # [(param1, param2), ...]

        all_ious.extend(ious_pert)

        # Find boundaries
        orig_class = iou_to_class(np.array([iou_orig]), thresholds)[0]
        pert_classes = iou_to_class(np.array(ious_pert), thresholds)

        for i, pert_class in enumerate(pert_classes):
            if pert_class != orig_class:
                all_boundaries.append({
                    'sample_idx': sample['sample_idx'],
                    'coords': pert_coords[i],
                    'iou_drop': iou_orig - ious_pert[i]
                })

    # Metric 1: IoU Variance
    iou_variance = np.var(all_ious) if len(all_ious) > 0 else 0

    # Metric 2: Boundary Rate
    total_perturbations = sum(len(s['ious_perturbed']) for s in sample_results)
    boundary_rate = len(all_boundaries) / total_perturbations if total_perturbations > 0 else 0

    # Metric 3: Clustering Score (only if enough boundaries)
    clustering_score = 0.0
    if len(all_boundaries) >= 3:
        coords = np.array([b['coords'] for b in all_boundaries])
        # Normalize coordinates to [0, 1] range for comparison
        coords_normalized = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0) + 1e-8)

        distances = pdist(coords_normalized)
        avg_dist = np.mean(distances)
        max_dist = np.sqrt(2)  # Max distance in normalized 2D space
        clustering_score = 1 - (avg_dist / max_dist)

    # Metric 4: Gradient Alignment (simplified - based on IoU drop variance)
    # If boundaries have consistent IoU drops, they're at similar gradient regions
    gradient_alignment = 0.0
    if len(all_boundaries) >= 3:
        iou_drops = [b['iou_drop'] for b in all_boundaries]
        # Low variance in drops = consistent gradient regions
        drop_variance = np.var(iou_drops)
        # Convert to alignment score (lower variance = higher alignment)
        gradient_alignment = 1.0 / (1.0 + drop_variance * 10)

    return {
        'category': category_name,
        'iou_variance': iou_variance,
        'boundary_rate': boundary_rate,
        'clustering_score': clustering_score,
        'gradient_alignment': gradient_alignment,
        'n_boundaries': len(all_boundaries),
        'boundaries': all_boundaries
    }


# ============================================================================
# MAIN SCREENING EXPERIMENT
# ============================================================================

def run_screening():
    """Run perturbation screening experiment."""

    print("=" * 70)
    print("PERTURBATION SCREENING EXPERIMENT")
    print("=" * 70)

    # Load configuration
    config = np.load('data/screening_config.npz', allow_pickle=True)
    test_samples = config['test_samples']

    # Load dataset
    print("\nLoading RefCOCO validation set...")
    dataset = load_refcoco('val')

    # Load validation indices
    validation_indices = np.load('validation_subset_indices.npy')

    # Load thresholds
    try:
        results_02 = np.load('data/threshold_optimization_results.npz')
        thresholds = results_02['optimal_thresholds_geo']
    except:
        thresholds = np.array([0.37, 0.47, 0.52])

    print(f"Using thresholds: {thresholds}")

    # Initialize VLM
    print("\nInitializing VLM...")
    vlm = VLMPredictor()

    # Define perturbation categories
    categories = {
        'Geometric': {
            'params': [
                {'rotation': -10, 'scale': 1.0},
                {'rotation': 0, 'scale': 1.0},
                {'rotation': 10, 'scale': 1.0},
            ],
            'apply_fn': apply_geometric_perturbation
        },
        'Occlusion': {
            'params': [
                {'coverage': 0.0},
                {'coverage': 0.15},
                {'coverage': 0.30},
            ],
            'apply_fn': apply_occlusion_perturbation
        },
        'Quality': {
            'params': [
                {'blur': 0, 'noise': 0.0},
                {'blur': 2, 'noise': 0.0},
                {'blur': 4, 'noise': 0.0},
            ],
            'apply_fn': apply_quality_perturbation
        }
    }

    # Run experiment
    results = {cat: [] for cat in categories.keys()}

    total_inferences = len(test_samples) * sum(len(cat['params']) for cat in categories.values())
    print(f"\nTotal inferences: {total_inferences}")
    print("Starting screening...\n")

    pbar = tqdm(total=total_inferences, desc="Screening")

    for sample_idx in test_samples:
        # Get sample
        sample = dataset[int(validation_indices[sample_idx])]
        info = get_sample_info(sample)
        image = info['image']
        expression = info['expressions'][0]
        bbox_gt = info['bbox_normalized']

        # Original prediction
        bbox_orig = vlm.predict_bbox(image, expression)
        iou_orig = compute_iou(bbox_orig, bbox_gt)
        pbar.set_postfix({'sample': sample_idx, 'expr': expression[:20]})

        # Test each category
        for cat_name, cat_config in categories.items():
            sample_result = {
                'sample_idx': sample_idx,
                'expression': expression,
                'iou_original': iou_orig,
                'ious_perturbed': [],
                'pert_coords': []
            }

            for pert_params in cat_config['params']:
                # Apply perturbation
                img_pert = cat_config['apply_fn'](image, **pert_params)

                # Predict
                bbox_pert = vlm.predict_bbox(img_pert, expression)
                iou_pert = compute_iou(bbox_pert, bbox_gt)

                sample_result['ious_perturbed'].append(iou_pert)

                # Store perturbation coordinates for clustering analysis
                coord_tuple = tuple(pert_params.values())
                sample_result['pert_coords'].append(coord_tuple)

                pbar.update(1)

            results[cat_name].append(sample_result)

    pbar.close()

    # Compute metrics for each category
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    category_metrics = {}
    for cat_name in categories.keys():
        metrics = compute_metrics(results[cat_name], cat_name, thresholds)
        category_metrics[cat_name] = metrics

        print(f"\n{cat_name}:")
        print(f"  IoU Variance:       {metrics['iou_variance']:.4f} (threshold: >0.05)")
        print(f"  Boundary Rate:      {metrics['boundary_rate']:.2%} (threshold: 20-40%)")
        print(f"  Clustering Score:   {metrics['clustering_score']:.3f} (threshold: >0.7)")
        print(f"  Gradient Alignment: {metrics['gradient_alignment']:.3f} (threshold: >0.5)")
        print(f"  Total Boundaries:   {metrics['n_boundaries']}")

    # Decision making
    print("\n" + "=" * 70)
    print("GO/NO-GO DECISION")
    print("=" * 70)

    criteria = {
        'iou_variance': 0.05,
        'boundary_rate': (0.20, 0.40),
        'clustering_score': 0.7,
        'gradient_alignment': 0.5
    }

    go_categories = []

    for cat_name, metrics in category_metrics.items():
        print(f"\n{cat_name}:")

        checks = {
            'IoU Variance': metrics['iou_variance'] > criteria['iou_variance'],
            'Boundary Rate': criteria['boundary_rate'][0] <= metrics['boundary_rate'] <= criteria['boundary_rate'][1],
            'Clustering': metrics['clustering_score'] > criteria['clustering_score'],
            'Gradient Align': metrics['gradient_alignment'] > criteria['gradient_alignment']
        }

        for check_name, passed in checks.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check_name}")

        if all(checks.values()):
            print(f"  → GO: All criteria met!")
            go_categories.append(cat_name)
        else:
            print(f"  → NO-GO: {sum(checks.values())}/4 criteria met")

    # Final recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    if len(go_categories) > 0:
        print(f"\n✓ PROCEED with deep analysis")
        print(f"\nPromising categories: {', '.join(go_categories)}")
        print(f"\nNext steps:")
        print(f"  1. Implement dense perturbation grid for {go_categories[0]}")
        print(f"  2. Run on 30-50 samples")
        print(f"  3. Full boundary detection analysis")
    else:
        print(f"\n✗ NO category shows sufficient structure")
        print(f"\nPossible reasons:")
        print(f"  - VLM predictions too unstable")
        print(f"  - Need different perturbation types")
        print(f"  - Grounding task too difficult for this approach")
        print(f"\nConsider:")
        print(f"  - Testing on different task (classification?)")
        print(f"  - Using different VLM (GPT-4V, Gemini)")
        print(f"  - Documenting as negative result")

    # Save results
    np.savez('data/screening_results.npz',
             category_metrics=category_metrics,
             go_categories=go_categories,
             results=results)

    print(f"\nResults saved to data/screening_results.npz")

    # Create visualization
    create_visualization(category_metrics, criteria)

    return category_metrics, go_categories


def create_visualization(category_metrics, criteria):
    """Create visualization of screening results."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    categories = list(category_metrics.keys())

    # 1. IoU Variance
    ax = axes[0, 0]
    values = [category_metrics[c]['iou_variance'] for c in categories]
    bars = ax.bar(categories, values, color=['green' if v > criteria['iou_variance'] else 'red' for v in values], alpha=0.7, edgecolor='black')
    ax.axhline(criteria['iou_variance'], color='blue', linestyle='--', linewidth=2, label=f'Threshold ({criteria["iou_variance"]})')
    ax.set_ylabel('IoU Variance', fontsize=12)
    ax.set_title('IoU Variance (Perturbation Effect)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 2. Boundary Rate
    ax = axes[0, 1]
    values = [category_metrics[c]['boundary_rate'] * 100 for c in categories]
    bars = ax.bar(categories, values, color=['green' if criteria['boundary_rate'][0]*100 <= v <= criteria['boundary_rate'][1]*100 else 'red' for v in values], alpha=0.7, edgecolor='black')
    ax.axhline(criteria['boundary_rate'][0] * 100, color='blue', linestyle='--', linewidth=2)
    ax.axhline(criteria['boundary_rate'][1] * 100, color='blue', linestyle='--', linewidth=2, label='Target Range (20-40%)')
    ax.set_ylabel('Boundary Rate (%)', fontsize=12)
    ax.set_title('Boundary Rate (Meaningfulness)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Clustering Score
    ax = axes[1, 0]
    values = [category_metrics[c]['clustering_score'] for c in categories]
    bars = ax.bar(categories, values, color=['green' if v > criteria['clustering_score'] else 'red' for v in values], alpha=0.7, edgecolor='black')
    ax.axhline(criteria['clustering_score'], color='blue', linestyle='--', linewidth=2, label=f'Threshold ({criteria["clustering_score"]})')
    ax.set_ylabel('Clustering Score', fontsize=12)
    ax.set_title('Clustering Score (Structure)', fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Gradient Alignment
    ax = axes[1, 1]
    values = [category_metrics[c]['gradient_alignment'] for c in categories]
    bars = ax.bar(categories, values, color=['green' if v > criteria['gradient_alignment'] else 'red' for v in values], alpha=0.7, edgecolor='black')
    ax.axhline(criteria['gradient_alignment'], color='blue', linestyle='--', linewidth=2, label=f'Threshold ({criteria["gradient_alignment"]})')
    ax.set_ylabel('Gradient Alignment', fontsize=12)
    ax.set_title('Gradient Alignment (Consistency)', fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Perturbation Screening Results - 4 Key Metrics', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/screening_results.png', dpi=150, bbox_inches='tight')
    print(f"Visualization saved to figures/screening_results.png")


if __name__ == '__main__':
    run_screening()
