"""
Classification Screening Experiment

Test boundary detection on image classification (CIFAR-10).
Compares to grounding task to see if simpler task shows structure.

Total: 63 inferences (7 samples × 9 perturbations), ~20-25 minutes.
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
import torch
import torchvision
import torchvision.transforms as transforms

# Import VLM utilities
import scripts.ollama_proxy as ollama

sns.set_style('whitegrid')


# ============================================================================
# CIFAR-10 DATASET
# ============================================================================

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def load_cifar10_samples():
    """Load CIFAR-10 and select 7 diverse samples."""

    print("Downloading CIFAR-10...")
    dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,  # Use test set
        download=True,
        transform=None
    )

    # Select specific samples (one per selected class)
    # Classes: 0=airplane, 1=automobile, 2=bird, 3=cat, 4=deer, 5=dog, 6=frog
    selected_classes = {
        'airplane': {'class_idx': 0, 'difficulty': 'easy'},
        'automobile': {'class_idx': 1, 'difficulty': 'easy'},
        'cat': {'class_idx': 3, 'difficulty': 'easy'},
        'bird': {'class_idx': 2, 'difficulty': 'medium'},
        'deer': {'class_idx': 4, 'difficulty': 'medium'},
        'dog': {'class_idx': 5, 'difficulty': 'hard'},  # Similar to cat
        'frog': {'class_idx': 6, 'difficulty': 'hard'},  # Small, harder to see
    }

    samples = []
    for class_name, info in selected_classes.items():
        class_idx = info['class_idx']
        difficulty = info['difficulty']

        # Find first image of this class
        for i, (img, label) in enumerate(dataset):
            if label == class_idx:
                # Upscale from 32×32 to 512×512
                img_upscaled = img.resize((512, 512), Image.Resampling.BICUBIC)

                samples.append({
                    'image': img_upscaled,
                    'class': class_name,
                    'class_idx': class_idx,
                    'difficulty': difficulty,
                    'dataset_idx': i
                })
                break

    print(f"Loaded {len(samples)} samples from CIFAR-10")
    return samples


# ============================================================================
# PERTURBATION IMPLEMENTATIONS (same as before)
# ============================================================================

def apply_geometric_perturbation(image: Image.Image, rotation=0, scale=1.0):
    """Apply geometric transformations."""
    img = image.copy()

    if rotation != 0:
        img = img.rotate(rotation, expand=False, fillcolor=(128, 128, 128))

    if scale != 1.0:
        w, h = img.size
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        if scale > 1.0:
            left = (new_w - w) // 2
            top = (new_h - h) // 2
            img = img.crop((left, top, left + w, top + h))
        else:
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

    np.random.seed(seed)
    mask = Image.new('L', (w, h), 255)
    draw = ImageDraw.Draw(mask)

    total_pixels = w * h
    target_pixels = int(total_pixels * coverage)
    covered = 0

    max_attempts = 20
    for _ in range(max_attempts):
        if covered >= target_pixels:
            break

        x1 = np.random.randint(0, w)
        y1 = np.random.randint(0, h)
        rect_w = np.random.randint(w // 10, w // 3)
        rect_h = np.random.randint(h // 10, h // 3)
        x2 = min(x1 + rect_w, w)
        y2 = min(y1 + rect_h, h)

        draw.rectangle([x1, y1, x2, y2], fill=0)
        covered += (x2 - x1) * (y2 - y1)

    gray = Image.new('RGB', (w, h), (128, 128, 128))
    img = Image.composite(img, gray, mask)

    return img


def apply_quality_perturbation(image: Image.Image, blur=0, noise=0.0):
    """Apply quality degradation."""
    img = image.copy()

    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))

    if noise > 0:
        arr = np.array(img).astype(np.float32)
        noise_arr = np.random.normal(0, noise * 255, arr.shape)
        arr = np.clip(arr + noise_arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    return img


# ============================================================================
# VLM CLASSIFIER
# ============================================================================

class VLMClassifier:
    """VLM-based image classifier."""

    def __init__(self, model_name="qwen3-vl:8b"):
        self.model_name = model_name
        self.parse_failures = 0
        self.total_calls = 0

    def predict_class(self, image: Image.Image, class_name: str, alternatives: list) -> int:
        """
        Predict if image matches class_name.

        Returns: 1 if correct, 0 if wrong
        """
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

        # Create multiple choice prompt
        all_options = [class_name] + alternatives
        options_str = ', '.join(f'"{opt}"' for opt in all_options)

        prompt = f'What is in this image? Choose ONE from: {options_str}. Answer with just the class name.'

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [img_b64]
                }]
            )

            content = response['message']['content'].lower().strip()

            # Check if correct class is in response
            if class_name.lower() in content:
                # Make sure it's not a false positive (e.g., "not a cat")
                # Simple heuristic: if "not" appears before class name, it's wrong
                class_pos = content.find(class_name.lower())
                not_pos = content.rfind("not", 0, class_pos)

                if not_pos == -1 or class_pos - not_pos > 10:
                    return 1  # Correct

            return 0  # Wrong

        except Exception as e:
            self.parse_failures += 1
            return 0  # Treat as wrong


# ============================================================================
# METRIC COMPUTATION (adapted for classification)
# ============================================================================

def compute_metrics(sample_results, category_name):
    """Compute 4 key metrics for a perturbation category."""

    # Extract data
    all_boundaries = []
    all_accuracies = []

    for sample in sample_results:
        acc_orig = sample['accuracy_original']
        accs_pert = sample['accuracies_perturbed']
        pert_coords = sample['pert_coords']

        all_accuracies.extend(accs_pert)

        # Find boundaries (class transitions)
        for i, acc_pert in enumerate(accs_pert):
            if acc_pert != acc_orig:  # Prediction changed
                all_boundaries.append({
                    'sample_idx': sample['sample_idx'],
                    'coords': pert_coords[i],
                    'accuracy_drop': acc_orig - acc_pert
                })

    # Metric 1: Accuracy Variance
    acc_variance = np.var(all_accuracies) if len(all_accuracies) > 0 else 0

    # Metric 2: Boundary Rate
    total_perturbations = sum(len(s['accuracies_perturbed']) for s in sample_results)
    boundary_rate = len(all_boundaries) / total_perturbations if total_perturbations > 0 else 0

    # Metric 3: Clustering Score
    clustering_score = 0.0
    if len(all_boundaries) >= 3:
        coords = np.array([b['coords'] for b in all_boundaries])
        coords_normalized = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0) + 1e-8)

        distances = pdist(coords_normalized)
        avg_dist = np.mean(distances)
        max_dist = np.sqrt(2)
        clustering_score = 1 - (avg_dist / max_dist)

    # Metric 4: Gradient Alignment
    gradient_alignment = 0.0
    if len(all_boundaries) >= 3:
        acc_drops = [b['accuracy_drop'] for b in all_boundaries]
        drop_variance = np.var(acc_drops)
        gradient_alignment = 1.0 / (1.0 + drop_variance * 10)

    return {
        'category': category_name,
        'accuracy_variance': acc_variance,
        'boundary_rate': boundary_rate,
        'clustering_score': clustering_score,
        'gradient_alignment': gradient_alignment,
        'n_boundaries': len(all_boundaries),
        'boundaries': all_boundaries
    }


# ============================================================================
# MAIN SCREENING EXPERIMENT
# ============================================================================

def run_classification_screening():
    """Run classification screening experiment."""

    print("=" * 70)
    print("CLASSIFICATION SCREENING EXPERIMENT (CIFAR-10)")
    print("=" * 70)

    # Load samples
    samples = load_cifar10_samples()

    print(f"\nSelected samples:")
    for i, s in enumerate(samples):
        print(f"  {i}. {s['class']:12s} ({s['difficulty']:6s})")

    # Initialize VLM
    print("\nInitializing VLM...")
    vlm = VLMClassifier()

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

    # Define alternatives for each class (for multiple choice)
    alternatives_map = {
        'airplane': ['helicopter', 'bird'],
        'automobile': ['truck', 'bus'],
        'cat': ['dog', 'tiger'],
        'bird': ['airplane', 'butterfly'],
        'deer': ['horse', 'dog'],
        'dog': ['cat', 'wolf'],
        'frog': ['lizard', 'fish'],
    }

    # Run experiment
    results = {cat: [] for cat in categories.keys()}

    total_inferences = len(samples) * sum(len(cat['params']) for cat in categories.values())
    print(f"\nTotal inferences: {total_inferences}")
    print("Starting screening...\n")

    pbar = tqdm(total=total_inferences, desc="Classification Screening")

    for sample_idx, sample in enumerate(samples):
        image = sample['image']
        class_name = sample['class']
        alternatives = alternatives_map[class_name]

        pbar.set_postfix({'sample': sample_idx, 'class': class_name})

        # Original prediction
        acc_orig = vlm.predict_class(image, class_name, alternatives)

        # Test each category
        for cat_name, cat_config in categories.items():
            sample_result = {
                'sample_idx': sample_idx,
                'class': class_name,
                'accuracy_original': acc_orig,
                'accuracies_perturbed': [],
                'pert_coords': []
            }

            for pert_params in cat_config['params']:
                # Apply perturbation
                img_pert = cat_config['apply_fn'](image, **pert_params)

                # Predict
                acc_pert = vlm.predict_class(img_pert, class_name, alternatives)

                sample_result['accuracies_perturbed'].append(acc_pert)
                sample_result['pert_coords'].append(tuple(pert_params.values()))

                pbar.update(1)

            results[cat_name].append(sample_result)

    pbar.close()

    # Compute metrics
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    category_metrics = {}
    for cat_name in categories.keys():
        metrics = compute_metrics(results[cat_name], cat_name)
        category_metrics[cat_name] = metrics

        print(f"\n{cat_name}:")
        print(f"  Accuracy Variance:  {metrics['accuracy_variance']:.4f} (threshold: >0.05)")
        print(f"  Boundary Rate:      {metrics['boundary_rate']:.2%} (threshold: 20-40%)")
        print(f"  Clustering Score:   {metrics['clustering_score']:.3f} (threshold: >0.7)")
        print(f"  Gradient Alignment: {metrics['gradient_alignment']:.3f} (threshold: >0.5)")
        print(f"  Total Boundaries:   {metrics['n_boundaries']}")

    # Decision making
    print("\n" + "=" * 70)
    print("GO/NO-GO DECISION")
    print("=" * 70)

    criteria = {
        'accuracy_variance': 0.05,
        'boundary_rate': (0.20, 0.40),
        'clustering_score': 0.7,
        'gradient_alignment': 0.5
    }

    go_categories = []

    for cat_name, metrics in category_metrics.items():
        print(f"\n{cat_name}:")

        checks = {
            'Accuracy Variance': metrics['accuracy_variance'] > criteria['accuracy_variance'],
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

    # Comparison with grounding
    print("\n" + "=" * 70)
    print("COMPARISON: CLASSIFICATION vs GROUNDING")
    print("=" * 70)

    # Load grounding results
    try:
        grounding_results = np.load('data/screening_results.npz', allow_pickle=True)
        grounding_metrics = grounding_results['category_metrics'].item()

        print("\nMetric Comparison:\n")
        print(f"{'Category':<12} {'Task':<12} {'AccVar':<10} {'BndRate':<10} {'Cluster':<10} {'GradAlign':<10}")
        print("-" * 70)

        for cat in categories.keys():
            # Grounding
            g = grounding_metrics[cat]
            print(f"{cat:<12} {'Grounding':<12} {g['iou_variance']:<10.3f} {g['boundary_rate']:<10.2%} {g['clustering_score']:<10.3f} {g['gradient_alignment']:<10.3f}")

            # Classification
            c = category_metrics[cat]
            print(f"{cat:<12} {'Classification':<12} {c['accuracy_variance']:<10.3f} {c['boundary_rate']:<10.2%} {c['clustering_score']:<10.3f} {c['gradient_alignment']:<10.3f}")
            print()

    except:
        print("\nGrounding results not found - cannot compare")

    # Final recommendation
    print("=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    if len(go_categories) > 0:
        print(f"\n✓ Classification SHOWS structure!")
        print(f"\nGO categories: {', '.join(go_categories)}")
        print(f"\nConclusion:")
        print(f"  • Framework works for classification")
        print(f"  • Grounding was too difficult/noisy")
        print(f"  • Thesis: Task-specific applicability of boundary detection")
    else:
        print(f"\n✗ Classification ALSO shows no structure")
        print(f"\nConclusion:")
        print(f"  • Framework limitation OR VLM too unstable")
        print(f"  • Consider:")
        print(f"    - Different VLM (GPT-4V, Gemini)")
        print(f"    - Different approach entirely")
        print(f"    - Document as negative result")

    # Save results
    np.savez('data/classification_screening_results.npz',
             category_metrics=category_metrics,
             go_categories=go_categories,
             results=results)

    print(f"\nResults saved to data/classification_screening_results.npz")

    # Create visualization
    create_visualization(category_metrics, grounding_metrics if 'grounding_metrics' in locals() else None)

    return category_metrics, go_categories


def create_visualization(class_metrics, ground_metrics=None):
    """Create comparison visualization."""

    if ground_metrics is None:
        # Single plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    else:
        # Comparison plot
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    categories = list(class_metrics.keys())
    criteria = {
        'accuracy_variance': 0.05,
        'boundary_rate': (0.20, 0.40),
        'clustering_score': 0.7,
        'gradient_alignment': 0.5
    }

    # Plot classification results
    ax = axes[0, 0] if ground_metrics else axes[0, 0]
    values = [class_metrics[c]['accuracy_variance'] for c in categories]
    bars = ax.bar(categories, values, color=['green' if v > criteria['accuracy_variance'] else 'red' for v in values], alpha=0.7, edgecolor='black')
    ax.axhline(criteria['accuracy_variance'], color='blue', linestyle='--', linewidth=2)
    ax.set_ylabel('Variance', fontsize=11)
    ax.set_title('Classification: Accuracy Variance', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[0, 1] if ground_metrics else axes[0, 1]
    values = [class_metrics[c]['boundary_rate'] * 100 for c in categories]
    bars = ax.bar(categories, values, color=['green' if criteria['boundary_rate'][0]*100 <= v <= criteria['boundary_rate'][1]*100 else 'red' for v in values], alpha=0.7, edgecolor='black')
    ax.axhline(criteria['boundary_rate'][0] * 100, color='blue', linestyle='--', linewidth=2)
    ax.axhline(criteria['boundary_rate'][1] * 100, color='blue', linestyle='--', linewidth=2)
    ax.set_ylabel('Rate (%)', fontsize=11)
    ax.set_title('Classification: Boundary Rate', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1, 0] if ground_metrics else axes[1, 0]
    values = [class_metrics[c]['clustering_score'] for c in categories]
    bars = ax.bar(categories, values, color=['green' if v > criteria['clustering_score'] else 'red' for v in values], alpha=0.7, edgecolor='black')
    ax.axhline(criteria['clustering_score'], color='blue', linestyle='--', linewidth=2)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Classification: Clustering', fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1, 1] if ground_metrics else axes[1, 1]
    values = [class_metrics[c]['gradient_alignment'] for c in categories]
    bars = ax.bar(categories, values, color=['green' if v > criteria['gradient_alignment'] else 'red' for v in values], alpha=0.7, edgecolor='black')
    ax.axhline(criteria['gradient_alignment'], color='blue', linestyle='--', linewidth=2)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Classification: Gradient Alignment', fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    # Add grounding comparison if available
    if ground_metrics:
        criteria_ground = {
            'iou_variance': 0.05,
            'boundary_rate': (0.20, 0.40),
            'clustering_score': 0.7,
            'gradient_alignment': 0.5
        }

        ax = axes[0, 2]
        values = [ground_metrics[c]['iou_variance'] for c in categories]
        bars = ax.bar(categories, values, color=['green' if v > criteria_ground['iou_variance'] else 'red' for v in values], alpha=0.7, edgecolor='black')
        ax.axhline(criteria_ground['iou_variance'], color='blue', linestyle='--', linewidth=2)
        ax.set_ylabel('Variance', fontsize=11)
        ax.set_title('Grounding: IoU Variance', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        ax = axes[0, 3]
        values = [ground_metrics[c]['boundary_rate'] * 100 for c in categories]
        bars = ax.bar(categories, values, color=['green' if criteria_ground['boundary_rate'][0]*100 <= v <= criteria_ground['boundary_rate'][1]*100 else 'red' for v in values], alpha=0.7, edgecolor='black')
        ax.axhline(criteria_ground['boundary_rate'][0] * 100, color='blue', linestyle='--', linewidth=2)
        ax.axhline(criteria_ground['boundary_rate'][1] * 100, color='blue', linestyle='--', linewidth=2)
        ax.set_ylabel('Rate (%)', fontsize=11)
        ax.set_title('Grounding: Boundary Rate', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        ax = axes[1, 2]
        values = [ground_metrics[c]['clustering_score'] for c in categories]
        bars = ax.bar(categories, values, color=['green' if v > criteria_ground['clustering_score'] else 'red' for v in values], alpha=0.7, edgecolor='black')
        ax.axhline(criteria_ground['clustering_score'], color='blue', linestyle='--', linewidth=2)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title('Grounding: Clustering', fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')

        ax = axes[1, 3]
        values = [ground_metrics[c]['gradient_alignment'] for c in categories]
        bars = ax.bar(categories, values, color=['green' if v > criteria_ground['gradient_alignment'] else 'red' for v in values], alpha=0.7, edgecolor='black')
        ax.axhline(criteria_ground['gradient_alignment'], color='blue', linestyle='--', linewidth=2)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title('Grounding: Gradient Alignment', fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')

    title = 'Classification vs Grounding Screening Results' if ground_metrics else 'Classification Screening Results'
    plt.suptitle(title, fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/classification_screening_results.png', dpi=150, bbox_inches='tight')
    print(f"Visualization saved to figures/classification_screening_results.png")


if __name__ == '__main__':
    run_classification_screening()
