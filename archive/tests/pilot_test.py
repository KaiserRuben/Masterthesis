#!/usr/bin/env python3
"""
Pilot Test: VLM Boundary Detection Feasibility Check

Quick feasibility test on 9 RefCOCO samples to verify:
1. Qwen3-VL can perform referring expression grounding
2. Perturbations affect IoU in a measurable way
3. Boundary samples exist in the dataset
4. System is ready for full-scale experiments

Expected runtime: ~10-15 minutes
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import ollama_proxy as ollama
import io
import base64
import re
from refcoco_loader import load_refcoco, get_sample_info, compute_bbox_difficulty
from collections import Counter


class VLMPredictor:
    """Simple VLM wrapper for pilot testing."""

    def __init__(self, model_name="qwen3-vl:8b"):
        self.model_name = model_name
        self.parse_failures = 0
        self.total_calls = 0
        print(f"Initialized VLM: {model_name}")

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64."""
        # Ensure RGB mode (Qwen3-VL requires RGB)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def predict_bbox(self, image: Image.Image, expression: str) -> np.ndarray:
        """
        Predict bbox for referring expression.

        Qwen3-VL uses [0, 1000] coordinate range with JSON format:
        {"bbox_2d": [x_min, y_min, x_max, y_max]}
        """
        self.total_calls += 1

        # Prompt for grounding task - ask for bbox_2d JSON format
        prompt = f'Where is "{expression}" in this image? Output the bounding box in format: {{"bbox_2d": [x_min, y_min, x_max, y_max]}} using coordinates 0-1000.'

        img_b64 = self._image_to_base64(image)

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [img_b64],
                }]
            )

            content = response['message']['content']

            # Debug: print first response to see format
            if not hasattr(self, '_printed_debug'):
                print(f"\nDEBUG - First model response: '{content}'")
                self._printed_debug = True

            return self.parse_bbox(content)

        except Exception as e:
            print(f"Error calling model: {e}")
            self.parse_failures += 1
            return np.array([0.0, 0.0, 0.0, 0.0])

    def parse_bbox(self, text: str) -> np.ndarray:
        """
        Parse bbox from Qwen3-VL output and convert to [0,1] normalized coords.

        Qwen3-VL outputs coordinates in [0, 1000] range.
        We convert to [0, 1] for our IoU calculations.

        Handles formats:
        - {"bbox_2d": [100, 200, 300, 400]}
        - [100, 200, 300, 400]
        - 100 200 300 400
        """
        # Try to extract JSON first
        json_match = re.search(r'\{"bbox_2d"\s*:\s*\[([^\]]+)\]\}', text)
        if json_match:
            coords_str = json_match.group(1)
            numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+', coords_str)
        else:
            # Fallback: extract any numbers
            numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+', text)

        if len(numbers) >= 4:
            # Parse coordinates (in [0, 1000] range from Qwen3-VL)
            bbox_1000 = np.array([float(n) for n in numbers[:4]])

            # Convert from [0, 1000] to [0, 1]
            bbox = bbox_1000 / 1000.0
            bbox = np.clip(bbox, 0, 1)

            # Validate bbox format (x1 < x2, y1 < y2)
            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                self.parse_failures += 1
                print(f"Warning: Invalid bbox {bbox} from: '{text[:100]}'")
                return np.array([0.0, 0.0, 0.0, 0.0])  # Zero-area box = IoU 0

            return bbox
        else:
            self.parse_failures += 1
            if self.parse_failures <= 3:  # Only print first 3 to avoid spam
                print(f"Warning: Could not parse bbox. Response length: {len(text)} chars")
                print(f"  First 200 chars: '{text[:200]}'")
            return np.array([0.0, 0.0, 0.0, 0.0])  # Zero-area box = IoU 0

    def print_stats(self):
        """Print parsing statistics."""
        success_rate = 100 * (1 - self.parse_failures / max(1, self.total_calls))
        print(f"\nVLM Statistics:")
        print(f"  Total calls: {self.total_calls}")
        print(f"  Parse failures: {self.parse_failures}")
        print(f"  Success rate: {success_rate:.1f}%")


def apply_perturbation(image: Image.Image, brightness=0, contrast=1.0, blur=0, noise=0) -> Image.Image:
    """Apply perturbations to image."""
    img = image.copy()

    if brightness != 0:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.0 + brightness)

    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast)

    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))

    if noise > 0:
        arr = np.array(img).astype(np.float32)
        noise_arr = np.random.normal(0, noise * 255, arr.shape)
        arr = np.clip(arr + noise_arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    return img


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


def run_pilot_test():
    """Run pilot test on 9 samples."""
    print("="*70)
    print("PILOT TEST: VLM Boundary Detection Feasibility Check")
    print("="*70)
    print()

    # Load pilot subset
    print("1. Loading pilot subset (9 samples)...")
    pilot_indices = np.load('pilot_subset_indices.npy')
    dataset = load_refcoco('val')
    pilot_samples = [dataset[int(idx)] for idx in pilot_indices]
    print(f"   Loaded {len(pilot_samples)} samples")
    print()

    # Initialize VLM
    print("2. Initializing VLM predictor...")
    vlm = VLMPredictor(model_name="qwen3-vl:8b")
    print()

    # Create simple perturbation grid (3x3 = 9 perturbations)
    print("3. Creating perturbation grid...")
    perturbations = [
        {'brightness': b, 'contrast': c, 'blur': 0, 'noise': 0}
        for b in [-0.2, 0, 0.2]
        for c in [0.8, 1.0, 1.2]
    ]
    print(f"   Grid size: {len(perturbations)} perturbations")
    print()

    # Run experiment
    print("4. Running pilot experiment...")
    print(f"   Total inferences: {len(pilot_samples) * (1 + len(perturbations))}")
    print(f"   Estimated time: ~{len(pilot_samples) * len(perturbations) * 3 / 60:.1f} minutes")
    print()

    results = {
        'iou_original': [],
        'iou_perturbed': [],
        'difficulties': [],
        'expressions': []
    }

    for i, sample in enumerate(pilot_samples):
        print(f"   Processing sample {i+1}/{len(pilot_samples)}...", end=" ")

        # Get sample info
        info = get_sample_info(sample)
        image = info['image']
        expression = info['expressions'][0]
        bbox_gt = info['bbox_normalized']
        difficulty = compute_bbox_difficulty(info['bbox_pixels'], info['image_size'])

        # Original prediction
        bbox_pred_orig = vlm.predict_bbox(image, expression)
        iou_orig = compute_iou(bbox_pred_orig, bbox_gt)

        results['iou_original'].append(iou_orig)
        results['difficulties'].append(difficulty)
        results['expressions'].append(expression)

        # Perturbed predictions
        ious_pert = []
        for pert_config in perturbations:
            img_pert = apply_perturbation(image, **pert_config)
            bbox_pred_pert = vlm.predict_bbox(img_pert, expression)
            iou_pert = compute_iou(bbox_pred_pert, bbox_gt)
            ious_pert.append(iou_pert)

        results['iou_perturbed'].append(ious_pert)

        print(f"IoU: {iou_orig:.3f} -> {np.mean(ious_pert):.3f} (avg pert), {difficulty}")

    print()

    # Convert to numpy
    results['iou_original'] = np.array(results['iou_original'])
    results['iou_perturbed'] = np.array(results['iou_perturbed'])

    # Analysis
    print("="*70)
    print("PILOT TEST RESULTS")
    print("="*70)
    print()

    print("1. VLM Performance:")
    print(f"   Mean IoU (original):  {results['iou_original'].mean():.3f}")
    print(f"   Mean IoU (perturbed): {results['iou_perturbed'].mean():.3f}")
    print(f"   IoU drop:             {(results['iou_original'].mean() - results['iou_perturbed'].mean()):.3f}")

    # Check if model meets baseline
    success_rate = (results['iou_original'] > 0.5).mean()
    print(f"   Success rate (IoU>0.5): {100*success_rate:.1f}%")

    if success_rate < 0.3:
        print("   ⚠️  WARNING: Low success rate! VLM may need tuning.")
    elif success_rate > 0.7:
        print("   ✓ Good baseline performance")
    else:
        print("   ⚠️  Moderate performance, acceptable for boundary testing")

    # Print VLM statistics
    vlm.print_stats()
    print()

    print("2. Perturbation Sensitivity:")
    iou_drops = results['iou_original'].reshape(-1, 1) - results['iou_perturbed']
    mean_drop_per_sample = iou_drops.mean(axis=1)
    max_drop_per_sample = iou_drops.max(axis=1)

    print(f"   Mean IoU drop per sample:  {mean_drop_per_sample.mean():.3f} ± {mean_drop_per_sample.std():.3f}")
    print(f"   Max IoU drop per sample:   {max_drop_per_sample.mean():.3f} ± {max_drop_per_sample.std():.3f}")

    if mean_drop_per_sample.mean() < 0.01:
        print("   ⚠️  WARNING: Very low sensitivity to perturbations!")
    elif mean_drop_per_sample.mean() > 0.2:
        print("   ⚠️  High sensitivity - perturbations may be too strong")
    else:
        print("   ✓ Reasonable sensitivity to perturbations")
    print()

    print("3. Boundary Detection:")
    baseline_thresholds = np.array([0.3, 0.5, 0.7])
    boundary_samples = []

    for i, iou_pert in enumerate(results['iou_perturbed']):
        # Check if sample crosses any threshold
        classes = np.zeros_like(iou_pert, dtype=int)
        for j, t in enumerate(baseline_thresholds):
            classes[iou_pert >= t] = j + 1

        if len(np.unique(classes)) > 1:
            boundary_samples.append(i)

    boundary_rate = 100 * len(boundary_samples) / len(pilot_samples)
    print(f"   Boundary samples: {len(boundary_samples)}/{len(pilot_samples)} ({boundary_rate:.1f}%)")

    if boundary_rate < 10:
        print("   ⚠️  Very few boundaries detected - may need stronger perturbations")
    elif boundary_rate > 70:
        print("   ⚠️  Too many boundaries - perturbations may be too strong")
    else:
        print("   ✓ Reasonable boundary detection rate")
    print()

    print("4. Difficulty Distribution:")
    diff_counts = Counter(results['difficulties'])
    for diff in ['easy', 'medium', 'hard']:
        count = diff_counts[diff]
        print(f"   {diff.capitalize():6s}: {count} samples")
    print()

    # Final verdict
    print("="*70)
    print("PILOT TEST VERDICT")
    print("="*70)

    # Check all criteria
    criteria_passed = 0
    total_criteria = 4

    if success_rate >= 0.3:
        print("✓ VLM has acceptable baseline performance")
        criteria_passed += 1
    else:
        print("✗ VLM baseline performance too low")

    if 0.01 <= mean_drop_per_sample.mean() <= 0.2:
        print("✓ Perturbation sensitivity in acceptable range")
        criteria_passed += 1
    else:
        print("✗ Perturbation sensitivity out of range")

    if 10 <= boundary_rate <= 70:
        print("✓ Boundary detection rate reasonable")
        criteria_passed += 1
    else:
        print("✗ Boundary detection rate out of acceptable range")

    if len(diff_counts) == 3:  # All difficulties present
        print("✓ All difficulty levels represented")
        criteria_passed += 1
    else:
        print("✗ Not all difficulty levels represented")

    print()
    if criteria_passed == total_criteria:
        print("🎉 PILOT TEST PASSED! Ready for full experiments.")
        print("   Next steps:")
        print("   1. Run notebook 02 (threshold optimization on 51 samples)")
        print("   2. Run notebook 03 (validation on 99 samples)")
    elif criteria_passed >= total_criteria - 1:
        print("⚠️  PILOT TEST MOSTLY PASSED with minor issues.")
        print("   You can proceed cautiously, but review the warnings above.")
    else:
        print("❌ PILOT TEST FAILED. Address the issues above before proceeding.")

    print()
    print("="*70)

    # Save results
    np.savez('pilot_test_results.npz', **results)
    print("\nResults saved to pilot_test_results.npz")


if __name__ == '__main__':
    np.random.seed(42)
    run_pilot_test()
