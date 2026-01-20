"""
Model Comparison Test: Ministral vs Qwen3-VL

Tests all available ministral and qwen3-vl models on 5 samples with 5 perturbations.
Collects comprehensive data for analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
import ollama_proxy as ollama
import io
import base64
import re
import time
import json
from datetime import datetime
from pathlib import Path
from tqdm.auto import tqdm

# Import RefCOCO loader
import sys
sys.path.append('/Users/kaiser/Desktop/Uni/Masterarbeit/notebooks')
from refcoco_loader import load_refcoco, get_sample_info


# Configuration
MODELS_TO_TEST = [
    'ministral-3:3b',
    'ministral-3:8b',
    'ministral-3:14b',
    'qwen3-vl:4b',
    'qwen3-vl:8b',
]

N_SAMPLES = 5
N_PERTURBATIONS = 5


class ModelTester:
    """Efficient model testing with comprehensive data collection."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.results = {
            'model': model_name,
            'timestamp': datetime.now().isoformat(),
            'samples': [],
            'stats': {
                'total_calls': 0,
                'parse_failures': 0,
                'total_time': 0,
                'avg_time_per_call': 0
            }
        }

    def image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL image to base64."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def parse_bbox(self, text: str) -> tuple[np.ndarray, bool]:
        """
        Extract bbox from model response.

        Returns:
            (bbox, success): bbox as [x1,y1,x2,y2] in [0,1], success flag
        """
        # Try JSON format
        json_match = re.search(r'\{"bbox_2d"\s*:\s*\[([^\]]+)\]\}', text)
        if json_match:
            coords_str = json_match.group(1)
            numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+', coords_str)
        else:
            # Fallback: extract any numbers
            numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+', text)

        if len(numbers) >= 4:
            # Parse from [0, 1000] range
            bbox_1000 = np.array([float(numbers[i]) for i in range(4)])
            bbox = bbox_1000 / 1000.0
            bbox = np.clip(bbox, 0, 1)

            # Validate
            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                return np.array([0.0, 0.0, 0.0, 0.0]), False

            return bbox, True
        else:
            return np.array([0.0, 0.0, 0.0, 0.0]), False

    def predict_bbox(self, image: Image.Image, expression: str) -> dict:
        """
        Predict bbox and collect metadata.

        Returns:
            {
                'bbox': [x1,y1,x2,y2],
                'success': bool,
                'response_text': str,
                'inference_time': float,
                'error': str or None
            }
        """
        prompt = f'Where is "{expression}" in this image? Output the bounding box in format: {{"bbox_2d": [x_min, y_min, x_max, y_max]}} using coordinates 0-1000.'
        img_b64 = self.image_to_base64(image)

        start_time = time.time()

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [img_b64]
                }]
            )

            inference_time = time.time() - start_time
            content = response['message']['content']
            bbox, success = self.parse_bbox(content)

            self.results['stats']['total_calls'] += 1
            self.results['stats']['total_time'] += inference_time
            if not success:
                self.results['stats']['parse_failures'] += 1

            return {
                'bbox': bbox.tolist(),
                'success': success,
                'response_text': content,
                'inference_time': inference_time,
                'error': None
            }

        except Exception as e:
            inference_time = time.time() - start_time
            self.results['stats']['total_calls'] += 1
            self.results['stats']['parse_failures'] += 1

            return {
                'bbox': [0.0, 0.0, 0.0, 0.0],
                'success': False,
                'response_text': '',
                'inference_time': inference_time,
                'error': str(e)
            }

    def finalize_stats(self):
        """Calculate final statistics."""
        stats = self.results['stats']
        if stats['total_calls'] > 0:
            stats['avg_time_per_call'] = stats['total_time'] / stats['total_calls']
            stats['parse_success_rate'] = 1 - (stats['parse_failures'] / stats['total_calls'])


def apply_perturbation(image: Image.Image, brightness=0, contrast=1.0, blur=0) -> Image.Image:
    """Apply perturbations to image."""
    img = image.copy()

    if brightness != 0:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.0 + brightness)

    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast)

    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur*2))

    return img


def perturbation_magnitude(brightness=0, contrast=1.0, blur=0) -> float:
    """Compute L2 norm of perturbation parameters."""
    return np.sqrt(brightness**2 + (contrast-1)**2 + blur**2)


def compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """Compute IoU between two normalized bboxes."""
    x1_inter = max(bbox1[0], bbox2[0])
    y1_inter = max(bbox1[1], bbox2[1])
    x2_inter = min(bbox1[2], bbox2[2])
    y2_inter = min(bbox1[3], bbox2[3])

    intersection = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def test_model(model_name: str, samples: list, perturbations: list) -> dict:
    """
    Test a single model on all samples and perturbations.

    Returns:
        Complete results dictionary with all metadata
    """
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")

    tester = ModelTester(model_name)
    total_predictions = len(samples) * (1 + len(perturbations))

    pbar = tqdm(total=total_predictions, desc=f"{model_name}", unit="pred")

    for sample_idx, sample in enumerate(samples):
        info = get_sample_info(sample)
        image = info['image']
        expression = info['expressions'][0]
        bbox_gt = np.array(info['bbox_normalized'])

        sample_result = {
            'sample_idx': sample_idx,
            'expression': expression,
            'bbox_gt': bbox_gt.tolist(),
            'image_size': info['image_size'],
            'original': None,
            'perturbations': []
        }

        # Original prediction
        pred = tester.predict_bbox(image, expression)
        iou = compute_iou(np.array(pred['bbox']), bbox_gt)
        sample_result['original'] = {
            **pred,
            'iou': iou
        }

        pbar.update(1)
        pbar.set_postfix({'Sample': f'{sample_idx+1}/{len(samples)}', 'IoU': f'{iou:.3f}'})

        # Perturbed predictions
        for pert_idx, pert_config in enumerate(perturbations):
            img_pert = apply_perturbation(image, **pert_config)
            pred_pert = tester.predict_bbox(img_pert, expression)
            iou_pert = compute_iou(np.array(pred_pert['bbox']), bbox_gt)

            sample_result['perturbations'].append({
                **pred_pert,
                'iou': iou_pert,
                'perturbation': pert_config,
                'magnitude': perturbation_magnitude(**pert_config)
            })

            pbar.update(1)

    pbar.close()

    tester.results['samples'] = sample_result
    tester.finalize_stats()

    # Print summary
    stats = tester.results['stats']
    print(f"\nResults for {model_name}:")
    print(f"  Total predictions: {stats['total_calls']}")
    print(f"  Parse success rate: {stats['parse_success_rate']*100:.1f}%")
    print(f"  Avg time per prediction: {stats['avg_time_per_call']:.2f}s")
    print(f"  Total time: {stats['total_time']/60:.1f} minutes")

    return tester.results


def main():
    """Run complete model comparison."""
    print("="*60)
    print("MODEL COMPARISON TEST")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Models: {len(MODELS_TO_TEST)}")
    print(f"  Samples: {N_SAMPLES}")
    print(f"  Perturbations: {N_PERTURBATIONS}")
    print(f"  Total predictions per model: {N_SAMPLES * (1 + N_PERTURBATIONS)}")
    print()

    # Load samples
    print("Loading RefCOCO samples...")
    try:
        small_indices = np.load('/Users/kaiser/Desktop/Uni/Masterarbeit/notebooks/small_subset_indices.npy')
        dataset = load_refcoco('val')
        samples = [dataset[int(idx)] for idx in small_indices[:N_SAMPLES]]
        print(f"✓ Loaded {len(samples)} samples")
    except Exception as e:
        print(f"✗ Error loading samples: {e}")
        return

    # Define perturbations (5 diverse configurations)
    perturbations = [
        {'brightness': -0.2, 'contrast': 0.8, 'blur': 0},
        {'brightness': -0.1, 'contrast': 0.9, 'blur': 0},
        {'brightness': 0, 'contrast': 1.0, 'blur': 1.0},
        {'brightness': 0.1, 'contrast': 1.1, 'blur': 0},
        {'brightness': 0.2, 'contrast': 1.2, 'blur': 0},
    ]
    print(f"✓ Defined {len(perturbations)} perturbations")
    print(f"  Magnitude range: [{min(perturbation_magnitude(**p) for p in perturbations):.3f}, {max(perturbation_magnitude(**p) for p in perturbations):.3f}]")

    # Test all models
    all_results = {
        'config': {
            'models': MODELS_TO_TEST,
            'n_samples': N_SAMPLES,
            'n_perturbations': N_PERTURBATIONS,
            'perturbations': perturbations,
            'timestamp': datetime.now().isoformat()
        },
        'models': {}
    }

    start_time = time.time()

    for model_name in MODELS_TO_TEST:
        try:
            results = test_model(model_name, samples, perturbations)
            all_results['models'][model_name] = results
        except Exception as e:
            print(f"\n✗ Error testing {model_name}: {e}")
            all_results['models'][model_name] = {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    total_time = time.time() - start_time

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'model_comparison_{timestamp}.json'

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"COMPLETE!")
    print(f"{'='*60}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Results saved to: {output_file}")

    # Print comparison table
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Success%':<10} {'Avg IoU':<10} {'Time/pred':<10}")
    print("-"*60)

    for model_name in MODELS_TO_TEST:
        if 'error' in all_results['models'][model_name]:
            print(f"{model_name:<20} {'ERROR':<10} {'-':<10} {'-':<10}")
        else:
            results = all_results['models'][model_name]
            stats = results['stats']

            # Calculate avg IoU
            ious = []
            for sample in results['samples']:
                if sample['original']:
                    ious.append(sample['original']['iou'])
                for pert in sample['perturbations']:
                    ious.append(pert['iou'])
            avg_iou = np.mean(ious) if ious else 0

            print(f"{model_name:<20} {stats['parse_success_rate']*100:<10.1f} {avg_iou:<10.3f} {stats['avg_time_per_call']:<10.2f}")

    print()


if __name__ == '__main__':
    main()
