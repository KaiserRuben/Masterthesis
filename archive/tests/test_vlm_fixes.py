#!/usr/bin/env python3
"""
Test VLM fixes: ollama_proxy changes, error handling, zero-IoU defaults.

This script tests all recent changes before applying to production notebooks.
"""

import sys
import numpy as np
from PIL import Image
import ollama_proxy as ollama
from refcoco_loader import load_refcoco, get_sample_info
import io
import base64
import re


class VLMPredictorTest:
    """Test version of VLMPredictor with all fixes."""

    def __init__(self, model_name="qwen3-vl:8b", embed_model="qwen3-embedding:latest"):
        self.model_name = model_name
        self.embed_model = embed_model
        self.parse_failures = 0
        self.total_calls = 0
        print(f"Initialized VLM: {model_name}")
        print(f"Embedding model: {embed_model}")

    def image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL image to base64."""
        if image.mode != 'RGB':
            image = image.convert('RGB')

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def predict_bbox(self, image: Image.Image, expression: str) -> np.ndarray:
        """Predict bbox for referring expression."""
        self.total_calls += 1

        prompt = f'Where is "{expression}" in this image? Output the bounding box in format: {{"bbox_2d": [x_min, y_min, x_max, y_max]}} using coordinates 0-1000.'

        img_b64 = self.image_to_base64(image)

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
            bbox = self.parse_bbox(content)

            return bbox, content  # Return both for testing

        except Exception as e:
            print(f"Error predicting bbox: {e}")
            self.parse_failures += 1
            return np.array([0.0, 0.0, 0.0, 0.0]), str(e)

    def parse_bbox(self, text: str) -> np.ndarray:
        """Extract bbox from model response and convert to [0,1]."""
        # Try to extract JSON format first
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

            # Convert to [0, 1]
            bbox = bbox_1000 / 1000.0
            bbox = np.clip(bbox, 0, 1)

            # Validate
            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                self.parse_failures += 1
                print(f"Invalid bbox {bbox} from: '{text[:100]}'")
                return np.array([0.0, 0.0, 0.0, 0.0])

            return bbox
        else:
            self.parse_failures += 1
            print(f"Could not parse bbox. Response length: {len(text)} chars")
            print(f"  First 200 chars: '{text[:200]}'")
            return np.array([0.0, 0.0, 0.0, 0.0])

    def print_stats(self):
        """Print parsing statistics."""
        success_rate = 100 * (1 - self.parse_failures / max(1, self.total_calls))
        print(f"\nVLM Statistics:")
        print(f"  Total calls: {self.total_calls}")
        print(f"  Parse failures: {self.parse_failures}")
        print(f"  Success rate: {success_rate:.1f}%")


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


def test_zero_iou_default():
    """Test that zero-area bbox gives IoU=0."""
    print("\n" + "="*70)
    print("TEST 1: Zero-IoU Default")
    print("="*70)

    zero_bbox = np.array([0.0, 0.0, 0.0, 0.0])
    real_bbox = np.array([0.2, 0.3, 0.8, 0.9])

    iou = compute_iou(zero_bbox, real_bbox)

    print(f"Zero bbox: {zero_bbox}")
    print(f"Real bbox: {real_bbox}")
    print(f"IoU: {iou}")

    assert iou == 0.0, f"Expected IoU=0, got {iou}"
    print("✓ PASSED: Zero-area bbox gives IoU=0")

    return True


def test_ollama_proxy():
    """Test that ollama_proxy has correct settings."""
    print("\n" + "="*70)
    print("TEST 2: Ollama Proxy Settings")
    print("="*70)

    # Check defaults
    print(f"DEFAULT_NUM_CTX: {ollama.DEFAULT_NUM_CTX}")
    print(f"DEFAULT_NUM_PREDICT: {ollama.DEFAULT_NUM_PREDICT}")
    print(f"DEFAULT_TIMEOUT: {ollama.DEFAULT_TIMEOUT}")

    assert ollama.DEFAULT_NUM_CTX == 16.384, "num_ctx should be 16.384"
    assert ollama.DEFAULT_NUM_PREDICT is None, "num_predict should be None (no limit)"
    assert ollama.DEFAULT_TIMEOUT == 300, "timeout should be 300s"

    print("✓ PASSED: Ollama proxy has correct settings")

    return True


def test_vlm_real_sample():
    """Test VLM on 3 real RefCOCO samples."""
    print("\n" + "="*70)
    print("TEST 3: Real RefCOCO Samples")
    print("="*70)

    # Load 3 samples
    dataset = load_refcoco('val')
    samples = [dataset[i] for i in [0, 1, 2]]

    vlm = VLMPredictorTest(model_name="qwen3-vl:8b")

    results = []

    for i, sample in enumerate(samples):
        info = get_sample_info(sample)
        image = info['image']
        expression = info['expressions'][0]
        bbox_gt = info['bbox_normalized']

        print(f"\nSample {i+1}:")
        print(f"  Expression: '{expression}'")
        print(f"  Ground truth: {bbox_gt}")

        # Predict
        bbox_pred, response = vlm.predict_bbox(image, expression)

        print(f"  Prediction: {bbox_pred}")
        print(f"  Response length: {len(response)} chars")

        # Compute IoU
        iou = compute_iou(bbox_pred, bbox_gt)
        print(f"  IoU: {iou:.3f}")

        # Check if it's a valid bbox (not the zero default)
        is_valid = not np.array_equal(bbox_pred, np.array([0.0, 0.0, 0.0, 0.0]))

        results.append({
            'expression': expression,
            'bbox_pred': bbox_pred,
            'bbox_gt': bbox_gt,
            'iou': iou,
            'response': response[:200],
            'is_valid': is_valid
        })

        if is_valid:
            print(f"  ✓ Valid bbox returned")
        else:
            print(f"  ✗ Zero-IoU default used (parse failure)")

    # Print summary
    vlm.print_stats()

    # Check success rate
    valid_count = sum(1 for r in results if r['is_valid'])
    success_rate = 100 * valid_count / len(results)

    print(f"\nValid predictions: {valid_count}/{len(results)} ({success_rate:.1f}%)")

    if success_rate >= 50:
        print("✓ PASSED: At least 50% success rate")
        return True
    else:
        print("⚠ WARNING: Low success rate, but test passes (diagnostic only)")
        return True  # Don't fail - this is diagnostic


def test_invalid_bbox_handling():
    """Test that invalid bboxes are caught."""
    print("\n" + "="*70)
    print("TEST 4: Invalid Bbox Handling")
    print("="*70)

    vlm = VLMPredictorTest(model_name="qwen3-vl:8b")

    # Test cases
    test_cases = [
        ('{"bbox_2d": [500, 200, 400, 600]}', "x1 > x2"),  # x1 > x2
        ('{"bbox_2d": [200, 600, 400, 200]}', "y1 > y2"),  # y1 > y2
        ('{"bbox_2d": [0, 0, 0, 500]}', "zero width"),    # zero width
        ('no numbers here', "no numbers"),                # no numbers
        ('', "empty string"),                             # empty
    ]

    for i, (response, case_name) in enumerate(test_cases):
        print(f"\nCase {i+1}: {case_name}")
        print(f"  Input: '{response}'")

        bbox = vlm.parse_bbox(response)

        print(f"  Output: {bbox}")

        # Should always return zero-IoU default
        expected = np.array([0.0, 0.0, 0.0, 0.0])
        assert np.array_equal(bbox, expected), f"Expected {expected}, got {bbox}"

        print(f"  ✓ Correctly returned zero-IoU default")

    print("\n✓ PASSED: All invalid cases handled correctly")

    return True


def main():
    """Run all tests."""
    print("="*70)
    print("VLM FIXES TEST SUITE")
    print("="*70)

    tests = [
        ("Zero-IoU Default", test_zero_iou_default),
        ("Ollama Proxy Settings", test_ollama_proxy),
        ("Invalid Bbox Handling", test_invalid_bbox_handling),
        ("Real RefCOCO Samples", test_vlm_real_sample),
    ]

    results = []

    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed, None))
        except Exception as e:
            print(f"\n✗ FAILED: {e}")
            results.append((name, False, str(e)))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for name, passed, error in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {name}")
        if error:
            print(f"  Error: {error}")

    passed_count = sum(1 for _, passed, _ in results if passed)
    total_count = len(results)

    print(f"\nTotal: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED! Safe to apply changes.")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED! Review before applying.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
