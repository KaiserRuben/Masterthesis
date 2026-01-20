#!/usr/bin/env python3
"""
Quick validation script for VLM integration components.
Run this before the full notebook to catch basic issues.
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import sys

print("="*70)
print("VLM Component Validation Script")
print("="*70)

# Test counter
tests_passed = 0
tests_failed = 0

def test_result(name, passed, message=""):
    global tests_passed, tests_failed
    symbol = "✅" if passed else "❌"
    if passed:
        tests_passed += 1
    else:
        tests_failed += 1
    print(f"{symbol} {name}: {message}")
    return passed

# ============================================================================
# Test 1: Perturbation Functions
# ============================================================================
print("\n[1/5] Testing Perturbation Functions...")

def perturbation_magnitude(params):
    return np.sqrt(
        params.get('brightness', 0)**2 +
        (params.get('contrast', 1.0) - 1)**2 +
        params.get('blur_sigma', 0)**2 +
        params.get('noise_sigma', 0)**2
    )

# Test magnitude calculation
identity_mag = perturbation_magnitude({'brightness': 0, 'contrast': 1.0, 'blur_sigma': 0, 'noise_sigma': 0})
test_result("Identity perturbation magnitude", abs(identity_mag) < 1e-6, f"mag={identity_mag:.6f}")

brightness_mag = perturbation_magnitude({'brightness': 0.3, 'contrast': 1.0, 'blur_sigma': 0, 'noise_sigma': 0})
test_result("Brightness perturbation magnitude", abs(brightness_mag - 0.3) < 1e-6, f"mag={brightness_mag:.3f}")

combined_mag = perturbation_magnitude({'brightness': 0.3, 'contrast': 1.4, 'blur_sigma': 0, 'noise_sigma': 0})
expected = np.sqrt(0.3**2 + 0.4**2)
test_result("Combined perturbation magnitude", abs(combined_mag - expected) < 1e-6, f"mag={combined_mag:.3f} (expected {expected:.3f})")

# Test image perturbation
def apply_perturbation(image, params):
    img = image.copy()

    if params.get('brightness', 0) != 0:
        enhancer = ImageEnhance.Brightness(img)
        factor = 1.0 + params['brightness']
        img = enhancer.enhance(factor)

    if params.get('contrast', 1.0) != 1.0:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(params['contrast'])

    if params.get('blur_sigma', 0) > 0:
        radius = params['blur_sigma'] * 2
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    if params.get('noise_sigma', 0) > 0:
        img_array = np.array(img).astype(np.float32)
        noise = np.random.normal(0, params['noise_sigma'] * 255, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)

    return img

test_img = Image.new('RGB', (100, 100), color=(128, 128, 128))
identity_result = apply_perturbation(test_img, {'brightness': 0, 'contrast': 1.0, 'blur_sigma': 0, 'noise_sigma': 0})
test_result("Identity perturbation preserves image", np.array_equal(np.array(test_img), np.array(identity_result)))

# Test brightness direction
bright_img = apply_perturbation(test_img, {'brightness': 0.3, 'contrast': 1.0, 'blur_sigma': 0, 'noise_sigma': 0})
dark_img = apply_perturbation(test_img, {'brightness': -0.3, 'contrast': 1.0, 'blur_sigma': 0, 'noise_sigma': 0})

orig_mean = np.array(test_img).mean()
bright_mean = np.array(bright_img).mean()
dark_mean = np.array(dark_img).mean()

test_result("Brightness ordering", bright_mean > orig_mean > dark_mean,
            f"dark={dark_mean:.1f} < orig={orig_mean:.1f} < bright={bright_mean:.1f}")

# ============================================================================
# Test 2: IoU Calculation
# ============================================================================
print("\n[2/5] Testing IoU Calculation...")

def compute_iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

# Perfect overlap
iou = compute_iou(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]))
test_result("IoU perfect overlap", abs(iou - 1.0) < 1e-6, f"IoU={iou:.4f}")

# No overlap
iou = compute_iou(np.array([0, 0, 0.5, 0.5]), np.array([0.5, 0.5, 1, 1]))
test_result("IoU no overlap", abs(iou - 0.0) < 1e-6, f"IoU={iou:.4f}")

# Half overlap
iou = compute_iou(np.array([0, 0, 1, 1]), np.array([0.5, 0.5, 1, 1]))
test_result("IoU half overlap", abs(iou - 0.25) < 1e-6, f"IoU={iou:.4f}")

# Symmetry
bbox_a = np.array([0.1, 0.1, 0.6, 0.6])
bbox_b = np.array([0.4, 0.4, 0.9, 0.9])
iou_ab = compute_iou(bbox_a, bbox_b)
iou_ba = compute_iou(bbox_b, bbox_a)
test_result("IoU symmetry", abs(iou_ab - iou_ba) < 1e-6, f"IoU(A,B)={iou_ab:.4f}, IoU(B,A)={iou_ba:.4f}")

# ============================================================================
# Test 3: Bbox Parsing
# ============================================================================
print("\n[3/5] Testing Bbox Parsing...")

import re

def parse_bbox(text):
    numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+', text)

    if len(numbers) >= 4:
        bbox = np.array([float(numbers[i]) for i in range(4)])
        bbox = np.clip(bbox, 0, 1)
        return bbox
    else:
        return np.array([0.25, 0.25, 0.75, 0.75])

test_cases = [
    ("[0.1, 0.2, 0.8, 0.9]", np.array([0.1, 0.2, 0.8, 0.9])),
    ("The box is (0.15, 0.25, 0.75, 0.85)", np.array([0.15, 0.25, 0.75, 0.85])),
    ("<box>0.2, 0.3, 0.7, 0.8</box>", np.array([0.2, 0.3, 0.7, 0.8])),
    ("No numbers", np.array([0.25, 0.25, 0.75, 0.75])),
]

for text, expected in test_cases:
    parsed = parse_bbox(text)
    passed = np.allclose(parsed, expected, atol=1e-6)
    test_result(f"Parse '{text[:30]}...'", passed, f"got {parsed}")

# ============================================================================
# Test 4: Test Image Creation
# ============================================================================
print("\n[4/5] Testing Test Image Creation...")

try:
    # Create image with clear bbox target
    test_img = Image.new('RGB', (400, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(test_img)
    draw.rectangle([100, 100, 300, 300], fill=(255, 0, 0), outline=(0, 0, 0), width=3)

    gt_bbox = np.array([100/400, 100/400, 300/400, 300/400])

    test_result("Test image creation", True, f"Created 400x400 with bbox at {gt_bbox}")

    # Verify bbox is correct
    expected_bbox = np.array([0.25, 0.25, 0.75, 0.75])
    test_result("Ground truth bbox", np.allclose(gt_bbox, expected_bbox), f"bbox={gt_bbox}")

except Exception as e:
    test_result("Test image creation", False, f"Error: {e}")

# ============================================================================
# Test 5: Check Ollama Availability
# ============================================================================
print("\n[5/5] Checking Ollama...")

try:
    import ollama_proxy as ollama
    response = ollama.list()

    # Extract model names from ListResponse object
    model_names = [m.model for m in response.models]

    vl_models = [m for m in model_names if 'qwen3-vl' in m.lower()]
    embed_models = [m for m in model_names if 'embed' in m.lower()]

    test_result("Ollama service", True, f"Found {len(model_names)} models")
    test_result("Qwen3-VL available", len(vl_models) > 0, f"Models: {vl_models}")
    test_result("Embedding model available", len(embed_models) > 0, f"Models: {embed_models}")

except Exception as e:
    import traceback
    test_result("Ollama service", False, f"Error: {e}\n{traceback.format_exc()}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Tests passed: {tests_passed}")
print(f"Tests failed: {tests_failed}")
print(f"Total tests:  {tests_passed + tests_failed}")
print("="*70)

if tests_failed == 0:
    print("✅ All basic components validated!")
    print("\nNext steps:")
    print("  1. Run the full test notebook: 02b_vlm_tests.ipynb")
    print("  2. Test with real Ollama API calls")
    print("  3. Run end-to-end perturbation pipeline")
    sys.exit(0)
else:
    print(f"❌ {tests_failed} test(s) failed - fix before proceeding")
    sys.exit(1)
