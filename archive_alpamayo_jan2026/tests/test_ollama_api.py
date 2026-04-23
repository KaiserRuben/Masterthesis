#!/usr/bin/env python3
"""
Test Ollama API integration with Qwen3-VL.
Tests real model inference to validate bbox prediction.
"""

import numpy as np
from PIL import Image, ImageDraw
import ollama_proxy as ollama
import io
import base64
import re
import time

print("="*70)
print("Ollama API Test Suite")
print("="*70)

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
# Test 1: Ollama Connection
# ============================================================================
print("\n[1/5] Testing Ollama Connection...")

try:
    response = ollama.list()

    # Extract model names from ListResponse object
    model_names = [m.model for m in response.models]

    vl_models = [m for m in model_names if 'qwen3-vl' in m.lower()]
    embed_models = [m for m in model_names if 'embed' in m.lower()]

    test_result("Ollama service", True, f"Found {len(model_names)} models")
    test_result("Qwen3-VL models", len(vl_models) > 0, f"{vl_models}")
    test_result("Embedding models", len(embed_models) > 0, f"{embed_models}")

except Exception as e:
    test_result("Ollama service", False, f"Error: {e}")
    print("\nCannot proceed without Ollama. Exiting.")
    exit(1)

# Select model to use - prefer 8b for speed
VLM_MODEL = "qwen3-vl:8b" if "qwen3-vl:8b" in vl_models else (vl_models[0] if vl_models else "qwen3-vl:8b")
EMBED_MODEL = "qwen3-embedding:latest" if "qwen3-embedding:latest" in embed_models else (embed_models[0] if embed_models else "qwen3-embedding")

print(f"\nUsing VLM: {VLM_MODEL}")
print(f"Using Embeddings: {EMBED_MODEL}")

# ============================================================================
# Test 2: Basic Image Chat
# ============================================================================
print("\n[2/5] Testing Basic Image Chat...")

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Create a simple test image with a red square
test_img = Image.new('RGB', (400, 400), color=(255, 255, 255))
draw = ImageDraw.Draw(test_img)
draw.rectangle([100, 150, 300, 350], fill=(255, 0, 0), outline=(0, 0, 0), width=3)

img_b64 = image_to_base64(test_img)

try:
    start = time.time()
    response = ollama.chat(
        model=VLM_MODEL,
        messages=[{
            'role': 'user',
            'content': 'Describe what you see in this image in one sentence.',
            'images': [img_b64]
        }]
    )
    elapsed = time.time() - start

    content = response['message']['content']
    test_result("Basic image chat", True, f"Response: '{content}' ({elapsed:.1f}s)")

except Exception as e:
    test_result("Basic image chat", False, f"Error: {e}")

# ============================================================================
# Test 3: Bounding Box Detection
# ============================================================================
print("\n[3/5] Testing Bounding Box Detection...")

def parse_bbox(text: str) -> np.ndarray:
    """Parse bounding box from model output."""
    numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+', text)

    if len(numbers) >= 4:
        bbox = np.array([float(numbers[i]) for i in range(4)])
        bbox = np.clip(bbox, 0, 1)
        return bbox
    else:
        print(f"Warning: Could not parse bbox from: {text}")
        return np.array([0.25, 0.25, 0.75, 0.75])

# Ground truth bbox for the red square (normalized)
gt_bbox = np.array([100/400, 150/400, 300/400, 350/400])  # [0.25, 0.375, 0.75, 0.875]

try:
    start = time.time()
    response = ollama.chat(
        model=VLM_MODEL,
        messages=[{
            'role': 'user',
            'content': 'Locate the red rectangle in this image. Return only the bounding box coordinates in the format [x1, y1, x2, y2] where values are normalized between 0 and 1. Top-left corner is (0,0), bottom-right is (1,1).',
            'images': [img_b64]
        }]
    )
    elapsed = time.time() - start

    content = response['message']['content']
    pred_bbox = parse_bbox(content)

    # Compute IoU with ground truth
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

    iou = compute_iou(pred_bbox, gt_bbox)

    test_result("Bbox prediction", True,
                f"GT={gt_bbox.round(3)}, Pred={pred_bbox.round(3)}, IoU={iou:.3f} ({elapsed:.1f}s)")
    test_result("Bbox accuracy (IoU > 0.3)", iou > 0.3, f"IoU={iou:.3f}")

    print(f"  Model response: {content}")

except Exception as e:
    test_result("Bbox prediction", False, f"Error: {e}")

# ============================================================================
# Test 4: Text Embeddings
# ============================================================================
print("\n[4/5] Testing Text Embeddings...")

try:
    test_texts = [
        "a red rectangle",
        "a blue circle",
        "a red square"
    ]

    embeddings = []
    for text in test_texts:
        response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
        embed = np.array(response['embedding'])
        embeddings.append(embed)

    test_result("Embedding extraction", True,
                f"Shape: {embeddings[0].shape}, dtype: {embeddings[0].dtype}")

    # Test semantic similarity
    sim_red = np.dot(embeddings[0], embeddings[2]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[2]))
    sim_color = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))

    test_result("Semantic similarity", sim_red > sim_color,
                f"sim('red rect', 'red square')={sim_red:.3f} > sim('red rect', 'blue circle')={sim_color:.3f}")

except Exception as e:
    test_result("Embedding extraction", False, f"Error: {e}")

# ============================================================================
# Test 5: Performance Benchmark
# ============================================================================
print("\n[5/5] Performance Benchmark...")

try:
    # Test inference speed
    n_runs = 3
    times = []

    for i in range(n_runs):
        start = time.time()
        response = ollama.chat(
            model=VLM_MODEL,
            messages=[{
                'role': 'user',
                'content': 'What color is the rectangle?',
                'images': [img_b64]
            }]
        )
        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = np.mean(times)
    std_time = np.std(times)

    test_result("Inference speed", True,
                f"Avg={avg_time:.2f}s, Std={std_time:.2f}s over {n_runs} runs")

    # Estimate throughput for experiments
    # Typical experiment: 100 samples x 10 perturbations = 1000 inferences
    estimated_time = avg_time * 1000 / 60  # minutes
    test_result("Experiment estimate", True,
                f"~{estimated_time:.1f} minutes for 100 samples x 10 perturbations")

except Exception as e:
    test_result("Performance benchmark", False, f"Error: {e}")

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
    print("✅ Ollama API working correctly!")
    print("\nReady for:")
    print("  - Dataset loading tests (test_dataset.py)")
    print("  - End-to-end pipeline (test_pipeline.py)")
else:
    print(f"❌ {tests_failed} test(s) failed")
    print("\nTroubleshooting:")
    print("  1. Check Ollama service is running: ollama list")
    print("  2. Try with qwen3-vl:8b (faster) or :30b (better quality)")
    print("  3. Adjust prompts for better bbox detection")
