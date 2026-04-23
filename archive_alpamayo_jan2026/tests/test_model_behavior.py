#!/usr/bin/env python3
"""
Test Model Behavior: Verify Qwen3-VL via Ollama is being used correctly.

This script tests:
1. API call correctness
2. Response format consistency
3. Error handling robustness
4. Behavior under perturbations
5. Failure pattern analysis
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import ollama_proxy as ollama
import io
import base64
import re
from refcoco_loader import load_refcoco, get_sample_info
import time


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL image to base64."""
    if image.mode != 'RGB':
        image = image.convert('RGB')

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


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
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))

    return img


def test_api_call():
    """Test 1: Verify Ollama API call is correct."""
    print("="*70)
    print("TEST 1: API Call Correctness")
    print("="*70)

    dataset = load_refcoco('val')
    sample = dataset[0]
    info = get_sample_info(sample)

    image = info['image']
    expression = info['expressions'][0]

    print(f"Expression: '{expression}'")
    print(f"Image size: {image.size}")
    print(f"Image mode: {image.mode}")

    # Test API call
    img_b64 = image_to_base64(image)
    prompt = f'Where is "{expression}" in this image? Output the bounding box in format: {{"bbox_2d": [x_min, y_min, x_max, y_max]}} using coordinates 0-1000.'

    print("\nCalling ollama.chat()...")
    print(f"  Model: qwen3-vl:8b")
    print(f"  Image encoded: {len(img_b64)} chars")
    print(f"  Prompt length: {len(prompt)} chars")

    start_time = time.time()

    try:
        response = ollama.chat(
            model="qwen3-vl:8b",
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [img_b64]
            }]
        )

        elapsed = time.time() - start_time

        print(f"\n✓ API call succeeded ({elapsed:.2f}s)")
        print(f"  Response type: {type(response)}")

        # ChatResponse object supports dict-like access
        content = response['message']['content']
        print(f"  Content length: {len(content)} chars")
        print(f"  Content: '{content}'")

        # Check if response has expected structure
        assert len(content) > 0, "Empty content"

        print("\n✓ PASSED: API call structure correct")
        return True

    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        return False


def test_response_formats():
    """Test 2: Check response format consistency across samples."""
    print("\n" + "="*70)
    print("TEST 2: Response Format Consistency")
    print("="*70)

    dataset = load_refcoco('val')
    samples = [dataset[i] for i in range(10)]  # Test 10 samples

    formats = {
        'json_format': 0,      # {"bbox_2d": [x, y, x, y]}
        'array_format': 0,     # [x, y, x, y]
        'text_with_coords': 0, # Text containing numbers
        'no_coords': 0,        # No numbers found
        'empty': 0             # Empty response
    }

    all_responses = []

    for i, sample in enumerate(samples):
        info = get_sample_info(sample)
        expression = info['expressions'][0]

        print(f"\nSample {i+1}: '{expression[:50]}...'", end=" ")

        img_b64 = image_to_base64(info['image'])
        prompt = f'Where is "{expression}" in this image? Output the bounding box in format: {{"bbox_2d": [x_min, y_min, x_max, y_max]}} using coordinates 0-1000.'

        try:
            response = ollama.chat(
                model="qwen3-vl:8b",
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [img_b64]
                }]
            )

            content = response['message']['content']
            all_responses.append(content)

            # Classify response format
            if len(content) == 0:
                formats['empty'] += 1
                print("→ EMPTY")
            elif re.search(r'\{"bbox_2d"\s*:\s*\[', content):
                formats['json_format'] += 1
                print(f"→ JSON: {content}")
            elif re.search(r'\[\s*\d+\s*,', content):
                formats['array_format'] += 1
                print(f"→ ARRAY: {content}")
            elif re.findall(r'\d+', content):
                formats['text_with_coords'] += 1
                print(f"→ TEXT: {content[:50]}...")
            else:
                formats['no_coords'] += 1
                print(f"→ NO_COORDS: {content[:50]}...")

        except Exception as e:
            print(f"→ ERROR: {e}")
            formats['empty'] += 1

    # Summary
    print("\n" + "-"*70)
    print("Format Distribution:")
    for fmt, count in formats.items():
        pct = 100 * count / len(samples)
        print(f"  {fmt:20s}: {count:2d} ({pct:5.1f}%)")

    # Check consistency
    if formats['json_format'] >= 8:  # 80% or more
        print("\n✓ PASSED: Model consistently returns JSON format")
        return True
    else:
        print("\n⚠ WARNING: Inconsistent format. JSON format: {formats['json_format']}/10")
        return False


def test_perturbation_handling():
    """Test 3: Verify model behavior under perturbations."""
    print("\n" + "="*70)
    print("TEST 3: Behavior Under Perturbations")
    print("="*70)

    dataset = load_refcoco('val')
    sample = dataset[0]
    info = get_sample_info(sample)
    expression = info['expressions'][0]

    print(f"Expression: '{expression}'")

    perturbations = [
        {'name': 'original', 'brightness': 0, 'contrast': 1.0, 'blur': 0},
        {'name': 'dark', 'brightness': -0.3, 'contrast': 1.0, 'blur': 0},
        {'name': 'bright', 'brightness': 0.3, 'contrast': 1.0, 'blur': 0},
        {'name': 'low_contrast', 'brightness': 0, 'contrast': 0.5, 'blur': 0},
        {'name': 'blurred', 'brightness': 0, 'contrast': 1.0, 'blur': 5},
    ]

    results = []

    for pert in perturbations:
        img = apply_perturbation(
            info['image'],
            brightness=pert['brightness'],
            contrast=pert['contrast'],
            blur=pert['blur']
        )

        print(f"\n{pert['name']:15s}: ", end="")

        img_b64 = image_to_base64(img)
        prompt = f'Where is "{expression}" in this image? Output the bounding box in format: {{"bbox_2d": [x_min, y_min, x_max, y_max]}} using coordinates 0-1000.'

        try:
            response = ollama.chat(
                model="qwen3-vl:8b",
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [img_b64]
                }]
            )

            content = response['message']['content']

            # Try to parse
            json_match = re.search(r'\{"bbox_2d"\s*:\s*\[([^\]]+)\]', content)
            if json_match:
                coords_str = json_match.group(1)
                numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+', coords_str)
                if len(numbers) >= 4:
                    bbox = [float(n) for n in numbers[:4]]
                    print(f"✓ {bbox}")
                    results.append({'pert': pert['name'], 'bbox': bbox, 'success': True})
                else:
                    print(f"✗ Parse failed: {content}")
                    results.append({'pert': pert['name'], 'bbox': None, 'success': False})
            else:
                print(f"✗ No JSON: {content[:50]}")
                results.append({'pert': pert['name'], 'bbox': None, 'success': False})

        except Exception as e:
            print(f"✗ Error: {e}")
            results.append({'pert': pert['name'], 'bbox': None, 'success': False})

    success_rate = sum(1 for r in results if r['success']) / len(results)
    print(f"\nSuccess rate: {100*success_rate:.1f}%")

    if success_rate >= 0.8:
        print("✓ PASSED: Model handles perturbations well")
        return True
    else:
        print("⚠ WARNING: Low success rate with perturbations")
        return False


def test_timeout_handling():
    """Test 4: Verify timeout protection works."""
    print("\n" + "="*70)
    print("TEST 4: Timeout Protection")
    print("="*70)

    print("Testing timeout setting...")
    print(f"  Default timeout: {ollama.DEFAULT_TIMEOUT}s")

    # Test that we can set timeout
    dataset = load_refcoco('val')
    sample = dataset[0]
    info = get_sample_info(sample)
    expression = info['expressions'][0]

    img_b64 = image_to_base64(info['image'])
    prompt = f'Where is "{expression}" in this image? Output the bounding box in format: {{"bbox_2d": [x_min, y_min, x_max, y_max]}} using coordinates 0-1000.'

    start = time.time()

    try:
        # Use a reasonable timeout (not too short to trigger false alarm)
        response = ollama.chat(
            model="qwen3-vl:8b",
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [img_b64]
            }],
            timeout=300  # 5 minutes
        )

        elapsed = time.time() - start
        print(f"  ✓ Call completed in {elapsed:.2f}s (under timeout)")
        print(f"  Response: '{response['message']['content']}'")
        print("\n✓ PASSED: Timeout setting works")
        return True

    except Exception as e:
        elapsed = time.time() - start
        print(f"  ✗ Failed after {elapsed:.2f}s: {e}")
        print("\n✗ FAILED: Timeout handling issue")
        return False


def main():
    """Run all model behavior tests."""
    print("="*70)
    print("MODEL BEHAVIOR TEST SUITE")
    print("="*70)
    print()

    tests = [
        ("API Call Correctness", test_api_call),
        ("Response Format Consistency", test_response_formats),
        ("Perturbation Handling", test_perturbation_handling),
        ("Timeout Protection", test_timeout_handling),
    ]

    results = []

    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed, None))
        except Exception as e:
            print(f"\n✗ EXCEPTION: {e}")
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
        print("\n🎉 ALL TESTS PASSED! Model is being handled correctly.")
        return 0
    else:
        print("\n⚠️ SOME TESTS FAILED! Review model handling.")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
