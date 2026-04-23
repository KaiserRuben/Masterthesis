#!/usr/bin/env python3
"""
Manual test to verify Qwen3-VL grounding format.

Tests different prompt formats to see what works.
"""

import ollama_proxy as ollama
from PIL import Image
import io
import base64
import numpy as np
from refcoco_loader import load_refcoco, get_sample_info


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL image to base64."""
    # Ensure RGB mode (Qwen3-VL might not like RGBA or other modes)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()

    print(f"  [DEBUG] Image encoded: {len(img_bytes)} bytes, mode={image.mode}, size={image.size}")

    return base64.b64encode(img_bytes).decode()


def test_grounding_prompt(image: Image.Image, expression: str, prompt_variant: str):
    """Test a specific prompt format."""

    prompts = {
        "json_format": f'Where is "{expression}" in this image? Output the bounding box in format: {{"bbox_2d": [x_min, y_min, x_max, y_max]}} using coordinates 0-1000.',

        "simple": f'Locate "{expression}" in this image and give me the bounding box.',

        "grounding": f'<ref>{expression}</ref>',

        "detection": f'Detect "{expression}" in this image.',

        "coordinates": f'What are the coordinates of "{expression}" in this image?',
    }

    prompt = prompts[prompt_variant]

    print(f"\n{'='*70}")
    print(f"PROMPT VARIANT: {prompt_variant}")
    print(f"{'='*70}")
    print(f"Prompt: {prompt}")
    print()

    img_b64 = image_to_base64(image)

    try:
        response = ollama.chat(
            model="qwen3-vl:8b",
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [img_b64],
            }]
        )

        content = response['message']['content']
        print(f"Response ({len(content)} chars):")
        print(f"{content}")
        print()

        return content

    except Exception as e:
        print(f"ERROR: {e}")
        return None


def main():
    print("Loading RefCOCO sample...")
    dataset = load_refcoco('val')
    sample = dataset[0]
    info = get_sample_info(sample)

    print(f"\nSample info:")
    print(f"  Image size: {info['image_size']}")
    print(f"  Expression: {info['expressions'][0]}")
    print(f"  Ground truth bbox (normalized [0,1]): {info['bbox_normalized']}")
    print(f"  Ground truth bbox (pixels): {info['bbox_pixels']}")

    # Convert ground truth to [0, 1000] format for comparison
    bbox_1000 = info['bbox_normalized'] * 1000
    print(f"  Ground truth bbox ([0,1000] format): {bbox_1000}")

    image = info['image']
    expression = info['expressions'][0]

    # Test different prompt formats
    print("\n" + "="*70)
    print("TESTING DIFFERENT PROMPT FORMATS")
    print("="*70)

    results = {}

    for variant in ["simple", "json_format", "grounding", "detection", "coordinates"]:
        response = test_grounding_prompt(image, expression, variant)
        results[variant] = response

        # Pause between requests
        import time
        time.sleep(1)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for variant, response in results.items():
        if response:
            has_numbers = any(char.isdigit() for char in response)
            has_bbox_keyword = any(word in response.lower() for word in ['box', 'bbox', 'coordinate', 'location'])
            print(f"\n{variant:15s}: {len(response):4d} chars, has_numbers={has_numbers}, has_bbox_keyword={has_bbox_keyword}")
            print(f"  First 100 chars: {response[:100]}...")


if __name__ == '__main__':
    main()
