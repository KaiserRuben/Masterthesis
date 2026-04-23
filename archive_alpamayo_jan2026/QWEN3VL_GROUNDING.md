# Qwen3-VL Grounding Task Documentation

**Last Updated:** 2026-01-11
**Model:** qwen3-vl:8b via Ollama

## Overview

Qwen3-VL supports referring expression grounding (bounding box prediction) through its text generation interface. Coordinates are treated as plain text tokens, not special constructs.

## Coordinate System

- **Range:** [0, 1000] (NOT [0, 1])
- **Format:** `[x_min, y_min, x_max, y_max]`
- **Conversion to normalized [0,1]:** Divide by 1000
- **Conversion from [0,1] to [0,1000]:** Multiply by 1000

### Example
```python
# Ground truth in pixels: [468, 0, 640, 117] on 640x428 image
# Normalized [0,1]: [0.731, 0.002, 1.000, 0.273]
# Qwen3-VL format [0,1000]: [731, 2, 1000, 273]

# Model prediction: [734, 0, 999, 243]
# Convert to [0,1]: [0.734, 0.000, 0.999, 0.243]
```

## Prompting Strategies

### ✅ RECOMMENDED: JSON Format

**Prompt:**
```python
f'Where is "{expression}" in this image? Output the bounding box in format: {{"bbox_2d": [x_min, y_min, x_max, y_max]}} using coordinates 0-1000.'
```

**Response:**
```
{"bbox_2d": [734, 0, 999, 243]}
```

**Advantages:**
- Clean, structured output
- Easy to parse with regex
- Shortest response (fastest inference)
- Most accurate in testing

### ✅ ALTERNATIVE: Simple Prompt

**Prompt:**
```python
f'Locate "{expression}" in this image and give me the bounding box.'
```

**Response:**
```json
[
    {"bbox_2d": [710, 0, 999, 285], "label": "bowl behind the others can only see part"}
]
```

**Advantages:**
- Returns label with bbox
- JSON array format
- Still easy to parse

### ✅ ALTERNATIVE: Natural Language

**Prompt:**
```python
f'What are the coordinates of "{expression}" in this image?'
```

**Response:**
```
The bowl that is behind the others and only partially visible in the image is the white ceramic bowl with a blue rim in the top-right corner. Its coordinates are [726, 0, 999, 275].
```

**Advantages:**
- More conversational
- Includes description
- May help with ambiguous references

**Disadvantages:**
- Requires extracting numbers from text
- Longer response (slower)

### ❌ NOT RECOMMENDED: Special Tokens

**Prompt:**
```python
f'<ref>{expression}</ref>'
```

**Response:**
```
The bowl behind the others that can only be seen partially is the white ceramic bowl...
[no coordinates]
```

**Result:** Only returns description, no coordinates.

### ❌ NOT RECOMMENDED: Detection

**Prompt:**
```python
f'Detect "{expression}" in this image.'
```

**Response:**
```
The bowl behind the others (the white bowl with blue rim in the foreground)...
[no coordinates]
```

**Result:** Only returns description, no coordinates.

## Parsing the Output

### Recommended Parser

```python
import re
import numpy as np

def parse_bbox(text: str) -> np.ndarray:
    """
    Parse bbox from Qwen3-VL output and convert to [0,1] normalized coords.
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
            raise ValueError(f"Invalid bbox: {bbox}")

        return bbox
    else:
        raise ValueError(f"Could not parse bbox from: {text}")
```

## Image Encoding

**Important:** Qwen3-VL requires RGB images.

```python
def image_to_base64(image: Image.Image) -> str:
    """Convert PIL image to base64 for Ollama."""
    # Ensure RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()
```

## Accuracy

Based on RefCOCO validation sample testing:

- **Ground Truth:** [732, 2, 1000, 273]
- **Predictions:**
  - JSON format: [734, 0, 999, 243] (error: 2-30 pixels)
  - Simple: [710, 0, 999, 285] (error: 12-22 pixels)
  - Coordinates: [726, 0, 999, 275] (error: 2-6 pixels)

**IoU range:** Typically 0.8-0.9 for correctly detected objects.

## Full Example

```python
import ollama_proxy as ollama
from PIL import Image
import numpy as np

# Load image
image = Image.open('image.jpg')

# Convert to base64
if image.mode != 'RGB':
    image = image.convert('RGB')

import io, base64
buffer = io.BytesIO()
image.save(buffer, format="PNG")
img_b64 = base64.b64encode(buffer.getvalue()).decode()

# Query model
expression = "red apple on the table"
prompt = f'Where is "{expression}" in this image? Output the bounding box in format: {{"bbox_2d": [x_min, y_min, x_max, y_max]}} using coordinates 0-1000.'

response = ollama.chat(
    model="qwen3-vl:8b",
    messages=[{
        'role': 'user',
        'content': prompt,
        'images': [img_b64],
    }]
)

# Parse response
import re
content = response['message']['content']
json_match = re.search(r'\{"bbox_2d"\s*:\s*\[([^\]]+)\]\}', content)
coords_str = json_match.group(1)
numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+', coords_str)

# Convert to [0,1]
bbox_1000 = np.array([float(n) for n in numbers[:4]])
bbox_normalized = bbox_1000 / 1000.0

print(f"Predicted bbox [0,1]: {bbox_normalized}")
# Output: [0.234 0.456 0.567 0.789]
```

## Troubleshooting

### Empty Responses

**Symptom:** Model returns empty string `""`

**Possible Causes:**
1. Image not in RGB mode → Convert with `image.convert('RGB')`
2. Image too large → Resize if needed
3. Ollama timeout → Increase timeout in options
4. Model not loaded → Check `ollama list`

### Invalid Coordinates

**Symptom:** Coordinates outside [0, 1000] or x1 >= x2

**Solutions:**
- Clip to [0, 1000] range
- Validate after parsing
- Check if model hallucinated (low confidence)

### Low Accuracy

**Symptom:** IoU < 0.3 consistently

**Solutions:**
- Improve expression clarity (e.g., "red bowl on left" vs "bowl")
- Try different prompt format
- Check if object is actually in image
- Consider model limitations (small objects, occlusion)

## References

- [Qwen3-VL Issue #1486](https://github.com/QwenLM/Qwen3-VL/issues/1486) - Bounding box format
- [Qwen3-VL Issue #1927](https://github.com/QwenLM/Qwen3-VL/issues/1927) - Coordinate mapping
- [Qwen3-VL GitHub](https://github.com/QwenLM/Qwen3-VL) - Official repository

## Testing

Run `test_qwen_grounding.py` to verify different prompting strategies on your setup.

Expected output:
```
json_format    :   31 chars, has_numbers=True, has_bbox_keyword=True
  First 100 chars: {"bbox_2d": [734, 0, 999, 243]}...
```
