# Testing Status & Next Steps

**Last Updated:** 2026-01-11

## Summary

We validated that Qwen3-VL can perform referring expression grounding via Ollama and identified the correct prompting format.

## Key Findings

### ✅ Qwen3-VL Grounding Works!

**Test Results (RefCOCO sample 0):**
- Ground Truth: `[732, 2, 1000, 273]` (in [0,1000] range)
- Prediction: `[734, 0, 999, 243]` (in [0,1000] range)
- **Error:** 2-30 pixels (very accurate!)
- **Format:** Returns clean JSON: `{"bbox_2d": [x, y, x, y]}`

### ✅ Coordinate System Confirmed

- **Input/Output:** [0, 1000] range
- **Conversion:** Divide by 1000 to get [0, 1] normalized
- **No special tokens needed** - coordinates are plain text

### ✅ Best Prompting Strategy

```python
prompt = f'Where is "{expression}" in this image? Output the bounding box in format: {{"bbox_2d": [x_min, y_min, x_max, y_max]}} using coordinates 0-1000.'
```

**Response:**
```json
{"bbox_2d": [734, 0, 999, 243]}
```

### ✅ Image Encoding Fixed

**Requirement:** Images must be RGB mode

```python
if image.mode != 'RGB':
    image = image.convert('RGB')
```

## Files Updated

All files now use the correct format:

1. ✅ **pilot_test.py** - Pilot test script (9 samples)
   - RGB conversion added
   - Correct [0,1000] parsing
   - Debug output on first response

2. ✅ **02_threshold_optimization.ipynb** - Threshold optimization (51 samples)
   - RGB conversion added
   - Correct prompting format
   - Proper bbox parsing

3. ✅ **03_vlm_pipeline.ipynb** - Validation pipeline (99 samples)
   - RGB conversion added
   - Correct prompting format
   - Proper bbox parsing

4. ✅ **test_qwen_grounding.py** - Manual testing script
   - Tests 5 different prompt formats
   - Shows raw responses
   - Validates assumptions

5. ✅ **QWEN3VL_GROUNDING.md** - Complete documentation
   - All prompting strategies
   - Parsing code
   - Troubleshooting guide
   - Full examples

## Test Results

### Manual Test (test_qwen_grounding.py)

Tested 5 prompting strategies:

| Prompt Type | Coords Returned | Accuracy | Parsability |
|-------------|----------------|----------|-------------|
| json_format | ✅ [734, 0, 999, 243] | Best | Easy |
| simple | ✅ [710, 0, 999, 285] | Good | Medium |
| coordinates | ✅ [726, 0, 999, 275] | Good | Hard |
| grounding | ❌ Description only | N/A | N/A |
| detection | ❌ Description only | N/A | N/A |

**Conclusion:** JSON format is optimal.

## Next Steps

### 1. Run Pilot Test

```bash
cd /Users/kaiser/Desktop/Uni/Masterarbeit/notebooks
python pilot_test.py
```

**Purpose:**
- Verify grounding works on 9 samples
- Test perturbation sensitivity
- Validate boundary detection
- Check runtime (~10-15 minutes)

**Success Criteria:**
- VLM success rate ≥30% (IoU > 0.5)
- Perturbation sensitivity 0.01-0.2
- Boundary rate 10-70%
- All difficulty levels present

### 2. If Pilot Passes → Run Threshold Optimization

Open and run: `02_threshold_optimization.ipynb`

**Purpose:**
- Optimize thresholds on 51 samples
- Use 5×5 perturbation grid (25 per sample)
- Validate MI improvement >5%
- Runtime: ~60 minutes

### 3. If Optimization Succeeds → Run Validation

Open and run: `03_vlm_pipeline.ipynb`

**Purpose:**
- Full validation on 99 samples
- Use 10×10 perturbation grid (100 per sample)
- Dual-space boundary detection
- Runtime: ~8-10 hours

## Current Status

- ✅ Qwen3-VL format validated
- ✅ All code updated
- ✅ Documentation complete
- ✅ All fixes tested and verified
- ✅ Fixes applied to production files
- ✅ Ready to run pilot test

## Known Issues

### Resolved
- ❌ ~~Empty responses~~ → Fixed: Removed num_predict limit in ollama_proxy
- ❌ ~~Wrong coordinate range~~ → Fixed: [0,1000] format with conversion
- ❌ ~~No coordinates returned~~ → Fixed: Correct JSON prompt
- ❌ ~~False positive IoU from default bbox~~ → Fixed: Zero-IoU default [0,0,0,0]
- ❌ ~~Poor error logging~~ → Fixed: First 200 chars logged, only first 3 failures shown
- ❌ ~~No statistics tracking~~ → Fixed: Added parse_failures, total_calls, print_stats()

### Potential Issues
- ⚠️ Runtime may be longer than expected (Qwen3-VL 8b is slower than expected)
- ⚠️ Some expressions may be ambiguous (use first expression only)
- ⚠️ Small objects (hard difficulty) may have lower accuracy

## Verification Tests

### test_vlm_fixes.py (4/4 passed ✅)
- Zero-IoU default returns IoU=0
- Ollama proxy settings verified
- Invalid bbox handling tested
- Real sample predictions (100% success, IoU: 0.919, 0.978, 0.846)

### test_model_behavior.py (4/4 passed ✅)
- API call structure verified
- Response format consistency: 100% JSON (10/10 samples)
- Perturbation handling: 100% success (5/5 perturbations)
- Timeout protection verified

## References

- `QWEN3VL_GROUNDING.md` - Complete grounding documentation
- `test_qwen_grounding.py` - Manual testing script
- `test_vlm_fixes.py` - Fix verification tests
- `test_model_behavior.py` - Model behavior tests
- `FIXES_APPLIED.md` - Complete documentation of applied fixes
- [Qwen3-VL GitHub Issues](https://github.com/QwenLM/Qwen3-VL/issues)

## Experiment Plan

| Phase | Samples | Perturbations | Inferences | Runtime | Status |
|-------|---------|---------------|------------|---------|--------|
| Pilot | 9 | 3×3 (9) | 90 | ~15 min | Ready |
| Optimization | 51 | 5×5 (25) | 1,326 | ~60 min | Ready |
| Validation | 99 | 10×10 (100) | 9,999 | ~8-10 hrs | Ready |

**Total Runtime:** ~10 hours for full pipeline
