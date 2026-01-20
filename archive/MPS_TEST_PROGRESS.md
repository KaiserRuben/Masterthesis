# MPS Inference Test - Progress Report

**Date:** 2026-01-16
**Status:** In Progress - Model Downloading

## Test Results So Far

### ✅ Phase 1: MPS Availability - PASSED
- PyTorch 2.8.0 with MPS support confirmed
- MPS backend available and working
- Basic tensor operations successful

### ✅ Phase 2: Dependency Resolution - PASSED (with fixes)

**Issues encountered:**
1. `huggingface-hub` version mismatch
   - Required: `<1.0`
   - Had: `1.3.2`
   - **Fixed:** Downgraded to `0.36.0`

2. `transformers` version mismatch
   - Required: `4.57.1`
   - Had: `4.55.4`
   - **Fixed:** Upgraded to `4.57.1`

3. `tokenizers` updated
   - From: `0.21.4`
   - To: `0.22.2`

### 🔄 Phase 3: Model Loading - IN PROGRESS

**Status:** Downloading model weights from HuggingFace
- Model: `nvidia/Alpamayo-R1-10B`
- Size: ~22GB
- Download started successfully
- Using SDPA attention (not flash-attn) for MPS compatibility

**Current log:**
```
Fetching 5 files: 0%|          | 0/5 [00:00<?, ?it/s]
```

## Key Findings

### MPS Compatibility Looks Promising

**What's working:**
- MPS backend fully operational
- bfloat16 dtype supported
- Model download initiated (good sign - config was parseable)
- SDPA attention fallback accepted

**What required changes:**
- `attn_implementation="sdpa"` instead of `"flash_attention_2"`
- Package version alignment

**What's unknown (testing next):**
- Full model loading to MPS
- Forward pass compatibility
- Custom trajectory components (tokenizer, diffusion)
- Inference pipeline end-to-end

## Next Test Phases

Once download completes:

### Phase 4: Model-to-Device Transfer
- Load model to MPS device
- Check memory usage
- Verify all layers compatible

### Phase 5: Data Pipeline
- Load sample from dataset
- Tokenize inputs
- Move to MPS

### Phase 6: Full Inference
- Run forward pass
- Generate trajectory
- Extract reasoning trace

## Expected Timeline

```
Model download:     ~20-30 min (22GB, depends on connection)
Model load to MPS:  ~2-3 min
First inference:    ~5-10 min
Total:              ~30-45 min
```

## Implications

### If This Works (likely ~80% chance)

**Immediate benefits:**
- Zero cloud costs
- Fast iteration during development
- Can run experiments locally

**Workflow:**
1. Develop perturbations locally
2. Test on single samples with MPS
3. Scale to larger batches (still local)
4. Final validation on cloud GPU (optional)

**Estimated performance:**
- MPS inference: 30-120 sec/sample (estimated)
- vs CUDA (H100): 10-30 sec/sample
- Still acceptable for thesis work

### If Specific Ops Fail (moderate chance)

**Likely culprits:**
- Custom trajectory tokenizer (VQ-VAE operations)
- Diffusion sampler (may use unsupported ops)
- Specific transformer layers

**Mitigation:**
- Check which ops fail
- Many will auto-fallback to CPU
- Profile to find bottlenecks
- Optimize critical paths only

### If Complete Failure (unlikely ~10%)

**Fallback plan:**
- Cloud GPU for inference (Lambda Labs H100)
- Keep local development for everything else
- Budget: ~$50-100 for thesis

## Code Changes Made

### test_mps_inference.py

**Key configuration:**
```python
model = AlpamayoR1.from_pretrained(
    "nvidia/Alpamayo-R1-10B",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",  # Critical change
    device_map="mps",           # Or manual .to('mps')
)
```

**Device handling:**
```python
# Instead of:
device = "cuda"

# Use:
device = "mps"
```

**Autocast:**
```python
# MPS doesn't support torch.autocast('mps') yet
# Use CPU autocast or none:
with torch.autocast("cpu", dtype=torch.bfloat16):
    # or just:
with torch.no_grad():
```

## Comparison with Alternatives

| Approach | Status | Effort | Performance |
|----------|--------|--------|-------------|
| **MPS** | Testing | ✅ Minimal | Medium |
| MLX | Not started | High (1 week) | Fast |
| Cloud GPU | Ready | None | Fastest |

## Recommendations Based on Test Outcome

### If MPS works smoothly (>90% ops successful):
→ **Use MPS for entire thesis**
- Fast local iteration
- Zero costs
- Simpler workflow

### If MPS partially works (70-90% ops, some CPU fallback):
→ **Hybrid: MPS for dev + Cloud for final runs**
- Develop and debug on MPS
- Final experiments on H100
- Best of both worlds

### If MPS fails (<70% ops):
→ **Cloud GPU primary, consider MLX for future**
- Lambda Labs for immediate needs
- Document MPS limitations
- MLX port if doing extended work

## Technical Notes

### Memory Estimate (M1 Max 64GB)

```
Model weights (fp16/bf16):  ~22 GB
Activations (batch=1):      ~6-8 GB
Data (images + history):    ~2 GB
OS + other:                 ~10 GB
Total:                      ~40 GB

Available:                  64 GB
Margin:                     ~24 GB ✅
```

Should fit comfortably even with some overhead.

### Performance Expectations

**MPS vs CUDA estimates:**
- Attention layers: 1.5-2× slower (SDPA vs flash-attn)
- Linear layers: Similar
- Custom ops: Varies (may fallback to CPU)
- Overall: 2-4× slower than H100

**Acceptable for research:**
- H100: ~20 sec/sample → ~33 min for 100 samples
- MPS: ~60 sec/sample → ~100 min for 100 samples
- Still very doable for thesis timeline

## Current Test Status

**Waiting for:**
- Model weight download to complete (22GB)
- Once downloaded, will test:
  1. Model loads to MPS ✓/✗
  2. Forward pass works ✓/✗
  3. Inference completes ✓/✗
  4. Results extractable ✓/✗

**Estimated completion:**
- Download: ~15-25 min remaining
- Load test: ~5 min
- Inference test: ~10 min
- **Total: ~30-40 min from now**

---

**Status:** Download in progress, will update when Phase 3 completes

**Monitoring:**
```bash
tail -f /private/tmp/claude/-Users-kaiser-Projects-Masterarbeit/tasks/bb758c1.output
```
