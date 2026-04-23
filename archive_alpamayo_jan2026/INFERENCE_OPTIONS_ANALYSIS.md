# Alpamayo-R1 Inference Options Analysis

**Date:** 2026-01-16
**System:** Apple M1 Max (64GB unified memory, 24-core GPU)

## Executive Summary

Alpamayo-R1 is a PyTorch model requiring:
- 22GB model weights
- 24GB+ VRAM for inference
- Originally designed for NVIDIA CUDA GPUs

We have **three options** to run it on M1 Max:

| Option | Effort | Speed | Viability |
|--------|--------|-------|-----------|
| **MPS (Metal)** | Low | Medium | **Testing now** |
| **MLX** | Medium | Fast | Requires porting |
| **Cloud GPU** | None | Fastest | Immediate |

## Option 1: PyTorch MPS Backend ⚡ (Testing)

### What is MPS?
- Metal Performance Shaders: Apple's GPU acceleration framework
- Built into PyTorch 2.0+
- Enables running PyTorch models on Apple Silicon GPUs

### Compatibility Analysis

**✓ Working:**
- PyTorch MPS is available and built
- Basic tensor operations (matrix multiplication, bfloat16)
- Most transformer operations

**? To Test:**
- Flash attention → Can disable, use SDPA instead
- Custom trajectory tokenizer
- Diffusion model operations
- Full inference pipeline

**Known MPS Limitations:**
1. Some ops not implemented (will fall back to CPU)
2. No `torch.autocast('mps')` yet (use CPU autocast)
3. Memory management less efficient than CUDA

### Implementation Changes Required

**Minimal:**
```python
# Instead of:
model = AlpamayoR1.from_pretrained(
    "nvidia/Alpamayo-R1-10B",
    dtype=torch.bfloat16,
).to("cuda")

# Use:
model = AlpamayoR1.from_pretrained(
    "nvidia/Alpamayo-R1-10B",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",  # Disable flash-attn
).to("mps")

# And change device strings:
device = "mps"  # instead of "cuda"
```

**No code changes needed in:**
- Data loading pipeline
- Perturbation framework
- Analysis code

### Memory Estimate

```
Model weights:     22 GB
Activations:       ~8 GB (with batch_size=1, num_traj_samples=1)
Working memory:    ~2 GB
Total:             ~32 GB
Available:         64 GB (M1 Max unified memory)
```

**✓ Should fit comfortably**

### Testing Script

Created: `test_mps_inference.py`
- Tests MPS availability
- Loads model with SDPA attention
- Runs minimal inference
- Reports any compatibility issues

### Expected Outcome

**Best case:** Full inference works, 10-20% slower than CUDA
**Likely case:** Inference works with some ops falling back to CPU
**Worst case:** Some operations fail, need workarounds

### Next Steps

1. Run `python test_mps_inference.py`
2. If successful → Use MPS for all experiments
3. If issues → Document failures, decide on MLX vs Cloud

---

## Option 2: MLX Framework 🍎 (Requires Porting)

### What is MLX?
- Apple's ML framework optimized for Apple Silicon
- Similar to JAX/PyTorch but designed for unified memory architecture
- Often faster than MPS for transformer models

### Porting Effort Required

**Core Model (~2-3 days):**
- Convert Qwen3VL backbone (largest component)
- Replace PyTorch ops with MLX equivalents
- Most ops have 1:1 mappings

**Custom Components (~1-2 days):**
- Trajectory tokenizer
- Diffusion sampler
- Action space encoder

**Testing & Debugging (~1-2 days):**
- Verify numerical equivalence
- Fix edge cases
- Benchmark performance

**Total estimate: 4-7 days of focused work**

### MLX Advantages

**Performance:**
- Optimized for unified memory (no CPU↔GPU transfers)
- Better memory efficiency than MPS
- 1.5-2× faster than MPS for transformers

**Development:**
- Cleaner API than PyTorch
- Better debugging (eager execution)
- Growing ecosystem (mlx-lm, mlx-vlm)

### MLX Disadvantages

**Compatibility:**
- New framework (less mature)
- Some PyTorch features missing
- Need to maintain separate codebase

**Ecosystem:**
- Smaller community
- Fewer pre-trained models
- May need custom implementations

### Implementation Strategy

If we go this route:

1. **Use existing MLX models as reference:**
   - mlx-vlm has Qwen2-VL support
   - Adapt to Qwen3-VL architecture
   - Reuse vision encoder logic

2. **Port in stages:**
   - Stage 1: VLM backbone only
   - Stage 2: Add trajectory components
   - Stage 3: Full pipeline with diffusion

3. **Validate numerically:**
   - Compare outputs with PyTorch version
   - Ensure <1e-5 difference in predictions

### Useful Resources

- MLX Examples: https://github.com/ml-explore/mlx-examples
- MLX VLM: https://github.com/Blaizzy/mlx-vlm
- Conversion guide: Convert PyTorch → MLX

### When to Choose MLX

- ✓ If MPS fails or is too slow
- ✓ If doing extensive local development
- ✓ If want best M1 performance
- ✗ If need results quickly (use cloud instead)

---

## Option 3: Cloud GPU ☁️ (Zero Porting)

### Fastest Path to Results

**Advantages:**
- Zero code changes
- Works immediately
- Faster inference than M1 (2-3×)
- Easy to scale

**Options:**

| Provider | GPU | VRAM | $/hour | Setup Time |
|----------|-----|------|--------|------------|
| Lambda Labs | H100 | 80GB | $2.49 | 5 min |
| Colab Pro+ | A100 | 40GB | $50/mo | 2 min |
| Vast.ai | RTX 4090 | 24GB | $0.30 | 10 min |
| AWS p3.2xlarge | V100 | 16GB | $3.06 | 15 min |

**Recommended: Lambda Labs H100**
- Best price/performance
- 80GB VRAM (can run multiple samples)
- No setup complexity
- Pay per minute

### Cost Estimate

**For thesis experiments:**
```
Model loading:        5 min × $2.49/hr = $0.21
Per inference:        ~30 sec
100 samples:          50 min = $2.08
Perturbation testing: 500 inferences = ~$10
Total thesis budget:  ~$50-100
```

**Very affordable for research.**

### Workflow

1. **Local:** Data prep, perturbation design, analysis
2. **Cloud:** Model inference only
3. **Local:** Results analysis, visualization

**Hybrid approach:**
- Develop locally on M1
- Run batches on cloud GPU
- Download results for analysis

### When to Choose Cloud

- ✓ Need results immediately
- ✓ Don't want to debug compatibility
- ✓ Willing to spend $50-100
- ✗ Doing extensive iteration (MPS better)

---

## Comparison Matrix

| Criterion | MPS | MLX | Cloud GPU |
|-----------|-----|-----|-----------|
| **Setup time** | 1 hour | 1 week | 15 min |
| **Code changes** | Minimal | Extensive | None |
| **Iteration speed** | Fast | Fast | Slow (upload/download) |
| **Cost** | Free | Free | $50-100 |
| **Performance** | Medium | Fast | Fastest |
| **Risk** | Medium | Low | None |
| **Learning value** | Medium | High | None |

## Recommendation

### Phase 1: Test MPS (Today)
Run `test_mps_inference.py` to see if MPS works.

**If successful:**
- Use MPS for all local development
- Fast iteration, zero cost
- Good enough for thesis

**If fails:**
- Document specific errors
- Decide between MLX vs Cloud

### Phase 2: Production Runs
For final thesis experiments:
- Use cloud GPU for large batches
- Ensures reproducibility
- No hardware dependencies in thesis

### Phase 3: (Optional) MLX Port
If doing extensive future work:
- Port to MLX for best M1 performance
- Contribute to mlx-vlm ecosystem
- Learn new framework

---

## Detailed MPS Compatibility Check

### Core Dependencies

**Transformers (Qwen3VL):**
- ✓ Attention: SDPA works on MPS
- ✓ Linear layers: Standard PyTorch
- ✓ LayerNorm: MPS-compatible
- ? Vision encoder: Should work
- ? RoPE embeddings: Should work

**Custom Components:**
- ? Trajectory tokenizer (VQ-VAE style)
- ? Diffusion sampler (DDPM/DDIM)
- ? Action space (custom operations)

**Known MPS Gaps:**
```python
# These may fail on MPS:
torch.complex64/128 ops  # → Use float workarounds
torch.linalg.eigh        # → May fall back to CPU
Some scatter ops         # → Check implementation
```

### Fallback Strategy

If specific ops fail:
1. Let them run on CPU (automatic fallback)
2. Profile to find bottlenecks
3. Optimize hot paths only

Most models work with ~5-10% of ops on CPU.

---

## Action Plan

### Immediate (Today):
```bash
# Test MPS inference
python test_mps_inference.py
```

**Expected runtime:** 10-15 minutes (first run downloads 22GB)

### If MPS works:
1. Update `alpamayo_minimal_inference.py` to use MPS
2. Run first real inference
3. Examine CoC reasoning format
4. Implement perturbation pipeline

### If MPS fails:
1. Document error messages
2. Decide: MLX port (1 week) vs Cloud GPU (today)
3. My recommendation: Cloud GPU for thesis timeline

### Long-term:
- Consider MLX port for future projects
- Contribute findings to mlx-vlm community

---

## Testing Checklist

- [ ] MPS availability confirmed
- [ ] Model loads to MPS
- [ ] Forward pass works
- [ ] Inference completes
- [ ] Reasoning trace extracted
- [ ] Trajectory predicted
- [ ] Results match CUDA baseline (if available)
- [ ] Memory usage acceptable (<64GB)
- [ ] Inference speed acceptable (<5min per sample)

---

## References

- PyTorch MPS: https://pytorch.org/docs/stable/notes/mps.html
- MLX: https://github.com/ml-explore/mlx
- MLX VLM: https://github.com/Blaizzy/mlx-vlm
- Transformers MPS support: https://huggingface.co/docs/transformers/perf_infer_gpu_one

---

**Status:** Ready to test MPS inference
**Next:** Run `python test_mps_inference.py`
