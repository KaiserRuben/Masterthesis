# Alpamayo-R1 Experiment Setup - Status Report

**Date:** 2026-01-16
**Status:** Step 2 Complete ✓

## Completed Tasks

### ✓ Step 1: Environment Setup

1. **Repository Cloned**
   - Location: `alpamayo/`
   - Branch: main
   - Source: https://github.com/NVlabs/alpamayo

2. **Dependencies Installed**
   - Core packages: torch, transformers, einops, hydra-core, av
   - Dataset loader: physical_ai_av
   - HuggingFace authentication: ✓ (KaiserRuben)

3. **Dataset Access**
   - Physical AI AV Dataset: ✓ Access granted
   - Sample data downloaded: ✓
   - Clip ID tested: `030c760c-ae38-49aa-9ad8-f5650a545d26`

### ✓ Step 2: Basic Inference Pipeline

1. **Data Loading** (`alpamayo_data_exploration.py`)
   - Successfully loads multi-camera images
   - Extracts ego trajectory (history + future)
   - Visualization created: `alpamayo_data_exploration.png`

2. **Data Format Understanding**
   ```
   Inputs:
   - image_frames: (4 cameras, 4 frames, 3 channels, 1080×1920)
   - ego_history_xyz: (1, 1, 16 steps, 3) — 1.6s @ 10Hz
   - ego_history_rot: (1, 1, 16 steps, 3×3) — rotation matrices

   Outputs (Expected):
   - Reasoning trace: Chain-of-Causation (natural language)
   - Predicted trajectory: (1, num_samples, 64 waypoints, 3) — 6.4s @ 10Hz
   ```

3. **Inference Template** (`alpamayo_minimal_inference.py`)
   - Complete pipeline documented
   - Reasoning parser scaffold
   - Trajectory analyzer with meta-action inference
   - Consistency checker framework

## Hardware Limitation Identified

**Issue:** Apple M1 Max (no NVIDIA GPU)
- Alpamayo-R1 requires NVIDIA GPU with 24GB+ VRAM
- Uses CUDA-specific operations (flash-attention, bfloat16)

**Resolution Options:**

1. **Cloud GPU** (Recommended for quick testing)
   - Lambda Labs (H100, A100)
   - Google Colab Pro (A100)
   - AWS/GCP (on-demand instances)

2. **MLX Conversion** (For local development)
   - Port PyTorch model to Apple MLX
   - Requires significant code adaptation
   - Benefits: Local inference, faster iteration

3. **Remote Jupyter** (Hybrid approach)
   - Keep data exploration local
   - Run inference on remote GPU
   - Sync results back

## Data Exploration Results

### Scene Analysis
- **Environment:** Urban construction zone
- **Ego Behavior:** Decelerating (9.5 → 6.5 m/s)
- **Scenario:** Navigating around construction equipment
- **Duration:** 6.4s future prediction

### Camera Views
1. **Cross Left 120°:** Left-side peripheral vision
2. **Front Wide 120°:** Main forward view (construction visible)
3. **Cross Right 120°:** Right-side peripheral vision
4. **Front Tele 30°:** Narrow forward focus

### Trajectory Characteristics
- **History displacement:** 13.57m (1.6s)
- **Future displacement:** 46.64m (6.4s)
- **Behavior:** Straight path, decelerating

## Next Steps (Step 3+)

### Step 3: Reasoning-Action Consistency Check

**Required:**
- [ ] Run inference to get actual CoC trace
- [ ] Implement reasoning parser (NLP-based)
- [ ] Map reasoning → intended meta-actions
- [ ] Compare with trajectory-inferred actions

**Key Questions:**
1. What format does the CoC trace use?
2. How explicit are meta-action statements?
3. Can we use LLM to parse reasoning?

### Step 4: Simple Perturbation Test

**Perturbation Candidates:**

1. **Occlusion** (Easiest to implement)
   - Black out regions of camera views
   - Test: Does reasoning mention occluded objects?
   - Expected: Consistency maintained if object not critical

2. **Object Manipulation** (More complex)
   - Remove construction equipment from scene
   - Test: Does model still decelerate?
   - Expected: Reasoning-action mismatch

3. **Weather Simulation**
   - Add synthetic rain/fog to images
   - Test: Safety margin changes
   - Expected: More conservative behavior

**Implementation Strategy:**
```python
def apply_perturbation(data, perturbation_type, **kwargs):
    """
    Apply semantic perturbation to input data.

    Args:
        data: Original data dict from load_physical_aiavdataset
        perturbation_type: "occlusion", "object_removal", "weather"
        **kwargs: Perturbation-specific parameters

    Returns:
        Modified data dict (same structure)
    """
    perturbed_data = data.copy()

    if perturbation_type == "occlusion":
        # Modify image_frames
        mask = create_occlusion_mask(**kwargs)
        perturbed_data["image_frames"] = apply_mask(
            data["image_frames"], mask
        )

    return perturbed_data
```

## Research Questions

Based on initial exploration, we can test:

### RQ1: Boundary Existence
*Do semantic perturbations create localizable decision boundaries?*

**Hypothesis:** Unlike pixel-space adversarial examples (which are everywhere), semantic perturbations (object removal, occlusion) may have structured boundaries.

**Test:** Apply graded perturbations (e.g., occlusion coverage: 0%, 25%, 50%, 75%, 100%) and measure:
- Reasoning consistency (does reasoning mention occluded object?)
- Action consistency (does trajectory change appropriately?)
- Boundary location (at what threshold does mismatch occur?)

### RQ2: Reasoning Quality
*Does the CoC reasoning accurately reflect decision factors?*

**Test Cases:**
- Remove critical object → Does reasoning still mention it?
- Add spurious object → Does reasoning over-react?
- Occlude non-critical region → Does reasoning ignore it appropriately?

### RQ3: Multi-Modal Robustness
*How do vision and language modalities interact under perturbation?*

**Perturbation Matrix:**
```
                Vision Intact    Vision Perturbed
Prompt Intact      Baseline       Vision-only test
Prompt Perturbed   Language test  Joint failure mode
```

## Files Created

1. **`alpamayo_data_exploration.py`**
   - Loads and visualizes dataset
   - No GPU required
   - Output: `alpamayo_data_exploration.png`

2. **`alpamayo_minimal_inference.py`**
   - Template for full inference pipeline
   - Documents expected outputs
   - Requires GPU to run

3. **`ALPAMAYO_SETUP_STATUS.md`** (this file)
   - Progress documentation
   - Next steps planning

## Resources

- Model: https://huggingface.co/nvidia/Alpamayo-R1-10B
- Dataset: https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles
- Paper: https://arxiv.org/abs/2511.00088
- Code: https://github.com/NVlabs/alpamayo

## Decision Points

### Immediate (Required to proceed)
- [ ] Choose GPU solution (cloud vs MLX vs remote)
- [ ] Run first inference to see actual CoC format
- [ ] Decide on reasoning parser approach (rule-based vs LLM)

### Short-term (Next 1-2 weeks)
- [ ] Implement perturbation framework
- [ ] Define consistency metrics
- [ ] Test on 5-10 diverse scenarios

### Long-term (Thesis scope)
- [ ] Scale to hundreds of samples
- [ ] Map perturbation-boundary relationships
- [ ] Compare with VLM boundary findings (from your previous work)

## Notes

- **VLM Boundary Context:** You previously found that VLM boundaries are ubiquitous in pixel space, with no localized structure. The VLA setup offers a new angle: semantic perturbations on a structured task (driving).

- **Key Difference:** Unlike vision-only models, VLAs explicitly reason about their actions. This reasoning trace provides a "ground truth" for intended behavior, enabling consistency testing.

- **Thesis Angle:** If semantic boundaries ARE localizable (unlike pixel boundaries), this suggests that the right perturbation space matters. The framework you developed (boundary rate, clustering, gradient alignment) can be adapted here.

---

**Status:** Ready to proceed to Step 3 (requires GPU access decision)
