"""
Test MPS (Metal Performance Shaders) Inference Feasibility

This script tests whether Alpamayo-R1 can run on Apple Silicon using PyTorch's MPS backend.

Strategy:
1. Load model config without flash-attn (use SDPA instead)
2. Try loading model to MPS device
3. Run minimal inference test
4. Document any compatibility issues
"""

import sys
sys.path.insert(0, 'tools/alpamayo/src')

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Workaround for OpenMP issue

import torch

print("=" * 80)
print("MPS INFERENCE FEASIBILITY TEST")
print("=" * 80)

# Check MPS availability
print("\n1. Checking MPS Backend:")
print(f"   PyTorch version: {torch.__version__}")
print(f"   MPS built: {torch.backends.mps.is_built()}")
print(f"   MPS available: {torch.backends.mps.is_available()}")

if not torch.backends.mps.is_available():
    print("\n✗ MPS not available. Cannot proceed.")
    sys.exit(1)

print("\n✓ MPS is available!")

# Test basic operations
print("\n2. Testing Basic MPS Operations:")
try:
    x = torch.randn(100, 100).to('mps')
    y = torch.randn(100, 100).to('mps')
    z = torch.matmul(x, y)
    print(f"   ✓ Matrix multiplication: {z.shape}")

    # Test bfloat16
    x_bf16 = x.to(torch.bfloat16)
    y_bf16 = y.to(torch.bfloat16)
    z_bf16 = torch.matmul(x_bf16, y_bf16)
    print(f"   ✓ bfloat16 operations: {z_bf16.dtype}")
except Exception as e:
    print(f"   ✗ Basic operations failed: {e}")
    sys.exit(1)

# Try loading the model
print("\n3. Attempting to Load Alpamayo-R1 Model:")
print("   Note: This will download 22GB of weights on first run")
print("   Using SDPA instead of flash-attention for MPS compatibility")

try:
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from transformers import AutoConfig

    # First, try to load config
    print("\n   Loading model config...")

    # We need to override attn_implementation
    model_id = "nvidia/Alpamayo-R1-10B"

    # Load with eager mode (standard PyTorch attention)
    print("   Loading model with eager attention...")
    model = AlpamayoR1.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        attn_implementation="eager",  # Use standard PyTorch attention
        device_map="mps",  # Try automatic device mapping to MPS
    )

    print("   ✓ Model loaded!")
    print(f"   Model device: {next(model.parameters()).device}")
    print(f"   Model dtype: {next(model.parameters()).dtype}")

except Exception as e:
    print(f"\n   ✗ Model loading failed: {e}")
    print("\n   Trying manual device placement...")

    try:
        # Try loading to CPU first, then move
        model = AlpamayoR1.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        print("   ✓ Model loaded to CPU")

        print("   Moving to MPS...")
        model = model.to('mps')
        print("   ✓ Model moved to MPS!")

    except Exception as e2:
        print(f"   ✗ Also failed: {e2}")
        print("\n" + "=" * 80)
        print("DIAGNOSIS: MPS may not support all operations")
        print("=" * 80)
        print("\nPossible issues:")
        print("  1. Some transformer operations not implemented in MPS")
        print("  2. Model size exceeds unified memory (64GB on M1 Max)")
        print("  3. Specific layer types not MPS-compatible")
        print("\nRecommendations:")
        print("  → Use cloud GPU (fastest path forward)")
        print("  → Or implement MLX conversion (more effort)")
        sys.exit(1)

# Try simple inference
print("\n4. Testing Inference:")
try:
    from alpamayo_r1 import load_physical_aiavdataset
    from alpamayo_r1 import helper

    CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
    T0_US = 5_100_000

    print(f"   Loading data for clip {CLIP_ID}...")
    data = load_physical_aiavdataset(CLIP_ID, t0_us=T0_US)

    print("   Preparing inputs...")
    processor = helper.get_processor(model.tokenizer)
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )

    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, "mps")

    print("   Running inference...")
    with torch.autocast("cpu", dtype=torch.bfloat16):  # MPS doesn't support autocast yet
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=256,
            return_extra=True,
        )

    print("\n✓ INFERENCE SUCCESSFUL!")
    print(f"   Predicted trajectory shape: {pred_xyz.shape}")
    print(f"   Reasoning trace: {extra['cot'][0][0][:100]}...")

    print("\n" + "=" * 80)
    print("SUCCESS: MPS inference is working!")
    print("=" * 80)

except Exception as e:
    print(f"\n✗ Inference failed: {e}")
    import traceback
    traceback.print_exc()

    print("\n" + "=" * 80)
    print("PARTIAL SUCCESS: Model loads but inference has issues")
    print("=" * 80)
    print("\nThe error above may indicate:")
    print("  1. Specific operations not MPS-compatible")
    print("  2. Memory issues (try reducing batch size)")
    print("  3. Missing MPS implementations for custom layers")

    sys.exit(1)
