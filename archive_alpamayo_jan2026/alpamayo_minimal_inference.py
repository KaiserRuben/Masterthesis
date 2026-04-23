"""
Alpamayo-R1 Minimal Inference Template

This script demonstrates the complete inference pipeline for Alpamayo-R1.

STATUS: Template only - requires NVIDIA GPU with 24GB+ VRAM to run
For M1/M2/M3 Macs: Consider MLX conversion or use cloud GPU

OUTPUTS:
1. Reasoning trace (Chain-of-Causation): Natural language explanation of driving decision
2. Predicted trajectory: 64 waypoints (6.4s @ 10Hz)

For full details, see: alpamayo/src/alpamayo_r1/test_inference.py
"""

import sys
sys.path.insert(0, 'archive/alpamayo/src')

import torch
import numpy as np
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import load_physical_aiavdataset
from alpamayo_r1 import helper

# Configuration
CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
T0_US = 5_100_000
NUM_TRAJ_SAMPLES = 1  # Number of trajectory samples to generate
DEVICE = "cuda"  # Change to "mps" for Apple Silicon with MLX (if ported)

def check_gpu():
    """Check if compatible GPU is available."""
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available")
        print("Options:")
        print("  1. Use cloud GPU (Lambda Labs, AWS, GCP)")
        print("  2. Convert model to MLX for Apple Silicon")
        print("  3. Continue as template (no inference)")
        return False

    # Check VRAM
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {gpu_mem_gb:.1f} GB")

    if gpu_mem_gb < 24:
        print("WARNING: Model requires 24GB+ VRAM")
        print(f"Available: {gpu_mem_gb:.1f}GB")
        return False

    return True


def load_model():
    """Load Alpamayo-R1 model from HuggingFace."""
    print("\nLoading model from HuggingFace...")
    print("Note: First run downloads 22GB of model weights")

    model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B",
        dtype=torch.bfloat16,
        # attn_implementation="sdpa"  # Uncomment if flash-attn issues
    ).to(DEVICE)

    processor = helper.get_processor(model.tokenizer)

    print("✓ Model loaded successfully")
    return model, processor


def prepare_inputs(data, processor):
    """Prepare data for model inference."""
    # Create message from image frames
    # Shape: (N_cameras * num_frames, 3, H, W)
    messages = helper.create_message(data["image_frames"].flatten(0, 1))

    # Tokenize inputs
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )

    # Prepare model inputs with ego history
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }

    # Move to device
    model_inputs = helper.to_device(model_inputs, DEVICE)

    return model_inputs


def run_inference(model, model_inputs):
    """Run inference to get reasoning trace and trajectory prediction."""
    print("\nRunning inference...")

    # Set random seed for reproducibility
    torch.cuda.manual_seed_all(42)

    with torch.autocast(DEVICE, dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=NUM_TRAJ_SAMPLES,
            max_generation_length=256,
            return_extra=True,
        )

    return pred_xyz, pred_rot, extra


def parse_reasoning_trace(cot_trace):
    """
    Parse the Chain-of-Causation reasoning trace.

    The CoC trace is a natural language explanation of the driving decision.
    We need to extract:
    1. Stated driving intent (e.g., "maintain speed", "decelerate", "lane change")
    2. Reasoning about scene (e.g., "construction ahead", "pedestrian detected")
    3. Safety considerations

    Returns:
        dict: Parsed reasoning components
    """
    # TODO: Implement reasoning parser
    # For now, just return the raw trace
    return {
        "raw_trace": cot_trace,
        "stated_intent": None,  # Extract from trace
        "scene_description": None,  # Extract from trace
        "safety_considerations": None,  # Extract from trace
    }


def analyze_trajectory(pred_xyz):
    """
    Analyze predicted trajectory to infer actual behavior.

    Maps trajectory to meta-actions:
    Longitudinal:
    - gentle_accelerate, strong_accelerate
    - gentle_decelerate, strong_decelerate
    - maintain_speed, stop, reverse

    Lateral:
    - go_straight, steer_left, steer_right
    - sharp_steer_left, sharp_steer_right

    Args:
        pred_xyz: Predicted trajectory (batch, num_traj_samples, 64, 3)

    Returns:
        dict: Inferred meta-actions
    """
    # Extract first trajectory sample
    traj = pred_xyz[0, 0].cpu().numpy()  # (64, 3)

    # Compute velocities
    dt = 0.1  # 10Hz
    velocities = np.linalg.norm(np.diff(traj, axis=0), axis=1) / dt

    # Compute accelerations
    accelerations = np.diff(velocities) / dt

    # Compute lateral displacement
    lateral_displacements = traj[:, 1]  # Y-axis in ego frame

    # Infer longitudinal behavior
    mean_acc = np.mean(accelerations)
    if abs(mean_acc) < 0.5:
        longitudinal = "maintain_speed"
    elif mean_acc > 2.0:
        longitudinal = "strong_accelerate"
    elif mean_acc > 0.5:
        longitudinal = "gentle_accelerate"
    elif mean_acc < -2.0:
        longitudinal = "strong_decelerate"
    elif mean_acc < -0.5:
        longitudinal = "gentle_decelerate"
    else:
        longitudinal = "unknown"

    # Infer lateral behavior
    final_lateral = lateral_displacements[-1]
    if abs(final_lateral) < 0.5:
        lateral = "go_straight"
    elif final_lateral > 2.0:
        lateral = "sharp_steer_left"
    elif final_lateral > 0.5:
        lateral = "steer_left"
    elif final_lateral < -2.0:
        lateral = "sharp_steer_right"
    elif final_lateral < -0.5:
        lateral = "steer_right"
    else:
        lateral = "unknown"

    return {
        "longitudinal": longitudinal,
        "lateral": lateral,
        "mean_velocity": np.mean(velocities),
        "mean_acceleration": mean_acc,
        "final_lateral_displacement": final_lateral,
    }


def check_reasoning_action_consistency(reasoning, trajectory_analysis):
    """
    Check if the reasoning trace is consistent with the predicted trajectory.

    This is the core of the VLA testing framework:
    - Does the stated intent match the actual behavior?
    - Are there contradictions between reasoning and action?

    Args:
        reasoning: Parsed reasoning trace
        trajectory_analysis: Inferred meta-actions from trajectory

    Returns:
        dict: Consistency analysis
    """
    # TODO: Implement consistency checker
    # This requires:
    # 1. Mapping reasoning text to intended meta-actions
    # 2. Comparing with inferred meta-actions
    # 3. Detecting mismatches

    return {
        "is_consistent": None,  # To be implemented
        "mismatches": [],  # List of detected inconsistencies
        "confidence": None,  # Confidence in consistency assessment
    }


def main():
    """Main inference pipeline."""
    print("=" * 80)
    print("ALPAMAYO-R1 MINIMAL INFERENCE PIPELINE")
    print("=" * 80)

    # Check GPU availability
    has_gpu = check_gpu()
    if not has_gpu:
        print("\n" + "=" * 80)
        print("TEMPLATE MODE: GPU not available")
        print("This script documents the inference pipeline for reference")
        print("=" * 80)
        return

    # Load data
    print(f"\nLoading data for clip: {CLIP_ID}")
    data = load_physical_aiavdataset(CLIP_ID, t0_us=T0_US)
    print("✓ Data loaded")

    # Load model
    model, processor = load_model()

    # Prepare inputs
    model_inputs = prepare_inputs(data, processor)

    # Run inference
    pred_xyz, pred_rot, extra = run_inference(model, model_inputs)

    # Extract outputs
    print("\n" + "=" * 80)
    print("OUTPUTS")
    print("=" * 80)

    # 1. Reasoning trace
    cot_traces = extra["cot"][0]  # List of traces (one per trajectory sample)
    print(f"\nChain-of-Causation Reasoning ({len(cot_traces)} samples):")
    for i, trace in enumerate(cot_traces):
        print(f"\nSample {i+1}:")
        print(f"  {trace}")

    # 2. Trajectory
    print(f"\nPredicted Trajectory:")
    print(f"  Shape: {pred_xyz.shape}")  # (batch, num_traj_samples, 64, 3)
    print(f"  Start: {pred_xyz[0, 0, 0].cpu().numpy()}")
    print(f"  End: {pred_xyz[0, 0, -1].cpu().numpy()}")

    # Parse reasoning
    print("\n" + "=" * 80)
    print("REASONING ANALYSIS")
    print("=" * 80)

    reasoning = parse_reasoning_trace(cot_traces[0])
    for key, value in reasoning.items():
        print(f"  {key}: {value}")

    # Analyze trajectory
    print("\n" + "=" * 80)
    print("TRAJECTORY ANALYSIS")
    print("=" * 80)

    trajectory_analysis = analyze_trajectory(pred_xyz)
    for key, value in trajectory_analysis.items():
        print(f"  {key}: {value}")

    # Check consistency
    print("\n" + "=" * 80)
    print("CONSISTENCY CHECK")
    print("=" * 80)

    consistency = check_reasoning_action_consistency(reasoning, trajectory_analysis)
    for key, value in consistency.items():
        print(f"  {key}: {value}")

    # Compute metrics
    print("\n" + "=" * 80)
    print("TRAJECTORY METRICS")
    print("=" * 80)

    # Compare with ground truth
    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    min_ade = diff.min()

    print(f"  minADE: {min_ade:.2f} meters")
    print("\n  Note: VLA models produce nondeterministic outputs")
    print("  Variance expected with num_traj_samples=1")

    print("\n" + "=" * 80)
    print("SUCCESS: Inference pipeline completed")
    print("=" * 80)


if __name__ == "__main__":
    main()
