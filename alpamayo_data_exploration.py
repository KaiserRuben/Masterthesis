"""
Alpamayo-R1 Data Exploration Script

This script loads sample data from the Physical AI AV dataset and explores its structure.
It does NOT require running the model (no GPU needed).

Purpose:
1. Understand the data format (multi-camera images, ego trajectories)
2. Visualize the inputs
3. Plan semantic perturbations for testing reasoning-action consistency
"""

import sys
sys.path.insert(0, 'archive/alpamayo/src')

import numpy as np
import torch
from alpamayo_r1 import load_physical_aiavdataset
import matplotlib.pyplot as plt

# Example clip ID from the dataset
# This matches the one used in test_inference.py
CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
T0_US = 5_100_000  # 5.1 seconds into the clip

print("=" * 80)
print("ALPAMAYO-R1 DATA EXPLORATION")
print("=" * 80)
print(f"\nLoading data for clip: {CLIP_ID}")
print(f"Timestamp (t0): {T0_US/1_000_000:.1f}s")
print("\nNote: This will download sample data from HuggingFace (~100MB)")
print("      You may need to authenticate with `huggingface-cli login` first")
print("=" * 80)

try:
    # Load data
    print("\nLoading dataset...")
    data = load_physical_aiavdataset(CLIP_ID, t0_us=T0_US)
    print("✓ Data loaded successfully!")

    # Print data structure
    print("\n" + "=" * 80)
    print("DATA STRUCTURE")
    print("=" * 80)

    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            print(f"\n{key}:")
            print(f"  Type: torch.Tensor")
            print(f"  Shape: {tuple(value.shape)}")
            print(f"  Dtype: {value.dtype}")
            if value.numel() < 10:
                print(f"  Values: {value}")
        elif isinstance(value, (int, str)):
            print(f"\n{key}: {value}")

    # Extract and analyze image data
    print("\n" + "=" * 80)
    print("CAMERA IMAGES")
    print("=" * 80)

    image_frames = data["image_frames"]  # (N_cameras, num_frames, 3, H, W)
    n_cameras, num_frames, channels, height, width = image_frames.shape

    print(f"\nNumber of cameras: {n_cameras}")
    print(f"Frames per camera: {num_frames}")
    print(f"Image size: {height} x {width}")
    print(f"Channels: {channels}")

    camera_names = [
        "Cross Left 120°",
        "Front Wide 120°",
        "Cross Right 120°",
        "Front Tele 30°"
    ]

    for i, cam_name in enumerate(camera_names[:n_cameras]):
        print(f"\n  Camera {i}: {cam_name}")
        print(f"    Index: {data['camera_indices'][i]}")
        print(f"    Timestamps (relative): {data['relative_timestamps'][i].numpy()}")

    # Analyze ego trajectory
    print("\n" + "=" * 80)
    print("EGO TRAJECTORY")
    print("=" * 80)

    ego_history_xyz = data["ego_history_xyz"][0, 0]  # (16, 3)
    ego_future_xyz = data["ego_future_xyz"][0, 0]   # (64, 3)

    print(f"\nHistory trajectory:")
    print(f"  Duration: 1.6s (16 steps @ 10Hz)")
    print(f"  Start position: {ego_history_xyz[0].numpy()}")
    print(f"  End position (t0): {ego_history_xyz[-1].numpy()}")
    print(f"  Total displacement: {torch.norm(ego_history_xyz[-1] - ego_history_xyz[0]):.2f}m")

    print(f"\nFuture trajectory (ground truth):")
    print(f"  Duration: 6.4s (64 steps @ 10Hz)")
    print(f"  Start position (t0): {ego_future_xyz[0].numpy()}")
    print(f"  End position: {ego_future_xyz[-1].numpy()}")
    print(f"  Total displacement: {torch.norm(ego_future_xyz[-1] - ego_future_xyz[0]):.2f}m")

    # Create visualization
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATION")
    print("=" * 80)

    fig = plt.figure(figsize=(16, 10))

    # Plot camera images (most recent frame from each camera)
    for i in range(min(4, n_cameras)):
        ax = plt.subplot(3, 4, i + 1)
        # Get last frame (t0)
        img = image_frames[i, -1].permute(1, 2, 0).numpy().astype(np.uint8)
        ax.imshow(img)
        ax.set_title(f"{camera_names[i]}\n(t={T0_US/1_000_000:.1f}s)", fontsize=10)
        ax.axis('off')

    # Plot ego trajectory (bird's eye view)
    ax_traj = plt.subplot(3, 2, 3)

    history_xy = ego_history_xyz[:, :2].numpy()
    future_xy = ego_future_xyz[:, :2].numpy()

    ax_traj.plot(history_xy[:, 0], history_xy[:, 1], 'b.-', label='History (1.6s)', linewidth=2, markersize=8)
    ax_traj.plot(future_xy[:, 0], future_xy[:, 1], 'r.-', label='Future GT (6.4s)', linewidth=2, markersize=4)
    ax_traj.scatter([0], [0], c='green', s=200, marker='*', label='t0 (ego position)', zorder=5)

    ax_traj.set_xlabel('X (meters)')
    ax_traj.set_ylabel('Y (meters)')
    ax_traj.set_title('Ego Trajectory (Bird\'s Eye View)')
    ax_traj.legend()
    ax_traj.grid(True, alpha=0.3)
    ax_traj.axis('equal')

    # Plot velocity over time
    ax_vel = plt.subplot(3, 2, 4)

    # Compute velocities (displacement between consecutive steps)
    history_times = np.arange(-1.5, 0.1, 0.1)
    future_times = np.arange(0.1, 6.5, 0.1)

    history_vel = np.linalg.norm(np.diff(ego_history_xyz.numpy(), axis=0), axis=1) / 0.1
    future_vel = np.linalg.norm(np.diff(ego_future_xyz.numpy(), axis=0), axis=1) / 0.1

    ax_vel.plot(history_times[:-1], history_vel, 'b.-', label='History velocity', linewidth=2)
    ax_vel.plot(future_times[:-1], future_vel, 'r.-', label='Future velocity', linewidth=2)
    ax_vel.axvline(0, color='green', linestyle='--', alpha=0.5, label='t0')

    ax_vel.set_xlabel('Time (seconds)')
    ax_vel.set_ylabel('Velocity (m/s)')
    ax_vel.set_title('Ego Velocity Over Time')
    ax_vel.legend()
    ax_vel.grid(True, alpha=0.3)

    # Add metadata text box
    ax_info = plt.subplot(3, 2, (5, 6))
    ax_info.axis('off')

    info_text = f"""
DATA SUMMARY
{'='*60}

Clip ID: {CLIP_ID}
Timestamp (t0): {T0_US/1_000_000:.1f}s

INPUTS:
  - Multi-camera images: {n_cameras} cameras × {num_frames} frames
  - Image resolution: {height} × {width} pixels
  - Ego history: {len(ego_history_xyz)} steps (1.6s @ 10Hz)

GROUND TRUTH:
  - Future trajectory: {len(ego_future_xyz)} waypoints (6.4s @ 10Hz)

EXPECTED MODEL OUTPUTS:
  - Reasoning trace (Chain-of-Causation)
  - Predicted trajectory (64 waypoints)

SEMANTIC PERTURBATION IDEAS:
  1. Occlusion: Add synthetic vehicles/objects in camera views
  2. Weather: Simulate rain, fog, glare
  3. Object manipulation: Move/remove pedestrians, vehicles
  4. Spatial: Shift camera positions, angles
  5. Temporal: Drop frames, introduce lag
    """

    ax_info.text(0.05, 0.95, info_text,
                 transform=ax_info.transAxes,
                 fontsize=9,
                 verticalalignment='top',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig('alpamayo_data_exploration.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved to: alpamayo_data_exploration.png")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("""
1. MODEL INFERENCE (requires NVIDIA GPU with 24GB+ VRAM):
   - Options: Cloud GPU (Lambda Labs, AWS, GCP), or MLX conversion for M1/M2/M3

2. REASONING-ACTION CONSISTENCY FRAMEWORK:
   - Parse reasoning trace to extract stated intent
   - Analyze trajectory to infer actual behavior
   - Compare: does reasoning match action?

3. SEMANTIC PERTURBATION TESTING:
   - Apply perturbations (occlusion, weather, object manipulation)
   - Re-run inference
   - Detect: when does reasoning-action misalignment occur?

4. BOUNDARY EXPLORATION:
   - Find minimal perturbations that cause consistency failures
   - Map the "decision boundary" between consistent and inconsistent states
    """)
    print("=" * 80)

except Exception as e:
    print(f"\n✗ Error: {e}")
    print("\nCommon issues:")
    print("  1. Not authenticated with HuggingFace:")
    print("     → Run: huggingface-cli login")
    print("  2. No access to dataset:")
    print("     → Request access at: https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles")
    import traceback
    traceback.print_exc()
