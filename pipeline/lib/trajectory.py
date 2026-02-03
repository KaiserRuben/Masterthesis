"""
Trajectory Classification

Computes trajectory classes from raw Alpamayo output.
36 classes total (4 direction x 3 speed x 3 lateral).

Based on definitions from:
- Pipeline Overview.md
- Trajectory Output Classes - Definition.md
"""

import numpy as np


# Direction thresholds (degrees)
TURN_THRESHOLD = 30.0  # |theta| > 30 = turn
STRAIGHT_THRESHOLD = 10.0  # |theta| < 10 = straight

# Speed threshold (m/s)
SPEED_THRESHOLD = 2.0  # |delta_v| > 2 = accel/decel

# Lateral threshold (meters)
LATERAL_THRESHOLD = 3.0  # |delta_y| > 3 = lane change


def _compute_heading_change(pred_xy: np.ndarray) -> float:
    """
    Compute total heading change from trajectory.

    Args:
        pred_xy: Shape (2, T) - x and y coordinates over T timesteps

    Returns:
        Heading change in degrees (positive = left turn)
    """
    x, y = pred_xy[0], pred_xy[1]

    # Compute heading at start (first few points)
    n_avg = min(5, len(x) // 4)
    if n_avg < 2:
        return 0.0

    dx_start = x[n_avg] - x[0]
    dy_start = y[n_avg] - y[0]
    theta_start = np.arctan2(dy_start, dx_start)

    # Compute heading at end (last few points)
    dx_end = x[-1] - x[-n_avg - 1]
    dy_end = y[-1] - y[-n_avg - 1]
    theta_end = np.arctan2(dy_end, dx_end)

    # Heading change in degrees
    delta_theta = np.degrees(theta_end - theta_start)

    # Normalize to [-180, 180]
    while delta_theta > 180:
        delta_theta -= 360
    while delta_theta < -180:
        delta_theta += 360

    return delta_theta


def _compute_speed_change(pred_xy: np.ndarray, dt: float = 0.1) -> float:
    """
    Compute velocity change from trajectory.

    Args:
        pred_xy: Shape (2, T) - x and y coordinates over T timesteps
        dt: Time step between consecutive points (seconds)

    Returns:
        Speed change in m/s (positive = acceleration)
    """
    x, y = pred_xy[0], pred_xy[1]

    if len(x) < 3:
        return 0.0

    # Compute velocities
    dx = np.diff(x) / dt
    dy = np.diff(y) / dt
    speeds = np.sqrt(dx**2 + dy**2)

    # Average speed at start vs end
    n_avg = max(1, len(speeds) // 4)
    v_start = np.mean(speeds[:n_avg])
    v_end = np.mean(speeds[-n_avg:])

    return v_end - v_start


def _compute_lateral_displacement(pred_xy: np.ndarray) -> float:
    """
    Compute lateral displacement from trajectory.

    Uses the initial heading direction as "forward", then computes
    how much the vehicle moved perpendicular to that direction.

    Args:
        pred_xy: Shape (2, T) - x and y coordinates over T timesteps

    Returns:
        Lateral displacement in meters (positive = left)
    """
    x, y = pred_xy[0], pred_xy[1]

    if len(x) < 2:
        return 0.0

    # Initial heading direction
    n_avg = min(5, len(x) // 4)
    if n_avg < 1:
        n_avg = 1

    dx_init = x[n_avg] - x[0]
    dy_init = y[n_avg] - y[0]
    dist = np.sqrt(dx_init**2 + dy_init**2)

    if dist < 0.01:
        # No significant initial movement
        return y[-1] - y[0]  # fallback to raw y displacement

    # Normalize forward direction
    fwd_x = dx_init / dist
    fwd_y = dy_init / dist

    # Left direction (perpendicular, 90 degrees counterclockwise)
    left_x = -fwd_y
    left_y = fwd_x

    # Total displacement vector
    total_dx = x[-1] - x[0]
    total_dy = y[-1] - y[0]

    # Project onto lateral direction
    lateral = total_dx * left_x + total_dy * left_y

    return lateral


def classify_direction(delta_theta: float) -> str:
    """Classify direction from heading change (degrees)."""
    if delta_theta > TURN_THRESHOLD:
        return "turn_left"
    elif delta_theta < -TURN_THRESHOLD:
        return "turn_right"
    elif abs(delta_theta) < STRAIGHT_THRESHOLD:
        return "straight"
    else:
        return "slight_curve"


def classify_speed(delta_v: float) -> str:
    """Classify speed from velocity change (m/s)."""
    if delta_v > SPEED_THRESHOLD:
        return "accelerate"
    elif delta_v < -SPEED_THRESHOLD:
        return "decelerate"
    else:
        return "constant"


def classify_lateral(delta_y: float) -> str:
    """Classify lateral movement from displacement (meters)."""
    if delta_y > LATERAL_THRESHOLD:
        return "lane_change_left"
    elif delta_y < -LATERAL_THRESHOLD:
        return "lane_change_right"
    else:
        return "lane_keep"


def classify_trajectory(
    pred_xy: np.ndarray,
    gt_xy: np.ndarray | None = None,
    dt: float = 0.1,
) -> dict:
    """
    Classify trajectory into direction, speed, and lateral classes.

    Args:
        pred_xy: Predicted trajectory, shape (2, T) - x,y coordinates
        gt_xy: Ground truth trajectory (optional, not used for classification)
        dt: Time step between consecutive points (seconds)

    Returns:
        Dict with keys:
        - direction: turn_left, turn_right, straight, slight_curve
        - speed: accelerate, decelerate, constant
        - lateral: lane_change_left, lane_change_right, lane_keep
        - combined: combined class string (e.g., "straight_constant_lane_keep")
        - delta_theta: heading change in degrees
        - delta_v: velocity change in m/s
        - delta_y: lateral displacement in meters
    """
    # Ensure correct shape
    if pred_xy.ndim == 1:
        raise ValueError(f"pred_xy must be 2D, got shape {pred_xy.shape}")

    if pred_xy.shape[0] != 2:
        # Try transposing if shape is (T, 2)
        if pred_xy.shape[1] == 2:
            pred_xy = pred_xy.T
        else:
            raise ValueError(f"pred_xy must have shape (2, T), got {pred_xy.shape}")

    # Compute metrics
    delta_theta = _compute_heading_change(pred_xy)
    delta_v = _compute_speed_change(pred_xy, dt=dt)
    delta_y = _compute_lateral_displacement(pred_xy)

    # Classify
    direction = classify_direction(delta_theta)
    speed = classify_speed(delta_v)
    lateral = classify_lateral(delta_y)

    # Combined class
    combined = f"{direction}_{speed}_{lateral}"

    return {
        "direction": direction,
        "speed": speed,
        "lateral": lateral,
        "combined": combined,
        "delta_theta": float(delta_theta),
        "delta_v": float(delta_v),
        "delta_y": float(delta_y),
    }


def trajectory_class_changed(
    traj1: dict,
    traj2: dict,
    check_direction: bool = True,
    check_speed: bool = True,
    check_lateral: bool = True,
) -> bool:
    """
    Check if trajectory class changed between two classifications.

    Args:
        traj1, traj2: Results from classify_trajectory()
        check_direction: Include direction in comparison
        check_speed: Include speed in comparison
        check_lateral: Include lateral in comparison

    Returns:
        True if any checked dimension changed
    """
    if check_direction and traj1["direction"] != traj2["direction"]:
        return True
    if check_speed and traj1["speed"] != traj2["speed"]:
        return True
    if check_lateral and traj1["lateral"] != traj2["lateral"]:
        return True
    return False
