#!/usr/bin/env python3
"""
Load existing threshold_optimization_results.npz and generate:
1. optimization_results.npz (detailed optimization for all 3 signals)
2. Visualizations (threshold comparison, MI comparison, boundary detection)

This script allows you to skip the expensive VLM inference and just run
the optimization and visualization steps.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize
from sklearn.metrics import mutual_info_score

sns.set_style('whitegrid')
np.random.seed(42)

print("="*60)
print("Threshold Optimization from Existing Data")
print("="*60)

# ============================================================================
# Load existing data
# ============================================================================
print("\n[1/4] Loading existing data...")
try:
    data = np.load('data/threshold_optimization_results.npz')
    results = {
        'iou_original': data['iou_original'],
        'iou_perturbed': data['iou_perturbed'],
        'perturbation_magnitude': data['perturbation_magnitude'],
        'embedding_distance': data['embedding_distance'],
        'sample_indices': data['sample_indices']
    }
    print(f"✓ Loaded data:")
    print(f"  Samples: {len(results['iou_original'])}")
    print(f"  Perturbations per sample: {results['iou_perturbed'].shape[1]}")
    print(f"  Mean IoU (original): {results['iou_original'].mean():.3f}")
    print(f"  Mean IoU (perturbed): {results['iou_perturbed'].mean():.3f}")
except FileNotFoundError:
    print("ERROR: threshold_optimization_results.npz not found!")
    print("Please run notebook 02 cell 11 first to generate the data.")
    exit(1)

# ============================================================================
# Data quality check
# ============================================================================
print("\n[2/4] Data Quality Check...")
print(f"  Samples: {len(results['iou_original'])}")
print(f"  Perturbations per sample: {results['iou_perturbed'].shape[1]}")
print(f"\nIoU Statistics:")
print(f"  Original - mean: {results['iou_original'].mean():.3f}, std: {results['iou_original'].std():.3f}")
print(f"  Perturbed - mean: {results['iou_perturbed'].mean():.3f}, std: {results['iou_perturbed'].std():.3f}")
print(f"\nIoU Degradation:")
iou_drop = results['iou_original'][:, np.newaxis] - results['iou_perturbed']
print(f"  Mean drop: {iou_drop.mean():.3f}")
print(f"  Std drop: {iou_drop.std():.3f}")
print(f"  Max drop: {iou_drop.max():.3f}")

# Check for sensitivity
iou_std = results['iou_perturbed'].std(axis=1).mean()
print(f"\nPerturbation Sensitivity:")
print(f"  Mean IoU std across perturbations: {iou_std:.3f}")
if iou_std > 0.1:
    print(f"  ✓ Model shows perturbation sensitivity")
else:
    print(f"  ⚠ Model may be too robust (low sensitivity)")

# ============================================================================
# Optimization functions
# ============================================================================
def iou_to_class(iou: np.ndarray, thresholds: list) -> np.ndarray:
    """Discretize IoU values into classes."""
    classes = np.zeros_like(iou, dtype=int)
    for i, t in enumerate(sorted(thresholds)):
        classes[iou >= t] = i + 1
    return classes

def compute_mutual_information(classes: np.ndarray, signal: np.ndarray, n_bins=20) -> float:
    """MI between discrete classes and continuous signal."""
    classes_flat = classes.flatten()
    signal_flat = signal.flatten()

    # Bin signal
    signal_binned = np.digitize(signal_flat, np.linspace(signal_flat.min(), signal_flat.max(), n_bins))

    return mutual_info_score(classes_flat, signal_binned)

def optimize_thresholds(iou: np.ndarray, signal: np.ndarray, n_thresholds=3, n_restarts=10):
    """Find thresholds maximizing MI via multi-start L-BFGS-B."""
    def objective(thresholds):
        thresholds = np.sort(np.clip(thresholds, 0.01, 0.99))
        classes = iou_to_class(iou, thresholds.tolist())
        mi = compute_mutual_information(classes, signal)
        return -mi  # Minimize negative MI

    best_result = None
    best_mi = -np.inf

    for _ in range(n_restarts):
        x0 = np.sort(np.random.uniform(0.1, 0.9, n_thresholds))
        bounds = [(0.05, 0.95)] * n_thresholds

        result = optimize.minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds
        )

        if -result.fun > best_mi:
            best_mi = -result.fun
            best_result = result

    return np.sort(best_result.x), best_mi

# ============================================================================
# Optimize thresholds for all signals
# ============================================================================
print("\n[3/4] Optimizing thresholds for different signals...")

# Define signals
signals = {
    'perturbation_magnitude': results['perturbation_magnitude'],
    'embedding_distance': results['embedding_distance'],
    'iou_drop': results['iou_original'][:, np.newaxis] - results['iou_perturbed']
}

# Optimize for each signal
optimization_results = {}

for signal_name, signal in signals.items():
    print(f"\n  Signal: {signal_name}")

    # Optimize
    optimal_thresh, optimal_mi = optimize_thresholds(
        results['iou_perturbed'],
        signal,
        n_thresholds=3,
        n_restarts=10
    )

    # Baseline (uniform thresholds)
    baseline_thresh = [0.3, 0.5, 0.7]
    baseline_classes = iou_to_class(results['iou_perturbed'], baseline_thresh)
    baseline_mi = compute_mutual_information(baseline_classes, signal)

    improvement = ((optimal_mi - baseline_mi) / baseline_mi) * 100 if baseline_mi > 0 else 0

    optimization_results[signal_name] = {
        'optimal_thresholds': optimal_thresh,
        'optimal_mi': optimal_mi,
        'baseline_mi': baseline_mi,
        'improvement_pct': improvement
    }

    print(f"    Optimal thresholds: {optimal_thresh.round(3)}")
    print(f"    Optimal MI: {optimal_mi:.4f}")
    print(f"    Baseline MI: {baseline_mi:.4f}")
    print(f"    Improvement: {improvement:+.1f}%")

# Save optimization results
np.savez('data/optimization_results.npz', **optimization_results)
print(f"\n✓ Saved optimization_results.npz")

# ============================================================================
# Visualizations
# ============================================================================
print("\n[4/4] Generating visualizations...")

# Figure 1: Threshold comparison and MI comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Threshold positions
ax = axes[0]
colors = plt.cm.Set2.colors

for i, (name, res) in enumerate(optimization_results.items()):
    thresholds = res['optimal_thresholds']
    y_pos = i

    for t in thresholds:
        ax.axvline(t, ymin=y_pos/len(optimization_results),
                   ymax=(y_pos+0.8)/len(optimization_results),
                   color=colors[i], linewidth=3)

    ax.text(0.02, y_pos + 0.4,
            f"{name}\nMI={res['optimal_mi']:.3f} (+{res['improvement_pct']:.1f}%)",
            transform=ax.get_yaxis_transform(), fontsize=9)

ax.set_xlim(0, 1)
ax.set_ylim(-0.5, len(optimization_results))
ax.set_xlabel('IoU Threshold')
ax.set_yticks([])
ax.set_title('Optimal Thresholds by Signal Type')

# Plot 2: MI comparison
ax = axes[1]
signal_names = list(optimization_results.keys())
optimal_mis = [optimization_results[s]['optimal_mi'] for s in signal_names]
baseline_mis = [optimization_results[s]['baseline_mi'] for s in signal_names]

x = np.arange(len(signal_names))
width = 0.35

ax.bar(x - width/2, baseline_mis, width, label='Baseline [0.3, 0.5, 0.7]', alpha=0.7)
ax.bar(x + width/2, optimal_mis, width, label='Optimized', alpha=0.7)

ax.set_ylabel('Mutual Information')
ax.set_title('MI: Baseline vs Optimized Thresholds')
ax.set_xticks(x)
ax.set_xticklabels([s.replace('_', '\n') for s in signal_names], fontsize=8)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/threshold_optimization_comparison.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved threshold_optimization_comparison.png")
plt.show()

# Figure 2: Boundary detection analysis
optimal_thresh = optimization_results['perturbation_magnitude']['optimal_thresholds']

# Classify samples
classes_orig = iou_to_class(results['iou_original'], optimal_thresh)
classes_pert = iou_to_class(results['iou_perturbed'], optimal_thresh)

# Detect boundaries: samples where ANY perturbation changes class
class_changes = classes_pert != classes_orig[:, np.newaxis]
boundary_mask = class_changes.any(axis=1)

# Severity: max class jump
severity = np.abs(classes_pert - classes_orig[:, np.newaxis]).max(axis=1)

print(f"\nBoundary Detection Results:")
print(f"  Optimal thresholds: {optimal_thresh.round(3)}")
print(f"  Boundary samples: {boundary_mask.sum()} / {len(boundary_mask)} ({100*boundary_mask.mean():.1f}%)")
print(f"  Severity distribution: {np.bincount(severity)}")
print(f"\nClass Distribution:")
print(f"  Original: {np.bincount(classes_orig)}")
print(f"  Perturbed: {np.bincount(classes_pert.flatten())}")

# Sanity check
if boundary_mask.mean() < 0.2 or boundary_mask.mean() > 0.6:
    print(f"\n  ⚠ Warning: Boundary rate {100*boundary_mask.mean():.1f}% outside expected range (20-60%)")
else:
    print(f"\n  ✓ Boundary rate within realistic range")

# Figure 3: IoU distributions and boundary visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: IoU distributions
ax = axes[0, 0]
ax.hist(results['iou_original'], bins=20, alpha=0.7, label='Original', edgecolor='black')
ax.hist(results['iou_perturbed'].flatten(), bins=20, alpha=0.5, label='Perturbed', edgecolor='black')
for t in optimal_thresh:
    ax.axvline(t, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('IoU')
ax.set_ylabel('Count')
ax.set_title('IoU Distribution (red lines = optimal thresholds)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Class distribution
ax = axes[0, 1]
class_labels = ['Class 0\n(IoU<{:.2f})'.format(optimal_thresh[0]),
                'Class 1\n({:.2f}≤IoU<{:.2f})'.format(optimal_thresh[0], optimal_thresh[1]),
                'Class 2\n({:.2f}≤IoU<{:.2f})'.format(optimal_thresh[1], optimal_thresh[2]),
                'Class 3\n(IoU≥{:.2f})'.format(optimal_thresh[2])]
orig_counts = np.bincount(classes_orig, minlength=4)
pert_counts = np.bincount(classes_pert.flatten(), minlength=4)

x = np.arange(4)
width = 0.35
ax.bar(x - width/2, orig_counts, width, label='Original', alpha=0.7)
ax.bar(x + width/2, pert_counts, width, label='Perturbed', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(class_labels, fontsize=8)
ax.set_ylabel('Count')
ax.set_title('Class Distribution')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: Boundary samples
ax = axes[1, 0]
boundary_colors = ['green' if not b else 'red' for b in boundary_mask]
ax.scatter(results['iou_original'], results['iou_perturbed'].mean(axis=1),
           c=boundary_colors, alpha=0.6, s=50)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='No degradation')
ax.set_xlabel('Original IoU')
ax.set_ylabel('Mean Perturbed IoU')
ax.set_title('Boundary Samples (red = boundary, green = stable)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Perturbation sensitivity
ax = axes[1, 1]
iou_std_per_sample = results['iou_perturbed'].std(axis=1)
colors_sensitivity = ['red' if b else 'green' for b in boundary_mask]
ax.scatter(results['iou_original'], iou_std_per_sample, c=colors_sensitivity, alpha=0.6, s=50)
ax.set_xlabel('Original IoU')
ax.set_ylabel('IoU Std Dev (across perturbations)')
ax.set_title('Perturbation Sensitivity')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/boundary_detection_analysis.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved boundary_detection_analysis.png")
plt.show()

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*60)
print("Summary")
print("="*60)
print(f"\nFiles generated:")
print(f"  ✓ optimization_results.npz")
print(f"  ✓ threshold_optimization_comparison.png")
print(f"  ✓ boundary_detection_analysis.png")

print(f"\nKey findings:")
for signal_name, res in optimization_results.items():
    print(f"  {signal_name}:")
    print(f"    Thresholds: {res['optimal_thresholds'].round(3)}")
    print(f"    MI improvement: {res['improvement_pct']:+.1f}%")

print(f"\nBoundary detection:")
print(f"  Boundary rate: {100*boundary_mask.mean():.1f}%")
print(f"  Expected range: 20-60%")

if all(res['improvement_pct'] > 5 for res in optimization_results.values() if res['baseline_mi'] > 0):
    print(f"\n✓ All signals show >5% MI improvement - proceed to notebook 03")
else:
    print(f"\n⚠ Some signals show <5% improvement - check data quality")

print("\n" + "="*60)
