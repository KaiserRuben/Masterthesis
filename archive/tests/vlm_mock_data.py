"""
Realistic mock data generation for VLM boundary testing.

Improves upon simple linear model with:
- Bimodal difficulty distribution (easy/hard samples)
- Threshold-based perturbation response
- Object-size dependent sensitivity
- Configurable geometric-semantic coupling
"""

import numpy as np

# Configuration constants
HARD_SAMPLE_RATIO = 0.3
PERTURBATION_MIN = 0.02
PERTURBATION_MAX = 0.25
PERTURBATION_THRESHOLD = 0.05  # Below this: minimal effect
SENSITIVITY_MIN = 0.1
SENSITIVITY_MAX = 3.0
HARD_SAMPLE_BASE_SENSITIVITY = 2.5
EASY_SAMPLE_BASE_SENSITIVITY = 0.8


def generate_mock_vlm_data(
    num_samples=1000,
    num_perturbations_per_sample=10,
    random_seed=42
):
    """
    Simulate realistic VLM grounding behavior under perturbation.

    Models key phenomena:
    - Easy samples (70%): high base IoU, low sensitivity
    - Hard samples (30%): low base IoU, high sensitivity
    - Small objects: more sensitive to perturbation
    - Perturbation threshold: minimal effect below threshold
    - Variable geometric-semantic coupling

    Expected boundary rate: ~30% (vs 92% in naive linear model)

    Args:
        num_samples: Number of image-expression pairs
        num_perturbations_per_sample: Perturbations applied to each sample
        random_seed: For reproducibility

    Returns:
        Dictionary with keys:
        - iou_original: (num_samples,) baseline IoU
        - iou_perturbed: (num_samples, num_perturbations_per_sample)
        - perturbation_magnitude: (num_samples, num_perturbations_per_sample)
        - embedding_distance: (num_samples, num_perturbations_per_sample)
        - sensitivity: (num_samples,) ground truth sensitivity
        - is_hard_sample: (num_samples,) bool
        - object_size: (num_samples,)
        - geometric_semantic_coupling: (num_samples,)
    """
    np.random.seed(random_seed)

    # Bimodal difficulty: 70% easy, 30% hard
    is_hard_sample = np.random.rand(num_samples) < HARD_SAMPLE_RATIO

    iou_before_perturbation = np.where(
        is_hard_sample,
        np.clip(np.random.beta(2, 3, num_samples), 0, 1),  # Hard: mean ~0.4
        np.clip(np.random.beta(8, 2, num_samples), 0, 1)   # Easy: mean ~0.8
    )

    # Object size: smaller objects are more fragile
    normalized_object_size = np.random.beta(2, 2, num_samples)

    # Sensitivity increases with: difficulty + small size
    size_penalty = 1.5 - normalized_object_size
    difficulty_factor = np.where(
        is_hard_sample,
        HARD_SAMPLE_BASE_SENSITIVITY,
        EASY_SAMPLE_BASE_SENSITIVITY
    )
    sample_sensitivity = difficulty_factor * size_penalty
    sample_sensitivity += np.random.normal(0, 0.3, num_samples)
    sample_sensitivity = np.clip(sample_sensitivity, SENSITIVITY_MIN, SENSITIVITY_MAX)

    # Perturbation magnitudes
    perturbation_magnitude = np.random.uniform(
        PERTURBATION_MIN,
        PERTURBATION_MAX,
        (num_samples, num_perturbations_per_sample)
    )

    # IoU degradation with threshold behavior
    iou_degradation = np.zeros((num_samples, num_perturbations_per_sample))

    for sample_idx in range(num_samples):
        for pert_idx in range(num_perturbations_per_sample):
            pert_magnitude = perturbation_magnitude[sample_idx, pert_idx]

            if pert_magnitude < PERTURBATION_THRESHOLD:
                # Subthreshold: minimal effect
                iou_drop = (
                    sample_sensitivity[sample_idx] * pert_magnitude * 0.2 +
                    np.random.normal(0, 0.02)
                )
            else:
                # Suprathreshold: linear response
                effective_magnitude = pert_magnitude - PERTURBATION_THRESHOLD
                iou_drop = (
                    sample_sensitivity[sample_idx] * effective_magnitude +
                    np.random.normal(0, 0.05)
                )

            iou_degradation[sample_idx, pert_idx] = max(0, iou_drop)

    iou_after_perturbation = np.clip(
        iou_before_perturbation[:, np.newaxis] - iou_degradation,
        0, 1
    )

    # Geometric-semantic coupling: simulate decoder fragility
    # High coupling (0.9): geometric ≈ semantic (most samples)
    # Low coupling (0.1): decoupled (fragile/robust decoder)
    geometric_semantic_coupling = np.random.choice(
        [0.9, 0.5, 0.1],
        size=num_samples,
        p=[0.6, 0.3, 0.1]
    )

    embedding_distance = np.zeros((num_samples, num_perturbations_per_sample))
    for sample_idx in range(num_samples):
        coupling_strength = geometric_semantic_coupling[sample_idx]
        embedding_distance[sample_idx] = (
            coupling_strength * iou_degradation[sample_idx] +
            (1 - coupling_strength) * np.random.uniform(0, 0.15, num_perturbations_per_sample)
        )

    return {
        'iou_original': iou_before_perturbation,
        'iou_perturbed': iou_after_perturbation,
        'perturbation_magnitude': perturbation_magnitude,
        'embedding_distance': embedding_distance,
        'sensitivity': sample_sensitivity,
        'is_hard_sample': is_hard_sample,
        'object_size': normalized_object_size,
        'geometric_semantic_coupling': geometric_semantic_coupling
    }


if __name__ == "__main__":
    # Validation: check data characteristics
    data = generate_mock_vlm_data()

    num_hard_samples = data['is_hard_sample'].sum()
    num_total_samples = len(data['iou_original'])

    print("Mock VLM Data Generation Report")
    print("=" * 50)
    print(f"Samples: {num_total_samples}")
    print(f"Perturbations per sample: {data['iou_perturbed'].shape[1]}")
    print()
    print(f"Hard samples: {num_hard_samples} ({100 * num_hard_samples / num_total_samples:.1f}%)")
    print(f"Mean IoU (easy): {data['iou_original'][~data['is_hard_sample']].mean():.3f}")
    print(f"Mean IoU (hard): {data['iou_original'][data['is_hard_sample']].mean():.3f}")
    print()
    print(f"Sensitivity: [{data['sensitivity'].min():.2f}, {data['sensitivity'].max():.2f}]")
    print(f"Object size: [{data['object_size'].min():.2f}, {data['object_size'].max():.2f}]")
    print()

    # Estimate boundary rate (requires iou_to_class from notebook)
    print("To compute boundary rate, import iou_to_class from notebook 01")
    print("Expected: ~30% boundary samples (vs 92% in simple linear model)")
