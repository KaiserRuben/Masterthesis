"""Visualization utilities for boundary analysis notebooks."""

from .style import THEME, plotly_layout, format_pct, format_count, axis_style, scene_style
from .data import (
    load_pipeline_data,
    classify_ade,
    ADE_BINS,
    ADE_LABELS,
    CLASSIFICATION_KEYS,
)
from .embedding import (
    create_3d_explorer,
    compute_boundary_sharpness,
)
from .boundary import (
    compute_boundary_density,
    create_boundary_geography,
    create_cost_surface_grid,
    create_sharpness_histograms,
)
from .transition import (
    create_transition_sankey,
    create_ade_transition_matrix,
    identify_danger_zones,
    create_danger_zone_plot,
)
from .pairs import (
    create_pair_scatter,
    create_pair_connections,
)
from .hypothesis import (
    # H1: Boundary-Error Correlation
    BoundaryMarginResult,
    compute_centroids,
    compute_boundary_margin,
    compute_boundary_proximity,
    create_h1_correlation_plot,
    create_h1_perkey_correlation_plot,
    compute_margin_sensitivity_alignment,
    # H2: Anisotropy
    compute_anisotropy_vector,
    create_h2_anisotropy_plot,
    # H3: Asymmetry
    compute_transition_asymmetry,
    create_h3_asymmetry_heatmap,
    create_h3_asymmetry_distribution,
    # Summary visualizations
    create_stability_map,
    create_divergence_curve_plot,
    create_three_level_summary,
    # Export
    export_figure_for_print,
)

__all__ = [
    # Style
    "THEME",
    "plotly_layout",
    "axis_style",
    "scene_style",
    "format_pct",
    "format_count",
    # Data
    "load_pipeline_data",
    "classify_ade",
    "ADE_BINS",
    "ADE_LABELS",
    "CLASSIFICATION_KEYS",
    # Embedding
    "create_3d_explorer",
    "compute_boundary_sharpness",
    # Boundary
    "compute_boundary_density",
    "create_boundary_geography",
    "create_cost_surface_grid",
    "create_sharpness_histograms",
    # Transition
    "create_transition_sankey",
    "create_ade_transition_matrix",
    "identify_danger_zones",
    "create_danger_zone_plot",
    # Pairs
    "create_pair_scatter",
    "create_pair_connections",
    # Hypothesis (H1-H4)
    "BoundaryMarginResult",
    "compute_centroids",
    "compute_boundary_margin",
    "compute_boundary_proximity",
    "create_h1_correlation_plot",
    "create_h1_perkey_correlation_plot",
    "compute_margin_sensitivity_alignment",
    "compute_anisotropy_vector",
    "create_h2_anisotropy_plot",
    "compute_transition_asymmetry",
    "create_h3_asymmetry_heatmap",
    "create_h3_asymmetry_distribution",
    "create_stability_map",
    "create_divergence_curve_plot",
    "create_three_level_summary",
    "export_figure_for_print",
]
