"""Visualization utilities for boundary analysis notebooks."""

from .style import THEME, plotly_layout, format_pct, format_count
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

__all__ = [
    # Style
    "THEME",
    "plotly_layout",
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
]
