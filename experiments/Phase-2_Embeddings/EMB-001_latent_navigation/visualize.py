"""
Interactive Plotly Visualization

Creates 3D visualization with:
- Dropdown to color by semantic key or cluster
- Scene points with hover info
- Text anchor markers (optional)
- k-NN edges (optional)
- Single-key-diff edges highlighted
"""

import numpy as np
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


def create_visualization(
    coords_3d: np.ndarray,
    scene_ids: list[str],
    classifications: dict[str, dict],
    cluster_labels: np.ndarray,
    graph: nx.Graph,
    text_anchor_coords: dict[str, np.ndarray] | None = None,  # {key_value: 3d_coord}
    keys_for_coloring: list[str] | None = None,
    output_path: Path | None = None,
    title: str = "Embedding Space Visualization",
) -> go.Figure:
    """
    Create interactive 3D Plotly visualization.

    Args:
        coords_3d: (N, 3) UMAP coordinates
        scene_ids: List of scene IDs
        classifications: {scene_id: {key: value}}
        cluster_labels: HDBSCAN cluster labels
        graph: NetworkX k-NN graph
        text_anchor_coords: Optional text anchor 3D coordinates
        keys_for_coloring: Keys to include in dropdown
        output_path: Where to save HTML (optional)
        title: Plot title

    Returns:
        Plotly Figure object
    """
    if keys_for_coloring is None:
        keys_for_coloring = []

    # Build hover text
    hover_texts = []
    for i, sid in enumerate(scene_ids):
        cls = classifications.get(sid, {})
        text_parts = [f"<b>{sid[:8]}...</b>"]
        for key, value in cls.items():
            if isinstance(value, (str, int, float, bool)):
                text_parts.append(f"{key}: {value}")
        text_parts.append(f"cluster: {cluster_labels[i]}")
        hover_texts.append("<br>".join(text_parts))

    # Create figure
    fig = go.Figure()

    # --- Cluster coloring (default visible) ---
    unique_clusters = sorted(set(cluster_labels))
    cluster_colors = _get_color_mapping(unique_clusters)

    fig.add_trace(go.Scatter3d(
        x=coords_3d[:, 0],
        y=coords_3d[:, 1],
        z=coords_3d[:, 2],
        mode="markers",
        marker=dict(
            size=5,
            color=[cluster_colors.get(c, "gray") for c in cluster_labels],
            opacity=0.7,
        ),
        text=hover_texts,
        hoverinfo="text",
        name="Scenes (by cluster)",
        visible=True,
    ))

    # --- Key-based colorings (hidden by default) ---
    for key in keys_for_coloring:
        key_values = []
        for sid in scene_ids:
            cls = classifications.get(sid, {})
            val = cls.get(key, "unknown")
            # Skip non-hashable values (dicts, lists)
            if isinstance(val, (dict, list)):
                val = "unknown"
            key_values.append(val)

        unique_values = sorted(set(key_values), key=str)
        value_colors = _get_color_mapping(unique_values)

        fig.add_trace(go.Scatter3d(
            x=coords_3d[:, 0],
            y=coords_3d[:, 1],
            z=coords_3d[:, 2],
            mode="markers",
            marker=dict(
                size=5,
                color=[value_colors.get(v, "gray") for v in key_values],
                opacity=0.7,
            ),
            text=hover_texts,
            hoverinfo="text",
            name=f"Scenes (by {key})",
            visible=False,
        ))

    # --- k-NN edges (hidden by default) ---
    edge_x, edge_y, edge_z = [], [], []
    single_key_edge_x, single_key_edge_y, single_key_edge_z = [], [], []

    for u, v, data in graph.edges(data=True):
        if data.get("is_single_key_diff", False):
            single_key_edge_x.extend([coords_3d[u, 0], coords_3d[v, 0], None])
            single_key_edge_y.extend([coords_3d[u, 1], coords_3d[v, 1], None])
            single_key_edge_z.extend([coords_3d[u, 2], coords_3d[v, 2], None])
        else:
            edge_x.extend([coords_3d[u, 0], coords_3d[v, 0], None])
            edge_y.extend([coords_3d[u, 1], coords_3d[v, 1], None])
            edge_z.extend([coords_3d[u, 2], coords_3d[v, 2], None])

    # Regular edges
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode="lines",
        line=dict(color="lightgray", width=1),
        hoverinfo="none",
        name="k-NN edges",
        visible=False,
    ))

    # Single-key-diff edges (highlighted)
    fig.add_trace(go.Scatter3d(
        x=single_key_edge_x, y=single_key_edge_y, z=single_key_edge_z,
        mode="lines",
        line=dict(color="red", width=2),
        hoverinfo="none",
        name="Single-key-diff edges",
        visible=False,
    ))

    # --- Text anchors (if provided, hidden by default) ---
    if text_anchor_coords:
        anchor_x, anchor_y, anchor_z, anchor_text = [], [], [], []
        for label, coord in text_anchor_coords.items():
            anchor_x.append(coord[0])
            anchor_y.append(coord[1])
            anchor_z.append(coord[2])
            anchor_text.append(label)

        fig.add_trace(go.Scatter3d(
            x=anchor_x, y=anchor_y, z=anchor_z,
            mode="markers+text",
            marker=dict(size=10, symbol="diamond", color="gold"),
            text=anchor_text,
            textposition="top center",
            hoverinfo="text",
            name="Text anchors",
            visible=False,
        ))

    # --- Build dropdown menu ---
    n_base_traces = 1 + len(keys_for_coloring)  # cluster + key colorings
    n_edge_traces = 2  # regular + single-key-diff
    n_anchor_traces = 1 if text_anchor_coords else 0
    total_traces = n_base_traces + n_edge_traces + n_anchor_traces

    buttons = []

    # Cluster view
    visibility = [False] * total_traces
    visibility[0] = True  # cluster trace
    buttons.append(dict(
        label="Color: Cluster",
        method="update",
        args=[{"visible": visibility}],
    ))

    # Key views
    for i, key in enumerate(keys_for_coloring):
        visibility = [False] * total_traces
        visibility[1 + i] = True  # key trace
        buttons.append(dict(
            label=f"Color: {key}",
            method="update",
            args=[{"visible": visibility}],
        ))

    # With edges
    visibility = [False] * total_traces
    visibility[0] = True  # cluster
    visibility[n_base_traces] = True  # regular edges
    visibility[n_base_traces + 1] = True  # single-key edges
    buttons.append(dict(
        label="Show edges",
        method="update",
        args=[{"visible": visibility}],
    ))

    # With text anchors (if available)
    if text_anchor_coords:
        visibility = [False] * total_traces
        visibility[0] = True  # cluster
        visibility[-1] = True  # anchors
        buttons.append(dict(
            label="Show text anchors",
            method="update",
            args=[{"visible": visibility}],
        ))

    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            zaxis_title="UMAP 3",
        ),
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            x=0.1,
            y=1.15,
            showactive=True,
            buttons=buttons,
        )],
        legend=dict(x=0.85, y=0.95),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    # Save if path provided
    if output_path:
        fig.write_html(str(output_path))
        print(f"Visualization saved to: {output_path}")

    return fig


def _get_color_mapping(values: list) -> dict:
    """Generate color mapping for categorical values."""
    # Plotly qualitative colors
    colors = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
        "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
        "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
    ]

    mapping = {}
    for i, v in enumerate(values):
        if v == -1:  # Noise cluster
            mapping[v] = "lightgray"
        else:
            mapping[v] = colors[i % len(colors)]

    return mapping


def create_comparison_figure(
    model_results: dict[str, dict],
    metric_name: str = "accuracy",
    keys: list[str] | None = None,
) -> go.Figure:
    """
    Create bar chart comparing models across keys.

    Args:
        model_results: {model_name: {key: {metric: value}}}
        metric_name: Which metric to plot
        keys: Keys to include (default: all)

    Returns:
        Plotly Figure
    """
    if keys is None:
        keys = list(next(iter(model_results.values())).keys())

    fig = go.Figure()

    for model_name, key_metrics in model_results.items():
        values = [key_metrics.get(k, {}).get(metric_name, 0) for k in keys]
        fig.add_trace(go.Bar(
            name=model_name,
            x=keys,
            y=values,
        ))

    fig.update_layout(
        title=f"{metric_name.replace('_', ' ').title()} by Key and Model",
        xaxis_title="Key",
        yaxis_title=metric_name.replace("_", " ").title(),
        barmode="group",
    )

    return fig
