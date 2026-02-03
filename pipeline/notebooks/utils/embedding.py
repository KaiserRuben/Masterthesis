"""
3D embedding space exploration and density visualization.

WHY 3D + COLOR:
- 3D projection preserves density structure that 2D projections collapse
- But on a 2D screen, depth is ambiguous - points at different Z look identical
- Color provides the semantic dimension: which class, which ADE level
- Without color, it's just a point cloud. With color, patterns emerge.

The chrome (axes, grid, background) stays monochrome so data color pops.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors

from .style import THEME, plotly_layout, scene_style
from .data import CLASSIFICATION_KEYS


def create_3d_explorer(
    scenes: pd.DataFrame,
    color_by: str = "ade_class",
    point_size_by_ade: bool = True,
    highlight_anchors: bool = True,
    opacity: float = 0.7,
) -> go.FigureWidget:
    """
    Create interactive 3D embedding space visualization.

    Preserves density information that 2D projections lose. Points are
    colored by the selected attribute; size optionally encodes ADE.

    Args:
        scenes: DataFrame with umap_x/y/z columns
        color_by: Column to color points by. Options:
            - "ade_class": ADE discrete classes (colored)
            - "is_anchor": VLM vs propagated labels
            - "label_confidence": Confidence score
            - Any CLASSIFICATION_KEY: Semantic attribute
        point_size_by_ade: Scale point size by ADE value
        highlight_anchors: Add ring markers around anchor points
        opacity: Point opacity (0-1)

    Returns:
        Interactive FigureWidget for Jupyter
    """
    df = scenes.copy()

    # Point sizes
    if point_size_by_ade and "ade" in df.columns:
        ade_normalized = df["ade"].fillna(df["ade"].median())
        ade_normalized = (ade_normalized - ade_normalized.min()) / (ade_normalized.max() - ade_normalized.min() + 1e-6)
        sizes = 3 + ade_normalized * 6
    else:
        sizes = 4

    # Build hover text
    hover_parts = []
    for _, row in df.iterrows():
        parts = [f"<b>{row['clip_id'][:12]}...</b>"]
        for key in CLASSIFICATION_KEYS:
            if key in df.columns:
                parts.append(f"{key}: {row[key]}")
        if pd.notna(row.get("ade")):
            parts.append(f"<b>ADE: {row['ade']:.2f} ({row.get('ade_class', 'N/A')})</b>")
        hover_parts.append("<br>".join(parts))
    df["hover"] = hover_parts

    # Use FigureWidget in Jupyter for interactivity, fall back to Figure otherwise
    try:
        fig = go.FigureWidget()
    except ImportError:
        fig = go.Figure()

    # Determine coloring
    if color_by == "ade_class":
        _add_ade_class_traces(fig, df, sizes, opacity)
    elif color_by == "is_anchor":
        _add_binary_traces(fig, df, "is_anchor", sizes, opacity)
    elif color_by == "label_confidence":
        _add_continuous_trace(fig, df, "label_confidence", sizes, opacity, "Confidence")
    elif color_by in CLASSIFICATION_KEYS:
        _add_categorical_traces(fig, df, color_by, sizes, opacity)
    else:
        # Fallback: monochrome
        fig.add_trace(go.Scatter3d(
            x=df["umap_x"], y=df["umap_y"], z=df["umap_z"],
            mode="markers",
            marker=dict(size=sizes, color=THEME["text_muted"], opacity=opacity),
            text=df["hover"],
            hoverinfo="text",
            name="Scenes",
        ))

    # Highlight anchors with ring markers
    if highlight_anchors and color_by != "is_anchor":
        anchors = df[df["is_anchor"] == True]
        if len(anchors) > 0:
            fig.add_trace(go.Scatter3d(
                x=anchors["umap_x"], y=anchors["umap_y"], z=anchors["umap_z"],
                mode="markers",
                marker=dict(
                    size=6,
                    color="rgba(0,0,0,0)",
                    line=dict(color=THEME["accent"], width=1.5),
                ),
                text=anchors["hover"],
                hoverinfo="text",
                name="Anchors",
            ))

    fig.update_layout(
        **plotly_layout(f"Embedding Space (color: {color_by})", height=650),
        scene=scene_style(),
    )

    return fig


def _add_ade_class_traces(
    fig: go.FigureWidget,
    df: pd.DataFrame,
    sizes: np.ndarray | float,
    opacity: float,
) -> None:
    """Add traces colored by ADE class."""
    ade_colors = THEME["ade"]

    for label in ["low", "medium", "high", "critical", None]:
        if label is None:
            mask = df["ade_class"].isna()
            color = ade_colors["missing"]
            name = "No ADE"
        else:
            mask = df["ade_class"] == label
            color = ade_colors[label]
            name = label.capitalize()

        if mask.sum() == 0:
            continue

        subset = df[mask]
        subset_sizes = sizes[mask] if isinstance(sizes, np.ndarray) else sizes

        fig.add_trace(go.Scatter3d(
            x=subset["umap_x"], y=subset["umap_y"], z=subset["umap_z"],
            mode="markers",
            marker=dict(size=subset_sizes, color=color, opacity=opacity),
            text=subset["hover"],
            hoverinfo="text",
            name=name,
        ))


def _add_binary_traces(
    fig: go.FigureWidget,
    df: pd.DataFrame,
    column: str,
    sizes: np.ndarray | float,
    opacity: float,
) -> None:
    """Add traces for binary column (e.g., is_anchor)."""
    colors = THEME["binary"]
    names = {True: "Anchors (VLM)", False: "Propagated"}

    # Show False first (background), True on top
    for value in [False, True]:
        mask = df[column] == value
        if mask.sum() == 0:
            continue

        subset = df[mask]
        subset_sizes = sizes[mask] if isinstance(sizes, np.ndarray) else sizes

        fig.add_trace(go.Scatter3d(
            x=subset["umap_x"], y=subset["umap_y"], z=subset["umap_z"],
            mode="markers",
            marker=dict(size=subset_sizes, color=colors[value], opacity=opacity),
            text=subset["hover"],
            hoverinfo="text",
            name=names[value],
        ))


def _add_continuous_trace(
    fig: go.FigureWidget,
    df: pd.DataFrame,
    column: str,
    sizes: np.ndarray | float,
    opacity: float,
    colorbar_title: str,
) -> None:
    """Add trace with continuous color scale."""
    fig.add_trace(go.Scatter3d(
        x=df["umap_x"], y=df["umap_y"], z=df["umap_z"],
        mode="markers",
        marker=dict(
            size=sizes if isinstance(sizes, (int, float)) else sizes.tolist(),
            color=df[column].fillna(0),
            colorscale="Greys",
            colorbar=dict(title=colorbar_title),
            opacity=opacity,
        ),
        text=df["hover"],
        hoverinfo="text",
        name=column,
    ))


def _add_categorical_traces(
    fig: go.FigureWidget,
    df: pd.DataFrame,
    column: str,
    sizes: np.ndarray | float,
    opacity: float,
) -> None:
    """Add traces for categorical column."""
    unique_vals = df[column].dropna().unique()
    colors = THEME["categorical"]

    for i, value in enumerate(sorted(unique_vals)):
        mask = df[column] == value
        if mask.sum() == 0:
            continue

        subset = df[mask]
        subset_sizes = sizes[mask] if isinstance(sizes, np.ndarray) else sizes

        fig.add_trace(go.Scatter3d(
            x=subset["umap_x"], y=subset["umap_y"], z=subset["umap_z"],
            mode="markers",
            marker=dict(
                size=subset_sizes,
                color=colors[i % len(colors)],
                opacity=opacity,
            ),
            text=subset["hover"],
            hoverinfo="text",
            name=str(value),
        ))


def compute_boundary_sharpness(
    scenes: pd.DataFrame,
    embeddings: np.ndarray,
    key: str,
    k: int = 10,
) -> np.ndarray:
    """
    Compute per-scene boundary sharpness for a semantic key.

    Sharpness = proportion of k nearest neighbors with a different label.
    - 0.0 = perfectly crisp (all neighbors same class)
    - 1.0 = maximally fuzzy (all neighbors different class)

    Args:
        scenes: DataFrame with labels
        embeddings: Full embedding matrix
        key: Classification key to analyze
        k: Number of neighbors to consider

    Returns:
        Array of sharpness scores (0-1) per scene
    """
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(embeddings)
    _, indices = nn.kneighbors(embeddings)

    labels = scenes[key].values
    sharpness = np.zeros(len(scenes))

    for i in range(len(scenes)):
        own_label = labels[i]
        neighbor_labels = labels[indices[i, 1:]]  # Exclude self
        sharpness[i] = np.mean(neighbor_labels != own_label)

    return sharpness
