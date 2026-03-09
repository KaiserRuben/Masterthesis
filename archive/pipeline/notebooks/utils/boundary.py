"""
Boundary analysis: geography, sharpness, and cost surfaces.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import ndimage
from scipy.interpolate import griddata
from sklearn.neighbors import KNeighborsClassifier

from .style import THEME, plotly_layout, axis_style
from .data import CLASSIFICATION_KEYS, get_idx_to_umap
from .embedding import compute_boundary_sharpness


def compute_boundary_density(
    scenes: pd.DataFrame,
    pairs: pd.DataFrame,
    key: str,
    resolution: int = 50,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Compute 2D density of boundary crossings for a semantic key.

    Boundary crossings are placed at the midpoint between paired scenes.
    Density is smoothed with Gaussian filter.

    Args:
        scenes: DataFrame with umap_x/y and emb_index
        pairs: DataFrame with idx_a, idx_b, diff_key
        key: Semantic key to analyze
        resolution: Grid resolution

    Returns:
        Tuple of (density, x_edges, y_edges) or (None, None, None) if no data
    """
    key_pairs = pairs[pairs["diff_key"] == key]
    if len(key_pairs) == 0:
        return None, None, None

    idx_to_umap = {
        row["emb_index"]: (row["umap_x"], row["umap_y"])
        for _, row in scenes.iterrows()
    }

    midpoints = []
    for _, row in key_pairs.iterrows():
        if row["idx_a"] in idx_to_umap and row["idx_b"] in idx_to_umap:
            xa, ya = idx_to_umap[row["idx_a"]]
            xb, yb = idx_to_umap[row["idx_b"]]
            midpoints.append(((xa + xb) / 2, (ya + yb) / 2))

    if not midpoints:
        return None, None, None

    midpoints = np.array(midpoints)

    x_range = (scenes["umap_x"].min() - 0.5, scenes["umap_x"].max() + 0.5)
    y_range = (scenes["umap_y"].min() - 0.5, scenes["umap_y"].max() + 0.5)

    density, x_edges, y_edges = np.histogram2d(
        midpoints[:, 0], midpoints[:, 1],
        bins=resolution,
        range=[x_range, y_range],
    )

    density = ndimage.gaussian_filter(density, sigma=1.5)

    return density, x_edges, y_edges


def create_boundary_geography(
    scenes: pd.DataFrame,
    pairs: pd.DataFrame,
    key: str | None = None,
    resolution: int = 50,
) -> go.Figure:
    """
    Create 2D boundary geography heatmap.

    Shows WHERE boundary crossings occur in embedding space.

    Args:
        scenes: DataFrame with UMAP coordinates
        pairs: DataFrame with boundary pairs
        key: Specific key to show, or None for all keys overlaid
        resolution: Grid resolution

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    # Background points (monochrome)
    fig.add_trace(go.Scatter(
        x=scenes["umap_x"], y=scenes["umap_y"],
        mode="markers",
        marker=dict(size=2, color=THEME["border"]),
        hoverinfo="skip",
        showlegend=False,
    ))

    keys_to_show = [key] if key else CLASSIFICATION_KEYS

    for i, k in enumerate(keys_to_show):
        density, x_edges, y_edges = compute_boundary_density(
            scenes, pairs, k, resolution
        )
        if density is None:
            continue

        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2

        # Monochrome density - darker = higher density
        if key:
            colorscale = [[0, "rgba(255,255,255,0)"], [1, THEME["text_secondary"]]]
            opacity = 0.6
            show_scale = True
        else:
            # Multiple keys: use grays with varying base intensity
            gray_val = 40 + i * 25  # Stagger grays
            colorscale = [[0, "rgba(255,255,255,0)"], [1, f"rgb({gray_val},{gray_val},{gray_val})"]]
            opacity = 0.35
            show_scale = False

        fig.add_trace(go.Contour(
            z=density.T,
            x=x_centers,
            y=y_centers,
            colorscale=colorscale,
            opacity=opacity,
            showscale=show_scale,
            name=k,
            contours=dict(showlines=True, coloring="heatmap"),
            colorbar=dict(title="Density") if show_scale else None,
        ))

    title = f"Boundary Geography: {key}" if key else "Boundary Hotspots (all keys)"

    fig.update_layout(
        **plotly_layout(title, height=500),
        xaxis={**axis_style("UMAP 1"), "scaleanchor": "y"},
        yaxis=axis_style("UMAP 2"),
    )

    return fig


def create_sharpness_histograms(
    scenes: pd.DataFrame,
    embeddings: np.ndarray,
    k: int = 10,
) -> go.Figure:
    """
    Create 2x3 grid of boundary sharpness distributions.

    Sharpness = % of neighbors with different label.
    Higher = fuzzier boundaries.

    Args:
        scenes: DataFrame with labels
        embeddings: Full embedding matrix
        k: Number of neighbors

    Returns:
        Plotly Figure with subplots
    """
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=CLASSIFICATION_KEYS,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    for i, key in enumerate(CLASSIFICATION_KEYS):
        row = i // 3 + 1
        col = i % 3 + 1

        sharpness = compute_boundary_sharpness(scenes, embeddings, key, k)

        fig.add_trace(
            go.Histogram(
                x=sharpness,
                nbinsx=20,
                marker_color=THEME["text_muted"],
                opacity=0.8,
                showlegend=False,
            ),
            row=row, col=col,
        )

        # Mean annotation
        fig.add_annotation(
            x=0.75, y=0.85,
            text=f"μ={sharpness.mean():.2f}",
            showarrow=False,
            xref=f"x{i+1 if i > 0 else ''} domain",
            yref=f"y{i+1 if i > 0 else ''} domain",
            font=dict(size=10, color=THEME["text_secondary"]),
        )

    fig.update_xaxes(range=[0, 1], title_text="Sharpness", row=2)
    fig.update_yaxes(title_text="Count", col=1)

    fig.update_layout(
        **plotly_layout("Boundary Sharpness (0=crisp, 1=fuzzy)", height=450),
    )

    return fig


def create_cost_surface_grid(
    scenes: pd.DataFrame,
    grid_resolution: int = 40,
) -> go.Figure:
    """
    Create 2x3 grid of cost surfaces with decision boundaries.

    Each panel shows:
    - Background heatmap: interpolated ADE
    - Contour lines: decision boundary entropy

    Args:
        scenes: DataFrame with UMAP coords and ADE
        grid_resolution: Number of grid points per axis

    Returns:
        Plotly Figure with subplots
    """
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=CLASSIFICATION_KEYS,
        horizontal_spacing=0.06,
        vertical_spacing=0.10,
    )

    # Filter scenes with ADE
    df_ade = scenes[scenes["ade"].notna()]

    # Build interpolation grid
    x_min, x_max = scenes["umap_x"].min(), scenes["umap_x"].max()
    y_min, y_max = scenes["umap_y"].min(), scenes["umap_y"].max()

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution),
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Interpolate ADE
    ade_interp = griddata(
        df_ade[["umap_x", "umap_y"]].values,
        df_ade["ade"].values,
        grid_points,
        method="linear",
    ).reshape(xx.shape)

    for i, key in enumerate(CLASSIFICATION_KEYS):
        row = i // 3 + 1
        col = i % 3 + 1

        # ADE heatmap (grayscale with hint of color)
        fig.add_trace(
            go.Heatmap(
                z=ade_interp,
                x=np.linspace(x_min, x_max, grid_resolution),
                y=np.linspace(y_min, y_max, grid_resolution),
                colorscale=THEME["ade_scale"],
                showscale=(i == 2),
                colorbar=dict(title="ADE", x=1.02, len=0.4, y=0.8) if i == 2 else None,
                opacity=0.5,
                hoverinfo="skip",
            ),
            row=row, col=col,
        )

        # Decision boundary via k-NN entropy
        label_map = {v: j for j, v in enumerate(scenes[key].dropna().unique())}
        labels = scenes[key].map(label_map).fillna(-1).astype(int).values

        valid_mask = labels >= 0
        if valid_mask.sum() < 10:
            continue

        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(
            scenes.loc[valid_mask, ["umap_x", "umap_y"]].values,
            labels[valid_mask],
        )

        proba = clf.predict_proba(grid_points)
        entropy = -np.sum(proba * np.log(proba + 1e-10), axis=1)
        entropy = entropy.reshape(xx.shape)

        # Boundary contours (monochrome)
        fig.add_trace(
            go.Contour(
                z=entropy,
                x=np.linspace(x_min, x_max, grid_resolution),
                y=np.linspace(y_min, y_max, grid_resolution),
                contours=dict(
                    start=entropy.max() * 0.5,
                    end=entropy.max(),
                    size=entropy.max() * 0.15,
                    coloring="lines",
                ),
                line=dict(color=THEME["accent"], width=1),
                showscale=False,
                hoverinfo="skip",
            ),
            row=row, col=col,
        )

        # Data points
        fig.add_trace(
            go.Scatter(
                x=df_ade["umap_x"], y=df_ade["umap_y"],
                mode="markers",
                marker=dict(size=2, color=THEME["text"], opacity=0.2),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row, col=col,
        )

    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=False)

    fig.update_layout(
        **plotly_layout("Cost Surfaces with Decision Boundaries", height=500),
    )

    return fig
