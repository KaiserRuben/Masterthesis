"""
Pair exploration: scatter plots and connection visualizations.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from .style import THEME, plotly_layout, axis_style
from .data import ADE_BINS, get_clip_to_umap


def create_pair_scatter(
    pairs: pd.DataFrame,
    key: str | None = None,
    min_delta: float = 0,
    show_ade_thresholds: bool = True,
) -> go.Figure:
    """
    Create ADE_a vs ADE_b scatter plot for pairs.

    Points away from diagonal = high sensitivity to semantic change.

    Args:
        pairs: DataFrame with ade_a, ade_b, diff_key
        key: Filter to specific semantic key, or None for all
        min_delta: Minimum ΔADE to include
        show_ade_thresholds: Show ADE class boundary lines

    Returns:
        Plotly Figure
    """
    df = pairs[pairs["rel_delta_ade"].notna()].copy()

    if key:
        df = df[df["diff_key"] == key]

    if min_delta > 0:
        df = df[df["rel_delta_ade"] >= min_delta]

    if len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No matching pairs", showarrow=False)
        return fig

    # Use categorical colors only for diff_key distinction
    unique_keys = df["diff_key"].unique()
    colors = THEME["categorical"]
    color_map = {k: colors[i % len(colors)] for i, k in enumerate(sorted(unique_keys))}

    fig = go.Figure()

    # Add traces per key for legend
    for k in sorted(unique_keys):
        subset = df[df["diff_key"] == k]
        fig.add_trace(go.Scatter(
            x=subset["ade_a"], y=subset["ade_b"],
            mode="markers",
            marker=dict(size=5, color=color_map[k], opacity=0.6),
            name=k,
            hovertemplate=(
                f"<b>{k}</b><br>"
                "%{customdata[0]} → %{customdata[1]}<br>"
                "ADE: %{x:.2f} → %{y:.2f}<br>"
                "ΔADE: %{customdata[2]:.3f}<extra></extra>"
            ),
            customdata=subset[["value_a", "value_b", "rel_delta_ade"]].values,
        ))

    # Diagonal reference line
    max_val = max(df["ade_a"].max(), df["ade_b"].max()) * 1.1
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines",
        line=dict(dash="dash", color=THEME["border"], width=1),
        showlegend=False,
        hoverinfo="skip",
    ))

    # ADE class threshold lines
    if show_ade_thresholds:
        for thresh in [1.0, 2.5, 5.0]:
            if thresh < max_val:
                fig.add_vline(x=thresh, line_dash="dot",
                             line_color=THEME["grid"], opacity=0.7)
                fig.add_hline(y=thresh, line_dash="dot",
                             line_color=THEME["grid"], opacity=0.7)

    fig.update_layout(
        **plotly_layout(f"Pair ADE Comparison ({len(df):,} pairs)", height=500),
        xaxis=axis_style("ADE Scene A"),
        yaxis=axis_style("ADE Scene B"),
    )

    return fig


def create_pair_connections(
    scenes: pd.DataFrame,
    pairs: pd.DataFrame,
    key: str,
    max_pairs: int = 50,
    use_3d: bool = True,
) -> go.Figure:
    """
    Visualize pairs as connected points in embedding space.

    Shows the spatial structure of boundary crossings.

    Args:
        scenes: DataFrame with UMAP coordinates
        pairs: DataFrame with pair data
        key: Semantic key to show
        max_pairs: Maximum number of pairs (top by ΔADE)
        use_3d: Use 3D projection (preserves density) vs 2D

    Returns:
        Plotly Figure
    """
    df = pairs[(pairs["diff_key"] == key) & (pairs["rel_delta_ade"].notna())]
    df = df.nlargest(max_pairs, "rel_delta_ade")

    if len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(text=f"No pairs for {key}", showarrow=False)
        return fig

    clip_to_umap = {
        row["clip_id"]: (row["umap_x"], row["umap_y"], row["umap_z"])
        for _, row in scenes.iterrows()
    }

    if use_3d:
        return _create_3d_connections(scenes, df, clip_to_umap, key)
    else:
        return _create_2d_connections(scenes, df, clip_to_umap, key)


def _create_3d_connections(
    scenes: pd.DataFrame,
    df: pd.DataFrame,
    clip_to_umap: dict,
    key: str,
) -> go.Figure:
    """Create 3D pair connection visualization."""
    fig = go.Figure()

    # Background points
    fig.add_trace(go.Scatter3d(
        x=scenes["umap_x"], y=scenes["umap_y"], z=scenes["umap_z"],
        mode="markers",
        marker=dict(size=2, color=THEME["border"], opacity=0.3),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Connection lines
    max_delta = df["rel_delta_ade"].max()
    lines_x, lines_y, lines_z = [], [], []
    line_colors = []

    for _, row in df.iterrows():
        if row["clip_a"] in clip_to_umap and row["clip_b"] in clip_to_umap:
            xa, ya, za = clip_to_umap[row["clip_a"]]
            xb, yb, zb = clip_to_umap[row["clip_b"]]

            lines_x.extend([xa, xb, None])
            lines_y.extend([ya, yb, None])
            lines_z.extend([za, zb, None])

            # Intensity based on ΔADE
            intensity = row["rel_delta_ade"] / max_delta
            gray = int(180 - 130 * intensity)
            line_colors.append(f"rgb({gray},{gray},{gray})")

    # Draw lines as single trace with color array
    for i in range(0, len(lines_x) - 2, 3):
        idx = i // 3
        fig.add_trace(go.Scatter3d(
            x=lines_x[i:i+3], y=lines_y[i:i+3], z=lines_z[i:i+3],
            mode="lines",
            line=dict(color=line_colors[idx], width=2),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Endpoint markers
    endpoints_x, endpoints_y, endpoints_z = [], [], []
    for _, row in df.iterrows():
        if row["clip_a"] in clip_to_umap and row["clip_b"] in clip_to_umap:
            xa, ya, za = clip_to_umap[row["clip_a"]]
            xb, yb, zb = clip_to_umap[row["clip_b"]]
            endpoints_x.extend([xa, xb])
            endpoints_y.extend([ya, yb])
            endpoints_z.extend([za, zb])

    fig.add_trace(go.Scatter3d(
        x=endpoints_x, y=endpoints_y, z=endpoints_z,
        mode="markers",
        marker=dict(size=4, color=THEME["accent"]),
        name=f"Endpoints ({len(df)} pairs)",
    ))

    from .style import scene_style
    fig.update_layout(
        **plotly_layout(f"Top {len(df)} High-ΔADE Pairs: {key}", height=550),
        scene=scene_style(),
    )

    return fig


def _create_2d_connections(
    scenes: pd.DataFrame,
    df: pd.DataFrame,
    clip_to_umap: dict,
    key: str,
) -> go.Figure:
    """Create 2D pair connection visualization."""
    fig = go.Figure()

    # Background points
    fig.add_trace(go.Scatter(
        x=scenes["umap_x"], y=scenes["umap_y"],
        mode="markers",
        marker=dict(size=2, color=THEME["border"]),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Connection lines
    max_delta = df["rel_delta_ade"].max()

    for _, row in df.iterrows():
        if row["clip_a"] in clip_to_umap and row["clip_b"] in clip_to_umap:
            xa, ya, _ = clip_to_umap[row["clip_a"]]
            xb, yb, _ = clip_to_umap[row["clip_b"]]

            intensity = row["rel_delta_ade"] / max_delta
            gray = int(180 - 130 * intensity)

            fig.add_trace(go.Scatter(
                x=[xa, xb], y=[ya, yb],
                mode="lines",
                line=dict(color=f"rgb({gray},{gray},{gray})", width=1 + intensity),
                hoverinfo="text",
                text=f"{row['value_a']} → {row['value_b']}<br>ΔADE: {row['rel_delta_ade']:.3f}",
                showlegend=False,
            ))

    # Endpoints
    endpoints_x, endpoints_y = [], []
    for _, row in df.iterrows():
        if row["clip_a"] in clip_to_umap and row["clip_b"] in clip_to_umap:
            xa, ya, _ = clip_to_umap[row["clip_a"]]
            xb, yb, _ = clip_to_umap[row["clip_b"]]
            endpoints_x.extend([xa, xb])
            endpoints_y.extend([ya, yb])

    fig.add_trace(go.Scatter(
        x=endpoints_x, y=endpoints_y,
        mode="markers",
        marker=dict(size=5, color=THEME["accent"]),
        name=f"Endpoints ({len(df)} pairs)",
    ))

    fig.update_layout(
        **plotly_layout(f"Top {len(df)} High-ΔADE Pairs: {key}", height=500),
        xaxis={**axis_style("UMAP 1"), "scaleanchor": "y"},
        yaxis=axis_style("UMAP 2"),
    )

    return fig
