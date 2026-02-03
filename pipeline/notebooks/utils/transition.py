"""
Transition analysis: flows, ADE matrices, and danger zones.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors

from .style import THEME, plotly_layout, axis_style
from .data import ADE_LABELS, CLASSIFICATION_KEYS


def create_transition_sankey(
    pairs: pd.DataFrame,
    key: str,
) -> go.Figure:
    """
    Create Sankey diagram showing class transitions for a semantic key.

    Node size = class frequency
    Edge width = number of pairs
    Edge color = mean ΔADE (darker = higher)

    Args:
        pairs: DataFrame with diff_key, value_a, value_b, rel_delta_ade
        key: Semantic key to analyze

    Returns:
        Plotly Figure
    """
    df = pairs[(pairs["diff_key"] == key) & (pairs["rel_delta_ade"].notna())]

    if len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(text=f"No data for {key}", showarrow=False)
        return fig

    # Aggregate transitions
    trans = df.groupby(["value_a", "value_b"]).agg({
        "rel_delta_ade": "mean",
        "clip_a": "count",
        "ade_class_changed": "mean",
    }).reset_index()
    trans.columns = ["source", "target", "mean_ade", "count", "ade_change_rate"]

    # Build node list
    labels = sorted(set(trans["source"]) | set(trans["target"]))
    label_to_idx = {l: i for i, l in enumerate(labels)}

    # Edge colors: ADE scale (green=low impact, red=high impact)
    # High ΔADE transitions should visually pop
    max_ade = trans["mean_ade"].max() if trans["mean_ade"].max() > 0 else 1
    edge_colors = []
    for ade in trans["mean_ade"]:
        ratio = ade / max_ade
        if ratio < 0.33:
            edge_colors.append("rgba(58, 158, 92, 0.5)")   # Green, low impact
        elif ratio < 0.66:
            edge_colors.append("rgba(230, 161, 50, 0.6)")  # Amber, medium
        else:
            edge_colors.append("rgba(214, 107, 43, 0.75)") # Orange, high impact

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            label=labels,
            color=[THEME["text_muted"]] * len(labels),
        ),
        link=dict(
            source=[label_to_idx[s] for s in trans["source"]],
            target=[label_to_idx[t] for t in trans["target"]],
            value=trans["count"],
            color=edge_colors,
            customdata=trans[["mean_ade", "ade_change_rate"]].values,
            hovertemplate=(
                "%{source.label} → %{target.label}<br>"
                "Count: %{value}<br>"
                "Mean ΔADE: %{customdata[0]:.3f}<br>"
                "ADE class change: %{customdata[1]:.1%}<extra></extra>"
            ),
        ),
    ))

    fig.update_layout(
        **plotly_layout(f"Transitions: {key}", height=400),
    )

    return fig


def create_ade_transition_matrix(
    pairs: pd.DataFrame,
    key: str | None = None,
) -> go.Figure:
    """
    Create heatmap of ADE class transitions.

    Shows how often boundary crossings cause ADE class changes.

    Args:
        pairs: DataFrame with ade_class_a, ade_class_b
        key: Filter to specific semantic key, or None for all

    Returns:
        Plotly Figure
    """
    df = pairs[pairs["rel_delta_ade"].notna()]
    if key:
        df = df[df["diff_key"] == key]

    if len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data", showarrow=False)
        return fig

    # Count transitions
    trans = df.groupby(["ade_class_a", "ade_class_b"]).size().unstack(fill_value=0)

    # Reorder to standard ADE class order
    order = [c for c in ADE_LABELS if c in trans.index]
    cols = [c for c in ADE_LABELS if c in trans.columns]
    trans = trans.reindex(index=order, columns=cols, fill_value=0)

    # Annotations
    annotations = []
    max_val = trans.values.max()
    for i, row_label in enumerate(trans.index):
        for j, col_label in enumerate(trans.columns):
            val = trans.iloc[i, j]
            text_color = "#ffffff" if val > max_val / 2 else THEME["text"]
            annotations.append(dict(
                x=col_label, y=row_label,
                text=str(int(val)),
                showarrow=False,
                font=dict(color=text_color, size=11),
            ))

    fig = go.Figure(go.Heatmap(
        z=trans.values,
        x=trans.columns.tolist(),
        y=trans.index.tolist(),
        colorscale="Greys",
        colorbar=dict(title="Count"),
    ))

    title = f"ADE Class Transitions ({key})" if key else "ADE Class Transitions"

    fig.update_layout(
        **plotly_layout(title, height=400),
        xaxis=axis_style("ADE Class B"),
        yaxis=axis_style("ADE Class A"),
        annotations=annotations,
    )

    return fig


def identify_danger_zones(
    scenes: pd.DataFrame,
    embeddings: np.ndarray,
    pairs: pd.DataFrame,
    k: int = 10,
    boundary_threshold: float = 0.3,
    ade_change_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Identify scenes in danger zones.

    Danger zone = near boundary (>threshold different-class neighbors)
                  AND high ADE class change rate

    Args:
        scenes: DataFrame with labels and UMAP coords
        embeddings: Full embedding matrix
        pairs: DataFrame with pair data
        k: Number of neighbors for boundary detection
        boundary_threshold: Min % different neighbors to be "near boundary"
        ade_change_threshold: Min ADE change rate to be "dangerous"

    Returns:
        DataFrame with danger scores per scene
    """
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(embeddings)
    _, indices = nn.kneighbors(embeddings)

    results = []

    for _, row in scenes.iterrows():
        clip_id = row["clip_id"]
        emb_idx = row["emb_index"]

        # Compute boundary proximity for each key
        boundary_scores = {}
        for key in CLASSIFICATION_KEYS:
            labels = scenes[key].values
            own_label = labels[emb_idx]
            neighbor_labels = labels[indices[emb_idx, 1:]]
            boundary_scores[key] = np.mean(neighbor_labels != own_label)

        # Get ADE change rate for pairs involving this scene
        scene_pairs = pairs[
            ((pairs["clip_a"] == clip_id) | (pairs["clip_b"] == clip_id)) &
            (pairs["ade_class_changed"].notna())
        ]

        if len(scene_pairs) > 0:
            ade_change_rate = scene_pairs["ade_class_changed"].mean()
            mean_delta_ade = scene_pairs["rel_delta_ade"].mean()
        else:
            ade_change_rate = 0
            mean_delta_ade = 0

        max_boundary = max(boundary_scores.values())
        boundary_key = max(boundary_scores, key=boundary_scores.get)

        # Danger score
        danger_score = max_boundary * ade_change_rate
        is_danger = (max_boundary > boundary_threshold and
                     ade_change_rate > ade_change_threshold)

        results.append({
            "clip_id": clip_id,
            "emb_index": emb_idx,
            "max_boundary_proximity": max_boundary,
            "boundary_key": boundary_key,
            "ade_change_rate": ade_change_rate,
            "mean_delta_ade": mean_delta_ade,
            "danger_score": danger_score,
            "is_danger": is_danger,
            "umap_x": row["umap_x"],
            "umap_y": row["umap_y"],
            "umap_z": row["umap_z"],
            "ade": row["ade"],
            "ade_class": row["ade_class"],
        })

    return pd.DataFrame(results)


def create_danger_zone_plot(danger_df: pd.DataFrame) -> go.Figure:
    """
    Visualize danger zones in 2D embedding space.

    Danger points have glow effect; all points colored by danger score.

    Args:
        danger_df: Output from identify_danger_zones()

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    # All points, colored by danger score
    # Danger should FEEL dangerous: neutral -> amber -> red
    danger_scale = [
        [0.0, THEME["point_inactive"]],   # Safe: neutral gray
        [0.3, THEME["ade"]["medium"]],    # Caution: amber
        [0.6, THEME["ade"]["high"]],      # Warning: orange
        [1.0, THEME["ade"]["critical"]],  # Danger: red
    ]

    fig.add_trace(go.Scatter(
        x=danger_df["umap_x"], y=danger_df["umap_y"],
        mode="markers",
        marker=dict(
            size=5,
            color=danger_df["danger_score"],
            colorscale=danger_scale,
            colorbar=dict(title="Danger"),
            cmin=0,
            cmax=danger_df["danger_score"].quantile(0.95),
        ),
        text=danger_df.apply(
            lambda r: (
                f"Boundary: {r['boundary_key']} ({r['max_boundary_proximity']:.1%})<br>"
                f"ADE change rate: {r['ade_change_rate']:.1%}<br>"
                f"Mean ΔADE: {r['mean_delta_ade']:.3f}"
            ),
            axis=1,
        ),
        hoverinfo="text",
        name="All scenes",
    ))

    # Danger zone highlights
    danger_pts = danger_df[danger_df["is_danger"]]
    if len(danger_pts) > 0:
        # Glow
        fig.add_trace(go.Scatter(
            x=danger_pts["umap_x"], y=danger_pts["umap_y"],
            mode="markers",
            marker=dict(size=18, color=THEME["danger_glow"]),
            hoverinfo="skip",
            showlegend=False,
        ))
        # Core
        fig.add_trace(go.Scatter(
            x=danger_pts["umap_x"], y=danger_pts["umap_y"],
            mode="markers",
            marker=dict(
                size=7,
                color=THEME["highlight"],
                line=dict(color="#ffffff", width=1),
            ),
            name=f"Danger zones ({len(danger_pts)})",
        ))

    fig.update_layout(
        **plotly_layout("Danger Zone Map", height=500),
        xaxis={**axis_style("UMAP 1"), "scaleanchor": "y"},
        yaxis=axis_style("UMAP 2"),
    )

    return fig
