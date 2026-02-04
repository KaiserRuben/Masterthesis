"""
Hypothesis-specific visualizations for the Unified Formalization.

Design principles:
- Monochromatic chrome (axes, grids, backgrounds)
- Color ONLY for data encoding and separation
- Elegant spacing and typography
- Clean, minimal annotations
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.neighbors import NearestNeighbors

from .style import THEME, plotly_layout, axis_style
from .data import CLASSIFICATION_KEYS, ADE_LABELS


# =============================================================================
# H1: BOUNDARY-ERROR CORRELATION
# =============================================================================

def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-10 else v


def compute_centroids(
    scenes: pd.DataFrame,
    embeddings: np.ndarray,
    anchors_only: bool = True,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Compute class centroids μ_c for each (key, value) pair.

    Per §4.1: μ_c = (1/n) Σ E(x_i) for x_i in class c

    Args:
        scenes: DataFrame with classification columns and emb_index
        embeddings: Full embedding matrix
        anchors_only: If True (default), compute centroids from anchor
            scenes only (VLM-labeled). Per §4.1, centroids should come
            from "representative samples", i.e., the anchors.

    Returns:
        Dict[key][value] = normalized centroid vector
    """
    # Use anchors for centroid computation (per §4.1)
    if anchors_only and "is_anchor" in scenes.columns:
        source = scenes[scenes["is_anchor"] == True]
    else:
        source = scenes

    centroids = {}

    for key in CLASSIFICATION_KEYS:
        if key not in source.columns:
            continue

        key_centroids = {}
        for value in source[key].dropna().unique():
            mask = source[key] == value
            indices = source.loc[mask, "emb_index"].values
            if len(indices) > 0:
                embs = embeddings[indices]
                centroid = normalize(np.mean(embs, axis=0))
                key_centroids[value] = centroid

        centroids[key] = key_centroids

    return centroids


def compute_boundary_margin(
    scenes: pd.DataFrame,
    embeddings: np.ndarray,
    centroids: dict[str, dict[str, np.ndarray]] | None = None,
) -> np.ndarray:
    """
    Compute d_B(z) - class-aware boundary margin per §4.3.

    margin = d(z, μ_nearest_other) - d(z, μ_own)

    where μ_own is the scene's own class centroid and μ_nearest_other
    is the closest competing class centroid.

    Higher margin = farther from competing classes = more stable.
    Negative margin = closer to competitor than to own class = unstable.

    This matches the BND-001 methodology that found r = -0.51.

    Args:
        scenes: DataFrame with classification columns
        embeddings: Full embedding matrix
        centroids: Precomputed centroids (computed if None)

    Returns:
        Array of margin values per scene (averaged across keys)
    """
    if centroids is None:
        centroids = compute_centroids(scenes, embeddings)

    n_scenes = len(scenes)
    margins_per_key = {key: np.full(n_scenes, np.nan) for key in CLASSIFICATION_KEYS}

    for key in CLASSIFICATION_KEYS:
        if key not in centroids or len(centroids[key]) < 2:
            continue

        key_centroids = centroids[key]

        for i, (_, row) in enumerate(scenes.iterrows()):
            emb_idx = row["emb_index"]
            scene_emb = embeddings[emb_idx]
            own_value = row.get(key)

            if pd.isna(own_value) or own_value not in key_centroids:
                continue

            scene_norm = normalize(scene_emb)

            # Distance to OWN class centroid
            own_centroid = key_centroids[own_value]
            dist_to_own = 1 - np.dot(own_centroid, scene_norm)

            # Distance to nearest OTHER class centroid (excluding own)
            other_distances = []
            for value, centroid in key_centroids.items():
                if value != own_value:
                    dist = 1 - np.dot(centroid, scene_norm)
                    other_distances.append(dist)

            if not other_distances:
                continue

            dist_to_nearest_other = min(other_distances)

            # Margin = distance to nearest competitor - distance to own
            # Positive = farther from competitors (stable)
            # Negative = closer to competitor than own class (unstable)
            margin = dist_to_nearest_other - dist_to_own
            margins_per_key[key][i] = margin

    # Average margin across keys (ignoring NaN)
    all_margins = np.array([margins_per_key[k] for k in CLASSIFICATION_KEYS])
    mean_margin = np.nanmean(all_margins, axis=0)

    return mean_margin


def compute_boundary_proximity(
    scenes: pd.DataFrame,
    embeddings: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """
    Compute boundary proximity using k-NN label agreement (§4.4 proxy).

    This is the BLACK-BOX approximation when centroids unavailable.
    For the precise white-box method, use compute_boundary_margin().

    Returns array where higher = farther from boundary.
    """
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(embeddings)
    _, indices = nn.kneighbors(embeddings)

    proximity = np.ones(len(scenes))

    for key in CLASSIFICATION_KEYS:
        if key not in scenes.columns:
            continue
        labels = scenes[key].values
        for i in range(len(scenes)):
            own_label = labels[i]
            neighbor_labels = labels[indices[i, 1:]]
            same_class = np.mean(neighbor_labels == own_label)
            proximity[i] = min(proximity[i], same_class)

    return proximity


def create_h1_correlation_plot(
    scenes: pd.DataFrame,
    embeddings: np.ndarray,
    k: int = 10,
    show_regression: bool = True,
    show_confidence: bool = True,
    method: str = "margin",  # "margin" (§4.3) or "proxy" (§4.4)
) -> go.Figure:
    """
    Scatter plot: boundary margin d_B vs trajectory error.

    Evidence for H1: Cor(d_B(z), epsilon_O(z)) < 0

    Args:
        method: "margin" uses centroid-based Voronoi margin (§4.3, precise)
                "proxy" uses k-NN label agreement (§4.4, black-box)
    """
    mask = scenes["ade"].notna()
    scenes_with_ade = scenes[mask].copy().reset_index(drop=True)

    if len(scenes_with_ade) < 10:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data", showarrow=False)
        return fig

    # Compute boundary metric using selected method
    if method == "margin":
        # Precise: centroid-based Voronoi margin (§4.3)
        d_B = compute_boundary_margin(scenes_with_ade, embeddings)
        x_label = "Boundary Margin d<sub>B</sub> (centroid-based)"
    else:
        # Proxy: k-NN label agreement (§4.4)
        embeddings_subset = embeddings[scenes_with_ade["emb_index"].values]
        d_B = compute_boundary_proximity(scenes_with_ade, embeddings_subset, k=k)
        x_label = "Boundary Proximity (k-NN proxy)"

    ade = scenes_with_ade["ade"].values

    # Filter out NaN margins
    valid = ~np.isnan(d_B)
    d_B = d_B[valid]
    ade = ade[valid]

    if len(d_B) < 10:
        fig = go.Figure()
        fig.add_annotation(text=f"Insufficient valid data (n={len(d_B)})", showarrow=False)
        return fig

    # Statistics
    r, p = stats.pearsonr(d_B, ade)
    rho, _ = stats.spearmanr(d_B, ade)

    # Bootstrap CI
    n = len(d_B)
    boot_r = []
    for _ in range(1000):
        idx = np.random.choice(n, n, replace=True)
        boot_r.append(stats.pearsonr(d_B[idx], ade[idx])[0])
    ci_low, ci_high = np.percentile(boot_r, [2.5, 97.5])

    fig = go.Figure()

    # Points - monochrome base, color only for ADE severity
    ade_colors = THEME["ade"]
    colors = [ade_colors["low"] if a < 1.0 else
              ade_colors["medium"] if a < 2.5 else
              ade_colors["high"] if a < 5.0 else
              ade_colors["critical"] for a in ade]

    fig.add_trace(go.Scatter(
        x=d_B,
        y=ade,
        mode="markers",
        marker=dict(
            size=7,
            color=colors,
            opacity=0.6,
        ),
        hovertemplate="d_B: %{x:.3f}<br>ADE: %{y:.2f}<extra></extra>",
        showlegend=False,
    ))

    # Regression line - subtle accent
    if show_regression:
        slope, intercept = np.polyfit(d_B, ade, 1)
        x_line = np.linspace(d_B.min(), d_B.max(), 50)
        y_line = slope * x_line + intercept

        if show_confidence:
            residuals = ade - (slope * d_B + intercept)
            se = np.std(residuals)
            fig.add_trace(go.Scatter(
                x=np.concatenate([x_line, x_line[::-1]]),
                y=np.concatenate([y_line + 1.96*se, (y_line - 1.96*se)[::-1]]),
                fill="toself",
                fillcolor="rgba(0,0,0,0.05)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            ))

        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            line=dict(color=THEME["text_secondary"], width=2),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Stats annotation - highlight significance
    sig_marker = "**" if p < 0.01 else "*" if p < 0.05 else ""
    fig.add_annotation(
        x=0.98, y=0.98,
        xref="paper", yref="paper",
        text=f"r = {r:.3f}{sig_marker}  p = {p:.4f}  n = {n}",
        showarrow=False,
        font=dict(size=11, family=THEME["font_mono"], color=THEME["text_secondary"]),
        xanchor="right", yanchor="top",
    )

    fig.update_layout(
        **plotly_layout("", height=450, show_legend=False, margin=dict(l=60, r=30, t=30, b=50)),
        xaxis={
            **axis_style(""),
            "title": dict(text=x_label, font=dict(size=12)),
        },
        yaxis={
            **axis_style(""),
            "title": dict(text="Trajectory Error (ADE)", font=dict(size=12)),
        },
    )

    return fig


# =============================================================================
# H2: ANISOTROPY
# =============================================================================

def compute_anisotropy_vector(
    pairs: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, float], float]:
    """Compute anisotropy vector a = (S_k1, ..., S_km)."""
    means, stds = {}, {}

    for key in CLASSIFICATION_KEYS:
        df = pairs[(pairs["diff_key"] == key) & (pairs["rel_delta_ade"].notna())]
        if len(df) > 0:
            means[key] = df["rel_delta_ade"].mean()
            stds[key] = df["rel_delta_ade"].std()
        else:
            means[key] = 0
            stds[key] = 0

    non_zero = [v for v in means.values() if v > 0]
    alpha = max(non_zero) / min(non_zero) if len(non_zero) >= 2 else 1.0

    return means, stds, alpha


def create_h2_anisotropy_plot(
    pairs: pd.DataFrame,
    style: str = "bar",
) -> go.Figure:
    """
    Horizontal bar chart of sensitivity per semantic key.

    Evidence for H2: sensitivity varies by axis.
    """
    means, stds, alpha = compute_anisotropy_vector(pairs)
    sorted_keys = sorted(means.keys(), key=lambda k: means[k], reverse=True)
    n_per_key = {k: len(pairs[(pairs["diff_key"] == k) & (pairs["rel_delta_ade"].notna())])
                 for k in sorted_keys}

    if style == "radar":
        fig = go.Figure()
        theta = sorted_keys + [sorted_keys[0]]
        r = [means[k] for k in sorted_keys] + [means[sorted_keys[0]]]

        fig.add_trace(go.Scatterpolar(
            r=r,
            theta=theta,
            fill="toself",
            fillcolor="rgba(100,100,100,0.15)",
            line=dict(color=THEME["text_secondary"], width=2),
        ))

        fig.update_layout(
            **plotly_layout("", height=450, show_legend=False, margin=dict(l=80, r=80, t=30, b=30)),
            polar=dict(
                bgcolor=THEME["bg"],
                radialaxis=dict(
                    visible=True,
                    range=[0, max(means.values()) * 1.2],
                    gridcolor=THEME["grid"],
                    linecolor=THEME["border"],
                ),
                angularaxis=dict(
                    gridcolor=THEME["grid"],
                    linecolor=THEME["border"],
                ),
            ),
        )
        return fig

    # Bar chart - monochrome with single accent for top
    fig = go.Figure()

    colors = [THEME["text"] if i == 0 else THEME["point_inactive"]
              for i in range(len(sorted_keys))]

    fig.add_trace(go.Bar(
        y=sorted_keys,
        x=[means[k] for k in sorted_keys],
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(width=0),
        ),
        error_x=dict(
            type="data",
            array=[stds[k] / np.sqrt(n_per_key[k]) * 1.96 for k in sorted_keys],
            color=THEME["text_muted"],
            thickness=1.5,
            width=3,
        ),
        hovertemplate="<b>%{y}</b><br>%{x:.3f} (n=%{customdata})<extra></extra>",
        customdata=[n_per_key[k] for k in sorted_keys],
    ))

    # Anisotropy ratio annotation
    fig.add_annotation(
        x=0.98, y=0.02,
        xref="paper", yref="paper",
        text=f"α = {alpha:.1f}",
        showarrow=False,
        font=dict(size=14, family=THEME["font_mono"], color=THEME["text_secondary"]),
        xanchor="right", yanchor="bottom",
    )

    fig.update_layout(
        **plotly_layout("", height=320, show_legend=False, margin=dict(l=120, r=40, t=20, b=50)),
        xaxis={
            **axis_style(""),
            "title": dict(text="Mean Sensitivity (ΔADE)", font=dict(size=11)),
            "zeroline": True,
            "zerolinecolor": THEME["border"],
        },
        yaxis=dict(
            tickfont=dict(size=11, color=THEME["text"]),
            showgrid=False,
        ),
        bargap=0.4,
    )

    return fig


# =============================================================================
# H3: ASYMMETRY
# =============================================================================

def compute_transition_asymmetry(
    pairs: pd.DataFrame,
    key: str,
) -> pd.DataFrame:
    """Compute asymmetry matrix S_ij - S_ji for a semantic key."""
    df = pairs[(pairs["diff_key"] == key) & (pairs["rel_delta_ade"].notna())]

    if len(df) == 0:
        return pd.DataFrame()

    forward = df.groupby(["value_a", "value_b"]).agg({
        "rel_delta_ade": ["mean", "count"],
    }).reset_index()
    forward.columns = ["source", "target", "forward_ade", "forward_n"]

    backward = df.groupby(["value_b", "value_a"]).agg({
        "rel_delta_ade": ["mean", "count"],
    }).reset_index()
    backward.columns = ["source", "target", "backward_ade", "backward_n"]

    merged = forward.merge(backward, on=["source", "target"], how="outer").fillna(0)
    merged["asymmetry"] = merged["forward_ade"] - merged["backward_ade"]

    return merged


def create_h3_asymmetry_heatmap(
    pairs: pd.DataFrame,
    key: str,
) -> go.Figure:
    """
    Heatmap of transition asymmetry S_ij - S_ji.

    Evidence for H3: transitions are directional.
    """
    asym = compute_transition_asymmetry(pairs, key)

    if len(asym) == 0:
        fig = go.Figure()
        fig.add_annotation(text=f"No data for {key}", showarrow=False)
        return fig

    labels = sorted(set(asym["source"]) | set(asym["target"]))
    n = len(labels)
    matrix = np.zeros((n, n))
    label_to_idx = {l: i for i, l in enumerate(labels)}

    for _, row in asym.iterrows():
        i, j = label_to_idx[row["source"]], label_to_idx[row["target"]]
        matrix[i, j] = row["forward_ade"]

    asymmetry = matrix - matrix.T
    max_abs = np.abs(asymmetry).max()

    fig = go.Figure()

    # Diverging colorscale from THEME
    fig.add_trace(go.Heatmap(
        z=asymmetry,
        x=labels,
        y=labels,
        colorscale=[
            [0.0, THEME["diverging"]["negative"]],
            [0.5, THEME["diverging"]["neutral"]],
            [1.0, THEME["diverging"]["positive"]],
        ],
        zmid=0,
        zmin=-max_abs if max_abs > 0 else -1,
        zmax=max_abs if max_abs > 0 else 1,
        colorbar=dict(
            title=dict(text="Δ", font=dict(size=11)),
            thickness=12,
            len=0.6,
            tickfont=dict(size=9),
        ),
        hovertemplate="%{y} → %{x}<br>Δ = %{z:.3f}<extra></extra>",
    ))

    # Cell annotations - only for non-diagonal, significant values
    for i in range(n):
        for j in range(n):
            if i != j and abs(asymmetry[i, j]) > 0.02:
                text_color = THEME["surface"] if abs(asymmetry[i, j]) > max_abs * 0.5 else THEME["text"]
                fig.add_annotation(
                    x=labels[j], y=labels[i],
                    text=f"{asymmetry[i, j]:.2f}",
                    showarrow=False,
                    font=dict(color=text_color, size=10),
                )

    fig.update_layout(
        **plotly_layout("", height=400, show_legend=False, margin=dict(l=80, r=60, t=20, b=80)),
        xaxis=dict(
            side="bottom",
            tickfont=dict(size=10),
            tickangle=45,
        ),
        yaxis=dict(
            tickfont=dict(size=10),
            autorange="reversed",
        ),
    )

    return fig


def create_h3_asymmetry_distribution(
    pairs: pd.DataFrame,
) -> go.Figure:
    """
    Histogram of asymmetry values across all transitions.

    If symmetric, distribution centers at 0.
    """
    all_asymmetries = []
    for key in CLASSIFICATION_KEYS:
        asym = compute_transition_asymmetry(pairs, key)
        if len(asym) > 0:
            all_asymmetries.extend(asym["asymmetry"].tolist())

    if not all_asymmetries:
        fig = go.Figure()
        fig.add_annotation(text="No data", showarrow=False)
        return fig

    asymmetries = np.array(all_asymmetries)
    mean_asym = np.mean(asymmetries)
    _, p_value = stats.ttest_1samp(asymmetries, 0)

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=asymmetries,
        nbinsx=25,
        marker=dict(
            color=THEME["point_inactive"],
            line=dict(color=THEME["border"], width=0.5),
        ),
        hovertemplate="Asymmetry: %{x:.2f}<br>Count: %{y}<extra></extra>",
    ))

    # Zero reference
    fig.add_vline(x=0, line=dict(color=THEME["text_secondary"], width=1.5, dash="dash"))

    # Mean line (subtle)
    if abs(mean_asym) > 0.01:
        fig.add_vline(x=mean_asym, line=dict(color=THEME["text_muted"], width=1))

    # Stats annotation
    fig.add_annotation(
        x=0.98, y=0.98,
        xref="paper", yref="paper",
        text=f"μ = {mean_asym:.3f}  p = {p_value:.3f}",
        showarrow=False,
        font=dict(size=10, family=THEME["font_mono"], color=THEME["text_secondary"]),
        xanchor="right", yanchor="top",
    )

    fig.update_layout(
        **plotly_layout("", height=300, show_legend=False, margin=dict(l=60, r=30, t=20, b=50)),
        xaxis={
            **axis_style(""),
            "title": dict(text="Asymmetry (S<sub>ij</sub> − S<sub>ji</sub>)", font=dict(size=11)),
        },
        yaxis={
            **axis_style(""),
            "title": dict(text="Count", font=dict(size=11)),
        },
        bargap=0.02,
    )

    return fig


# =============================================================================
# STABILITY MAP
# =============================================================================

def create_stability_map(
    pairs: pd.DataFrame,
    show_change_rate: bool = True,
) -> go.Figure:
    """
    Primary summary: sensitivity ranking by semantic key.
    """
    metrics = []
    for key in CLASSIFICATION_KEYS:
        df = pairs[(pairs["diff_key"] == key) & (pairs["rel_delta_ade"].notna())]
        if len(df) > 0:
            metrics.append({
                "key": key,
                "mean_delta_ade": df["rel_delta_ade"].mean(),
                "std_delta_ade": df["rel_delta_ade"].std(),
                "ade_change_rate": df["ade_class_changed"].mean(),
                "n": len(df),
            })

    if not metrics:
        fig = go.Figure()
        fig.add_annotation(text="No data", showarrow=False)
        return fig

    df_m = pd.DataFrame(metrics).sort_values("mean_delta_ade", ascending=True)

    if show_change_rate:
        fig = make_subplots(
            rows=1, cols=2,
            horizontal_spacing=0.2,
            column_widths=[0.55, 0.45],
        )

        # Left: Sensitivity (lollipop style)
        fig.add_trace(go.Scatter(
            x=df_m["mean_delta_ade"],
            y=df_m["key"],
            mode="markers",
            marker=dict(size=10, color=THEME["text"]),
            error_x=dict(
                type="data",
                array=df_m["std_delta_ade"] / np.sqrt(df_m["n"]) * 1.96,
                color=THEME["text_muted"],
                thickness=1.5,
            ),
            hovertemplate="<b>%{y}</b><br>%{x:.3f}<extra></extra>",
            showlegend=False,
        ), row=1, col=1)

        # Stems
        for _, row in df_m.iterrows():
            fig.add_shape(
                type="line",
                x0=0, x1=row["mean_delta_ade"],
                y0=row["key"], y1=row["key"],
                line=dict(color=THEME["border"], width=1),
                row=1, col=1,
            )

        # Right: Change rate (simple bars)
        fig.add_trace(go.Bar(
            x=df_m["ade_change_rate"] * 100,
            y=df_m["key"],
            orientation="h",
            marker=dict(color=THEME["point_inactive"]),
            hovertemplate="<b>%{y}</b><br>%{x:.1f}%<extra></extra>",
            showlegend=False,
        ), row=1, col=2)

        fig.update_xaxes(
            title_text="Mean ΔADE",
            title_font_size=10,
            row=1, col=1,
            gridcolor=THEME["grid"],
            zeroline=True, zerolinecolor=THEME["border"],
        )
        fig.update_xaxes(
            title_text="Class Change %",
            title_font_size=10,
            row=1, col=2,
            gridcolor=THEME["grid"],
        )
        fig.update_yaxes(showticklabels=True, tickfont_size=10, row=1, col=1)
        fig.update_yaxes(showticklabels=False, row=1, col=2)

        fig.update_layout(
            **plotly_layout("", height=280, show_legend=False, margin=dict(l=110, r=30, t=20, b=45)),
        )
    else:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=df_m["key"],
            x=df_m["mean_delta_ade"],
            orientation="h",
            marker=dict(color=THEME["text_muted"]),
        ))
        fig.update_layout(
            **plotly_layout("", height=280, show_legend=False, margin=dict(l=110, r=30, t=20, b=45)),
            xaxis=axis_style("Mean ΔADE"),
        )

    return fig


# =============================================================================
# DIVERGENCE CURVES
# =============================================================================

def create_divergence_curve_plot(
    pairs: pd.DataFrame,
    key: str,
    n_curves: int = 5,
) -> go.Figure:
    """
    Divergence curves delta(t) along transition paths.
    """
    df = pairs[(pairs["diff_key"] == key) & (pairs["rel_delta_ade"].notna())]

    if len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(text=f"No data for {key}", showarrow=False)
        return fig

    transitions = df.groupby(["value_a", "value_b"]).agg({
        "ade_a": "mean",
        "ade_b": "mean",
        "rel_delta_ade": "mean",
        "clip_a": "count",
    }).reset_index()
    transitions.columns = ["source", "target", "ade_start", "ade_end", "mean_delta", "n"]

    top = transitions.nlargest(n_curves, "mean_delta")

    fig = go.Figure()

    # Use THEME grays sequence
    grays = THEME["grays"]

    for i, (_, row) in enumerate(top.iterrows()):
        t = np.linspace(0, 1, 30)
        delta = row["ade_start"] + (row["ade_end"] - row["ade_start"]) * (3*t**2 - 2*t**3)
        label = f"{row['source']} → {row['target']}"

        fig.add_trace(go.Scatter(
            x=t,
            y=delta,
            mode="lines",
            line=dict(color=grays[i % len(grays)], width=2),
            name=label,
            hovertemplate=f"<b>{label}</b><br>t: %{{x:.2f}}<br>δ: %{{y:.2f}}<extra></extra>",
        ))

    # Reference line at t=0.5
    fig.add_vline(x=0.5, line=dict(color=THEME["grid"], width=1, dash="dot"))

    fig.update_layout(
        **plotly_layout("", height=350, show_legend=True, margin=dict(l=50, r=30, t=20, b=45)),
        xaxis={
            **axis_style(""),
            "title": dict(text="t", font=dict(size=11)),
            "range": [0, 1],
        },
        yaxis={
            **axis_style(""),
            "title": dict(text="δ(t)", font=dict(size=11)),
        },
        legend=dict(
            x=0.02, y=0.98,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=THEME["border"],
            borderwidth=1,
            font=dict(size=9),
        ),
    )

    return fig


# =============================================================================
# THREE-LEVEL SUMMARY
# =============================================================================

def create_three_level_summary(
    scenes: pd.DataFrame,
    pairs: pd.DataFrame,
    embeddings: np.ndarray,
) -> go.Figure:
    """
    3-panel overview of boundary levels: Input, Embedding, Output.
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Input (κ)", "Embedding (d<sub>B</sub>)", "Output (C)"],
        horizontal_spacing=0.08,
    )

    # Level 1: Key cardinality
    key_counts = {k: scenes[k].nunique() for k in CLASSIFICATION_KEYS}
    fig.add_trace(go.Bar(
        x=list(key_counts.keys()),
        y=list(key_counts.values()),
        marker=dict(color=THEME["text_muted"]),
        showlegend=False,
    ), row=1, col=1)

    # Level 2: Boundary proximity
    mask = scenes["ade"].notna()
    if mask.sum() > 10:
        proximity = compute_boundary_proximity(
            scenes[mask],
            embeddings[scenes[mask]["emb_index"].values]
        )
        fig.add_trace(go.Histogram(
            x=proximity,
            nbinsx=15,
            marker=dict(color=THEME["text_muted"]),
            showlegend=False,
        ), row=1, col=2)

    # Level 3: ADE classes
    ade_counts = scenes["ade_class"].value_counts()
    ade_order = ["low", "medium", "high", "critical"]
    ordered_counts = [ade_counts.get(c, 0) for c in ade_order if c in ade_counts.index]
    ordered_labels = [c for c in ade_order if c in ade_counts.index]
    ade_colors = [THEME["ade"].get(c, THEME["text_muted"]) for c in ordered_labels]

    fig.add_trace(go.Bar(
        x=ordered_labels,
        y=ordered_counts,
        marker=dict(color=ade_colors),
        showlegend=False,
    ), row=1, col=3)

    # Clean up axes
    fig.update_xaxes(tickangle=45, tickfont_size=9, row=1, col=1)
    fig.update_xaxes(tickfont_size=9, row=1, col=2)
    fig.update_xaxes(tickfont_size=9, row=1, col=3)

    fig.update_yaxes(title_text="# values", title_font_size=10, row=1, col=1)
    fig.update_yaxes(title_text="count", title_font_size=10, row=1, col=2)
    fig.update_yaxes(title_text="count", title_font_size=10, row=1, col=3)

    fig.update_layout(
        **plotly_layout("", height=280, show_legend=False, margin=dict(l=50, r=30, t=40, b=60)),
    )

    # Style subplot titles
    for annotation in fig.layout.annotations:
        annotation.font.size = 11
        annotation.font.color = THEME["text_secondary"]

    return fig


# =============================================================================
# EXPORT
# =============================================================================

def export_figure_for_print(
    fig: go.Figure,
    filename: str,
    width: int = 800,
    height: int = 500,
    scale: int = 3,
) -> None:
    """Export figure as high-resolution PNG/PDF/HTML."""
    import os
    os.makedirs("figures", exist_ok=True)

    fig.update_layout(
        width=width,
        height=height,
        font=dict(size=12),
    )

    fig.write_image(f"figures/{filename}.png", scale=scale)
    fig.write_image(f"figures/{filename}.pdf")
    fig.write_html(f"figures/{filename}.html", include_plotlyjs="cdn")

    print(f"Exported: figures/{filename}.{{png,pdf,html}}")
