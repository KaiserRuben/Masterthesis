#!/usr/bin/env python3
"""
BND-004: Dense 3D Visualizations

Creates information-dense visualizations where axes are computed metrics,
not raw categories. Each visualization packs multiple dimensions of insight.

Usage:
    python visualize_3d.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Project root
PROJECT_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from archive.pipeline.step_1_embed import TEXT_VOCABULARY

# Paths
DATA_DIR = PROJECT_ROOT / "data" / "BND-004"
RESULTS_FILE = DATA_DIR / "perturbation_results.parquet"
ANALYSIS_FILE = DATA_DIR / "perturbation_analysis.json"
FIG_DIR = DATA_DIR / "figures"

# Key categories for color coding
KEY_CATEGORIES = {
    "weather": "scene_context",
    "time_of_day": "scene_context",
    "road_type": "scene_context",
    "traffic_situation": "scene_context",
    "depth_complexity": "spatial",
    "occlusion_level": "spatial",
    "visual_degradation": "perceptual",
    "required_action": "safety",
    "safety_criticality": "safety",
    "lane_marking_type": "spatial",
    "pedestrians_present": "safety",
    "cyclists_present": "safety",
    "construction_activity": "scene_context",
    "traffic_signals_visible": "safety",
    "similar_object_confusion": "perceptual",
}

CATEGORY_COLORS = {
    "scene_context": "#4C78A8",  # blue
    "spatial": "#F58518",        # orange
    "perceptual": "#E45756",     # red
    "safety": "#72B7B2",         # teal
}


def load_data():
    """Load results and analysis."""
    df = pd.read_parquet(RESULTS_FILE)
    with open(ANALYSIS_FILE) as f:
        analysis = json.load(f)
    return df, analysis


def generate_modality_trust_landscape(df: pd.DataFrame, analysis: dict, output_dir: Path):
    """
    The main visualization: Modality Trust Landscape

    A single 3D plot where each key is positioned by computed metrics:
    - X: Alignment Effect (← text trusted | image trusted →)
    - Y: Sensitivity (low → high variance in ADE)
    - Z: Effect Size (Cohen's d)

    Visual encodings:
    - Point size: Statistical significance (-log10 p-value)
    - Color: Key category (scene/spatial/perceptual/safety)
    - Labels: Key names with arrows showing trust direction

    This reveals: Which keys matter? Which modality does the model trust for each?
    """
    import plotly.graph_objects as go

    print("Generating Modality Trust Landscape...")

    alignment = analysis.get("alignment_effects", {})
    sensitivity = analysis.get("key_sensitivity", {})

    if not alignment:
        print("  No alignment data")
        return

    # Compute metrics for each key
    keys = []
    x_vals = []  # alignment effect
    y_vals = []  # sensitivity
    z_vals = []  # cohen's d
    sizes = []   # significance
    colors = []  # category
    hovers = []  # hover text

    for key in alignment.keys():
        if key not in sensitivity:
            continue

        ae = alignment[key]
        sens = sensitivity[key]

        keys.append(key)
        x_vals.append(ae["alignment_effect"])
        y_vals.append(sens["mean_sensitivity"])
        z_vals.append(abs(ae["cohens_d"]))  # absolute effect size

        # Size by significance: -log10(p) clamped to [1, 4]
        sig_size = min(4, max(1, -np.log10(ae["p_value"] + 1e-10)))
        sizes.append(sig_size * 15)

        # Color by category
        cat = KEY_CATEGORIES.get(key, "scene_context")
        colors.append(CATEGORY_COLORS[cat])

        # Rich hover text
        trust = "IMAGE" if ae["alignment_effect"] > 0 else "TEXT"
        sig_label = "***" if ae["p_value"] < 0.001 else ("**" if ae["p_value"] < 0.01 else ("*" if ae["p_value"] < 0.05 else ""))
        hovers.append(
            f"<b>{key.replace('_', ' ').title()}</b><br>"
            f"─────────────────<br>"
            f"Trust: <b>{trust}</b> {sig_label}<br>"
            f"Effect: {ae['alignment_effect']:+.3f}<br>"
            f"Cohen's d: {ae['cohens_d']:.2f}<br>"
            f"Sensitivity: {sens['mean_sensitivity']:.3f}<br>"
            f"p-value: {ae['p_value']:.4f}<br>"
            f"─────────────────<br>"
            f"Aligned ADE: {ae['mean_ade_aligned']:.2f}±{ae['std_ade_aligned']:.2f}<br>"
            f"Misaligned ADE: {ae['mean_ade_misaligned']:.2f}±{ae['std_ade_misaligned']:.2f}"
        )

    fig = go.Figure()

    # Add key points
    fig.add_trace(go.Scatter3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        mode='markers+text',
        marker=dict(
            size=sizes,
            color=colors,
            opacity=0.8,
            line=dict(width=1, color='white'),
        ),
        text=[k.replace("_", " ") for k in keys],
        textposition='top center',
        textfont=dict(size=9),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hovers,
    ))

    # Add reference planes
    # Zero-effect plane (X=0)
    x_range = [min(x_vals) - 0.1, max(x_vals) + 0.1]
    y_range = [min(y_vals) - 0.1, max(y_vals) + 0.1]
    z_range = [0, max(z_vals) + 0.1]

    # Vertical plane at x=0 (trust boundary)
    fig.add_trace(go.Surface(
        x=[[0, 0], [0, 0]],
        y=[[y_range[0], y_range[1]], [y_range[0], y_range[1]]],
        z=[[z_range[0], z_range[0]], [z_range[1], z_range[1]]],
        colorscale=[[0, 'rgba(128,128,128,0.1)'], [1, 'rgba(128,128,128,0.1)']],
        showscale=False,
        hoverinfo='skip',
    ))

    # Add category legend as annotations
    for i, (cat, color) in enumerate(CATEGORY_COLORS.items()):
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            name=cat.replace("_", " ").title(),
            showlegend=True,
        ))

    fig.update_layout(
        title=dict(
            text="<b>Modality Trust Landscape</b><br>"
                 "<sup>X: Alignment Effect (←text | image→) · Y: Sensitivity · Z: Effect Size · Size: Significance</sup>",
            x=0.5,
        ),
        scene=dict(
            xaxis=dict(
                title="← Text Trusted | Image Trusted →",
                zeroline=True,
                zerolinewidth=3,
                zerolinecolor='gray',
            ),
            yaxis=dict(title="Sensitivity (ADE σ)"),
            zaxis=dict(title="|Cohen's d|"),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            annotations=[
                dict(
                    x=x_range[0], y=y_range[1], z=z_range[1],
                    text="TEXT TRUSTED",
                    showarrow=False,
                    font=dict(size=12, color="blue"),
                ),
                dict(
                    x=x_range[1], y=y_range[1], z=z_range[1],
                    text="IMAGE TRUSTED",
                    showarrow=False,
                    font=dict(size=12, color="green"),
                ),
            ],
        ),
        legend=dict(
            x=0.02, y=0.98,
            bgcolor="rgba(255,255,255,0.9)",
            title="Key Category",
        ),
        width=1100,
        height=900,
    )

    fig.write_html(output_dir / "modality_trust_landscape.html")
    fig.write_image(output_dir / "modality_trust_landscape.png", scale=2)
    print(f"  Saved: modality_trust_landscape.html")


def generate_delta_ade_heatmap(df: pd.DataFrame, analysis: dict, output_dir: Path):
    """
    Dense heatmap showing ΔADE from aligned baseline for all key-value combinations.

    Rows: Keys (sorted by alignment effect)
    Columns: Perturbed values (grouped by key)
    Color: ΔADE (green = better than baseline, red = worse)

    Diagonal pattern within each key block shows aligned (should be ~0) vs misaligned.
    """
    import plotly.graph_objects as go

    print("Generating Delta ADE Heatmap...")

    alignment = analysis.get("alignment_effects", {})

    # Sort keys by alignment effect
    sorted_keys = sorted(
        [k for k in df["key"].unique() if k in alignment],
        key=lambda k: alignment[k]["alignment_effect"]
    )

    # Build matrix: rows = keys, columns = all values (grouped)
    all_values = []
    value_labels = []
    key_boundaries = [0]

    for key in sorted_keys:
        values = sorted(TEXT_VOCABULARY.get(key, {}).keys())
        all_values.extend([(key, v) for v in values])
        value_labels.extend([f"{v}" for v in values])
        key_boundaries.append(len(all_values))

    n_keys = len(sorted_keys)
    n_values = len(all_values)

    # Build ΔADE matrix
    z_matrix = np.full((n_keys, n_values), np.nan)
    hover_matrix = [[None] * n_values for _ in range(n_keys)]

    for i, key in enumerate(sorted_keys):
        key_data = df[df["key"] == key]
        baseline_ade = alignment[key]["mean_ade_aligned"]

        for j, (val_key, val) in enumerate(all_values):
            if val_key != key:
                continue

            val_data = key_data[key_data["perturbed_value"] == val]
            if len(val_data) > 0:
                mean_ade = val_data["ade"].mean()
                delta = mean_ade - baseline_ade
                z_matrix[i, j] = delta

                is_aligned = val_data["is_aligned"].iloc[0] if len(val_data) > 0 else False
                hover_matrix[i][j] = (
                    f"<b>{key}</b> → {val}<br>"
                    f"{'✓ Aligned' if is_aligned else '✗ Misaligned'}<br>"
                    f"ΔADE: {delta:+.3f}<br>"
                    f"ADE: {mean_ade:.3f}"
                )

    # Create heatmap
    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=z_matrix,
        x=value_labels,
        y=[k.replace("_", " ") for k in sorted_keys],
        colorscale='RdYlGn_r',
        zmid=0,
        colorbar=dict(
            title=dict(text="ΔADE", side="right"),
            tickformat="+.2f",
        ),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_matrix,
    ))

    # Add vertical lines between key groups
    for b in key_boundaries[1:-1]:
        fig.add_vline(x=b - 0.5, line=dict(color="white", width=2))

    # Add key group labels at top
    annotations = []
    for i, key in enumerate(sorted_keys):
        start = key_boundaries[i]
        end = key_boundaries[i + 1]
        mid = (start + end) / 2 - 0.5
        annotations.append(dict(
            x=mid, y=n_keys,
            text=key.replace("_", " ").title(),
            showarrow=False,
            font=dict(size=8),
            textangle=-45,
        ))

    fig.update_layout(
        title=dict(
            text="<b>Perturbation Effect Matrix</b><br>"
                 "<sup>ΔADE from aligned baseline · Green=improvement · Red=degradation</sup>",
            x=0.5,
        ),
        xaxis=dict(
            title="Perturbed Value",
            tickangle=-45,
            side="bottom",
        ),
        yaxis=dict(
            title="Key (sorted by trust direction →)",
            autorange="reversed",
        ),
        annotations=annotations,
        width=1400,
        height=600,
        margin=dict(t=150),
    )

    fig.write_html(output_dir / "delta_ade_heatmap.html")
    fig.write_image(output_dir / "delta_ade_heatmap.png", scale=2)
    print(f"  Saved: delta_ade_heatmap.html")


def generate_scene_key_stability(df: pd.DataFrame, output_dir: Path):
    """
    3D surface showing stability across scenes and keys.

    X: Scene (clip_id)
    Y: Key
    Z: ADE standard deviation (stability measure)

    Reveals which scene-key combinations are volatile.
    """
    import plotly.graph_objects as go

    print("Generating Scene-Key Stability Surface...")

    scenes = sorted(df["clip_id"].unique())
    keys = sorted(df["key"].unique())

    # Build stability matrix
    z_matrix = np.full((len(keys), len(scenes)), np.nan)
    hover_matrix = [[None] * len(scenes) for _ in range(len(keys))]

    for i, key in enumerate(keys):
        key_data = df[df["key"] == key]
        for j, scene in enumerate(scenes):
            scene_key_data = key_data[key_data["clip_id"] == scene]
            if len(scene_key_data) >= 2:
                std = scene_key_data["ade"].std()
                mean = scene_key_data["ade"].mean()
                z_matrix[i, j] = std
                hover_matrix[i][j] = (
                    f"<b>{scene[:20]}...</b><br>"
                    f"Key: {key}<br>"
                    f"Stability (σ): {std:.3f}<br>"
                    f"Mean ADE: {mean:.3f}<br>"
                    f"n={len(scene_key_data)}"
                )

    fig = go.Figure()

    fig.add_trace(go.Surface(
        z=z_matrix,
        colorscale='Viridis',
        colorbar=dict(title=dict(text="ADE σ", side="right")),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_matrix,
    ))

    fig.update_layout(
        title=dict(
            text="<b>Scene × Key Stability Surface</b><br>"
                 "<sup>Higher peaks = more volatile (perturbations affect predictions more)</sup>",
            x=0.5,
        ),
        scene=dict(
            xaxis=dict(
                title="Scene",
                tickvals=list(range(0, len(scenes), max(1, len(scenes)//5))),
                ticktext=[s[:15] + "..." for s in scenes[::max(1, len(scenes)//5)]],
            ),
            yaxis=dict(
                title="Key",
                tickvals=list(range(len(keys))),
                ticktext=[k.replace("_", " ")[:12] for k in keys],
            ),
            zaxis=dict(title="Stability (ADE σ)"),
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.0)),
        ),
        width=1100,
        height=800,
    )

    fig.write_html(output_dir / "scene_key_stability.html")
    fig.write_image(output_dir / "scene_key_stability.png", scale=2)
    print(f"  Saved: scene_key_stability.html")


def generate_trust_spectrum(df: pd.DataFrame, analysis: dict, output_dir: Path):
    """
    2D visualization: Trust spectrum with error bars.

    Clear, publication-ready figure showing:
    - X: Alignment effect with confidence intervals
    - Y: Keys sorted by effect
    - Color: Category
    - Annotations: Significance stars
    """
    import plotly.graph_objects as go

    print("Generating Trust Spectrum...")

    alignment = analysis.get("alignment_effects", {})

    # Sort keys by alignment effect
    sorted_keys = sorted(
        alignment.keys(),
        key=lambda k: alignment[k]["alignment_effect"]
    )

    effects = []
    errors = []
    colors = []
    annotations = []

    for key in sorted_keys:
        ae = alignment[key]
        effects.append(ae["alignment_effect"])

        # Compute 95% CI using pooled standard error
        n_a, n_m = ae["n_aligned"], ae["n_misaligned"]
        se = np.sqrt(ae["std_ade_aligned"]**2/n_a + ae["std_ade_misaligned"]**2/n_m)
        ci = 1.96 * se
        errors.append(ci)

        colors.append(CATEGORY_COLORS.get(KEY_CATEGORIES.get(key, "scene_context")))

        # Significance annotation
        if ae["p_value"] < 0.001:
            annotations.append("***")
        elif ae["p_value"] < 0.01:
            annotations.append("**")
        elif ae["p_value"] < 0.05:
            annotations.append("*")
        else:
            annotations.append("")

    fig = go.Figure()

    # Add horizontal bars with error bars
    fig.add_trace(go.Bar(
        y=[k.replace("_", " ").title() for k in sorted_keys],
        x=effects,
        orientation='h',
        marker=dict(color=colors),
        error_x=dict(type='data', array=errors, color='black', thickness=1.5),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Effect: %{x:+.3f}<br>"
            "<extra></extra>"
        ),
    ))

    # Add significance stars
    for i, (effect, ann) in enumerate(zip(effects, annotations)):
        if ann:
            fig.add_annotation(
                x=effect + errors[i] + 0.05 if effect > 0 else effect - errors[i] - 0.05,
                y=i,
                text=ann,
                showarrow=False,
                font=dict(size=14),
            )

    # Add vertical line at 0
    fig.add_vline(x=0, line=dict(color="black", width=2, dash="dash"))

    # Add region labels
    fig.add_annotation(
        x=min(effects) - 0.1, y=len(sorted_keys),
        text="← TEXT TRUSTED",
        showarrow=False,
        font=dict(size=12, color="blue"),
    )
    fig.add_annotation(
        x=max(effects) + 0.1, y=len(sorted_keys),
        text="IMAGE TRUSTED →",
        showarrow=False,
        font=dict(size=12, color="green"),
    )

    fig.update_layout(
        title=dict(
            text="<b>Modality Trust Spectrum</b><br>"
                 "<sup>Alignment Effect with 95% CI · *p<0.05 **p<0.01 ***p<0.001</sup>",
            x=0.5,
        ),
        xaxis=dict(
            title="Alignment Effect (Misaligned ADE − Aligned ADE)",
            zeroline=True,
        ),
        yaxis=dict(title=""),
        showlegend=False,
        width=900,
        height=700,
        margin=dict(l=200),
    )

    fig.write_html(output_dir / "trust_spectrum.html")
    fig.write_image(output_dir / "trust_spectrum.png", scale=2)
    print(f"  Saved: trust_spectrum.html")


def main():
    print("=" * 60)
    print("BND-004: DENSE 3D VISUALIZATIONS")
    print("=" * 60)

    # Check dependencies
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("ERROR: plotly not installed. Run: pip install plotly kaleido")
        return 1

    # Create output directory
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    if not RESULTS_FILE.exists():
        print(f"ERROR: Results not found: {RESULTS_FILE}")
        return 1

    if not ANALYSIS_FILE.exists():
        print(f"ERROR: Analysis not found. Run analyze_results.py first.")
        return 1

    df, analysis = load_data()
    print(f"Loaded {len(df)} results from {df['clip_id'].nunique()} scenes")

    # Generate visualizations
    print("\n" + "-" * 60)
    generate_modality_trust_landscape(df, analysis, FIG_DIR)
    generate_trust_spectrum(df, analysis, FIG_DIR)
    generate_delta_ade_heatmap(df, analysis, FIG_DIR)
    generate_scene_key_stability(df, FIG_DIR)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Figures saved to: {FIG_DIR}")
    print("\nKey visualizations:")
    print(f"  1. Modality Trust Landscape (3D): {FIG_DIR / 'modality_trust_landscape.html'}")
    print(f"  2. Trust Spectrum (2D bars): {FIG_DIR / 'trust_spectrum.html'}")
    print(f"  3. Delta ADE Heatmap: {FIG_DIR / 'delta_ade_heatmap.html'}")
    print(f"  4. Scene-Key Stability: {FIG_DIR / 'scene_key_stability.html'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
