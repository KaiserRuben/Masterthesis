#!/usr/bin/env python3
"""
Interactive Visualization for Pipeline Results

Generates an HTML dashboard with:
- Sensitivity ranking by semantic key
- ADE scatter plots for pairs
- Trajectory class change analysis
- Transition heatmaps
- 3D embedding space with PCA/UMAP

Usage:
    python pipeline/visualize.py
    python pipeline/visualize.py --output my_analysis.html
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent))
from lib.io import load_config, load_embeddings, get_repo_root
from lib.schema import CLASSIFICATION_KEYS


def load_data(repo_root: Path, config: dict):
    """Load all pipeline data."""
    scenes_file = repo_root / config["paths"]["scenes_file"]
    results_dir = repo_root / config["paths"]["results_dir"]

    scenes = pd.read_parquet(scenes_file)
    pairs = pd.read_parquet(results_dir / "pairs.parquet")

    with open(results_dir / "stability_map.json") as f:
        stability = json.load(f)

    with open(results_dir / "summary.json") as f:
        summary = json.load(f)

    return scenes, pairs, stability, summary


def create_sensitivity_chart(stability: dict, summary: dict) -> go.Figure:
    """Create sensitivity ranking bar chart."""
    ranking = summary.get("sensitivity_ranking", [])

    keys = [r[0] for r in ranking]
    values = [r[1] for r in ranking]

    # Get additional data
    n_pairs = [stability[k].get("n_with_ade", 0) for k in keys]
    traj_rates = [stability[k].get("traj_change_rate", 0) or 0 for k in keys]

    fig = go.Figure()

    # Sensitivity bars
    fig.add_trace(go.Bar(
        name="Relative ΔADE",
        x=keys,
        y=values,
        marker_color=px.colors.sequential.Viridis[::2][:len(keys)],
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Rel ΔADE: %{y:.3f}<br>Pairs: %{customdata[0]}<br>Traj Change: %{customdata[1]:.1%}<extra></extra>",
        customdata=list(zip(n_pairs, traj_rates)),
    ))

    fig.update_layout(
        title=dict(
            text="<b>Sensitivity Ranking by Semantic Key</b><br><sup>Higher = more impact on trajectory prediction error</sup>",
            x=0.5,
        ),
        xaxis_title="Semantic Key",
        yaxis_title="Relative ΔADE",
        showlegend=False,
        height=450,
        margin=dict(t=80, b=60),
    )

    return fig


def create_traj_change_chart(stability: dict) -> go.Figure:
    """Create trajectory class change rate chart."""
    keys = list(stability.keys())
    rates = [stability[k].get("traj_change_rate", 0) or 0 for k in keys]
    n_pairs = [stability[k].get("n_with_ade", 0) for k in keys]

    # Sort by rate
    sorted_data = sorted(zip(keys, rates, n_pairs), key=lambda x: x[1], reverse=True)
    keys, rates, n_pairs = zip(*sorted_data)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=keys,
        y=[r * 100 for r in rates],
        marker_color=px.colors.sequential.Reds[2:],
        text=[f"{r:.1%}" for r in rates],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Traj Change Rate: %{y:.1f}%<br>Pairs: %{customdata}<extra></extra>",
        customdata=n_pairs,
    ))

    fig.update_layout(
        title=dict(
            text="<b>Trajectory Class Change Rate</b><br><sup>% of pairs where direction/speed/lateral class changed</sup>",
            x=0.5,
        ),
        xaxis_title="Semantic Key",
        yaxis_title="Change Rate (%)",
        showlegend=False,
        height=400,
        margin=dict(t=80, b=60),
    )

    return fig


def create_pairs_scatter(pairs: pd.DataFrame) -> go.Figure:
    """Create scatter plot of ADE pairs."""
    # Filter to pairs with ADE
    df = pairs[pairs["rel_delta_ade"].notna()].copy()

    if len(df) == 0:
        return go.Figure().add_annotation(text="No pairs with ADE data", showarrow=False)

    # Add jitter for better visibility
    df["ade_a_jitter"] = df["ade_a"] + np.random.normal(0, 0.02, len(df))
    df["ade_b_jitter"] = df["ade_b"] + np.random.normal(0, 0.02, len(df))

    fig = px.scatter(
        df,
        x="ade_a_jitter",
        y="ade_b_jitter",
        color="diff_key",
        size="rel_delta_ade",
        size_max=15,
        hover_data={
            "ade_a_jitter": False,
            "ade_b_jitter": False,
            "ade_a": ":.3f",
            "ade_b": ":.3f",
            "rel_delta_ade": ":.3f",
            "diff_key": True,
            "value_a": True,
            "value_b": True,
            "clip_a": True,
            "clip_b": True,
            "traj_changed": True,
        },
        color_discrete_sequence=px.colors.qualitative.Set2,
    )

    # Add diagonal line
    max_val = max(df["ade_a"].max(), df["ade_b"].max()) * 1.1
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode="lines",
        line=dict(dash="dash", color="gray"),
        showlegend=False,
        hoverinfo="skip",
    ))

    fig.update_layout(
        title=dict(
            text="<b>ADE Pairs Comparison</b><br><sup>Points away from diagonal = high sensitivity</sup>",
            x=0.5,
        ),
        xaxis_title="ADE Scene A (m)",
        yaxis_title="ADE Scene B (m)",
        height=500,
        legend_title="Differing Key",
    )

    return fig


def create_transition_heatmap(pairs: pd.DataFrame, key: str) -> go.Figure:
    """Create transition heatmap for a specific key."""
    df = pairs[(pairs["diff_key"] == key) & (pairs["rel_delta_ade"].notna())].copy()

    if len(df) == 0:
        return go.Figure().add_annotation(text=f"No data for {key}", showarrow=False)

    # Create transition matrix
    transitions = df.groupby(["value_a", "value_b"]).agg({
        "rel_delta_ade": ["mean", "count"],
        "traj_changed": "mean",
    }).reset_index()
    transitions.columns = ["value_a", "value_b", "mean_ade", "count", "traj_rate"]

    # Get unique values
    all_values = sorted(set(transitions["value_a"]) | set(transitions["value_b"]))

    # Create matrix
    matrix = np.zeros((len(all_values), len(all_values)))
    count_matrix = np.zeros((len(all_values), len(all_values)))

    value_to_idx = {v: i for i, v in enumerate(all_values)}

    for _, row in transitions.iterrows():
        i, j = value_to_idx[row["value_a"]], value_to_idx[row["value_b"]]
        matrix[i, j] = row["mean_ade"]
        count_matrix[i, j] = row["count"]
        # Make symmetric
        matrix[j, i] = row["mean_ade"]
        count_matrix[j, i] = row["count"]

    # Create hover text
    hover_text = []
    for i, va in enumerate(all_values):
        row_text = []
        for j, vb in enumerate(all_values):
            if count_matrix[i, j] > 0:
                row_text.append(f"{va} → {vb}<br>ΔADE: {matrix[i,j]:.3f}<br>n={int(count_matrix[i,j])}")
            else:
                row_text.append("")
        hover_text.append(row_text)

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=all_values,
        y=all_values,
        colorscale="YlOrRd",
        hoverinfo="text",
        text=hover_text,
        colorbar=dict(title="Rel ΔADE"),
    ))

    fig.update_layout(
        title=f"<b>Transitions: {key}</b>",
        xaxis_title="Value B",
        yaxis_title="Value A",
        height=400,
    )

    return fig


def create_coverage_gauge(summary: dict) -> go.Figure:
    """Create data coverage gauge chart."""
    data_state = summary.get("data_state", {})

    total = data_state.get("total_scenes", 1)
    with_ade = data_state.get("has_ade", 0)
    pairs_total = data_state.get("single_key_pairs", 1)
    pairs_with_ade = data_state.get("pairs_with_ade", 0)

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "indicator"}, {"type": "indicator"}]],
    )

    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=100 * with_ade / total,
        title={"text": "Scene ADE Coverage"},
        delta={"reference": 50, "suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, 33], "color": "lightcoral"},
                {"range": [33, 66], "color": "khaki"},
                {"range": [66, 100], "color": "lightgreen"},
            ],
            "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 50},
        },
        number={"suffix": "%"},
    ), row=1, col=1)

    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=100 * pairs_with_ade / pairs_total,
        title={"text": "Pair ADE Coverage"},
        delta={"reference": 50, "suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkgreen"},
            "steps": [
                {"range": [0, 33], "color": "lightcoral"},
                {"range": [33, 66], "color": "khaki"},
                {"range": [66, 100], "color": "lightgreen"},
            ],
            "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 50},
        },
        number={"suffix": "%"},
    ), row=1, col=2)

    fig.update_layout(height=300, margin=dict(t=50, b=20))

    return fig


def create_ade_distribution(pairs: pd.DataFrame) -> go.Figure:
    """Create ADE distribution by key."""
    df = pairs[(pairs["diff_key"].notna()) & (pairs["rel_delta_ade"].notna())].copy()

    if len(df) == 0:
        return go.Figure().add_annotation(text="No data", showarrow=False)

    fig = px.box(
        df,
        x="diff_key",
        y="rel_delta_ade",
        color="diff_key",
        color_discrete_sequence=px.colors.qualitative.Set2,
        points="outliers",
    )

    fig.update_layout(
        title=dict(
            text="<b>ΔADE Distribution by Semantic Key</b>",
            x=0.5,
        ),
        xaxis_title="Differing Key",
        yaxis_title="Relative ΔADE",
        showlegend=False,
        height=400,
    )

    return fig


def create_dashboard(scenes: pd.DataFrame, pairs: pd.DataFrame,
                     stability: dict, summary: dict, output_path: Path) -> None:
    """Create full HTML dashboard."""

    # Create all figures
    fig_sensitivity = create_sensitivity_chart(stability, summary)
    fig_traj = create_traj_change_chart(stability)
    fig_scatter = create_pairs_scatter(pairs)
    fig_coverage = create_coverage_gauge(summary)
    fig_dist = create_ade_distribution(pairs)

    # Create transition heatmaps for top keys
    ranking = summary.get("sensitivity_ranking", [])
    top_keys = [r[0] for r in ranking[:3]]
    fig_heatmaps = [create_transition_heatmap(pairs, k) for k in top_keys]

    # Build HTML
    html_parts = [
        """
<!DOCTYPE html>
<html>
<head>
    <title>Pipeline Analysis Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .header h1 { margin: 0 0 10px 0; }
        .header p { margin: 0; opacity: 0.9; }
        .stats-row {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            flex: 1;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stat-card h3 { margin: 0 0 5px 0; color: #666; font-size: 14px; }
        .stat-card .value { font-size: 32px; font-weight: bold; color: #333; }
        .stat-card .detail { font-size: 12px; color: #999; }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .row { display: flex; gap: 20px; }
        .col-2 { flex: 1; }
        .heatmaps { display: flex; gap: 20px; flex-wrap: wrap; }
        .heatmap { flex: 1; min-width: 300px; }
        .key-finding {
            background: #e8f5e9;
            border-left: 4px solid #4caf50;
            padding: 15px 20px;
            margin-bottom: 20px;
            border-radius: 0 10px 10px 0;
        }
        .key-finding h4 { margin: 0 0 5px 0; color: #2e7d32; }
        .key-finding p { margin: 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Semantic Boundary Analysis</h1>
        <p>Pipeline results: sensitivity of trajectory prediction to semantic input changes</p>
    </div>
""",
    ]

    # Stats cards
    ds = summary.get("data_state", {})
    key_result = summary.get("key_result", {})

    html_parts.append(f"""
    <div class="stats-row">
        <div class="stat-card">
            <h3>Total Scenes</h3>
            <div class="value">{ds.get('total_scenes', 0):,}</div>
            <div class="detail">{ds.get('anchors', 0)} anchors</div>
        </div>
        <div class="stat-card">
            <h3>With ADE</h3>
            <div class="value">{ds.get('has_ade', 0):,}</div>
            <div class="detail">{100*ds.get('has_ade', 0)/max(ds.get('total_scenes', 1), 1):.0f}% coverage</div>
        </div>
        <div class="stat-card">
            <h3>Single-Key Pairs</h3>
            <div class="value">{ds.get('single_key_pairs', 0):,}</div>
            <div class="detail">{ds.get('pairs_with_ade', 0):,} with ADE</div>
        </div>
        <div class="stat-card">
            <h3>Most Sensitive Key</h3>
            <div class="value" style="font-size: 24px;">{key_result.get('top_sensitive_key', 'N/A')}</div>
            <div class="detail">ΔADE ratio: {key_result.get('sensitivity_ratio', 0):.3f}</div>
        </div>
    </div>

    <div class="key-finding">
        <h4>🔍 Key Finding</h4>
        <p><strong>{key_result.get('top_sensitive_key', 'N/A')}</strong> causes the largest trajectory prediction changes.
        When this attribute differs between similar scenes, ADE changes by {key_result.get('sensitivity_ratio', 0):.1%} on average,
        with trajectory class changing {100*stability.get(key_result.get('top_sensitive_key', ''), {}).get('traj_change_rate', 0) or 0:.0f}% of the time.</p>
    </div>
""")

    # Coverage gauges
    html_parts.append('<div class="chart-container">')
    html_parts.append(f'<div id="coverage"></div>')
    html_parts.append(f'<script>Plotly.newPlot("coverage", {fig_coverage.to_json()});</script>')
    html_parts.append('</div>')

    # Sensitivity charts
    html_parts.append('<div class="row">')
    html_parts.append('<div class="col-2 chart-container">')
    html_parts.append(f'<div id="sensitivity"></div>')
    html_parts.append(f'<script>Plotly.newPlot("sensitivity", {fig_sensitivity.to_json()});</script>')
    html_parts.append('</div>')
    html_parts.append('<div class="col-2 chart-container">')
    html_parts.append(f'<div id="traj_change"></div>')
    html_parts.append(f'<script>Plotly.newPlot("traj_change", {fig_traj.to_json()});</script>')
    html_parts.append('</div>')
    html_parts.append('</div>')

    # Scatter and distribution
    html_parts.append('<div class="row">')
    html_parts.append('<div class="col-2 chart-container">')
    html_parts.append(f'<div id="scatter"></div>')
    html_parts.append(f'<script>Plotly.newPlot("scatter", {fig_scatter.to_json()});</script>')
    html_parts.append('</div>')
    html_parts.append('<div class="col-2 chart-container">')
    html_parts.append(f'<div id="dist"></div>')
    html_parts.append(f'<script>Plotly.newPlot("dist", {fig_dist.to_json()});</script>')
    html_parts.append('</div>')
    html_parts.append('</div>')

    # Heatmaps
    html_parts.append('<div class="chart-container"><h3>Transition Heatmaps (Top 3 Sensitive Keys)</h3><div class="heatmaps">')
    for i, (key, fig) in enumerate(zip(top_keys, fig_heatmaps)):
        html_parts.append(f'<div class="heatmap"><div id="heatmap_{i}"></div>')
        html_parts.append(f'<script>Plotly.newPlot("heatmap_{i}", {fig.to_json()});</script></div>')
    html_parts.append('</div></div>')

    # Footer
    html_parts.append(f"""
    <div style="text-align: center; padding: 20px; color: #999;">
        Generated: {summary.get('timestamp', 'N/A')} | Git: {summary.get('git_hash', 'N/A')}
    </div>
</body>
</html>
""")

    # Write HTML
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(html_parts))

    print(f"Dashboard saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate interactive visualization")
    parser.add_argument("--output", type=str, default=None, help="Output HTML file")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    args = parser.parse_args()

    # Load config
    config_path = args.config or (Path(__file__).parent / "config.yaml")
    config = load_config(config_path)
    repo_root = get_repo_root()

    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = repo_root / config["paths"]["results_dir"] / "dashboard.html"

    print("Loading data...")
    scenes, pairs, stability, summary = load_data(repo_root, config)

    print(f"Creating dashboard ({len(scenes)} scenes, {len(pairs)} pairs)...")
    create_dashboard(scenes, pairs, stability, summary, output_path)

    print(f"\nOpen in browser: file://{output_path.absolute()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
