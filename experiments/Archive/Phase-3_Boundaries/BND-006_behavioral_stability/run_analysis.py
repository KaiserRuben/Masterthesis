#!/usr/bin/env python3
"""
BND-006: Behavioral Boundary Stability

Tests whether semantic Voronoi boundaries predict behavioral class changes,
not just continuous error variation.

Behavioral H1: Cor(d_B, behavioral_change_rate) < 0

  - d_B = boundary margin (distance to nearest competing centroid)
  - behavioral_change_rate = fraction of a scene's single-key-diff pairs
    where traj_changed == True

Scenes near boundaries (low d_B) should show behavioral class changes
more often (high behavioral_change_rate), giving a negative correlation.

Phases:
    A. Load data (scenes, OpenCLIP embeddings)
    B. Filter scenes (trajectory data + ADE + embedding)
    C. k-NN pair analysis
    D. Compute per-scene behavioral change rate
    E. Behavioral H1: Cor(d_B, behavioral_change_rate)
    F. Save results + figures

Usage:
    python run_analysis.py
"""

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

# Add project root to path
PROJECT_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from archive.pipeline.lib.schema import load_scenes, CLASSIFICATION_KEYS
from archive.pipeline.lib.io import load_embeddings, get_git_hash
from archive.pipeline.notebooks.utils.hypothesis import (
    compute_centroids,
    compute_boundary_margin,
)
from archive.pipeline.step_4_analyze import build_knn_graph, find_pairs
from archive.pipeline.notebooks.utils import THEME, plotly_layout, axis_style

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = PROJECT_ROOT / "data" / "pipeline"
BND_DIR = PROJECT_ROOT / "data" / "BND-006"
FIG_DIR = BND_DIR / "figures"

N_BOOTSTRAP = 1000
TRAJ_DIMS = ["direction", "speed", "lateral"]


# =============================================================================
# PHASE A: LOAD DATA
# =============================================================================

def load_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Load scenes and OpenCLIP embeddings."""
    print("=" * 60)
    print("PHASE A: LOAD DATA")
    print("=" * 60)

    scenes_file = DATA_DIR / "scenes.parquet"
    df = load_scenes(scenes_file)
    print(f"Scenes: {len(df)}")

    openclip_file = DATA_DIR / "embeddings.npz"
    embeddings = load_embeddings(openclip_file, key="embeddings")
    print(f"OpenCLIP embeddings: {embeddings.shape}")

    return df, embeddings


# =============================================================================
# PHASE B: FILTER SCENES
# =============================================================================

def filter_scenes(df: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    """
    Filter to scenes with trajectory data, ADE, and valid embedding.

    Requires: traj_direction, traj_speed, traj_lateral all non-null,
    has_ade == True, valid emb_index.
    """
    print("\n" + "=" * 60)
    print("PHASE B: FILTER SCENES")
    print("=" * 60)

    mask = (
        df["traj_direction"].notna()
        & df["traj_speed"].notna()
        & df["traj_lateral"].notna()
        & (df["has_ade"] == True)
        & df["emb_index"].notna()
        & (df["emb_index"] < len(embeddings))
    )
    df_f = df[mask].copy().reset_index(drop=True)

    print(f"Scenes with trajectory + ADE + embedding: {len(df_f)}")
    print(f"ADE range: [{df_f['ade'].min():.2f}, {df_f['ade'].max():.2f}]")

    # Trajectory class distribution
    for dim in TRAJ_DIMS:
        col = f"traj_{dim}"
        counts = df_f[col].value_counts()
        print(f"  {col}: {dict(counts)}")

    return df_f


# =============================================================================
# PHASE C: k-NN PAIR ANALYSIS
# =============================================================================

def run_knn_analysis(
    scenes: pd.DataFrame,
    embeddings: np.ndarray,
    k: int = 20,
) -> tuple[dict, pd.DataFrame]:
    """Build k-NN graph and find single-key-diff pairs."""
    print("\n" + "=" * 60)
    print("PHASE C: k-NN PAIR ANALYSIS")
    print("=" * 60)

    # Remap emb_index to contiguous 0..N-1 for the filtered subset
    subset_emb = embeddings[scenes["emb_index"].values]
    df_contiguous = scenes.copy()
    df_contiguous["emb_index"] = pd.array(range(len(df_contiguous)), dtype="Int64")

    edges = build_knn_graph(subset_emb, k=k)
    print(f"Edges: {len(edges)}")

    pairs = find_pairs(df_contiguous, edges, max_key_diff=1)
    n_total = len(pairs)
    n_single = (pairs["hamming"] == 1).sum()

    # Count pairs where both sides have trajectory data
    traj_mask = pairs["traj_direction_a"].notna() & pairs["traj_direction_b"].notna()
    n_with_traj = traj_mask.sum()

    print(f"Total pairs: {n_total}")
    print(f"Single-key-diff: {n_single}")
    print(f"With trajectory data: {n_with_traj}")
    print(f"traj_changed == True: {pairs.loc[traj_mask, 'traj_changed'].sum()}")

    stats_dict = {
        "k": k,
        "n_edges": len(edges),
        "n_total_pairs": n_total,
        "n_single_key_pairs": int(n_single),
        "n_with_traj": int(n_with_traj),
        "n_traj_changed": int(pairs.loc[traj_mask, "traj_changed"].sum()),
    }

    return stats_dict, pairs, df_contiguous, subset_emb


# =============================================================================
# PHASE D: PER-SCENE BEHAVIORAL CHANGE RATE
# =============================================================================

def compute_behavioral_change_rate(
    pairs_df: pd.DataFrame,
    scenes: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Compute per-scene behavioral change rate from single-key-diff pairs.

    For each scene, collect all its single-key-diff pairs where both sides
    have trajectory data. Compute:
      behavioral_change_rate = count(traj_changed) / count(pairs)

    Also computes per-key rates and per-dimension rates.

    Returns:
        (overall_rate, per_key_rates, per_dim_rates)
        - overall_rate: array aligned with scenes index (NaN where no pairs)
        - per_key_rates: dict[key] -> array aligned with scenes index
        - per_dim_rates: dict[dim] -> array aligned with scenes index
    """
    print("\n" + "=" * 60)
    print("PHASE D: PER-SCENE BEHAVIORAL CHANGE RATE")
    print("=" * 60)

    # Filter to single-key-diff pairs with trajectory data on both sides
    single = pairs_df[
        (pairs_df["hamming"] == 1)
        & pairs_df["traj_direction_a"].notna()
        & pairs_df["traj_direction_b"].notna()
    ]

    n = len(scenes)
    clip_to_idx = {cid: i for i, cid in enumerate(scenes["clip_id"].values)}

    # Collect per-scene behavioral change counts
    scene_changed = defaultdict(list)       # clip_id -> [bool, ...]
    scene_key_changed = defaultdict(lambda: defaultdict(list))
    scene_dim_changed = defaultdict(lambda: defaultdict(list))

    for _, row in single.iterrows():
        changed = bool(row["traj_changed"])
        diff_key = row["diff_key"]

        # Per-dimension changes for this pair
        dim_changes = {}
        for dim in TRAJ_DIMS:
            va = row[f"traj_{dim}_a"]
            vb = row[f"traj_{dim}_b"]
            dim_changes[dim] = (pd.notna(va) and pd.notna(vb) and va != vb)

        for clip in [row["clip_a"], row["clip_b"]]:
            scene_changed[clip].append(changed)
            if diff_key:
                scene_key_changed[clip][diff_key].append(changed)
            for dim in TRAJ_DIMS:
                scene_dim_changed[clip][dim].append(dim_changes[dim])

    # Build arrays aligned with scenes
    overall_rate = np.full(n, np.nan)
    for clip, changes in scene_changed.items():
        if clip in clip_to_idx:
            overall_rate[clip_to_idx[clip]] = np.mean(changes)

    per_key_rates = {}
    for key in CLASSIFICATION_KEYS:
        arr = np.full(n, np.nan)
        for clip, key_map in scene_key_changed.items():
            if key in key_map and clip in clip_to_idx:
                arr[clip_to_idx[clip]] = np.mean(key_map[key])
        per_key_rates[key] = arr

    per_dim_rates = {}
    for dim in TRAJ_DIMS:
        arr = np.full(n, np.nan)
        for clip, dim_map in scene_dim_changed.items():
            if dim in dim_map and clip in clip_to_idx:
                arr[clip_to_idx[clip]] = np.mean(dim_map[dim])
        per_dim_rates[dim] = arr

    n_with_rate = np.sum(~np.isnan(overall_rate))
    mean_rate = np.nanmean(overall_rate)
    print(f"Scenes with behavioral change data: {n_with_rate}/{n}")
    print(f"Mean behavioral change rate: {mean_rate:.3f}")

    for dim in TRAJ_DIMS:
        dim_mean = np.nanmean(per_dim_rates[dim])
        print(f"  {dim} change rate: {dim_mean:.3f}")

    return overall_rate, per_key_rates, per_dim_rates


# =============================================================================
# PHASE E: BEHAVIORAL H1
# =============================================================================

def _correlate_with_ci(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """Compute Pearson + Spearman correlation with bootstrap 95% CI."""
    n = len(x)
    r_pearson, p_pearson = stats.pearsonr(x, y)
    r_spearman, p_spearman = stats.spearmanr(x, y)

    boot_r = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(n, n, replace=True)
        boot_r.append(stats.pearsonr(x[idx], y[idx])[0])
    ci_low, ci_high = np.percentile(boot_r, [2.5, 97.5])

    return {
        "pearson_r": float(r_pearson),
        "pearson_p": float(p_pearson),
        "spearman_r": float(r_spearman),
        "spearman_p": float(p_spearman),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n": n,
    }


def run_behavioral_h1(
    scenes: pd.DataFrame,
    embeddings: np.ndarray,
    overall_rate: np.ndarray,
    perkey_rates: dict[str, np.ndarray],
    perdim_rates: dict[str, np.ndarray],
) -> dict:
    """
    Behavioral H1: Cor(d_B, behavioral_change_rate) < 0.

    Scenes near boundaries (low d_B) should show more behavioral changes.
    """
    print("\n" + "=" * 60)
    print("PHASE E: BEHAVIORAL H1")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # Compute boundary margins
    # anchors_only=False because the filtered set (traj+ADE) has no anchors
    centroids = compute_centroids(scenes, embeddings, anchors_only=False)
    margin_result = compute_boundary_margin(
        scenes, embeddings, centroids=centroids, return_full=True,
    )

    # --- Aggregate: Cor(mean d_B, behavioral_change_rate) ---
    valid = ~np.isnan(margin_result.mean) & ~np.isnan(overall_rate)
    d_B = margin_result.mean[valid]
    rate = overall_rate[valid]
    n = len(d_B)

    if n < 10:
        print(f"Insufficient data: {n} valid scenes")
        return {"error": "insufficient_data", "n": n}

    agg = _correlate_with_ci(d_B, rate, rng)
    print(f"\nAggregate: Cor(d_B, behavioral_change_rate)")
    print(f"  n = {agg['n']}")
    print(f"  Pearson  r = {agg['pearson_r']:.4f}  (p = {agg['pearson_p']:.4f})")
    print(f"  Spearman r = {agg['spearman_r']:.4f}  (p = {agg['spearman_p']:.4f})")
    print(f"  95% CI: [{agg['ci_low']:.4f}, {agg['ci_high']:.4f}]")

    # --- Per-key: Cor(d_B_k, behavioral_change_rate_k) ---
    per_key = {}
    print(f"\nPer-key: Cor(d_B_k, behavioral_change_rate_k)")
    for key in CLASSIFICATION_KEYS:
        d_B_k = margin_result.per_key[key]
        rate_k = perkey_rates[key]
        valid_k = ~np.isnan(d_B_k) & ~np.isnan(rate_k)
        n_k = int(valid_k.sum())

        if n_k >= 10:
            result_k = _correlate_with_ci(d_B_k[valid_k], rate_k[valid_k], rng)
            per_key[key] = result_k
            sig = "**" if result_k["pearson_p"] < 0.01 else "* " if result_k["pearson_p"] < 0.05 else "  "
            print(f"  {key}: r={result_k['pearson_r']:+.4f} (p={result_k['pearson_p']:.4f}) {sig} n={n_k}")
        else:
            per_key[key] = {"pearson_r": None, "pearson_p": None, "n": n_k}
            print(f"  {key}: insufficient data (n={n_k})")

    # --- Per-dimension: Cor(d_B, dim_change_rate) ---
    per_dim = {}
    print(f"\nPer-dimension: Cor(d_B, dim_change_rate)")
    for dim in TRAJ_DIMS:
        rate_dim = perdim_rates[dim]
        valid_dim = ~np.isnan(margin_result.mean) & ~np.isnan(rate_dim)
        n_dim = int(valid_dim.sum())

        if n_dim >= 10:
            result_dim = _correlate_with_ci(
                margin_result.mean[valid_dim], rate_dim[valid_dim], rng,
            )
            per_dim[dim] = result_dim
            sig = "**" if result_dim["pearson_p"] < 0.01 else "* " if result_dim["pearson_p"] < 0.05 else "  "
            print(f"  {dim}: r={result_dim['pearson_r']:+.4f} (p={result_dim['pearson_p']:.4f}) {sig} n={n_dim}")
        else:
            per_dim[dim] = {"pearson_r": None, "pearson_p": None, "n": n_dim}
            print(f"  {dim}: insufficient data (n={n_dim})")

    # --- Cross-table: 6 semantic keys x 3 trajectory dimensions ---
    crosstab = {}
    print(f"\nCross-table: semantic key x trajectory dimension")
    print(f"{'Key':<25} {'direction':>10} {'speed':>10} {'lateral':>10}")
    print("-" * 58)

    for key in CLASSIFICATION_KEYS:
        d_B_k = margin_result.per_key[key]
        crosstab[key] = {}

        row_vals = []
        for dim in TRAJ_DIMS:
            rate_dim = perdim_rates[dim]
            valid_kd = ~np.isnan(d_B_k) & ~np.isnan(rate_dim)
            n_kd = int(valid_kd.sum())

            if n_kd >= 10:
                r_kd, p_kd = stats.pearsonr(d_B_k[valid_kd], rate_dim[valid_kd])
                crosstab[key][dim] = {"r": float(r_kd), "p": float(p_kd), "n": n_kd}
                sig = "**" if p_kd < 0.01 else "* " if p_kd < 0.05 else "  "
                row_vals.append(f"{r_kd:+.3f}{sig}")
            else:
                crosstab[key][dim] = {"r": None, "p": None, "n": n_kd}
                row_vals.append(f"{'N/A':>6}  ")

        print(f"{key:<25} {row_vals[0]:>10} {row_vals[1]:>10} {row_vals[2]:>10}")

    return {
        "n": n,
        "aggregate": agg,
        "per_key": per_key,
        "per_dim": per_dim,
        "crosstab": crosstab,
        # Raw arrays for figures (stripped before JSON save)
        "_d_B": d_B.tolist(),
        "_rate": rate.tolist(),
    }


# =============================================================================
# FIGURES
# =============================================================================

def create_behavioral_h1_figure(results: dict) -> go.Figure:
    """Scatter: d_B vs behavioral_change_rate."""
    d_B = np.array(results["_d_B"])
    rate = np.array(results["_rate"])
    agg = results["aggregate"]

    fig = go.Figure()

    # Color by change rate
    colors = [
        THEME["ade"]["low"] if r < 0.2 else
        THEME["ade"]["medium"] if r < 0.5 else
        THEME["ade"]["high"] if r < 0.8 else
        THEME["ade"]["critical"]
        for r in rate
    ]

    fig.add_trace(go.Scatter(
        x=d_B, y=rate,
        mode="markers",
        marker=dict(size=6, color=colors, opacity=0.5),
        hovertemplate="d_B: %{x:.3f}<br>Change rate: %{y:.3f}<extra></extra>",
        showlegend=False,
    ))

    # Regression line
    slope, intercept = np.polyfit(d_B, rate, 1)
    x_line = np.linspace(d_B.min(), d_B.max(), 50)
    y_line = slope * x_line + intercept

    fig.add_trace(go.Scatter(
        x=x_line, y=y_line,
        mode="lines",
        line=dict(color=THEME["text_secondary"], width=2),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Stats annotation
    sig = "**" if agg["pearson_p"] < 0.01 else "*" if agg["pearson_p"] < 0.05 else ""
    fig.add_annotation(
        x=0.98, y=0.98,
        xref="paper", yref="paper",
        text=(
            f"r = {agg['pearson_r']:.3f}{sig}<br>"
            f"p = {agg['pearson_p']:.4f}<br>"
            f"95% CI: [{agg['ci_low']:.3f}, {agg['ci_high']:.3f}]<br>"
            f"n = {agg['n']}"
        ),
        showarrow=False,
        font=dict(size=10, family=THEME["font_mono"], color=THEME["text_secondary"]),
        xanchor="right", yanchor="top",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor=THEME["border"],
        borderwidth=1,
    )

    fig.update_layout(
        **plotly_layout("", height=450, show_legend=False, margin=dict(l=60, r=30, t=30, b=50)),
        xaxis={
            **axis_style(""),
            "title": dict(text="Boundary Margin d<sub>B</sub>", font=dict(size=12)),
        },
        yaxis={
            **axis_style(""),
            "title": dict(text="Behavioral Change Rate", font=dict(size=12)),
        },
    )

    return fig


def create_perkey_behavioral_figure(results: dict) -> go.Figure:
    """Per-key Cor(d_B_k, behavioral_change_rate_k) bars."""
    fig = go.Figure()

    keys = CLASSIFICATION_KEYS
    r_vals = [results["per_key"].get(k, {}).get("pearson_r") for k in keys]
    p_vals = [results["per_key"].get(k, {}).get("pearson_p") for k in keys]

    # Sort by correlation
    order = sorted(range(len(keys)), key=lambda i: (r_vals[i] or 0))
    sorted_keys = [keys[i] for i in order]
    sorted_r = [r_vals[i] for i in order]
    sorted_p = [p_vals[i] for i in order]

    colors = [
        THEME["ade"]["critical"] if p is not None and p < 0.01 else
        THEME["ade"]["high"] if p is not None and p < 0.05 else
        THEME["point_inactive"]
        for p in sorted_p
    ]

    fig.add_trace(go.Bar(
        y=sorted_keys,
        x=[r if r is not None else 0 for r in sorted_r],
        orientation="h",
        marker=dict(color=colors),
        hovertemplate="<b>%{y}</b><br>r = %{x:.3f}<extra></extra>",
    ))

    fig.add_vline(x=0, line=dict(color=THEME["border"], width=1.5))

    # Significance legend
    fig.add_annotation(
        x=0.98, y=0.02,
        xref="paper", yref="paper",
        text="p<0.01  p<0.05  n.s.",
        showarrow=False,
        font=dict(size=9, family=THEME["font_mono"], color=THEME["text_muted"]),
        xanchor="right", yanchor="bottom",
    )

    fig.update_layout(
        **plotly_layout("", height=380, show_legend=False, margin=dict(l=130, r=40, t=30, b=50)),
        xaxis={
            **axis_style(""),
            "title": dict(
                text="Cor(d<sub>B</sub><sup>k</sup>, behavioral change rate<sup>k</sup>)",
                font=dict(size=11),
            ),
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


def create_dimension_crosstab_figure(results: dict) -> go.Figure:
    """6x3 heatmap: semantic key x trajectory dimension change rates."""
    crosstab = results["crosstab"]

    keys = CLASSIFICATION_KEYS
    dims = TRAJ_DIMS

    # Build correlation matrix
    z = []
    hover_text = []
    for key in keys:
        row_z = []
        row_hover = []
        for dim in dims:
            entry = crosstab.get(key, {}).get(dim, {})
            r = entry.get("r")
            p = entry.get("p")
            n = entry.get("n", 0)
            if r is not None:
                row_z.append(r)
                sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
                row_hover.append(f"r={r:.3f}{sig}<br>p={p:.4f}<br>n={n}")
            else:
                row_z.append(0)
                row_hover.append(f"N/A (n={n})")
        z.append(row_z)
        hover_text.append(row_hover)

    z_arr = np.array(z)
    max_abs = max(np.abs(z_arr).max(), 0.01)

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=z_arr,
        x=dims,
        y=keys,
        colorscale=[
            [0.0, THEME["diverging"]["negative"]],
            [0.5, THEME["diverging"]["neutral"]],
            [1.0, THEME["diverging"]["positive"]],
        ],
        zmid=0,
        zmin=-max_abs,
        zmax=max_abs,
        colorbar=dict(
            title=dict(text="r", font=dict(size=11)),
            thickness=12,
            len=0.6,
            tickfont=dict(size=9),
        ),
        hovertext=hover_text,
        hovertemplate="%{y} x %{x}<br>%{hovertext}<extra></extra>",
    ))

    # Cell annotations
    for i in range(len(keys)):
        for j in range(len(dims)):
            entry = crosstab.get(keys[i], {}).get(dims[j], {})
            r = entry.get("r")
            p = entry.get("p")
            if r is not None:
                sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
                text_color = THEME["surface"] if abs(r) > max_abs * 0.5 else THEME["text"]
                fig.add_annotation(
                    x=dims[j], y=keys[i],
                    text=f"{r:.2f}{sig}",
                    showarrow=False,
                    font=dict(color=text_color, size=10),
                )

    fig.update_layout(
        **plotly_layout("", height=380, show_legend=False, margin=dict(l=130, r=60, t=30, b=50)),
        xaxis=dict(
            side="bottom",
            tickfont=dict(size=11, color=THEME["text"]),
        ),
        yaxis=dict(
            tickfont=dict(size=11, color=THEME["text"]),
            autorange="reversed",
        ),
    )

    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("BND-006: BEHAVIORAL BOUNDARY STABILITY")
    print("=" * 60)
    print("Behavioral H1: Cor(d_B, behavioral_change_rate) < 0")
    print("  behavioral_change_rate = fraction of pairs where traj_changed")

    # Phase A
    df, embeddings = load_data()

    # Phase B
    df_filtered = filter_scenes(df, embeddings)

    # Phase C
    knn_stats, pairs, df_contiguous, subset_emb = run_knn_analysis(
        df_filtered, embeddings,
    )

    # Phase D
    overall_rate, perkey_rates, perdim_rates = compute_behavioral_change_rate(
        pairs, df_contiguous,
    )

    # Phase E
    h1_results = run_behavioral_h1(
        df_contiguous, subset_emb,
        overall_rate, perkey_rates, perdim_rates,
    )

    # Phase F: Save results + figures
    print("\n" + "=" * 60)
    print("PHASE F: SAVE RESULTS + FIGURES")
    print("=" * 60)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Strip raw arrays before saving JSON
    def strip_arrays(d):
        return {k: v for k, v in d.items() if not k.startswith("_")}

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "git_hash": get_git_hash(),
            "n_bootstrap": N_BOOTSTRAP,
            "test": "Cor(d_B, behavioral_change_rate) — Behavioral H1",
        },
        "knn": knn_stats,
        "h1": strip_arrays(h1_results),
    }

    results_file = BND_DIR / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {results_file}")

    # Generate figures
    if "error" not in h1_results:
        print("\nGenerating figures...")

        fig_h1 = create_behavioral_h1_figure(h1_results)
        fig_h1.write_html(FIG_DIR / "behavioral_h1.html", include_plotlyjs="cdn")
        print(f"  Saved: behavioral_h1.html")

        fig_pk = create_perkey_behavioral_figure(h1_results)
        fig_pk.write_html(FIG_DIR / "perkey_behavioral.html", include_plotlyjs="cdn")
        print(f"  Saved: perkey_behavioral.html")

        fig_ct = create_dimension_crosstab_figure(h1_results)
        fig_ct.write_html(FIG_DIR / "dimension_crosstab.html", include_plotlyjs="cdn")
        print(f"  Saved: dimension_crosstab.html")
    else:
        print("Skipping figures due to insufficient data.")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if "error" not in h1_results:
        agg = h1_results["aggregate"]
        print(f"\nAggregate: Cor(d_B, behavioral_change_rate)")
        print(f"  Pearson  r = {agg['pearson_r']:+.4f}  (p = {agg['pearson_p']:.4f})")
        print(f"  Spearman r = {agg['spearman_r']:+.4f}  (p = {agg['spearman_p']:.4f})")
        print(f"  95% CI: [{agg['ci_low']:.4f}, {agg['ci_high']:.4f}]")
        print(f"  n = {agg['n']}")

        print(f"\nPer-key correlations:")
        print(f"{'Key':<25} {'r':>10} {'p':>10}")
        print("-" * 48)
        for key in CLASSIFICATION_KEYS:
            k_data = h1_results["per_key"].get(key, {})
            r_val = k_data.get("pearson_r")
            p_val = k_data.get("pearson_p")
            if r_val is not None:
                sig = "**" if p_val < 0.01 else "* " if p_val < 0.05 else "  "
                print(f"{key:<25} {r_val:+10.4f} {p_val:10.4f} {sig}")
            else:
                print(f"{key:<25} {'N/A':>10} {'N/A':>10}")

        print(f"\nPer-dimension correlations:")
        for dim in TRAJ_DIMS:
            d_data = h1_results["per_dim"].get(dim, {})
            r_val = d_data.get("pearson_r")
            p_val = d_data.get("pearson_p")
            if r_val is not None:
                sig = "**" if p_val < 0.01 else "* " if p_val < 0.05 else "  "
                print(f"  {dim}: r={r_val:+.4f} (p={p_val:.4f}) {sig}")
            else:
                print(f"  {dim}: N/A")

    print(f"\nResults: {results_file}")
    print(f"Figures: {FIG_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
