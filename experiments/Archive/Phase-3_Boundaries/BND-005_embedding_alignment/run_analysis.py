#!/usr/bin/env python3
"""
BND-005: Embedding Space Alignment Analysis

Re-tests H1 (boundary-sensitivity correlation) in qwen3-embedding space and
compares with OpenCLIP ViT-bigG/14 baseline.

Corrected H1: Cor(d_B(z), sensitivity(z)) < 0
  - d_B = boundary margin (distance to nearest competing centroid)
  - sensitivity = mean rel_delta_ade across single-key-diff pairs the scene
    participates in (how much ADE changes when a neighbor flips one key)

Scenes near boundaries (low d_B) should be MORE sensitive (high ΔADE),
giving a negative correlation.

Phases:
    A. Load data (scenes, qwen3 embeddings, OpenCLIP embeddings)
    B. Prepare scenes with remapped embedding indices
    C. k-NN pair analysis in both spaces (must run before H1)
    D. Compute per-scene sensitivity from pairs
    E. Corrected H1: Cor(d_B, sensitivity) in both spaces
    F. Save results + comparison figures

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
from plotly.subplots import make_subplots
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
BND_DIR = PROJECT_ROOT / "data" / "BND-005"
FIG_DIR = BND_DIR / "figures"

N_BOOTSTRAP = 1000


# =============================================================================
# PHASE A: LOAD DATA
# =============================================================================

def load_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Load scenes, qwen3 embeddings, OpenCLIP embeddings, and optionally VLM embeddings.

    Returns:
        (scenes_df, qwen3_embeddings, qwen3_clip_ids, openclip_embeddings,
         vlm_embeddings_or_None, vlm_clip_ids_or_None)
    """
    print("=" * 60)
    print("PHASE A: LOAD DATA")
    print("=" * 60)

    # Scenes
    scenes_file = DATA_DIR / "scenes.parquet"
    df = load_scenes(scenes_file)
    print(f"Scenes: {len(df)}")

    # Qwen3 embeddings
    qwen3_file = BND_DIR / "qwen3_embeddings.npz"
    if not qwen3_file.exists():
        print(f"\nError: {qwen3_file} not found. Run run_embedding.py first.")
        sys.exit(1)

    qwen3_data = np.load(qwen3_file, allow_pickle=True)
    qwen3_emb = qwen3_data["embeddings"]
    qwen3_ids = qwen3_data["clip_ids"]
    print(f"Qwen3 embeddings: {qwen3_emb.shape}")
    print(f"Qwen3 clip_ids: {len(qwen3_ids)}")

    # OpenCLIP embeddings
    openclip_file = DATA_DIR / "embeddings.npz"
    openclip_emb = load_embeddings(openclip_file, key="embeddings")
    print(f"OpenCLIP embeddings: {openclip_emb.shape}")

    # VLM embeddings (optional)
    vlm_file = BND_DIR / "vlm_embeddings.npz"
    vlm_emb = None
    vlm_ids = None
    if vlm_file.exists():
        vlm_data = np.load(vlm_file, allow_pickle=True)
        vlm_emb = vlm_data["embeddings"]
        vlm_ids = vlm_data["clip_ids"]
        print(f"VLM embeddings: {vlm_emb.shape}")
        print(f"VLM clip_ids: {len(vlm_ids)}")
    else:
        print("VLM embeddings: not found (skipping VLM space)")

    return df, qwen3_emb, qwen3_ids, openclip_emb, vlm_emb, vlm_ids


# =============================================================================
# PHASE B: PREPARE SCENES
# =============================================================================

def prepare_scenes(
    df: pd.DataFrame,
    qwen3_ids: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Build index mapping from clip_id to qwen3 embedding index.
    Filter to scenes that have both qwen3 embeddings AND ADE.

    Returns:
        (filtered scenes DataFrame with emb_index remapped, clip_id->qwen3_idx map)
    """
    print("\n" + "=" * 60)
    print("PHASE B: PREPARE SCENES")
    print("=" * 60)

    # Build clip_id -> qwen3 index mapping
    qwen3_map = {cid: i for i, cid in enumerate(qwen3_ids)}

    # Filter scenes: must have qwen3 embedding AND ADE
    mask = df["clip_id"].isin(qwen3_map) & (df["has_ade"] == True)
    df_q = df[mask].copy().reset_index(drop=True)

    # Remap emb_index to qwen3 indices
    df_q["emb_index"] = df_q["clip_id"].map(qwen3_map).astype("Int64")

    print(f"Scenes with qwen3 + ADE: {len(df_q)}")
    print(f"ADE range: [{df_q['ade'].min():.2f}, {df_q['ade'].max():.2f}]")

    return df_q, qwen3_map


def prepare_openclip_scenes(
    df_full: pd.DataFrame,
    df_qwen: pd.DataFrame,
    openclip_emb: np.ndarray,
) -> pd.DataFrame:
    """
    Prepare the same subset of scenes with OpenCLIP emb_index.

    Returns scenes DataFrame with original emb_index restored,
    filtered to the same clip_ids as the qwen3 subset.
    """
    qwen3_clip_ids = set(df_qwen["clip_id"].values)

    mask = df_full["clip_id"].isin(qwen3_clip_ids)
    df_oc = df_full[mask].copy().reset_index(drop=True)

    # Verify emb_index is valid for OpenCLIP embeddings
    valid = df_oc["emb_index"].notna() & (df_oc["emb_index"] < len(openclip_emb))
    df_oc = df_oc[valid].reset_index(drop=True)

    return df_oc


def prepare_vlm_scenes(
    df_full: pd.DataFrame,
    vlm_ids: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Build index mapping from clip_id to VLM embedding index.
    Filter to scenes that have both VLM embeddings AND ADE.

    Returns:
        (filtered scenes DataFrame with emb_index remapped, clip_id->vlm_idx map)
    """
    vlm_map = {cid: i for i, cid in enumerate(vlm_ids)}

    mask = df_full["clip_id"].isin(vlm_map) & (df_full["has_ade"] == True)
    df_v = df_full[mask].copy().reset_index(drop=True)

    df_v["emb_index"] = df_v["clip_id"].map(vlm_map).astype("Int64")

    print(f"Scenes with VLM + ADE: {len(df_v)}")
    print(f"ADE range: [{df_v['ade'].min():.2f}, {df_v['ade'].max():.2f}]")

    return df_v, vlm_map


# =============================================================================
# PHASE C: k-NN PAIR ANALYSIS
# =============================================================================

def run_knn_analysis(
    scenes: pd.DataFrame,
    embeddings: np.ndarray,
    space_name: str,
    k: int = 20,
) -> tuple[dict, pd.DataFrame]:
    """
    Build k-NN graph and find single-key-diff pairs.

    Returns:
        (stats_dict, pairs_dataframe)
    """
    print(f"\n--- k-NN Analysis: {space_name} ---")

    edges = build_knn_graph(embeddings, k=k)
    print(f"  Edges: {len(edges)}")

    pairs = find_pairs(scenes, edges, max_key_diff=1)
    n_total = len(pairs)
    n_single = (pairs["hamming"] == 1).sum()
    n_with_ade = pairs["rel_delta_ade"].notna().sum()

    print(f"  Total pairs: {n_total}")
    print(f"  Single-key-diff: {n_single}")
    print(f"  With ADE: {n_with_ade}")

    # Quality metrics for single-key-diff pairs with ADE
    single_ade = pairs[(pairs["hamming"] == 1) & (pairs["rel_delta_ade"].notna())]

    per_key_pairs = {}
    for key in CLASSIFICATION_KEYS:
        kp = single_ade[single_ade["diff_key"] == key]
        if len(kp) > 0:
            per_key_pairs[key] = {
                "n_pairs": len(kp),
                "mean_rel_delta_ade": float(kp["rel_delta_ade"].mean()),
                "std_rel_delta_ade": float(kp["rel_delta_ade"].std()),
                "mean_similarity": float(kp["similarity"].mean()),
            }

    stats_dict = {
        "space": space_name,
        "k": k,
        "n_edges": len(edges),
        "n_total_pairs": n_total,
        "n_single_key_pairs": int(n_single),
        "n_pairs_with_ade": int(n_with_ade),
        "per_key": per_key_pairs,
    }

    return stats_dict, pairs


# =============================================================================
# PHASE D: COMPUTE PER-SCENE SENSITIVITY
# =============================================================================

def compute_scene_sensitivity(
    pairs_df: pd.DataFrame,
    scenes: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Compute per-scene sensitivity from single-key-diff pairs.

    For each scene, sensitivity = mean rel_delta_ade across all
    single-key-diff pairs it participates in (as either side).

    Returns:
        (overall_sensitivity, per_key_sensitivity)
        - overall: array aligned with scenes index (NaN where no pairs)
        - per_key: dict[key] -> array aligned with scenes index
    """
    single = pairs_df[
        (pairs_df["hamming"] == 1) & (pairs_df["rel_delta_ade"].notna())
    ]

    n = len(scenes)
    clip_to_idx = {cid: i for i, cid in enumerate(scenes["clip_id"].values)}

    # Collect per-scene ΔADE values
    scene_deltas = defaultdict(list)
    scene_key_deltas = defaultdict(lambda: defaultdict(list))

    for _, row in single.iterrows():
        delta = row["rel_delta_ade"]
        diff_key = row["diff_key"]
        for clip in [row["clip_a"], row["clip_b"]]:
            scene_deltas[clip].append(delta)
            if diff_key:
                scene_key_deltas[clip][diff_key].append(delta)

    # Build arrays aligned with scenes
    overall = np.full(n, np.nan)
    for clip, deltas in scene_deltas.items():
        if clip in clip_to_idx:
            overall[clip_to_idx[clip]] = np.mean(deltas)

    per_key = {}
    for key in CLASSIFICATION_KEYS:
        arr = np.full(n, np.nan)
        for clip, key_map in scene_key_deltas.items():
            if key in key_map and clip in clip_to_idx:
                arr[clip_to_idx[clip]] = np.mean(key_map[key])
        per_key[key] = arr

    n_with_sens = np.sum(~np.isnan(overall))
    print(f"  Scenes with sensitivity data: {n_with_sens}/{n}")

    return overall, per_key


# =============================================================================
# PHASE E: CORRECTED H1 — Cor(d_B, sensitivity)
# =============================================================================

def run_h1_sensitivity(
    scenes: pd.DataFrame,
    embeddings: np.ndarray,
    overall_sensitivity: np.ndarray,
    perkey_sensitivity: dict[str, np.ndarray],
    space_name: str,
) -> dict:
    """
    Corrected H1: correlate boundary margin d_B with per-scene sensitivity.

    H1 predicts: Cor(d_B, sensitivity) < 0
    Scenes near boundaries (low d_B) should be more sensitive (high ΔADE).
    """
    print(f"\n--- H1 Sensitivity: {space_name} ---")

    # Compute boundary margins
    centroids = compute_centroids(scenes, embeddings)
    margin_result = compute_boundary_margin(
        scenes, embeddings, centroids=centroids, return_full=True
    )

    sensitivity = overall_sensitivity

    # Aggregate: Cor(mean d_B, sensitivity)
    valid = ~np.isnan(margin_result.mean) & ~np.isnan(sensitivity)
    d_B = margin_result.mean[valid]
    sens = sensitivity[valid]
    n = len(d_B)

    if n < 10:
        print(f"  Insufficient data: {n} valid scenes")
        return {"error": "insufficient_data", "n": n}

    r_pearson, p_pearson = stats.pearsonr(d_B, sens)
    r_spearman, p_spearman = stats.spearmanr(d_B, sens)

    # Bootstrap CI
    rng = np.random.default_rng(42)
    boot_r = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(n, n, replace=True)
        boot_r.append(stats.pearsonr(d_B[idx], sens[idx])[0])
    ci_low, ci_high = np.percentile(boot_r, [2.5, 97.5])

    print(f"  n = {n}")
    print(f"  Pearson  r = {r_pearson:.4f}  (p = {p_pearson:.4f})")
    print(f"  Spearman r = {r_spearman:.4f}  (p = {p_spearman:.4f})")
    print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

    # Per-key: Cor(d_B_k, sensitivity_k)
    per_key = {}
    for key in CLASSIFICATION_KEYS:
        d_B_k = margin_result.per_key[key]
        sens_k = perkey_sensitivity[key]
        valid_k = ~np.isnan(d_B_k) & ~np.isnan(sens_k)
        n_k = int(valid_k.sum())

        if n_k >= 10:
            d_B_kv = d_B_k[valid_k]
            sens_kv = sens_k[valid_k]
            r_k, p_k = stats.pearsonr(d_B_kv, sens_kv)
            rho_k, _ = stats.spearmanr(d_B_kv, sens_kv)

            # Bootstrap CI
            boot_r_k = []
            for _ in range(N_BOOTSTRAP):
                idx = rng.choice(n_k, n_k, replace=True)
                boot_r_k.append(stats.pearsonr(d_B_kv[idx], sens_kv[idx])[0])
            ci_k_low, ci_k_high = np.percentile(boot_r_k, [2.5, 97.5])

            per_key[key] = {
                "r": float(r_k),
                "p": float(p_k),
                "rho": float(rho_k),
                "n": n_k,
                "ci_low": float(ci_k_low),
                "ci_high": float(ci_k_high),
            }
            sig = "**" if p_k < 0.01 else "* " if p_k < 0.05 else "  "
            print(f"  {key}: r={r_k:+.4f} (p={p_k:.4f}) {sig} n={n_k}")
        else:
            per_key[key] = {"r": None, "p": None, "n": n_k}
            print(f"  {key}: insufficient data (n={n_k})")

    return {
        "space": space_name,
        "n": n,
        "aggregate": {
            "pearson_r": float(r_pearson),
            "pearson_p": float(p_pearson),
            "spearman_r": float(r_spearman),
            "spearman_p": float(p_spearman),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
        },
        "per_key": per_key,
        # Raw arrays for figures (stripped before JSON save)
        "d_B": d_B.tolist(),
        "sensitivity": sens.tolist(),
    }


# =============================================================================
# FIGURES
# =============================================================================

def create_h1_comparison_figure(
    qwen3_results: dict,
    openclip_results: dict,
    vlm_results: dict | None = None,
) -> go.Figure:
    """Scatter: d_B vs sensitivity for embedding spaces, side by side."""
    spaces = [("OpenCLIP ViT-bigG/14", openclip_results), ("Qwen3-Embedding", qwen3_results)]
    if vlm_results is not None:
        spaces.append(("Alpamayo VLM", vlm_results))

    n_cols = len(spaces)
    fig = make_subplots(
        rows=1, cols=n_cols,
        subplot_titles=[name for name, _ in spaces],
        horizontal_spacing=0.08,
    )

    for col, (_, results) in enumerate(spaces, start=1):
        d_B = np.array(results["d_B"])
        sens = np.array(results["sensitivity"])
        agg = results["aggregate"]

        # Color by sensitivity magnitude
        ade_colors = THEME["ade"]
        colors = [
            ade_colors["low"] if s < 0.5 else
            ade_colors["medium"] if s < 1.0 else
            ade_colors["high"] if s < 1.5 else
            ade_colors["critical"]
            for s in sens
        ]

        fig.add_trace(go.Scatter(
            x=d_B, y=sens,
            mode="markers",
            marker=dict(size=5, color=colors, opacity=0.5),
            hovertemplate="d_B: %{x:.3f}<br>Sensitivity: %{y:.3f}<extra></extra>",
            showlegend=False,
        ), row=1, col=col)

        # Regression line
        slope, intercept = np.polyfit(d_B, sens, 1)
        x_line = np.linspace(d_B.min(), d_B.max(), 50)
        y_line = slope * x_line + intercept

        fig.add_trace(go.Scatter(
            x=x_line, y=y_line,
            mode="lines",
            line=dict(color=THEME["text_secondary"], width=2),
            showlegend=False,
            hoverinfo="skip",
        ), row=1, col=col)

        # Stats annotation
        x_ref = "x domain" if col == 1 else f"x{col} domain"
        y_ref = "y domain" if col == 1 else f"y{col} domain"
        sig = "**" if agg["pearson_p"] < 0.01 else "*" if agg["pearson_p"] < 0.05 else ""
        fig.add_annotation(
            x=0.98, y=0.98,
            xref=x_ref, yref=y_ref,
            text=(
                f"r = {agg['pearson_r']:.3f}{sig}<br>"
                f"p = {agg['pearson_p']:.4f}<br>"
                f"95% CI: [{agg['ci_low']:.3f}, {agg['ci_high']:.3f}]<br>"
                f"n = {results['n']}"
            ),
            showarrow=False,
            font=dict(size=10, family=THEME["font_mono"], color=THEME["text_secondary"]),
            xanchor="right", yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=THEME["border"],
            borderwidth=1,
        )

    for col in range(1, n_cols + 1):
        fig.update_xaxes(title_text="Boundary Margin d<sub>B</sub>", row=1, col=col, **axis_style(""))
        fig.update_yaxes(title_text="Sensitivity (mean ΔADE)", row=1, col=col, **axis_style(""))

    width = 450 * n_cols
    fig.update_layout(
        **plotly_layout("", height=450, width=width, show_legend=False, margin=dict(l=60, r=30, t=50, b=50)),
    )

    # Style subplot titles
    for annotation in fig.layout.annotations:
        if hasattr(annotation, "font"):
            annotation.font.size = 12
            annotation.font.color = THEME["text"]

    return fig


def create_perkey_comparison_figure(
    qwen3_results: dict,
    openclip_results: dict,
    vlm_results: dict | None = None,
) -> go.Figure:
    """Per-key Cor(d_B_k, sensitivity_k) bars: compare embedding spaces."""
    fig = go.Figure()

    keys = CLASSIFICATION_KEYS
    q_r = [qwen3_results["per_key"].get(k, {}).get("r") for k in keys]
    o_r = [openclip_results["per_key"].get(k, {}).get("r") for k in keys]
    q_p = [qwen3_results["per_key"].get(k, {}).get("p") for k in keys]
    o_p = [openclip_results["per_key"].get(k, {}).get("p") for k in keys]

    # VLM per-key values (computed once, used for sorting and bars)
    v_r = [vlm_results["per_key"].get(k, {}).get("r") for k in keys] if vlm_results is not None else None
    v_p = [vlm_results["per_key"].get(k, {}).get("p") for k in keys] if vlm_results is not None else None

    # Sort by VLM correlation if available, else qwen3
    sort_r = v_r if v_r is not None else q_r

    order = sorted(range(len(keys)), key=lambda i: (sort_r[i] or 0))
    sorted_keys = [keys[i] for i in order]
    sorted_qr = [q_r[i] for i in order]
    sorted_or = [o_r[i] for i in order]
    sorted_qp = [q_p[i] for i in order]
    sorted_op = [o_p[i] for i in order]

    def _sig_colors(p_vals):
        return [
            THEME["ade"]["critical"] if p is not None and p < 0.01 else
            THEME["ade"]["high"] if p is not None and p < 0.05 else
            THEME["point_inactive"]
            for p in p_vals
        ]

    # OpenCLIP bars
    fig.add_trace(go.Bar(
        y=sorted_keys,
        x=[r if r is not None else 0 for r in sorted_or],
        orientation="h",
        name="OpenCLIP",
        marker=dict(color=_sig_colors(sorted_op), opacity=0.5),
        offsetgroup=0,
    ))

    # Qwen3 bars
    fig.add_trace(go.Bar(
        y=sorted_keys,
        x=[r if r is not None else 0 for r in sorted_qr],
        orientation="h",
        name="Qwen3",
        marker=dict(color=_sig_colors(sorted_qp)),
        offsetgroup=1,
    ))

    # VLM bars
    if vlm_results is not None:
        sorted_vr = [v_r[i] for i in order]
        sorted_vp = [v_p[i] for i in order]
        fig.add_trace(go.Bar(
            y=sorted_keys,
            x=[r if r is not None else 0 for r in sorted_vr],
            orientation="h",
            name="VLM",
            marker=dict(color=_sig_colors(sorted_vp), opacity=0.8),
            offsetgroup=2,
        ))

    fig.add_vline(x=0, line=dict(color=THEME["border"], width=1.5))

    layout = plotly_layout("", height=380, show_legend=True, margin=dict(l=130, r=40, t=30, b=50))
    layout["legend"] = dict(
        orientation="h", y=1.05, x=0.5, xanchor="center", font=dict(size=11),
    )
    fig.update_layout(
        **layout,
        barmode="group",
        bargap=0.3,
        bargroupgap=0.1,
        xaxis={
            **axis_style(""),
            "title": dict(
                text="Cor(d<sub>B</sub><sup>k</sup>, sensitivity<sup>k</sup>)",
                font=dict(size=11),
            ),
            "zeroline": True,
            "zerolinecolor": THEME["border"],
        },
        yaxis=dict(
            tickfont=dict(size=11, color=THEME["text"]),
            showgrid=False,
        ),
    )

    return fig


def create_knn_quality_figure(
    qwen3_knn: dict,
    openclip_knn: dict,
    vlm_knn: dict | None = None,
) -> go.Figure:
    """Comparison of k-NN pair quality between spaces."""
    fig = go.Figure()

    all_knn = [qwen3_knn, openclip_knn]
    if vlm_knn is not None:
        all_knn.append(vlm_knn)

    keys = [k for k in CLASSIFICATION_KEYS
            if any(k in knn["per_key"] for knn in all_knn)]

    keys_sorted = sorted(
        keys,
        key=lambda k: qwen3_knn["per_key"].get(k, {}).get("mean_rel_delta_ade", 0),
        reverse=True,
    )

    q_vals = [qwen3_knn["per_key"].get(k, {}).get("mean_rel_delta_ade", 0) for k in keys_sorted]
    o_vals = [openclip_knn["per_key"].get(k, {}).get("mean_rel_delta_ade", 0) for k in keys_sorted]
    q_n = [qwen3_knn["per_key"].get(k, {}).get("n_pairs", 0) for k in keys_sorted]
    o_n = [openclip_knn["per_key"].get(k, {}).get("n_pairs", 0) for k in keys_sorted]

    fig.add_trace(go.Bar(
        y=keys_sorted,
        x=o_vals,
        orientation="h",
        name="OpenCLIP",
        marker=dict(color=THEME["point_inactive"]),
        customdata=o_n,
        hovertemplate="<b>%{y}</b><br>ΔADE=%{x:.3f} (n=%{customdata})<extra></extra>",
        offsetgroup=0,
    ))

    fig.add_trace(go.Bar(
        y=keys_sorted,
        x=q_vals,
        orientation="h",
        name="Qwen3",
        marker=dict(color=THEME["text"]),
        customdata=q_n,
        hovertemplate="<b>%{y}</b><br>ΔADE=%{x:.3f} (n=%{customdata})<extra></extra>",
        offsetgroup=1,
    ))

    if vlm_knn is not None:
        v_vals = [vlm_knn["per_key"].get(k, {}).get("mean_rel_delta_ade", 0) for k in keys_sorted]
        v_n = [vlm_knn["per_key"].get(k, {}).get("n_pairs", 0) for k in keys_sorted]
        fig.add_trace(go.Bar(
            y=keys_sorted,
            x=v_vals,
            orientation="h",
            name="VLM",
            marker=dict(color=THEME["ade"]["critical"], opacity=0.7),
            customdata=v_n,
            hovertemplate="<b>%{y}</b><br>ΔADE=%{x:.3f} (n=%{customdata})<extra></extra>",
            offsetgroup=2,
        ))

    q_total = qwen3_knn["n_single_key_pairs"]
    o_total = openclip_knn["n_single_key_pairs"]
    pairs_text = f"Pairs: Qwen3={q_total}, OpenCLIP={o_total}"
    if vlm_knn is not None:
        pairs_text += f", VLM={vlm_knn['n_single_key_pairs']}"
    fig.add_annotation(
        x=0.98, y=0.02,
        xref="paper", yref="paper",
        text=pairs_text,
        showarrow=False,
        font=dict(size=9, family=THEME["font_mono"], color=THEME["text_muted"]),
        xanchor="right", yanchor="bottom",
    )

    layout = plotly_layout("", height=380, show_legend=True, margin=dict(l=130, r=40, t=30, b=50))
    layout["legend"] = dict(
        orientation="h", y=1.05, x=0.5, xanchor="center", font=dict(size=11),
    )
    fig.update_layout(
        **layout,
        barmode="group",
        bargap=0.3,
        bargroupgap=0.1,
        xaxis={
            **axis_style(""),
            "title": dict(text="Mean Relative ΔADE", font=dict(size=11)),
        },
        yaxis=dict(
            tickfont=dict(size=11, color=THEME["text"]),
            showgrid=False,
        ),
    )

    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("BND-005: EMBEDDING SPACE ALIGNMENT ANALYSIS")
    print("=" * 60)
    print("Corrected H1: Cor(d_B, sensitivity) < 0")
    print("  sensitivity = mean ΔADE across single-key-diff pairs")

    # Phase A: Load data
    df, qwen3_emb, qwen3_ids, openclip_emb, vlm_emb, vlm_ids = load_data()
    has_vlm = vlm_emb is not None

    # Phase B: Prepare scenes
    df_qwen, qwen3_map = prepare_scenes(df, qwen3_ids)

    # Prepare OpenCLIP subset (same scenes)
    df_openclip = prepare_openclip_scenes(df, df_qwen, openclip_emb)
    print(f"OpenCLIP subset: {len(df_openclip)} scenes")

    # Prepare VLM subset
    df_vlm = None
    if has_vlm:
        print("\nVLM space:")
        df_vlm, vlm_map = prepare_vlm_scenes(df, vlm_ids)

    # Phase C: k-NN pair analysis (BEFORE H1 — we need pairs for sensitivity)
    print("\n" + "=" * 60)
    print("PHASE C: k-NN PAIR ANALYSIS")
    print("=" * 60)

    qwen3_knn, qwen3_pairs = run_knn_analysis(
        df_qwen, qwen3_emb, "qwen3-embedding"
    )

    # OpenCLIP: remap emb_index to contiguous 0..N-1 for subset
    openclip_subset_emb = openclip_emb[df_openclip["emb_index"].values]
    df_oc_contiguous = df_openclip.copy()
    df_oc_contiguous["emb_index"] = pd.array(
        range(len(df_oc_contiguous)), dtype="Int64"
    )
    openclip_knn, openclip_pairs = run_knn_analysis(
        df_oc_contiguous, openclip_subset_emb, "OpenCLIP-ViT-bigG/14"
    )

    vlm_knn = None
    vlm_pairs = None
    if has_vlm:
        vlm_knn, vlm_pairs = run_knn_analysis(
            df_vlm, vlm_emb, "Alpamayo-VLM"
        )

    # Phase D: Compute per-scene sensitivity
    print("\n" + "=" * 60)
    print("PHASE D: COMPUTE PER-SCENE SENSITIVITY")
    print("=" * 60)

    print("\nQwen3 space:")
    q_sens_overall, q_sens_perkey = compute_scene_sensitivity(qwen3_pairs, df_qwen)

    print("\nOpenCLIP space:")
    o_sens_overall, o_sens_perkey = compute_scene_sensitivity(
        openclip_pairs, df_oc_contiguous
    )

    v_sens_overall = None
    v_sens_perkey = None
    if has_vlm:
        print("\nVLM space:")
        v_sens_overall, v_sens_perkey = compute_scene_sensitivity(vlm_pairs, df_vlm)

    # Phase E: Corrected H1
    print("\n" + "=" * 60)
    print("PHASE E: CORRECTED H1 — Cor(d_B, sensitivity)")
    print("=" * 60)

    qwen3_h1 = run_h1_sensitivity(
        df_qwen, qwen3_emb, q_sens_overall, q_sens_perkey, "qwen3-embedding"
    )

    openclip_h1 = run_h1_sensitivity(
        df_oc_contiguous, openclip_subset_emb,
        o_sens_overall, o_sens_perkey, "OpenCLIP-ViT-bigG/14"
    )

    vlm_h1 = None
    if has_vlm:
        vlm_h1 = run_h1_sensitivity(
            df_vlm, vlm_emb,
            v_sens_overall, v_sens_perkey, "Alpamayo-VLM"
        )

    # Phase F: Save results + figures
    print("\n" + "=" * 60)
    print("PHASE F: SAVE RESULTS + FIGURES")
    print("=" * 60)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Strip raw arrays for JSON
    def strip_arrays(d):
        return {k: v for k, v in d.items() if k not in ("d_B", "sensitivity")}

    q_agg = qwen3_h1.get("aggregate", {})
    o_agg = openclip_h1.get("aggregate", {})
    v_agg = vlm_h1.get("aggregate", {}) if vlm_h1 else {}

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "git_hash": get_git_hash(),
            "n_bootstrap": N_BOOTSTRAP,
            "test": "Cor(d_B, sensitivity) — corrected H1",
        },
        "qwen3_h1": strip_arrays(qwen3_h1),
        "openclip_h1": strip_arrays(openclip_h1),
        "qwen3_knn": qwen3_knn,
        "openclip_knn": openclip_knn,
        "comparison": {
            "delta_r": (
                (q_agg.get("pearson_r", 0) or 0)
                - (o_agg.get("pearson_r", 0) or 0)
            ),
            "qwen3_stronger": (
                abs(q_agg.get("pearson_r", 0) or 0)
                > abs(o_agg.get("pearson_r", 0) or 0)
            ),
        },
    }

    if vlm_h1 is not None:
        results["vlm_h1"] = strip_arrays(vlm_h1)
    if vlm_knn is not None:
        results["vlm_knn"] = vlm_knn
    if v_agg:
        results["comparison"]["vlm_delta_r"] = (
            (v_agg.get("pearson_r", 0) or 0)
            - (o_agg.get("pearson_r", 0) or 0)
        )
        results["comparison"]["vlm_stronger_than_openclip"] = (
            abs(v_agg.get("pearson_r", 0) or 0)
            > abs(o_agg.get("pearson_r", 0) or 0)
        )
        results["comparison"]["vlm_stronger_than_qwen3"] = (
            abs(v_agg.get("pearson_r", 0) or 0)
            > abs(q_agg.get("pearson_r", 0) or 0)
        )

    results_file = BND_DIR / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {results_file}")

    # Generate figures
    h1_spaces_ok = "error" not in qwen3_h1 and "error" not in openclip_h1
    vlm_h1_ok = vlm_h1 is not None and "error" not in vlm_h1

    if h1_spaces_ok:
        print("\nGenerating figures...")

        fig_h1 = create_h1_comparison_figure(
            qwen3_h1, openclip_h1,
            vlm_results=vlm_h1 if vlm_h1_ok else None,
        )
        fig_h1.write_html(FIG_DIR / "h1_comparison.html", include_plotlyjs="cdn")
        print(f"  Saved: h1_comparison.html")

        fig_pk = create_perkey_comparison_figure(
            qwen3_h1, openclip_h1,
            vlm_results=vlm_h1 if vlm_h1_ok else None,
        )
        fig_pk.write_html(FIG_DIR / "perkey_comparison.html", include_plotlyjs="cdn")
        print(f"  Saved: perkey_comparison.html")

    fig_knn = create_knn_quality_figure(
        qwen3_knn, openclip_knn,
        vlm_knn=vlm_knn,
    )
    fig_knn.write_html(FIG_DIR / "knn_quality.html", include_plotlyjs="cdn")
    print(f"  Saved: knn_quality.html")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY: Cor(d_B, sensitivity)")
    print("=" * 60)

    vlm_col = f"{'VLM':>12}" if has_vlm else ""
    print(f"\n{'Metric':<25} {'OpenCLIP':>12} {'Qwen3':>12}{vlm_col}")
    print("-" * (52 + (14 if has_vlm else 0)))
    for label, key in [
        ("Pearson r", "pearson_r"),
        ("p-value", "pearson_p"),
        ("95% CI low", "ci_low"),
        ("95% CI high", "ci_high"),
        ("Spearman r", "spearman_r"),
    ]:
        o_v = o_agg.get(key)
        q_v = q_agg.get(key)
        o_s = f"{o_v:>12.4f}" if o_v is not None else f"{'N/A':>12}"
        q_s = f"{q_v:>12.4f}" if q_v is not None else f"{'N/A':>12}"
        row = f"{label:<25} {o_s} {q_s}"
        if has_vlm:
            v_v = v_agg.get(key)
            v_s = f"{v_v:>12.4f}" if v_v is not None else f"{'N/A':>12}"
            row += f" {v_s}"
        print(row)

    print(f"\nPer-key Cor(d_B_k, sensitivity_k):")
    print(f"{'Key':<25} {'OpenCLIP':>12} {'Qwen3':>12}{vlm_col}")
    print("-" * (52 + (14 if has_vlm else 0)))
    for key in CLASSIFICATION_KEYS:
        o_k = openclip_h1.get("per_key", {}).get(key, {})
        q_k = qwen3_h1.get("per_key", {}).get(key, {})
        o_val = f"{o_k['r']:+.4f}" if o_k.get("r") is not None else "N/A"
        q_val = f"{q_k['r']:+.4f}" if q_k.get("r") is not None else "N/A"
        # Significance markers
        o_sig = "**" if (o_k.get("p") or 1) < 0.01 else "* " if (o_k.get("p") or 1) < 0.05 else "  "
        q_sig = "**" if (q_k.get("p") or 1) < 0.01 else "* " if (q_k.get("p") or 1) < 0.05 else "  "
        row = f"{key:<25} {o_val:>10}{o_sig} {q_val:>10}{q_sig}"
        if has_vlm:
            v_k = vlm_h1.get("per_key", {}).get(key, {})
            v_val = f"{v_k['r']:+.4f}" if v_k.get("r") is not None else "N/A"
            v_sig = "**" if (v_k.get("p") or 1) < 0.01 else "* " if (v_k.get("p") or 1) < 0.05 else "  "
            row += f" {v_val:>10}{v_sig}"
        print(row)

    stronger = results["comparison"]["qwen3_stronger"]
    print(f"\nQwen3 stronger than OpenCLIP: {stronger}")
    print(f"Delta r (Qwen3 - OpenCLIP): {results['comparison']['delta_r']:.4f}")

    if has_vlm:
        print(f"VLM stronger than OpenCLIP: {results['comparison']['vlm_stronger_than_openclip']}")
        print(f"VLM stronger than Qwen3: {results['comparison']['vlm_stronger_than_qwen3']}")
        print(f"Delta r (VLM - OpenCLIP): {results['comparison']['vlm_delta_r']:.4f}")

    print(f"\nResults: {results_file}")
    print(f"Figures: {FIG_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
