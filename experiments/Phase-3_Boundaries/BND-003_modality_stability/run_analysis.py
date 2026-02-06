#!/usr/bin/env python3
"""
BND-003: Modality Stability Analysis

Investigates which modality (image vs text) better predicts ADE stability,
and visualizes key-pair interactions in 3D stability landscapes.

Research Questions:
1. Does image similarity or text similarity better predict trajectory stability?
2. How do different semantic key combinations affect prediction error?
3. Are there interaction effects between keys?

Outputs:
- 3D stability surface: Key1 × Key2 × Stability
- Modality comparison metrics
- Interactive HTML visualizations
"""

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.lib.schema import load_scenes, CLASSIFICATION_KEYS
from pipeline.lib.io import load_config, load_embeddings, load_text_vocabulary, get_git_hash
from pipeline.step_1_embed import (
    OpenCLIPBigGProvider,
    TEXT_VOCABULARY,
    generate_text_vocabulary_embeddings,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = PROJECT_ROOT / "data" / "pipeline"
OUTPUT_DIR = PROJECT_ROOT / "data" / "BND-003"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "mps"  # Use MPS on Apple Silicon


# =============================================================================
# DATA LOADING
# =============================================================================

def load_pipeline_data():
    """Load all required data from the pipeline."""
    print("=" * 60)
    print("LOADING PIPELINE DATA")
    print("=" * 60)

    scenes_file = DATA_DIR / "scenes.parquet"
    embeddings_file = DATA_DIR / "embeddings.npz"
    pairs_file = DATA_DIR / "results" / "pairs.parquet"

    # Load scenes
    df = load_scenes(scenes_file)
    print(f"Scenes: {len(df)}")

    # Load image embeddings
    image_emb = load_embeddings(embeddings_file, key="embeddings")
    print(f"Image embeddings: {image_emb.shape}")

    # Load text vocabulary (if exists)
    text_emb, vocab_map = load_text_vocabulary(embeddings_file)
    if len(text_emb) == 0:
        print("Text embeddings not found - will generate")
    else:
        print(f"Text vocabulary: {text_emb.shape} ({len(vocab_map)} keys)")

    # Load pairs
    pairs_df = pd.read_parquet(pairs_file)
    print(f"Pairs: {len(pairs_df)}")
    print(f"  With ADE: {pairs_df['rel_delta_ade'].notna().sum()}")

    return df, image_emb, text_emb, vocab_map, pairs_df


def ensure_text_embeddings(embeddings_file: Path, device: str = "mps"):
    """Generate text embeddings if they don't exist."""
    text_emb, vocab_map = load_text_vocabulary(embeddings_file)

    if len(text_emb) > 0:
        print(f"Text embeddings already exist: {text_emb.shape}")
        return text_emb, vocab_map

    print("\nGenerating text vocabulary embeddings...")
    provider = OpenCLIPBigGProvider(device=device)
    text_emb, vocab_map = generate_text_vocabulary_embeddings(provider)

    # Save to embeddings file
    from pipeline.lib.io import save_embeddings
    save_embeddings(embeddings_file, text_embeddings=text_emb, text_vocab_map=vocab_map)
    print(f"Saved text embeddings: {text_emb.shape}")

    return text_emb, vocab_map


# =============================================================================
# MODALITY ANALYSIS
# =============================================================================

def compute_scene_text_vector(
    row: pd.Series,
    text_emb: np.ndarray,
    vocab_map: dict,
) -> np.ndarray | None:
    """Compute aggregated text embedding for a scene based on its labels."""
    vectors = []

    for key, value_map in vocab_map.items():
        val = row.get(key)
        if pd.isna(val):
            continue

        # Handle boolean keys stored as strings
        if isinstance(val, bool):
            val = "true" if val else "false"
        val = str(val).lower()

        if val in value_map:
            idx = value_map[val]
            vectors.append(text_emb[idx])

    if not vectors:
        return None

    # Average and normalize
    mean_vec = np.mean(vectors, axis=0)
    return mean_vec / np.linalg.norm(mean_vec)


def compute_modality_correlations(
    df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    image_emb: np.ndarray,
    text_emb: np.ndarray,
    vocab_map: dict,
) -> dict:
    """Compute correlation between modality similarity and ADE stability."""
    print("\n" + "=" * 60)
    print("MODALITY CORRELATION ANALYSIS")
    print("=" * 60)

    # Build index mappings
    emb_to_df = {}
    for idx, row in df.iterrows():
        if pd.notna(row.get("emb_index")):
            emb_to_df[int(row["emb_index"])] = idx
    df_to_emb = {v: k for k, v in emb_to_df.items()}

    # Precompute text vectors for all scenes
    print("Computing scene text vectors...")
    scene_text_vectors = {}
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        vec = compute_scene_text_vector(row, text_emb, vocab_map)
        if vec is not None:
            scene_text_vectors[idx] = vec

    print(f"Scenes with text vectors: {len(scene_text_vectors)}")

    # Collect modality data for pairs
    results = {
        "image_sim": [],
        "text_sim": [],
        "delta_ade": [],
        "diff_key": [],
        "pair_idx": [],
    }

    valid_pairs = pairs_df[pairs_df["rel_delta_ade"].notna()]
    print(f"Analyzing {len(valid_pairs)} pairs with ADE data...")

    for pair_idx, row in tqdm(valid_pairs.iterrows(), total=len(valid_pairs)):
        idx_a, idx_b = row["idx_a"], row["idx_b"]
        emb_a, emb_b = df_to_emb.get(idx_a), df_to_emb.get(idx_b)

        if emb_a is None or emb_b is None:
            continue

        # Image similarity
        img_sim = float(image_emb[emb_a] @ image_emb[emb_b])

        # Text similarity
        if idx_a not in scene_text_vectors or idx_b not in scene_text_vectors:
            continue
        txt_sim = float(scene_text_vectors[idx_a] @ scene_text_vectors[idx_b])

        results["image_sim"].append(img_sim)
        results["text_sim"].append(txt_sim)
        results["delta_ade"].append(row["rel_delta_ade"])
        results["diff_key"].append(row.get("diff_key", "unknown"))
        results["pair_idx"].append(pair_idx)

    n = len(results["image_sim"])
    print(f"Valid pairs for analysis: {n}")

    if n < 10:
        return {"error": "Insufficient data", "n_pairs": n}

    # Convert to arrays
    img_sims = np.array(results["image_sim"])
    txt_sims = np.array(results["text_sim"])
    delta_ades = np.array(results["delta_ade"])

    # Compute correlations
    img_corr, img_p = stats.spearmanr(img_sims, delta_ades)
    txt_corr, txt_p = stats.spearmanr(txt_sims, delta_ades)

    # Partial correlations
    img_resid = img_sims - np.polyval(np.polyfit(txt_sims, img_sims, 1), txt_sims)
    ade_resid_txt = delta_ades - np.polyval(np.polyfit(txt_sims, delta_ades, 1), txt_sims)
    partial_img, partial_img_p = stats.spearmanr(img_resid, ade_resid_txt)

    txt_resid = txt_sims - np.polyval(np.polyfit(img_sims, txt_sims, 1), img_sims)
    ade_resid_img = delta_ades - np.polyval(np.polyfit(img_sims, delta_ades, 1), img_sims)
    partial_txt, partial_txt_p = stats.spearmanr(txt_resid, ade_resid_img)

    analysis = {
        "n_pairs": n,
        "image": {
            "correlation": float(img_corr),
            "p_value": float(img_p),
            "partial_correlation": float(partial_img),
            "partial_p_value": float(partial_img_p),
            "mean_sim": float(np.mean(img_sims)),
            "std_sim": float(np.std(img_sims)),
        },
        "text": {
            "correlation": float(txt_corr),
            "p_value": float(txt_p),
            "partial_correlation": float(partial_txt),
            "partial_p_value": float(partial_txt_p),
            "mean_sim": float(np.mean(txt_sims)),
            "std_sim": float(np.std(txt_sims)),
        },
        "winner": "image" if abs(partial_img) > abs(partial_txt) else "text",
        "raw_data": results,
    }

    # Print results
    print(f"\nIMAGE MODALITY:")
    print(f"  Correlation with ΔADE: {img_corr:.4f} (p={img_p:.4f})")
    print(f"  Partial correlation:   {partial_img:.4f} (p={partial_img_p:.4f})")

    print(f"\nTEXT MODALITY:")
    print(f"  Correlation with ΔADE: {txt_corr:.4f} (p={txt_p:.4f})")
    print(f"  Partial correlation:   {partial_txt:.4f} (p={partial_txt_p:.4f})")

    print(f"\n→ More predictive: {analysis['winner'].upper()}")

    return analysis


# =============================================================================
# KEY-PAIR STABILITY ANALYSIS
# =============================================================================

def compute_key_pair_stability(
    df: pd.DataFrame,
    pairs_df: pd.DataFrame,
) -> dict:
    """
    Compute stability metrics for all key-pair combinations.

    Returns a matrix of stability values for 3D visualization.
    """
    print("\n" + "=" * 60)
    print("KEY-PAIR STABILITY ANALYSIS")
    print("=" * 60)

    # Get all keys that appear in single-key-diff pairs
    valid_pairs = pairs_df[(pairs_df["hamming"] == 1) & (pairs_df["rel_delta_ade"].notna())]
    keys_in_data = valid_pairs["diff_key"].dropna().unique().tolist()
    print(f"Keys with data: {keys_in_data}")

    # For each pair, record which keys are SAME (context) vs DIFFERENT
    key_context_stability = defaultdict(lambda: defaultdict(list))

    for _, row in tqdm(valid_pairs.iterrows(), total=len(valid_pairs), desc="Analyzing key contexts"):
        diff_key = row["diff_key"]
        delta_ade = row["rel_delta_ade"]
        idx_a, idx_b = row["idx_a"], row["idx_b"]

        row_a = df.loc[idx_a]
        row_b = df.loc[idx_b]

        # Find which other keys have the same value (the context)
        for ctx_key in CLASSIFICATION_KEYS:
            if ctx_key == diff_key:
                continue

            val_a = row_a.get(ctx_key)
            val_b = row_b.get(ctx_key)

            if pd.isna(val_a) or pd.isna(val_b):
                continue

            if val_a == val_b:
                # This key provides context (same value)
                # Record: when diff_key differs, in context of ctx_key=val_a, what's the stability?
                key_context_stability[diff_key][(ctx_key, val_a)].append(delta_ade)

    # Build stability matrix for key pairs
    # Matrix[i][j] = mean stability when key_i differs, given key_j is same
    stability_matrix = {}

    for diff_key, contexts in key_context_stability.items():
        stability_matrix[diff_key] = {}
        for (ctx_key, ctx_val), ade_values in contexts.items():
            if len(ade_values) >= 3:  # Minimum sample size
                stability_matrix[diff_key][f"{ctx_key}={ctx_val}"] = {
                    "mean_delta_ade": float(np.mean(ade_values)),
                    "std_delta_ade": float(np.std(ade_values)),
                    "n_samples": len(ade_values),
                }

    # Also compute per-key overall statistics
    per_key_stats = {}
    for key in keys_in_data:
        key_pairs = valid_pairs[valid_pairs["diff_key"] == key]
        if len(key_pairs) >= 3:
            ades = key_pairs["rel_delta_ade"].values
            per_key_stats[key] = {
                "mean_delta_ade": float(np.mean(ades)),
                "std_delta_ade": float(np.std(ades)),
                "n_pairs": len(ades),
            }

    print(f"\nPer-key statistics:")
    for key, stats in sorted(per_key_stats.items(), key=lambda x: x[1]["mean_delta_ade"], reverse=True):
        print(f"  {key}: ΔADE={stats['mean_delta_ade']:.3f} ± {stats['std_delta_ade']:.3f} (n={stats['n_pairs']})")

    return {
        "per_key": per_key_stats,
        "key_context": stability_matrix,
        "keys_in_data": keys_in_data,
    }


# =============================================================================
# 3D VISUALIZATION
# =============================================================================

def generate_3d_visualizations(
    modality_analysis: dict,
    key_pair_analysis: dict,
    output_dir: Path,
):
    """Generate 3D visualizations using Plotly."""
    print("\n" + "=" * 60)
    print("GENERATING 3D VISUALIZATIONS")
    print("=" * 60)

    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        print("ERROR: plotly not installed. Run: pip install plotly kaleido")
        return

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # =========================================================================
    # FIGURE 1: 3D Modality Space
    # X = Image Similarity, Y = Text Similarity, Z = Delta ADE
    # =========================================================================
    print("\n1. Generating Modality-Stability 3D scatter...")

    raw = modality_analysis.get("raw_data", {})
    if raw:
        img_sims = np.array(raw["image_sim"])
        txt_sims = np.array(raw["text_sim"])
        delta_ades = np.array(raw["delta_ade"])
        diff_keys = raw["diff_key"]

        # Create scatter with color by key
        fig1 = go.Figure()

        unique_keys = [k for k in set(diff_keys) if k is not None]
        colors = px.colors.qualitative.Set2

        for i, key in enumerate(unique_keys):
            mask = np.array([k == key for k in diff_keys])
            if mask.sum() == 0:
                continue

            key_label = key.replace("_", " ").title() if key else "Unknown"
            fig1.add_trace(go.Scatter3d(
                x=img_sims[mask],
                y=txt_sims[mask],
                z=delta_ades[mask],
                mode='markers',
                name=key_label,
                marker=dict(
                    size=4,
                    color=colors[i % len(colors)],
                    opacity=0.7,
                ),
                hovertemplate=(
                    f"<b>{key_label}</b><br>"
                    "Image Sim: %{x:.3f}<br>"
                    "Text Sim: %{y:.3f}<br>"
                    "Δ ADE: %{z:.3f}<extra></extra>"
                ),
            ))

        # Add regression plane
        A = np.column_stack([img_sims, txt_sims, np.ones_like(img_sims)])
        coeffs, _, _, _ = np.linalg.lstsq(A, delta_ades, rcond=None)

        x_range = np.linspace(img_sims.min(), img_sims.max(), 15)
        y_range = np.linspace(txt_sims.min(), txt_sims.max(), 15)
        X, Y = np.meshgrid(x_range, y_range)
        Z = coeffs[0] * X + coeffs[1] * Y + coeffs[2]

        fig1.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            opacity=0.3,
            colorscale='Greys',
            showscale=False,
            name='Regression Plane',
            hoverinfo='skip',
        ))

        fig1.update_layout(
            title=dict(
                text="<b>Modality-Stability Space</b><br>"
                     "<sup>How Image & Text Similarity Relate to ADE Instability</sup>",
                x=0.5,
            ),
            scene=dict(
                xaxis_title="Image Embedding Similarity",
                yaxis_title="Text Embedding Similarity",
                zaxis_title="Relative Δ ADE",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
            ),
            legend=dict(x=0.02, y=0.98),
            width=1000,
            height=800,
        )

        fig1.write_html(fig_dir / "modality_stability_3d.html")
        fig1.write_image(fig_dir / "modality_stability_3d.png", scale=2)
        print(f"  Saved: modality_stability_3d.html")

    # =========================================================================
    # FIGURE 2: 3D Key-Pair Stability Surface
    # X = Key 1 (differing), Y = Key 2 (context), Z = Stability
    # =========================================================================
    print("\n2. Generating Key-Pair Stability 3D surface...")

    per_key = key_pair_analysis.get("per_key", {})
    keys = list(per_key.keys())

    if len(keys) >= 2:
        # Build matrix: row = differing key, col = context key
        n_keys = len(keys)
        stability_matrix = np.zeros((n_keys, n_keys))
        count_matrix = np.zeros((n_keys, n_keys))

        # Diagonal: single-key stability (no specific context)
        for i, key in enumerate(keys):
            stability_matrix[i, i] = per_key[key]["mean_delta_ade"]
            count_matrix[i, i] = per_key[key]["n_pairs"]

        # Off-diagonal: key-in-context stability
        key_context = key_pair_analysis.get("key_context", {})
        for diff_key, contexts in key_context.items():
            if diff_key not in keys:
                continue
            i = keys.index(diff_key)

            for ctx_str, stats in contexts.items():
                # Parse "ctx_key=value"
                ctx_key = ctx_str.split("=")[0]
                if ctx_key not in keys:
                    continue
                j = keys.index(ctx_key)

                # Average if multiple values for same context key
                if count_matrix[i, j] == 0:
                    stability_matrix[i, j] = stats["mean_delta_ade"]
                    count_matrix[i, j] = stats["n_samples"]
                else:
                    # Weighted average
                    old_n = count_matrix[i, j]
                    new_n = stats["n_samples"]
                    stability_matrix[i, j] = (
                        stability_matrix[i, j] * old_n + stats["mean_delta_ade"] * new_n
                    ) / (old_n + new_n)
                    count_matrix[i, j] = old_n + new_n

        # Create 3D surface
        fig2 = go.Figure()

        x = np.arange(n_keys)
        y = np.arange(n_keys)
        X, Y = np.meshgrid(x, y)

        fig2.add_trace(go.Surface(
            x=X,
            y=Y,
            z=stability_matrix,
            colorscale='RdYlGn_r',
            colorbar=dict(title=dict(text="Mean Δ ADE", side="right")),
            hovertemplate=(
                "Diff: %{customdata[0]}<br>"
                "Context: %{customdata[1]}<br>"
                "Δ ADE: %{z:.3f}<br>"
                "n: %{customdata[2]}<extra></extra>"
            ),
            customdata=np.dstack([
                np.array([[keys[i] for _ in range(n_keys)] for i in range(n_keys)]),
                np.array([[keys[j] for j in range(n_keys)] for _ in range(n_keys)]),
                count_matrix,
            ]),
        ))

        key_labels = [k.replace("_", " ").title() for k in keys]

        fig2.update_layout(
            title=dict(
                text="<b>Key-Pair Stability Landscape</b><br>"
                     "<sup>X: Differing Key | Y: Context Key (same) | Z: Instability</sup>",
                x=0.5,
            ),
            scene=dict(
                xaxis=dict(
                    title="Differing Key",
                    tickvals=list(range(n_keys)),
                    ticktext=key_labels,
                ),
                yaxis=dict(
                    title="Context Key",
                    tickvals=list(range(n_keys)),
                    ticktext=key_labels,
                ),
                zaxis_title="Mean Relative Δ ADE",
                camera=dict(eye=dict(x=1.8, y=1.8, z=1.0)),
            ),
            width=1000,
            height=800,
        )

        fig2.write_html(fig_dir / "key_pair_stability_3d.html")
        fig2.write_image(fig_dir / "key_pair_stability_3d.png", scale=2)
        print(f"  Saved: key_pair_stability_3d.html")

    # =========================================================================
    # FIGURE 3: Per-Key Stability Bars (3D effect)
    # =========================================================================
    print("\n3. Generating Per-Key Stability 3D bars...")

    if per_key:
        sorted_keys = sorted(per_key.keys(), key=lambda k: per_key[k]["mean_delta_ade"], reverse=True)

        fig3 = go.Figure()

        colors = px.colors.sequential.Reds
        max_ade = max(per_key[k]["mean_delta_ade"] for k in sorted_keys)

        for i, key in enumerate(sorted_keys):
            stats = per_key[key]
            ade = stats["mean_delta_ade"]
            color_idx = int((ade / max_ade) * (len(colors) - 1))

            fig3.add_trace(go.Scatter3d(
                x=[i, i],
                y=[0, 0],
                z=[0, ade],
                mode='lines+markers',
                line=dict(width=20, color=colors[color_idx]),
                marker=dict(size=[0, 8], color=colors[color_idx]),
                name=key.replace("_", " ").title(),
                hovertemplate=(
                    f"<b>{key}</b><br>"
                    f"Δ ADE: {ade:.3f}<br>"
                    f"n: {stats['n_pairs']}<extra></extra>"
                ),
            ))

        fig3.update_layout(
            title=dict(
                text="<b>Per-Key ADE Instability</b><br>"
                     "<sup>Height = Mean Relative Δ ADE when key differs</sup>",
                x=0.5,
            ),
            scene=dict(
                xaxis=dict(
                    title="",
                    tickvals=list(range(len(sorted_keys))),
                    ticktext=[k.replace("_", " ").title() for k in sorted_keys],
                ),
                yaxis=dict(title="", showticklabels=False, range=[-0.5, 0.5]),
                zaxis_title="Mean Relative Δ ADE",
                camera=dict(eye=dict(x=0.1, y=-2.0, z=0.8)),
            ),
            showlegend=False,
            width=1000,
            height=700,
        )

        fig3.write_html(fig_dir / "key_stability_bars_3d.html")
        fig3.write_image(fig_dir / "key_stability_bars_3d.png", scale=2)
        print(f"  Saved: key_stability_bars_3d.html")

    # =========================================================================
    # FIGURE 4: Modality Comparison (2D summary)
    # =========================================================================
    print("\n4. Generating Modality Comparison chart...")

    if "image" in modality_analysis and "text" in modality_analysis:
        img = modality_analysis["image"]
        txt = modality_analysis["text"]

        metrics = ["Correlation", "Partial Corr.", "Unique R²"]
        img_vals = [
            abs(img["correlation"]),
            abs(img["partial_correlation"]),
            img["partial_correlation"] ** 2,
        ]
        txt_vals = [
            abs(txt["correlation"]),
            abs(txt["partial_correlation"]),
            txt["partial_correlation"] ** 2,
        ]

        fig4 = go.Figure()

        fig4.add_trace(go.Bar(
            name="Image Modality",
            x=metrics,
            y=img_vals,
            marker_color='steelblue',
            text=[f"{v:.3f}" for v in img_vals],
            textposition='outside',
        ))

        fig4.add_trace(go.Bar(
            name="Text Modality",
            x=metrics,
            y=txt_vals,
            marker_color='coral',
            text=[f"{v:.3f}" for v in txt_vals],
            textposition='outside',
        ))

        winner = modality_analysis.get("winner", "unknown")
        fig4.update_layout(
            title=dict(
                text=f"<b>Modality Importance for ADE Prediction</b><br>"
                     f"<sup>Winner: {winner.upper()} modality</sup>",
                x=0.5,
            ),
            xaxis_title="Metric",
            yaxis_title="Absolute Value",
            barmode='group',
            legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
            width=800,
            height=500,
        )

        fig4.write_html(fig_dir / "modality_comparison.html")
        fig4.write_image(fig_dir / "modality_comparison.png", scale=2)
        print(f"  Saved: modality_comparison.html")

    print(f"\nAll figures saved to: {fig_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("BND-003: MODALITY STABILITY ANALYSIS")
    print("=" * 60)
    print(f"Output: {OUTPUT_DIR}")
    print(f"Device: {DEVICE}")

    # Load data
    df, image_emb, text_emb, vocab_map, pairs_df = load_pipeline_data()

    # Ensure text embeddings exist
    embeddings_file = DATA_DIR / "embeddings.npz"
    if len(text_emb) == 0:
        text_emb, vocab_map = ensure_text_embeddings(embeddings_file, device=DEVICE)

    # Modality correlation analysis
    modality_analysis = compute_modality_correlations(
        df, pairs_df, image_emb, text_emb, vocab_map
    )

    # Key-pair stability analysis
    key_pair_analysis = compute_key_pair_stability(df, pairs_df)

    # Generate visualizations
    generate_3d_visualizations(modality_analysis, key_pair_analysis, OUTPUT_DIR)

    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "git_hash": get_git_hash(),
            "n_scenes": len(df),
            "n_pairs": len(pairs_df),
        },
        "modality_analysis": {
            k: v for k, v in modality_analysis.items() if k != "raw_data"
        },
        "key_pair_analysis": key_pair_analysis,
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {OUTPUT_DIR / 'results.json'}")

    # Save raw data for further analysis
    if "raw_data" in modality_analysis:
        raw = modality_analysis["raw_data"]
        np.savez_compressed(
            OUTPUT_DIR / "modality_raw_data.npz",
            image_sim=np.array(raw["image_sim"]),
            text_sim=np.array(raw["text_sim"]),
            delta_ade=np.array(raw["delta_ade"]),
        )
        print(f"Saved: {OUTPUT_DIR / 'modality_raw_data.npz'}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✓ Modality winner: {modality_analysis.get('winner', 'N/A').upper()}")
    print(f"✓ Keys analyzed: {len(key_pair_analysis.get('per_key', {}))}")
    print(f"✓ 3D visualizations generated")
    print(f"\nView interactive plots:")
    print(f"  open {OUTPUT_DIR / 'figures' / 'modality_stability_3d.html'}")
    print(f"  open {OUTPUT_DIR / 'figures' / 'key_pair_stability_3d.html'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
