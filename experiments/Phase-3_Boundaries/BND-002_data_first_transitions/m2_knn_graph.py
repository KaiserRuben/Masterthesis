#!/usr/bin/env python3
"""
BND-002 Milestone 2: k-NN Graph Construction

Builds a nearest-neighbor graph in embedding space for the 100 usable scenes.
"""

import json
import numpy as np
import networkx as nx
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# Paths
DATA_DIR = Path("/Users/kaiser/Projects/Masterarbeit/data")
OUTPUT_DIR = DATA_DIR / "BND-002"

# Configuration
K_NEIGHBORS = 10  # Number of nearest neighbors

# =============================================================================
# 1. Load Data
# =============================================================================
print("=" * 60)
print("M2: k-NN GRAPH CONSTRUCTION")
print("=" * 60)

# Load M1 summary to get usable scene IDs
with open(OUTPUT_DIR / "m1_data_exploration.json") as f:
    m1_summary = json.load(f)

usable_clip_ids = set(m1_summary["usable_scenes"]["clip_ids"])
print(f"Usable scenes: {len(usable_clip_ids)}")

# Load embeddings
emb_path = DATA_DIR / "EMB-001/v2/openclip_bigg_top_20260129_043407/embeddings.npz"
emb_data = np.load(emb_path, allow_pickle=True)

all_embeddings = emb_data["embeddings"]
all_scene_ids = [str(sid) for sid in emb_data["scene_ids"]]

# Filter to usable scenes only
scene_to_idx = {sid: i for i, sid in enumerate(all_scene_ids)}
usable_indices = [scene_to_idx[sid] for sid in usable_clip_ids if sid in scene_to_idx]
usable_ids = [all_scene_ids[i] for i in usable_indices]

embeddings = all_embeddings[usable_indices]
print(f"Filtered embeddings shape: {embeddings.shape}")

# Create ID to index mapping for the filtered set
id_to_idx = {sid: i for i, sid in enumerate(usable_ids)}
idx_to_id = {i: sid for i, sid in enumerate(usable_ids)}

# =============================================================================
# 2. Compute Pairwise Cosine Similarity
# =============================================================================
print("\nComputing pairwise cosine similarity...")

# Since embeddings are L2-normalized, cosine similarity = dot product
similarity_matrix = cosine_similarity(embeddings)

print(f"Similarity matrix shape: {similarity_matrix.shape}")
print(f"Similarity range: [{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]")

# Exclude self-similarity for statistics
np.fill_diagonal(similarity_matrix, 0)
non_zero_sims = similarity_matrix[similarity_matrix > 0]
print(f"Non-self similarity stats:")
print(f"  Min: {non_zero_sims.min():.4f}")
print(f"  Max: {non_zero_sims.max():.4f}")
print(f"  Mean: {non_zero_sims.mean():.4f}")
print(f"  Std: {non_zero_sims.std():.4f}")

# =============================================================================
# 3. Construct k-NN Graph
# =============================================================================
print(f"\nConstructing k-NN graph (k={K_NEIGHBORS})...")

G = nx.Graph()

# Add all nodes with their clip_id as attribute
for i, clip_id in enumerate(usable_ids):
    G.add_node(i, clip_id=clip_id)

# For each node, connect to k nearest neighbors
edges_added = 0
for i in range(len(usable_ids)):
    # Get similarities to all other nodes
    sims = similarity_matrix[i].copy()
    sims[i] = -1  # Exclude self

    # Get indices of k most similar nodes
    top_k_indices = np.argsort(sims)[-K_NEIGHBORS:]

    for j in top_k_indices:
        if not G.has_edge(i, j):
            G.add_edge(i, j,
                       similarity=float(sims[j]),
                       distance=float(1 - sims[j]))
            edges_added += 1

print(f"Edges added: {edges_added}")

# =============================================================================
# 4. Graph Analysis
# =============================================================================
print("\n" + "=" * 60)
print("GRAPH STATISTICS")
print("=" * 60)

print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Density: {nx.density(G):.4f}")

# Degree distribution
degrees = [d for n, d in G.degree()]
print(f"\nDegree distribution:")
print(f"  Min: {min(degrees)}")
print(f"  Max: {max(degrees)}")
print(f"  Mean: {np.mean(degrees):.2f}")

# Connected components
components = list(nx.connected_components(G))
print(f"\nConnected components: {len(components)}")
if len(components) > 1:
    print(f"  Component sizes: {sorted([len(c) for c in components], reverse=True)}")
else:
    print("  Graph is fully connected")

# Edge weight (similarity) distribution
edge_sims = [d['similarity'] for _, _, d in G.edges(data=True)]
print(f"\nEdge similarity distribution:")
print(f"  Min: {min(edge_sims):.4f}")
print(f"  Max: {max(edge_sims):.4f}")
print(f"  Mean: {np.mean(edge_sims):.4f}")

# Average clustering coefficient
avg_clustering = nx.average_clustering(G)
print(f"\nAverage clustering coefficient: {avg_clustering:.4f}")

# =============================================================================
# 5. Save Graph
# =============================================================================
print("\n" + "=" * 60)
print("SAVING OUTPUTS")
print("=" * 60)

# Save graph as pickle
graph_path = OUTPUT_DIR / "knn_graph.pkl"
with open(graph_path, "wb") as f:
    pickle.dump(G, f)
print(f"Graph saved to: {graph_path}")

# Save metadata
metadata = {
    "generated_at": datetime.now().isoformat(),
    "k_neighbors": K_NEIGHBORS,
    "n_nodes": G.number_of_nodes(),
    "n_edges": G.number_of_edges(),
    "density": nx.density(G),
    "n_connected_components": len(components),
    "is_connected": len(components) == 1,
    "degree_stats": {
        "min": min(degrees),
        "max": max(degrees),
        "mean": float(np.mean(degrees))
    },
    "edge_similarity_stats": {
        "min": float(min(edge_sims)),
        "max": float(max(edge_sims)),
        "mean": float(np.mean(edge_sims))
    },
    "avg_clustering_coefficient": avg_clustering,
    "id_to_idx": id_to_idx,
    "idx_to_id": idx_to_id
}

metadata_path = OUTPUT_DIR / "knn_graph_metadata.json"
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Metadata saved to: {metadata_path}")

# Save similarity matrix for later use
sim_path = OUTPUT_DIR / "similarity_matrix.npz"
np.savez(sim_path,
         similarity=similarity_matrix,
         scene_ids=usable_ids)
print(f"Similarity matrix saved to: {sim_path}")

# =============================================================================
# 6. Checkpoint
# =============================================================================
print("\n" + "=" * 60)
print("M2 CHECKPOINT")
print("=" * 60)

if len(components) == 1 and G.number_of_edges() > 0:
    print(f"✅ k-NN graph constructed with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"✅ Graph is connected (1 component)")
    print(f"✅ Average similarity on edges: {np.mean(edge_sims):.4f}")
    print("✅ Ready to proceed with M3: Single-Key-Diff Pair Detection")
else:
    print(f"⚠️  Graph has {len(components)} connected components")
    print("   Consider increasing k to connect the graph")
