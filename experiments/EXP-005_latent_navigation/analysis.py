"""
Shared Analysis Functions

Functions for:
- Dimensionality reduction (PCA, UMAP)
- Clustering (HDBSCAN)
- Alignment metrics (ARI, silhouette, intra/inter ratio)
- Navigation graph (k-NN)
- Text anchor alignment
"""

import numpy as np
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import umap
import hdbscan


# =============================================================================
# DIMENSIONALITY REDUCTION
# =============================================================================

def compute_pca(
    embeddings: np.ndarray,
    variance_threshold: float = 0.95,
) -> tuple[np.ndarray, int, PCA]:
    """
    PCA reduction preserving specified variance.

    Args:
        embeddings: (N, D) array
        variance_threshold: Target cumulative variance ratio

    Returns:
        (reduced_embeddings, n_components, pca_model)
    """
    pca = PCA(n_components=variance_threshold, svd_solver="full")
    reduced = pca.fit_transform(embeddings)
    return reduced, pca.n_components_, pca


def compute_umap_3d(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    """
    UMAP reduction to 3D for visualization.

    Args:
        embeddings: (N, D) array
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        metric: Distance metric
        random_state: Random seed for reproducibility

    Returns:
        (N, 3) array of 3D coordinates
    """
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings)


# =============================================================================
# CLUSTERING
# =============================================================================

def compute_clusters(
    embeddings: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int = 3,
    metric: str = "euclidean",
) -> tuple[np.ndarray, hdbscan.HDBSCAN]:
    """
    HDBSCAN clustering.

    Args:
        embeddings: (N, D) array
        min_cluster_size: Minimum cluster size
        min_samples: Minimum samples for core point
        metric: Distance metric

    Returns:
        (labels, clusterer)
        Labels are -1 for noise points
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
    )
    labels = clusterer.fit_predict(embeddings)
    return labels, clusterer


def compute_cluster_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """
    Compute cluster quality metrics.

    Returns:
        n_clusters: Number of clusters (excluding noise)
        noise_ratio: Fraction of points labeled as noise
        silhouette_score: Overall silhouette (NaN if <2 clusters)
    """
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_ratio = (labels == -1).sum() / len(labels)

    # Silhouette requires at least 2 clusters and some non-noise points
    non_noise_mask = labels != -1
    if n_clusters >= 2 and non_noise_mask.sum() > n_clusters:
        sil = silhouette_score(embeddings[non_noise_mask], labels[non_noise_mask])
    else:
        sil = float("nan")

    return {
        "n_clusters": n_clusters,
        "noise_ratio": float(noise_ratio),
        "silhouette_score": float(sil),
    }


# =============================================================================
# ALIGNMENT METRICS
# =============================================================================

def compute_alignment_metrics(
    embeddings: np.ndarray,
    classifications: dict[str, dict],
    scene_ids: list[str],
    cluster_labels: np.ndarray,
    keys_to_evaluate: list[str],
) -> dict:
    """
    Compute alignment metrics between embeddings and semantic keys.

    Args:
        embeddings: (N, D) array
        classifications: {scene_id: {key: value}}
        scene_ids: List of scene IDs matching embeddings order
        cluster_labels: HDBSCAN cluster labels
        keys_to_evaluate: List of semantic keys to evaluate

    Returns:
        per_key metrics: ari, silhouette, intra_inter_ratio
    """
    results = {}

    for key in keys_to_evaluate:
        # Get key values for each scene
        key_values = []
        valid_indices = []
        for i, sid in enumerate(scene_ids):
            if sid in classifications and key in classifications[sid]:
                val = classifications[sid][key]
                # Skip non-hashable values (dicts, lists)
                if isinstance(val, (dict, list)):
                    continue
                key_values.append(val)
                valid_indices.append(i)

        if len(valid_indices) < 10:
            results[key] = {
                "ari": float("nan"),
                "silhouette": float("nan"),
                "intra_inter_ratio": float("nan"),
                "n_samples": len(valid_indices),
            }
            continue

        valid_embeddings = embeddings[valid_indices]
        valid_labels = cluster_labels[valid_indices]

        # Convert key values to numeric labels
        unique_values = list(set(key_values))
        key_labels = np.array([unique_values.index(v) for v in key_values])

        # ARI: cluster vs key values
        non_noise = valid_labels != -1
        if non_noise.sum() > 1 and len(set(valid_labels[non_noise])) > 1:
            ari = adjusted_rand_score(key_labels[non_noise], valid_labels[non_noise])
        else:
            ari = float("nan")

        # Silhouette grouped by key value
        if len(unique_values) >= 2:
            sil = silhouette_score(valid_embeddings, key_labels)
        else:
            sil = float("nan")

        # Intra/inter distance ratio
        intra_inter = compute_intra_inter_ratio(valid_embeddings, key_labels)

        results[key] = {
            "ari": float(ari),
            "silhouette": float(sil),
            "intra_inter_ratio": float(intra_inter),
            "n_samples": len(valid_indices),
            "n_unique_values": len(unique_values),
        }

    return results


def compute_intra_inter_ratio(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute ratio of mean intra-class to inter-class distances.

    Lower is better (tight clusters, well separated).
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return float("nan")

    # Compute pairwise distances
    distances = cdist(embeddings, embeddings, metric="cosine")

    intra_distances = []
    inter_distances = []

    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            if labels[i] == labels[j]:
                intra_distances.append(distances[i, j])
            else:
                inter_distances.append(distances[i, j])

    if not intra_distances or not inter_distances:
        return float("nan")

    return np.mean(intra_distances) / np.mean(inter_distances)


# =============================================================================
# NAVIGATION GRAPH
# =============================================================================

def build_navigation_graph(
    embeddings: np.ndarray,
    scene_ids: list[str],
    classifications: dict[str, dict],
    keys_to_compare: list[str],
    k: int = 10,
) -> nx.Graph:
    """
    Build k-NN navigation graph with edge attributes.

    Args:
        embeddings: (N, D) array
        scene_ids: List of scene IDs
        classifications: {scene_id: {key: value}}
        keys_to_compare: Keys to check for differences
        k: Number of nearest neighbors

    Returns:
        NetworkX graph with nodes and edges
        Edge attributes: distance, keys_changed, is_single_key_diff
    """
    # Fit k-NN
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    # Build graph
    G = nx.Graph()

    # Add nodes with classifications
    for i, sid in enumerate(scene_ids):
        node_attrs = {"scene_id": sid}
        if sid in classifications:
            node_attrs.update(classifications[sid])
        G.add_node(i, **node_attrs)

    # Add edges
    for i in range(len(scene_ids)):
        sid_i = scene_ids[i]
        cls_i = classifications.get(sid_i, {})

        for j_idx in range(1, k + 1):  # Skip self (index 0)
            j = indices[i, j_idx]
            dist = distances[i, j_idx]

            if G.has_edge(i, j):
                continue

            sid_j = scene_ids[j]
            cls_j = classifications.get(sid_j, {})

            # Find keys that differ
            keys_changed = []
            for key in keys_to_compare:
                val_i = cls_i.get(key)
                val_j = cls_j.get(key)
                if val_i is not None and val_j is not None and val_i != val_j:
                    keys_changed.append(key)

            G.add_edge(
                i, j,
                distance=float(dist),
                keys_changed=keys_changed,
                is_single_key_diff=len(keys_changed) == 1,
            )

    return G


def compute_navigation_metrics(
    graph: nx.Graph,
    scene_ids: list[str],
    classifications: dict[str, dict],
    keys_to_evaluate: list[str],
) -> dict:
    """
    Compute navigation graph metrics.

    Returns:
        n_edges: Total edges
        n_single_key_diff: Edges with exactly one key different
        single_key_diff_ratio: Ratio of single-key-diff edges
        coverage: Per-key coverage (% of scenes with single-key-diff neighbor for that key)
        graph_connected: Is the graph connected?
    """
    edges = list(graph.edges(data=True))
    n_edges = len(edges)

    single_key_edges = [e for e in edges if e[2].get("is_single_key_diff", False)]
    n_single_key_diff = len(single_key_edges)

    # Coverage per key
    coverage = {}
    for key in keys_to_evaluate:
        # Count scenes that have a neighbor differing only in this key
        scenes_with_coverage = set()
        for u, v, data in single_key_edges:
            if data.get("keys_changed") == [key]:
                scenes_with_coverage.add(u)
                scenes_with_coverage.add(v)

        # Only count scenes that have this key defined
        scenes_with_key = [
            i for i, sid in enumerate(scene_ids)
            if sid in classifications and key in classifications[sid]
        ]
        if scenes_with_key:
            coverage[key] = len(scenes_with_coverage & set(scenes_with_key)) / len(scenes_with_key)
        else:
            coverage[key] = 0.0

    return {
        "n_edges": n_edges,
        "n_single_key_diff": n_single_key_diff,
        "single_key_diff_ratio": n_single_key_diff / n_edges if n_edges > 0 else 0.0,
        "coverage_per_key": coverage,
        "mean_coverage": np.mean(list(coverage.values())) if coverage else 0.0,
        "graph_connected": nx.is_connected(graph),
    }


# =============================================================================
# TEXT ANCHOR ALIGNMENT
# =============================================================================

def compute_text_anchor_alignment(
    image_embeddings: np.ndarray,
    text_embeddings: dict[str, np.ndarray],  # {value: embedding}
    scene_ids: list[str],
    classifications: dict[str, dict],
    key_name: str,
    key_values: list[str],
) -> dict:
    """
    Compute text-image alignment for a specific key.

    Args:
        image_embeddings: (N, D) array
        text_embeddings: {value: (D,) embedding}
        scene_ids: List of scene IDs
        classifications: {scene_id: {key: value}}
        key_name: The key to evaluate
        key_values: Ordered list of possible values

    Returns:
        accuracy: % of images closer to correct text anchor
        mean_rank: Mean rank of correct anchor (1 = best)
        per_value_accuracy: Accuracy broken down by value
    """
    # Stack text embeddings
    text_matrix = np.stack([text_embeddings[v] for v in key_values])  # (V, D)

    # Get ground truth and filter valid samples
    ground_truth = []
    valid_indices = []
    for i, sid in enumerate(scene_ids):
        if sid in classifications and key_name in classifications[sid]:
            gt = classifications[sid][key_name]
            if gt in key_values:
                ground_truth.append(gt)
                valid_indices.append(i)

    if not valid_indices:
        return {
            "accuracy": float("nan"),
            "mean_rank": float("nan"),
            "n_samples": 0,
        }

    valid_embeddings = image_embeddings[valid_indices]

    # Compute similarities
    similarities = valid_embeddings @ text_matrix.T  # (N_valid, V)

    # Predictions and ground truth indices
    pred_indices = similarities.argmax(axis=1)
    gt_indices = np.array([key_values.index(gt) for gt in ground_truth])

    # Accuracy
    accuracy = (pred_indices == gt_indices).mean()

    # Mean rank
    ranks = []
    for i, gt_idx in enumerate(gt_indices):
        sims = similarities[i]
        rank = (sims > sims[gt_idx]).sum() + 1
        ranks.append(rank)
    mean_rank = np.mean(ranks)

    # Per-value accuracy
    per_value_accuracy = {}
    for v_idx, v in enumerate(key_values):
        mask = gt_indices == v_idx
        if mask.sum() > 0:
            per_value_accuracy[v] = float((pred_indices[mask] == v_idx).mean())

    return {
        "accuracy": float(accuracy),
        "mean_rank": float(mean_rank),
        "n_samples": len(valid_indices),
        "per_value_accuracy": per_value_accuracy,
    }
