"""Precompute semantic taxonomy for all 1000 ImageNet classes.

fastText word-embedding (wiki-news-subwords-300, averaged over tokens)
→ UMAP 2D → HDBSCAN auto-clusters. WordNet hypernym lookup in parallel
as interpretable super-category. Full pairwise distance matrices saved
for downstream analysis.

Raw-first: every per-class field, raw embeddings, and full 1000×1000
distance matrices (cosine + WordNet path-similarity) go to disk.
Post-processing filters whatever it needs.

Outputs to `runs/preprocessing/taxonomy/`:
    category_taxonomy.parquet       per-class main table
    embeddings_fasttext.npy         (1000, 300) averaged word-vec embeddings
    pair_distances_fasttext.npy     (1000, 1000) cosine distance matrix
    pair_distances_wordnet.npy      (1000, 1000) WordNet path-similarity
    umap_scatter.png                2D UMAP scatter, colored by cluster
    context.json                    run metadata

CPU-only; no GPU/MPS needed. ~2 min total wall time.

Why fastText over CLIP: we want *linguistic* semantic structure over the
labels, not the joint vision-text space. fastText matches the embedding
model already used by the text manipulator (``fasttext-wiki-news-subwords-300``
via gensim), so taxonomy and word-mutation live in the same embedding.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data import ImageNetCache  # noqa: E402


FASTTEXT_MODEL_ID = "fasttext-wiki-news-subwords-300"
OUT_DIR = REPO / "runs" / "taxonomy"

# Curated WordNet super-categories, each a valid WN noun synset.
# Walk each class's hypernyms and assign the first super-cat that appears.
# Order matters: more specific (closer to leaves) listed first so
# "sporting_dog" wins over generic "mammal".
WN_SUPER_CATS = [
    "fish.n.01", "bird.n.01", "reptile.n.01", "amphibian.n.03",
    "arachnid.n.01", "insect.n.01", "crustacean.n.01", "mollusk.n.01",
    "dog.n.01", "feline.n.01", "primate.n.02", "rodent.n.01",
    "ungulate.n.01", "aquatic_mammal.n.01", "carnivore.n.01",
    "mammal.n.01", "invertebrate.n.01", "plant.n.02",
    "food.n.02", "food.n.01", "clothing.n.01", "tool.n.01",
    "container.n.01", "instrument.n.01", "wheeled_vehicle.n.01",
    "vessel.n.02", "aircraft.n.01", "vehicle.n.01",
    "building.n.01", "structure.n.01", "furniture.n.01",
    "device.n.01", "instrumentality.n.03", "covering.n.02",
    "substance.n.01", "geological_formation.n.01",
    "natural_object.n.01", "artifact.n.01",
]


def load_labels() -> list[str]:
    """Return the canonical 1000 ImageNet class names (human-readable)."""
    cache = ImageNetCache(dirs=(
        Path("~/.cache/imagenet").expanduser(),
        Path("/Volumes/SanDisk/Cache/imagenet"),
    ))
    labels = cache.labels()
    if len(labels) != 1000:
        raise RuntimeError(
            f"Expected 1000 ImageNet classes, got {len(labels)}"
        )
    return labels


def embed_with_fasttext(labels: list[str]) -> tuple[np.ndarray, str, list[int]]:
    """Encode each label as the mean of its token fastText word vectors.

    Uses the same ``fasttext-wiki-news-subwords-300`` KeyedVectors that the
    text manipulator loads. Labels like "great white shark" are tokenized
    (lower-cased, comma-stripped) and per-word vectors are averaged.

    Returns:
        embeddings: (N, 300) float32 array.
        model_id: the gensim model name used.
        oov_token_counts: per-label count of tokens missing from the vocabulary.
    """
    import gensim.downloader as api

    print(
        f"[{datetime.now():%H:%M:%S}] Loading fastText '{FASTTEXT_MODEL_ID}' "
        f"(gensim KeyedVectors)...",
        flush=True,
    )
    kv = api.load(FASTTEXT_MODEL_ID)
    emb_dim = int(kv.vector_size)

    feats: list[np.ndarray] = []
    oov_counts: list[int] = []
    for lbl in tqdm(labels, desc="fastText embed"):
        # "great white shark, Carcharodon carcharias" → first segment, lower-cased
        phrase = lbl.split(",")[0].lower()
        tokens = [t for t in phrase.replace("-", " ").split() if t]
        vecs = []
        oov = 0
        for t in tokens:
            if t in kv:
                vecs.append(kv[t])
            else:
                oov += 1
        if vecs:
            feats.append(np.mean(vecs, axis=0))
        else:
            # All tokens OOV — use zero vector (will be an outlier in UMAP)
            feats.append(np.zeros(emb_dim, dtype=np.float32))
        oov_counts.append(oov)

    emb = np.asarray(feats, dtype=np.float32)
    total_oov = sum(oov_counts)
    n_empty = sum(1 for v in feats if not np.any(v))
    print(
        f"[{datetime.now():%H:%M:%S}] fastText done. "
        f"shape {emb.shape}, {total_oov} OOV tokens, "
        f"{n_empty} labels with all-OOV tokens (zero vectors)",
        flush=True,
    )
    return emb, FASTTEXT_MODEL_ID, oov_counts


def cosine_distance_matrix(emb: np.ndarray) -> np.ndarray:
    """Return (N, N) cosine distance matrix (1 - cosine_similarity)."""
    normed = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    sim = normed @ normed.T
    dist = 1.0 - sim
    np.clip(dist, 0.0, 2.0, out=dist)
    return dist.astype(np.float32)


def umap_2d(emb: np.ndarray, seed: int = 0) -> np.ndarray:
    """2D UMAP projection using cosine metric."""
    import umap

    print(f"[{datetime.now():%H:%M:%S}] UMAP 2D (cosine)...", flush=True)
    reducer = umap.UMAP(
        n_components=2, metric="cosine", random_state=seed,
        n_neighbors=15, min_dist=0.1,
    )
    return reducer.fit_transform(emb).astype(np.float32)


def hdbscan_cluster(coords: np.ndarray, min_cluster_size: int = 8) -> np.ndarray:
    """HDBSCAN on 2D UMAP coords. Returns per-class int labels (-1 = noise)."""
    import hdbscan

    print(f"[{datetime.now():%H:%M:%S}] HDBSCAN (min_cluster_size={min_cluster_size})...", flush=True)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=3,
        cluster_selection_method="eom",
    )
    return clusterer.fit_predict(coords).astype(np.int64)


def wordnet_synset_for_label(label: str):
    """Best-effort WordNet synset lookup for an ImageNet label.

    Tries (in order): full label with underscores, first segment before comma,
    first word. Returns ``None`` if nothing matches.
    """
    from nltk.corpus import wordnet as wn

    candidates = [
        label.replace(" ", "_"),
        label.split(",")[0].strip().replace(" ", "_"),
        label.split()[0],
    ]
    seen = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        syns = wn.synsets(c, pos=wn.NOUN)
        if syns:
            return syns[0]
    return None


def wordnet_super_category(synset) -> str:
    """Assign a super-category label from the curated WN_SUPER_CATS list."""
    from nltk.corpus import wordnet as wn

    if synset is None:
        return "unknown"

    supers_as_syn = {wn.synset(n) for n in WN_SUPER_CATS}
    for path in synset.hypernym_paths():
        # Walk from leaf upward (path is root→leaf; we reverse)
        for ancestor in reversed(path):
            if ancestor in supers_as_syn:
                return ancestor.name().split(".")[0]
    return "other"


def wordnet_distance_matrix(synsets: list) -> np.ndarray:
    """(N, N) WordNet path-similarity (Wu-Palmer) where available, else 0."""
    n = len(synsets)
    m = np.zeros((n, n), dtype=np.float32)
    print(
        f"[{datetime.now():%H:%M:%S}] WordNet path similarity "
        f"({n*(n-1)//2} upper-triangle pairs)...",
        flush=True,
    )
    for i in tqdm(range(n), desc="WN sim"):
        si = synsets[i]
        if si is None:
            continue
        m[i, i] = 1.0
        for j in range(i + 1, n):
            sj = synsets[j]
            if sj is None:
                continue
            try:
                sim = si.wup_similarity(sj) or 0.0
            except Exception:
                sim = 0.0
            m[i, j] = sim
            m[j, i] = sim
    return m


def render_scatter(
    umap_coords: np.ndarray,
    clusters: np.ndarray,
    wn_supers: list[str],
    labels: list[str],
    out_path: Path,
) -> None:
    """Two-panel UMAP scatter: HDBSCAN clusters | WordNet super-cats."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Panel 1: HDBSCAN clusters
    unique = np.unique(clusters)
    n_clusters = int((unique != -1).sum())
    n_noise = int((clusters == -1).sum())
    for c in unique:
        mask = clusters == c
        color = "lightgrey" if c == -1 else None
        alpha = 0.25 if c == -1 else 0.75
        axes[0].scatter(
            umap_coords[mask, 0], umap_coords[mask, 1],
            c=color, s=8, alpha=alpha, label=f"c{c}" if c != -1 else "noise",
        )
    axes[0].set_title(
        f"HDBSCAN clusters: {n_clusters} clusters, {n_noise} noise"
    )
    axes[0].axis("off")

    # Panel 2: WordNet super-cats
    super_unique = sorted(set(wn_supers))
    cmap = plt.get_cmap("tab20", len(super_unique))
    for i, s in enumerate(super_unique):
        mask = np.array([x == s for x in wn_supers])
        axes[1].scatter(
            umap_coords[mask, 0], umap_coords[mask, 1],
            s=8, alpha=0.7, color=cmap(i), label=s,
        )
    axes[1].legend(loc="best", fontsize=7, markerscale=2, ncol=2)
    axes[1].set_title(f"WordNet super-cats ({len(super_unique)} labels)")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{datetime.now():%H:%M:%S}] scatter saved to {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min-cluster-size", type=int, default=8,
        help="HDBSCAN min_cluster_size (default 8 → ~30-50 clusters on 1000)",
    )
    args = parser.parse_args()

    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -- 1. Labels
    labels = load_labels()
    print(f"[{datetime.now():%H:%M:%S}] {len(labels)} labels loaded", flush=True)

    # -- 2. fastText text embeddings
    embeddings, embedder_id, oov_counts = embed_with_fasttext(labels)
    np.save(OUT_DIR / "embeddings_fasttext.npy", embeddings)

    # -- 3. Pairwise cosine distance matrix
    dist = cosine_distance_matrix(embeddings)
    np.save(OUT_DIR / "pair_distances_fasttext.npy", dist)
    print(
        f"[{datetime.now():%H:%M:%S}] fastText dist matrix: "
        f"shape {dist.shape}, mean {dist.mean():.3f}, "
        f"median {np.median(dist):.3f}",
        flush=True,
    )

    # -- 4. UMAP 2D
    umap_coords = umap_2d(embeddings)

    # -- 5. HDBSCAN
    clusters = hdbscan_cluster(umap_coords, args.min_cluster_size)
    n_clusters = int((np.unique(clusters) != -1).sum())
    n_noise = int((clusters == -1).sum())
    print(
        f"[{datetime.now():%H:%M:%S}] HDBSCAN: "
        f"{n_clusters} clusters, {n_noise} noise points",
        flush=True,
    )

    # -- 6. WordNet lookup
    import nltk
    nltk.download("wordnet", quiet=True)
    print(f"[{datetime.now():%H:%M:%S}] WordNet lookup per class...", flush=True)
    synsets = [wordnet_synset_for_label(lbl) for lbl in labels]
    n_matched = sum(1 for s in synsets if s is not None)
    print(
        f"[{datetime.now():%H:%M:%S}] WordNet matched "
        f"{n_matched}/{len(labels)} classes",
        flush=True,
    )

    wn_supers = [wordnet_super_category(s) for s in synsets]
    wn_dist = wordnet_distance_matrix(synsets)
    np.save(OUT_DIR / "pair_distances_wordnet.npy", wn_dist)

    # -- 7. Main taxonomy parquet
    df = pd.DataFrame({
        "class_idx": np.arange(len(labels), dtype=np.int64),
        "class_name": labels,
        "umap_x": umap_coords[:, 0],
        "umap_y": umap_coords[:, 1],
        "vlm_cluster": clusters,
        "wordnet_super": wn_supers,
        "wordnet_synset": [s.name() if s else None for s in synsets],
    })
    df.to_parquet(OUT_DIR / "category_taxonomy.parquet", index=False)
    print(
        f"[{datetime.now():%H:%M:%S}] taxonomy saved to "
        f"{OUT_DIR / 'category_taxonomy.parquet'}",
        flush=True,
    )

    # -- 8. Scatter plot
    render_scatter(
        umap_coords, clusters, wn_supers, labels,
        OUT_DIR / "umap_scatter.png",
    )

    # -- 9. Context / run metadata
    from collections import Counter

    context = {
        "date": datetime.now().isoformat(),
        "embedder": embedder_id,
        "embedder_family": "fasttext",
        "n_classes": len(labels),
        "embedding_dim": int(embeddings.shape[1]),
        "n_oov_tokens_total": int(sum(oov_counts)),
        "n_labels_all_oov": int(sum(1 for v in embeddings if not np.any(v))),
        "umap_params": {
            "n_components": 2, "metric": "cosine",
            "n_neighbors": 15, "min_dist": 0.1, "seed": 0,
        },
        "hdbscan_params": {
            "min_cluster_size": args.min_cluster_size, "min_samples": 3,
            "cluster_selection_method": "eom",
            "n_clusters": n_clusters, "n_noise": n_noise,
        },
        "wordnet_params": {
            "super_categories_tried": WN_SUPER_CATS,
            "n_matched_synsets": n_matched,
            "super_cat_distribution": dict(Counter(wn_supers).most_common()),
        },
        "wall_time_s": round(time.time() - start, 2),
        "artefacts": {
            "taxonomy": "category_taxonomy.parquet",
            "embeddings": "embeddings_fasttext.npy",
            "pair_dist_fasttext": "pair_distances_fasttext.npy",
            "pair_dist_wordnet": "pair_distances_wordnet.npy",
            "scatter_plot": "umap_scatter.png",
        },
    }
    with (OUT_DIR / "context.json").open("w") as f:
        json.dump(context, f, indent=2)
    print(
        f"[{datetime.now():%H:%M:%S}] context saved to "
        f"{OUT_DIR / 'context.json'}",
        flush=True,
    )
    print(
        f"[{datetime.now():%H:%M:%S}] DONE. Total wall time "
        f"{context['wall_time_s']}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
