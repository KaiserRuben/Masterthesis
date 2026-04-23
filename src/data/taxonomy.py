"""Query API over the curated ImageNet-1k class taxonomy.

Wraps ``src.data.imagenet_class_mapping.imagenet_clusters`` with convenience
lookups

The raw mapping is a dict ``{class_name: [level_0, level_1, level_2]}`` where
each level is a string cluster label, most specific first. Some classes have
only 1 or 2 levels (the list is left-truncated in those cases).

Conventions used in this module:

* Level indexing is left-anchored (level 0 = finest cluster, level 2 = most
  abstract) to match the raw mapping.
* ``common_ancestor_level(a, b)`` returns the *smallest* level at which two
  classes share a cluster label; ``None`` if they never agree.
* All queries return lists/tuples of class names verbatim as they appear in
  the canonical 1000-class ImageNet list (human-readable, with spaces).

This module has no dependency on pandas or numpy — the goal is a cheap,
zero-overhead lookup layer. Higher-level code (pair samplers, analysis
scripts) can join with the ``category_taxonomy.parquet`` artefact as needed.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
from typing import Iterator

from .imagenet_class_mapping import imagenet_clusters

N_LEVELS = 3

# Reverse index: (level, cluster_label) -> list of class names.
# Built lazily on first query.
_level_index: dict[int, dict[str, list[str]]] | None = None


def _build_level_index() -> dict[int, dict[str, list[str]]]:
    """Invert the mapping into level → cluster → members."""
    index: dict[int, dict[str, list[str]]] = {
        lvl: defaultdict(list) for lvl in range(N_LEVELS)
    }
    for cls_name, path in imagenet_clusters.items():
        # Align left-truncated paths to the right: a 2-level path like
        # ["frog", "amphibian"] means level 0 = "frog", level 1 = "amphibian".
        # Classes without a level-2 super-cat simply don't appear at level 2.
        for lvl, label in enumerate(path):
            index[lvl][label].append(cls_name)
    return {lvl: dict(d) for lvl, d in index.items()}


def _ensure_index() -> dict[int, dict[str, list[str]]]:
    global _level_index
    if _level_index is None:
        _level_index = _build_level_index()
    return _level_index


# ---------------------------------------------------------------------------
# Single-class queries
# ---------------------------------------------------------------------------


def cluster_of(class_name: str, level: int = 0) -> str | None:
    """Cluster label of ``class_name`` at ``level`` (0 = finest, 2 = super-cat).

    Returns ``None`` when the class's path is shorter than ``level + 1``
    (e.g. a class with only 2 levels has no level-2 super-cat).
    """
    if level < 0 or level >= N_LEVELS:
        raise ValueError(f"level must be in [0, {N_LEVELS}); got {level}")
    path = imagenet_clusters.get(class_name)
    if path is None:
        raise KeyError(f"unknown class {class_name!r}")
    return path[level] if level < len(path) else None


def path_of(class_name: str) -> tuple[str, ...]:
    """Full cluster path for a class, tuple of length 1-3."""
    path = imagenet_clusters.get(class_name)
    if path is None:
        raise KeyError(f"unknown class {class_name!r}")
    return tuple(path)


def siblings(class_name: str, level: int = 0) -> list[str]:
    """Classes sharing the same cluster as ``class_name`` at ``level``.

    Excludes ``class_name`` itself. Returns ``[]`` if the class has no
    cluster at that level.
    """
    label = cluster_of(class_name, level=level)
    if label is None:
        return []
    members = _ensure_index()[level].get(label, [])
    return [m for m in members if m != class_name]


# ---------------------------------------------------------------------------
# Pairwise queries
# ---------------------------------------------------------------------------


def common_ancestor_level(a: str, b: str) -> int | None:
    """Smallest level at which ``a`` and ``b`` share a cluster label.

    Returns 0 if they share a fine-grained cluster; 1 if only at mid-level;
    2 if only at super-cat; ``None`` if they are in different super-cats
    (or either class lacks a path reaching the common level).
    """
    path_a = imagenet_clusters.get(a)
    path_b = imagenet_clusters.get(b)
    if path_a is None:
        raise KeyError(f"unknown class {a!r}")
    if path_b is None:
        raise KeyError(f"unknown class {b!r}")

    # Both paths are left-anchored. Compare at each level, stop at first match.
    for lvl in range(min(len(path_a), len(path_b))):
        if path_a[lvl] == path_b[lvl]:
            return lvl
    return None


def pair_bucket(a: str, b: str) -> str:
    """Human-readable bucket label for a pair.

    - ``"same_L0"``  fine-grained sibling (e.g. two specific shark species)
    - ``"same_L1"``  mid-level sibling, different L0
    - ``"same_L2"``  same super-cat, different L1
    - ``"cross"``    different super-cats (distant semantic distance)
    """
    lvl = common_ancestor_level(a, b)
    if lvl is None:
        return "cross"
    if lvl == 0:
        return "same_L0"
    if lvl == 1:
        return "same_L1"
    return "same_L2"


# ---------------------------------------------------------------------------
# Cluster-level queries
# ---------------------------------------------------------------------------


def members(cluster_label: str, level: int) -> list[str]:
    """All classes in ``cluster_label`` at ``level``."""
    if level < 0 or level >= N_LEVELS:
        raise ValueError(f"level must be in [0, {N_LEVELS}); got {level}")
    return list(_ensure_index()[level].get(cluster_label, []))


def cluster_labels(level: int) -> list[str]:
    """All cluster labels present at ``level``."""
    if level < 0 or level >= N_LEVELS:
        raise ValueError(f"level must be in [0, {N_LEVELS}); got {level}")
    return sorted(_ensure_index()[level].keys())


def cluster_sizes(level: int) -> dict[str, int]:
    """Cluster-size histogram at ``level`` (label → member count)."""
    if level < 0 or level >= N_LEVELS:
        raise ValueError(f"level must be in [0, {N_LEVELS}); got {level}")
    return {k: len(v) for k, v in _ensure_index()[level].items()}


def pairs_within(cluster_label: str, level: int) -> Iterator[tuple[str, str]]:
    """All unordered pairs of class names within a cluster.

    Iterates C(n, 2) pairs lazily; wrap in ``list(...)`` when materialising.
    """
    ms = members(cluster_label, level)
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            yield ms[i], ms[j]


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def coverage_report() -> dict[str, object]:
    """Quick summary of the taxonomy for sanity checks."""
    idx = _ensure_index()
    depths = Counter(len(p) for p in imagenet_clusters.values())
    top_lvl2 = Counter()
    for p in imagenet_clusters.values():
        if len(p) >= 3:
            top_lvl2[p[2]] += 1
    return {
        "n_classes": len(imagenet_clusters),
        "depth_distribution": dict(depths),
        "n_labels_L0": len(idx[0]),
        "n_labels_L1": len(idx[1]),
        "n_labels_L2": len(idx[2]),
        "L2_histogram": dict(top_lvl2.most_common()),
    }


__all__ = [
    "N_LEVELS",
    "cluster_of",
    "cluster_sizes",
    "cluster_labels",
    "common_ancestor_level",
    "coverage_report",
    "members",
    "pair_bucket",
    "pairs_within",
    "path_of",
    "siblings",
]
