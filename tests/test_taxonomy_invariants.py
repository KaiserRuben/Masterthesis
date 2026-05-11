"""Hard invariants for the ImageNet-1k taxonomy L2 layer.

These tests guard the contract expected by Exp-100 (class-geometry experiment)
and the ``src.data.taxonomy`` query API:

- Every class has exactly 3 levels.
- ``path[2]`` is drawn from the closed ``CANONICAL_L2`` set.
- L2 labels never appear at L0 or L1 (no leakage across levels).
- No path has repeated labels.
"""

from __future__ import annotations

from src.data.imagenet_class_mapping import CANONICAL_L2, imagenet_clusters


def test_canonical_l2_is_tuple_of_strings():
    assert isinstance(CANONICAL_L2, tuple)
    assert all(isinstance(x, str) for x in CANONICAL_L2)
    assert len(CANONICAL_L2) == len(set(CANONICAL_L2))


def test_all_paths_have_length_three():
    bad = {k: v for k, v in imagenet_clusters.items() if len(v) != 3}
    assert not bad, f"{len(bad)} classes have non-3 path length; e.g. {list(bad.items())[:3]}"


def test_l2_is_in_canonical_set():
    canonical = set(CANONICAL_L2)
    bad = {
        k: v for k, v in imagenet_clusters.items() if v[2] not in canonical
    }
    assert not bad, f"non-canonical L2 labels: {list(bad.items())[:5]}"


def test_l2_does_not_leak_into_l1():
    l1s = {v[1] for v in imagenet_clusters.values()}
    l2s = {v[2] for v in imagenet_clusters.values()}
    leak = l2s & l1s
    assert not leak, f"L2 labels leaked into L1: {sorted(leak)}"


def test_l2_does_not_leak_into_l0():
    l0s = {v[0] for v in imagenet_clusters.values()}
    l2s = {v[2] for v in imagenet_clusters.values()}
    leak = l2s & l0s
    assert not leak, f"L2 labels leaked into L0: {sorted(leak)}"


def test_no_repeated_labels_within_path():
    bad = {k: v for k, v in imagenet_clusters.items() if len(set(v)) != len(v)}
    assert not bad, f"paths with repeated labels: {list(bad.items())[:5]}"


def test_all_canonical_l2_buckets_are_populated():
    l2s = {v[2] for v in imagenet_clusters.values()}
    missing = set(CANONICAL_L2) - l2s
    assert not missing, f"canonical L2 buckets with zero members: {sorted(missing)}"


def test_class_count_unchanged():
    assert len(imagenet_clusters) == 1000


def test_canonical_l2_has_nineteen_buckets():
    """After Iteration 2 (object split, arthropod merge), the canonical L2 set
    has exactly 19 buckets — 9 biological/natural + 10 artifact/human-made."""
    assert len(CANONICAL_L2) == 19


def test_every_l2_bucket_has_at_least_one_member():
    from collections import Counter

    sizes = Counter(v[2] for v in imagenet_clusters.values())
    for label in CANONICAL_L2:
        assert sizes.get(label, 0) >= 1, f"L2 bucket {label!r} is empty"
