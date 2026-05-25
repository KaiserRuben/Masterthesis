"""Taxonomy-abstraction helpers for the roster seed pipeline.

Thin wrappers around :func:`src.data.taxonomy.cluster_of` that enforce
the Exp-100 invariant: every class participating in a roster run must
expose a complete L0/L1/L2 path. Classes with shorter (left-anchored)
paths are rejected during config validation, so downstream code can
treat ``resolve_label`` as total over ``levels in (0, 1, 2)``.

Lives under ``src.common`` because both the seed pool and the
combinatorial pair generator consume it.
"""

from __future__ import annotations

from typing import Iterable

from src.data.taxonomy import N_LEVELS, cluster_of, path_of


def validate_class_list(class_list: Iterable[str]) -> None:
    """Verify every class has a complete L0/L1/L2 path.

    :param class_list: Concrete (L0) ImageNet class names.
    :raises ValueError: If any class has a path shorter than ``N_LEVELS``,
        or is unknown to the taxonomy. Aggregates all failures into a
        single error so the user can fix the config in one pass.
    """
    bad_unknown: list[str] = []
    bad_short: list[tuple[str, tuple[str, ...]]] = []
    for name in class_list:
        try:
            p = path_of(name)
        except KeyError:
            bad_unknown.append(name)
            continue
        if len(p) < N_LEVELS:
            bad_short.append((name, p))
    if not bad_unknown and not bad_short:
        return
    parts: list[str] = []
    if bad_unknown:
        parts.append(
            f"unknown to taxonomy: {bad_unknown!r}"
        )
    if bad_short:
        msgs = [f"{n!r} has only path {list(p)} (need {N_LEVELS} levels)"
                for n, p in bad_short]
        parts.append("incomplete taxonomy paths: " + "; ".join(msgs))
    raise ValueError(
        "Roster class_list validation failed — "
        + " | ".join(parts)
    )


def resolve_label(class_concrete: str, level: int) -> str:
    """Return the cluster label of *class_concrete* at *level*.

    :param class_concrete: An ImageNet class name (lookup key into the
        taxonomy). Must have been validated via :func:`validate_class_list`
        (or otherwise be known to have a full path); we still defensively
        check here and raise on any None return from the underlying API.
    :param level: ``-1`` (concrete = the ImageNet class name itself,
        bypassing the taxonomy), ``0`` (fine), ``1`` (mid) or ``2`` (super).
        ``-1`` is the right choice when you want the prompt to use the
        raw ImageNet label (e.g. "hammerhead shark") rather than its
        coarser L0 cluster ("shark").
    :returns: The label string for use in the prompt.
    :raises ValueError: If the class has no label at the requested level.
    :raises KeyError: If the class is unknown to the taxonomy.
    """
    if level == -1:
        return class_concrete
    label = cluster_of(class_concrete, level=level)
    if label is None:
        raise ValueError(
            f"class {class_concrete!r} has no label at level {level}; "
            "validate_class_list should have caught this earlier."
        )
    return label


__all__ = ["resolve_label", "validate_class_list"]
