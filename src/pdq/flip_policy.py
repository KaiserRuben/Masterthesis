"""Flip-detection policies for PDQ Stage 1 / Stage 2.

A *flip predicate* decides whether a scored candidate counts as a
boundary crossing relative to the anchor.  Two policies exist:

``any_non_anchor``
    Canonical AutoBVA semantics: the full-category argmax label differs
    from the anchor label.  Any crossing out of the anchor's basin
    counts, regardless of which class it lands on.

``pair_target``
    Pair-space semantics for the boundary-pair pipeline: only the raw
    logprobs of the pair's two concrete classes are compared.  The
    anchor's pair side is fixed from the anchor logprobs
    (``class_a`` when ``lp[a] >= lp[b]``, matching the tie rule of
    :func:`src.boundary_pair.runner._pair_softmax_argmax`); a candidate
    flips iff its pair-restricted argmax lands on the *other* side.
    For the common case where the anchor sits on the ``class_a``
    (anchor-class) side this reduces exactly to
    ``lp[target_class] > lp[anchor_class]``.

    The SUT is still scored over the full configured category list —
    only the flip *detection* is pair-restricted.  This avoids the
    generic-attractor degeneracy (e.g. LLaVA-OV-INT8 labelling almost
    every perturbed input "boa constrictor"): a candidate whose
    full-category argmax is an off-pair attractor is a flip iff it is
    on the target side of the pair boundary.
"""

from __future__ import annotations

from typing import Callable, Sequence

# (logprobs, full_category_argmax_label) -> flipped?
FlipPredicate = Callable[[Sequence[float], str], bool]


def make_flip_predicate(
    policy: str,
    *,
    categories: tuple[str, ...],
    anchor_label: str,
    anchor_logprobs: Sequence[float],
    pair_classes: tuple[str, str] | None,
) -> FlipPredicate:
    """Build a flip predicate ``(logprobs, label) -> bool`` for *policy*.

    :param policy: ``"any_non_anchor"`` or ``"pair_target"``.
    :param categories: Full SUT category tuple (same order as logprobs).
    :param anchor_label: Anchor label — full-category argmax for
        canonical PDQ, pair-softmax argmax for the boundary-pair
        pipeline.  Used only by ``any_non_anchor``.
    :param anchor_logprobs: Logprobs of the anchor SUT call.  Used only
        by ``pair_target`` to fix the anchor's pair side.
    :param pair_classes: ``(anchor_class_concrete, target_class_concrete)``
        — the pair's two concrete roster classes.  Required for
        ``pair_target``; both must be in *categories*.
    :returns: Predicate taking ``(logprobs, label)`` where *label* is
        the full-category argmax of *logprobs*.
    :raises ValueError: Unknown policy, missing pair for ``pair_target``,
        or pair class not present in *categories*.
    """
    if policy == "any_non_anchor":
        return lambda logprobs, label: label != anchor_label

    if policy == "pair_target":
        if pair_classes is None:
            raise ValueError(
                "flip_policy='pair_target' requires the seed's concrete "
                "(anchor_class, target_class) pair, but none was supplied. "
                "The boundary-pair pipeline provides it automatically; for "
                "standalone PDQ set flip_policy / flip_preserve_policy to "
                "'any_non_anchor'."
            )
        class_a, class_b = pair_classes
        for cls in (class_a, class_b):
            if cls not in categories:
                raise ValueError(
                    f"flip_policy='pair_target': pair class {cls!r} is not "
                    f"in the configured categories {list(categories)}. The "
                    "pair must use the SUT's concrete roster class names "
                    "(seed.metadata['anchor_class_concrete'] / "
                    "['target_class_concrete'])."
                )
        idx_a = categories.index(class_a)
        idx_b = categories.index(class_b)
        # Anchor pair side, tie -> class_a (same rule as
        # _pair_softmax_argmax, which assigns boundary-pair anchor labels).
        anchor_on_a = anchor_logprobs[idx_a] >= anchor_logprobs[idx_b]

        def _pair_flipped(logprobs: Sequence[float], _label: str) -> bool:
            cand_on_a = logprobs[idx_a] >= logprobs[idx_b]
            return cand_on_a != anchor_on_a

        return _pair_flipped

    raise ValueError(
        f"Unknown flip policy {policy!r}. "
        "Valid: ['any_non_anchor', 'pair_target']"
    )


__all__ = ["FlipPredicate", "make_flip_predicate"]
