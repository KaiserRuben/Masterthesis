"""Targeted balance criterion for VLM decision boundaries.

Returns ``|lp_A - lp_B|`` — the absolute difference of the two target
classes' length-normalized log-probabilities. This is a monotone
stand-in for "distance to the A-vs-B decision boundary" that is
(a) linear at the boundary (no tanh compression),
(b) retains the full FP32 dynamic range of the underlying scorer
    (resolution ~1e-7 vs. the ~1e-4 of softmax-then-subtract),
(c) free of the N-class-softmax failure mode where
    ``|P(A) - P(B)| → 0`` when the model is confident about a *third*
    class C and both p(A) and p(B) are tiny.

Why no softmax: schema v2 stores the full N-dim log-prob vector per
individual in the trace, so any probability-scaled or
distribution-normalised form is trivially recoverable post-hoc. The
optimizer only needs a monotone scalar per individual for selection,
and the raw log-prob gap is the cheapest, most numerically well-behaved
monotone function available. See Diary
2026-04-15-Exp05-Numerical-Floor-Discovery for context.
"""

from __future__ import annotations

from typing import Any

from smoo.objectives import Criterion
from torch import Tensor


class TargetedBalance(Criterion):
    """Measures imbalance between two target classes as ``|lp_A - lp_B|``.

    Returns 0.0 when the two log-probs are equal (the A-vs-B decision
    boundary), and grows unboundedly as one class' log-prob pulls away
    from the other. Unlike a softmax-based variant, this does not
    saturate near the boundary and does not interact with the
    probability mass of any third class — it reads only the two columns
    indicated by *target_classes*.

    :param inverse: Whether the criterion should be inverted.
    """

    _name: str = "TgtBal"

    def __init__(self, inverse: bool = False) -> None:
        super().__init__(inverse=inverse, allow_batched=True)

    def evaluate(
        self,
        *,
        logits: Tensor,
        target_classes: tuple[int, int],
        batch_dim: int | None = None,
        **_: Any,
    ) -> list[float]:
        """Compute ``|lp_A - lp_B|`` per sample.

        :param logits: Log-prob tensor of shape ``(pop_size, n_classes)``
            when ``batch_dim=0``, or ``(n_classes,)`` when
            ``batch_dim is None``. ``n_classes`` may be 2 (legacy runs
            on schema v1) or N (schema v2) — only the two columns
            indicated by *target_classes* are read.
        :param target_classes: Indices ``(A, B)`` of the two target
            classes into the last dimension of *logits*.
        :param batch_dim: ``0`` for batched input, ``None`` for a single
            logit vector.
        :param _: Unused kwargs.
        :returns: List of non-negative log-prob gap values. 0.0 exactly
            at the boundary; grows linearly with ``|lp_A - lp_B|``.
        """
        if batch_dim is None:
            logits = logits.unsqueeze(0)

        a, b = target_classes
        gap = (logits[:, a] - logits[:, b]).abs()
        return gap.tolist()
