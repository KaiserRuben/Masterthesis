"""Targeted balance criterion for VLM decision boundaries.

Measures ``|P(A) - P(B)|`` where P is the softmax distribution over
log-prob logits.  Returns 0.0 at a perfect decision boundary and
approaches 1.0 when one target class dominates the other.
"""

from __future__ import annotations

from typing import Any

import torch.nn.functional as F
from smoo.objectives import Criterion
from torch import Tensor


class TargetedBalance(Criterion):
    """Measures imbalance between two target classes: |P(A) - P(B)|.

    Returns 0.0 when perfectly balanced, approaches 1.0 when one dominates.
    Inputs are raw logits (log-probs); softmax is applied internally.

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
        """Compute |P(A) - P(B)| per sample.

        :param logits: Logit tensor of shape ``(pop_size, n_classes)`` when
            ``batch_dim=0``, or ``(n_classes,)`` when ``batch_dim is None``.
        :param target_classes: Indices ``(A, B)`` of the two target classes.
        :param batch_dim: ``0`` for batched input, ``None`` for a single
            logit vector.
        :param _: Unused kwargs.
        :returns: List of balance values in [0, 1].
        """
        if batch_dim is None:
            logits = logits.unsqueeze(0)

        probs = F.softmax(logits, dim=-1)
        a, b = target_classes
        balance = (probs[:, a] - probs[:, b]).abs()
        return balance.tolist()
