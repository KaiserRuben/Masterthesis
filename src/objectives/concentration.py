"""Concentration criterion for VLM decision boundaries.

Measures how much probability mass leaks to classes outside the target
pair: ``sum P(c) for c not in {A, B}``.  Returns 0.0 when all mass is
concentrated on the two target classes and approaches 1.0 when
probability is spread across non-target classes.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from smoo.objectives import Criterion
from torch import Tensor


class Concentration(Criterion):
    """Measures probability mass outside the target pair.

    Computes ``sum P(c) for c not in {A, B}`` after applying softmax to the
    raw logits.  Returns 0.0 when all mass is on the target pair,
    approaches 1.0 when probability leaks to other classes.

    :param inverse: Whether the criterion should be inverted.
    """

    _name: str = "Conc"

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
        """Compute sum of P(c) for all c not in the target pair.

        :param logits: Logit tensor of shape ``(pop_size, n_classes)`` when
            ``batch_dim=0``, or ``(n_classes,)`` when ``batch_dim is None``.
        :param target_classes: Indices ``(A, B)`` of the two target classes.
        :param batch_dim: ``0`` for batched input, ``None`` for a single
            logit vector.
        :param _: Unused kwargs.
        :returns: List of concentration values in [0, 1].
        """
        if batch_dim is None:
            logits = logits.unsqueeze(0)

        probs = F.softmax(logits, dim=-1)
        a, b = target_classes
        mask = torch.ones(probs.shape[-1], dtype=torch.bool, device=probs.device)
        mask[a] = False
        mask[b] = False
        outside = probs[:, mask].sum(dim=-1)
        return outside.tolist()
