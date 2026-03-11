"""Boundary-testing criteria for VLM decision boundaries.

Two criteria measure how close a VLM's prediction distribution is to the
decision boundary between two specific target classes (A, B):

- **TargetedBalance**: |P(A) - P(B)|  -- 0 at boundary, ~1 when one dominates
- **Concentration**: sum of P(c) for c not in {A, B}  -- 0 when all mass on
  the target pair, approaches 1 when probability leaks to other classes

Both accept raw log-prob logits (not pre-softmaxed) and softmax internally.
"""

from typing import Any

import torch
import torch.nn.functional as F
from smoo.objectives.classifier_criteria._classifier_criterion import ClassifierCriterion
from torch import Tensor


class TargetedBalance(ClassifierCriterion):
    """Measures imbalance between two target classes: |P(A) - P(B)|.

    Returns 0.0 when perfectly balanced, approaches 1.0 when one dominates.
    Inputs are raw logits (log-probs); softmax is applied internally.

    :param target_pair: Indices (A, B) of the two target classes.
    :param inverse: Whether the criterion should be inverted.
    """

    _name: str = "TargetedBalance"

    def __init__(
        self, target_pair: tuple[int, int], inverse: bool = False,
    ) -> None:
        super().__init__(inverse=inverse, allow_batched=True)
        self._target_pair = target_pair

    def evaluate(self, *, logits: Tensor, **_: Any) -> list[float]:
        """Compute |P(A) - P(B)| per sample.

        :param logits: Logit tensor of shape (batch, n_classes).
        :param _: Unused kwargs.
        :returns: List of balance values in [0, 1].
        """
        probs = F.softmax(logits, dim=-1)
        a, b = self._target_pair
        balance = (probs[:, a] - probs[:, b]).abs()
        return balance.tolist()


class Concentration(ClassifierCriterion):
    """Measures probability mass outside the target pair: sum P(c) for c not in {A, B}.

    Returns 0.0 when all mass is on the target pair, approaches 1.0 when
    probability is spread across other classes.

    :param target_pair: Indices (A, B) of the two target classes.
    :param inverse: Whether the criterion should be inverted.
    """

    _name: str = "Concentration"

    def __init__(
        self, target_pair: tuple[int, int], inverse: bool = False,
    ) -> None:
        super().__init__(inverse=inverse, allow_batched=True)
        self._target_pair = target_pair

    def evaluate(self, *, logits: Tensor, **_: Any) -> list[float]:
        """Compute sum of P(c) for all c not in the target pair.

        :param logits: Logit tensor of shape (batch, n_classes).
        :param _: Unused kwargs.
        :returns: List of concentration values in [0, 1].
        """
        probs = F.softmax(logits, dim=-1)
        a, b = self._target_pair
        mask = torch.ones(probs.shape[-1], dtype=torch.bool, device=probs.device)
        mask[a] = False
        mask[b] = False
        outside = probs[:, mask].sum(dim=-1)
        return outside.tolist()
