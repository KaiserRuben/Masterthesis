"""SMOO-compatible VLM system-under-test.

Wraps a :class:`~src.sut.scorer.VLMScorer` into SMOO's
:class:`SUT` interface so the boundary tester can call
``sut.process_input(image, text)`` and get back a log-prob tensor.
"""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image
from smoo.sut import SUT
from torch import Tensor

from .config import VLMSUTConfig
from .scorer import create_scorer


class VLMSUT(SUT):
    """VLM system-under-test with teacher-forced scoring.

    Wraps a VLM scorer into SMOO's SUT interface.  For each input
    (image + optional text), runs teacher-forced decoding to produce a
    log-prob vector over the category set.

    :param config: Configuration object.  Uses defaults when ``None``.
    """

    def __init__(self, config: VLMSUTConfig | None = None) -> None:
        self._config = config or VLMSUTConfig()
        self._device = torch.device(self._config.device)
        self._scorer = create_scorer(
            model_id=self._config.model_id,
            device=self._config.device,
            enable_thinking=self._config.enable_thinking,
            max_thinking_tokens=self._config.max_thinking_tokens,
            max_pixels=self._config.max_pixels,
        )
        self._prompt = self._config.prompt_template.format(
            categories=", ".join(self._config.categories)
        )

    # ------------------------------------------------------------------
    # SUT interface
    # ------------------------------------------------------------------

    def process_input(
        self,
        image: Image.Image,
        text: str | None = None,
        categories: tuple[str, ...] | None = None,
    ) -> Tensor:
        """Score an image against categories.

        :param image: PIL image to classify.
        :param text: Override prompt text.  If ``None``, uses the prompt
            built from ``config.prompt_template`` and ``config.categories``.
            If provided, should be the complete prompt (template already
            filled).
        :param categories: Override categories for this call.  If ``None``,
            uses ``config.categories``.  This changes which labels are
            force-decoded.
        :returns: Tensor of shape ``(n_categories,)`` with *log_prob_norm*
            values.  ``tensor[i]`` corresponds to ``categories[i]``.
        """
        cats = categories if categories is not None else self._config.categories
        prompt = text if text is not None else self._prompt
        return self._scorer.score_categories_tensor(image, prompt, cats)

    def input_valid(self, inpt: Any, cond: Any) -> tuple[bool, Any]:
        """Validate that the VLM's top prediction matches the condition.

        :param inpt: Either a PIL image, or a tuple ``(image, text)``
            where *text* is the prompt override (may be ``None``).
        :param cond: Expected top-1 category (``str``).
        :returns: ``(is_valid, logprobs_tensor)`` where *is_valid* is
            ``True`` when the highest log-prob category equals *cond*.
        """
        if isinstance(inpt, tuple):
            image, text = inpt
        else:
            image, text = inpt, None

        logprobs = self.process_input(image, text=text)
        top_idx = int(logprobs.argmax().item())
        top_label = self._config.categories[top_idx]
        return top_label == cond, logprobs
