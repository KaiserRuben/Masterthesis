"""Adapter wrapping :class:`VLMSUT` as a :class:`SupportsLogprobs` source.

The class-target builder talks to a SUT through a narrow ``predict_logprobs``
interface so unit tests can inject a fake. In production the runner
supplies the live :class:`VLMSUT`; this adapter bridges the two surfaces
without dragging SUT-loading concerns into the manipulator package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from PIL import Image

    from src.sut import VLMSUT


class VLMSUTLogprobsAdapter:
    """Bridge :class:`VLMSUT.process_input` → ``predict_logprobs``.

    The pairwise precheck uses the same SUT that will score the optimizer
    run. Wrapping it via an adapter (instead of teaching the class-target
    builder about :class:`VLMSUT` directly) keeps the manipulator package
    decoupled from ``src.sut`` and easy to test in isolation.

    :param sut: Shared :class:`VLMSUT` instance.
    :param prompt: Prompt template used during precheck — must be the
        full prompt (template + answer suffix) that the run will use.
        Otherwise the SUT's classification depends on a context the run
        loop never produces, which would defeat the purpose of the precheck.
    """

    def __init__(self, sut: "VLMSUT", *, prompt: str) -> None:
        self._sut = sut
        self._prompt = prompt

    def predict_logprobs(
        self,
        image: "Image.Image",
        candidates: tuple[str, ...],
    ) -> Mapping[str, float]:
        """Return per-candidate log-probabilities for ``image``.

        Calls :meth:`VLMSUT.process_input` with the configured prompt and
        the provided candidate tuple, then maps each candidate name to
        its log-prob.
        """
        logprobs = self._sut.process_input(
            image, text=self._prompt, categories=candidates,
        )
        arr = logprobs.detach().cpu().tolist()
        return {name: float(arr[i]) for i, name in enumerate(candidates)}


__all__ = ["VLMSUTLogprobsAdapter"]
