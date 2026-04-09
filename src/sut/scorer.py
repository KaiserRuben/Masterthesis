"""VLM scorers: teacher-forced log-prob scoring over a closed label set.

Extracted from ``experiments/imagenet_vlm_probing/run.py`` and cleaned up
for reuse in the boundary-testing pipeline.

Provides:

* :class:`VLMScorer` -- abstract base with model-agnostic KV-cache scoring.
* :class:`Qwen3VLScorer` -- concrete scorer for Qwen3-VL models.
* :class:`Qwen35Scorer` -- concrete scorer for Qwen3.5 models (DeltaNet).
* :func:`create_scorer` -- factory that picks the right scorer from a
  model id string.
"""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

THINK_END_TOKEN = "</think>"


class VLMScorer(ABC):
    """Abstract VLM scorer using teacher-forced KV-cache decoding.

    Lifecycle:

    1.  ``__init__`` loads the model and processor.
    2.  ``score_categories`` runs the prompt+image prefix once, then
        force-decodes each candidate label and returns per-label log-probs.
    3.  ``score_categories_tensor`` wraps ``score_categories`` to return a
        :class:`torch.Tensor` ordered by the input categories list.

    :param model_id: HuggingFace model identifier.
    :param device: Torch device string.
    :param enable_thinking: Allow ``<think>`` trace before scoring.
    :param max_thinking_tokens: Token budget for the thinking trace.
    :param dtype: Model dtype (defaults to ``float16``).
    :param max_pixels: Pixel cap forwarded to the processor.
    """

    _model: AutoModelForImageTextToText
    _processor: AutoProcessor
    _device: torch.device
    _enable_thinking: bool
    _max_thinking_tokens: int

    def __init__(
        self,
        model_id: str,
        device: str,
        enable_thinking: bool = True,
        max_thinking_tokens: int = 2000,
        dtype: torch.dtype | None = None,
        max_pixels: int | None = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ) -> None:
        self._device = torch.device(device)
        self._enable_thinking = enable_thinking
        self._max_thinking_tokens = max_thinking_tokens

        proc_kwargs: dict = {}
        if max_pixels is not None:
            proc_kwargs["max_pixels"] = max_pixels
        self._processor = AutoProcessor.from_pretrained(model_id, **proc_kwargs)

        model_kwargs: dict = {}
        if load_in_8bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["device_map"] = device
        elif load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            model_kwargs["device_map"] = device
        else:
            model_kwargs["torch_dtype"] = dtype or torch.float16
            model_kwargs["device_map"] = device

        self._model = AutoModelForImageTextToText.from_pretrained(
            model_id, **model_kwargs,
        )
        self._model.eval()

    # ------------------------------------------------------------------
    # Abstract
    # ------------------------------------------------------------------

    @abstractmethod
    def _prepare_inputs(
        self,
        image: Image.Image,
        prompt: str,
        enable_thinking: bool,
    ) -> dict:
        """Tokenize messages into model inputs on device.

        :param image: PIL image.
        :param prompt: Text prompt.
        :param enable_thinking: Whether to enable thinking for this call.
        :returns: Dict suitable for ``model(**inputs)``.
        """
        ...

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def tokenizer(self):
        """Shortcut to the underlying tokenizer."""
        return self._processor.tokenizer

    # ------------------------------------------------------------------
    # Generation (optional -- kept for completeness)
    # ------------------------------------------------------------------

    def _find_think_end(self, token_ids: torch.Tensor) -> int | None:
        """Return position of the ``</think>`` token, or ``None``."""
        think_end_id = self.tokenizer.convert_tokens_to_ids(THINK_END_TOKEN)
        for i, tid in enumerate(token_ids):
            if tid.item() == think_end_id:
                return i
        return None

    def generate(
        self,
        image: Image.Image,
        prompt: str,
    ) -> tuple[str, str | None, torch.Tensor | None]:
        """Free-generate a response.

        :param image: PIL image.
        :param prompt: Text prompt.
        :returns: ``(answer_text, thinking_text, thinking_ids)``
            where *thinking_text* and *thinking_ids* are ``None`` when
            thinking is disabled or the model didn't produce a thinking
            trace.
        """
        inputs = self._prepare_inputs(image, prompt, self._enable_thinking)
        prefix_len = inputs["input_ids"].shape[1]

        max_gen = (
            self._max_thinking_tokens + 50 if self._enable_thinking else 50
        )
        with torch.no_grad():
            gen_ids = self._model.generate(**inputs, max_new_tokens=max_gen)

        generated_tokens = gen_ids[0][prefix_len:]
        answer_tokens = generated_tokens
        thinking_ids: torch.Tensor | None = None
        thinking_text: str | None = None

        if self._enable_thinking:
            think_end_pos = self._find_think_end(generated_tokens)
            if think_end_pos is not None:
                thinking_ids = generated_tokens[: think_end_pos + 1]
                answer_tokens = generated_tokens[think_end_pos + 1 :]
                thinking_text = self.tokenizer.decode(
                    thinking_ids, skip_special_tokens=True
                ).strip()

        answer_text: str = self.tokenizer.decode(
            answer_tokens, skip_special_tokens=True
        ).strip()
        return answer_text, thinking_text, thinking_ids

    # ------------------------------------------------------------------
    # Teacher-forced scoring
    # ------------------------------------------------------------------

    def score_categories(
        self,
        image: Image.Image,
        prompt: str,
        categories: list[str] | tuple[str, ...],
        thinking_ids: torch.Tensor | None = None,
    ) -> list[tuple[str, float, float, int]]:
        """Force-score each category via KV-cache continuation.

        If *thinking_ids* is provided (from :meth:`generate`), the
        thinking trace is appended to the prompt so scoring is
        conditioned on the same context as free generation.

        :param image: PIL image.
        :param prompt: Text prompt.
        :param categories: Labels to score.
        :param thinking_ids: Optional thinking token ids from
            :meth:`generate`.
        :returns: List of ``(label, log_prob, log_prob_norm, n_tokens)``
            tuples sorted descending by *log_prob_norm*.
        """
        if thinking_ids is not None and self._enable_thinking:
            inputs = self._prepare_inputs(image, prompt, enable_thinking=True)
            n_think = len(thinking_ids)
            inputs["input_ids"] = torch.cat(
                [inputs["input_ids"], thinking_ids.unsqueeze(0)], dim=1
            )
            for key in ("attention_mask", "mm_token_type_ids"):
                if key in inputs:
                    pad_val = 1 if key == "attention_mask" else 0
                    extra = torch.full(
                        (1, n_think),
                        pad_val,
                        device=self._device,
                        dtype=inputs[key].dtype,
                    )
                    inputs[key] = torch.cat([inputs[key], extra], dim=1)
            with torch.no_grad():
                prefix_out = self._model(**inputs, use_cache=True)
        else:
            inputs = self._prepare_inputs(image, prompt, enable_thinking=False)
            with torch.no_grad():
                prefix_out = self._model(**inputs, use_cache=True)

        prefix_kvs = prefix_out.past_key_values
        last_logits = prefix_out.logits[0, -1, :]

        scored: list[tuple[str, float, float, int]] = []
        for lbl in categories:
            label_tok_ids = self.tokenizer.encode(lbl, add_special_tokens=False)
            label_ids = torch.tensor(label_tok_ids, device=self._device)
            n_tokens = len(label_tok_ids)

            if n_tokens == 0:
                scored.append((lbl, float("-inf"), float("-inf"), 0))
                continue

            total_lp = F.log_softmax(last_logits, dim=-1)[label_ids[0]].item()

            if n_tokens > 1:
                with torch.no_grad():
                    cont_out = self._model(
                        input_ids=label_ids[:-1].unsqueeze(0),
                        past_key_values=prefix_kvs,
                    )
                for i in range(n_tokens - 1):
                    lp = F.log_softmax(cont_out.logits[0, i, :], dim=-1)
                    total_lp += lp[label_ids[i + 1]].item()

            norm_lp = total_lp / n_tokens
            scored.append((lbl, total_lp, norm_lp, n_tokens))

        return sorted(scored, key=lambda x: x[2], reverse=True)

    def score_categories_tensor(
        self,
        image: Image.Image,
        prompt: str,
        categories: list[str] | tuple[str, ...],
        thinking_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score categories and return a tensor preserving input order.

        Unlike :meth:`score_categories` (which sorts by score),
        this returns a tensor where ``tensor[i]`` is the
        *log_prob_norm* for ``categories[i]``.

        :param image: PIL image.
        :param prompt: Text prompt.
        :param categories: Labels to score.
        :param thinking_ids: Optional thinking token ids.
        :returns: Tensor of shape ``(len(categories),)`` with
            log_prob_norm values in input order.
        """
        scored = self.score_categories(
            image, prompt, categories, thinking_ids=thinking_ids
        )
        # Build a lookup from the sorted result list.
        lp_by_label = {label: norm_lp for label, _, norm_lp, _ in scored}
        return torch.tensor(
            [lp_by_label[cat] for cat in categories], dtype=torch.float32
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def score_image(
        self,
        image: Image.Image,
        prompt: str,
        categories: list[str] | tuple[str, ...],
    ) -> tuple[str, str | None, list[tuple[str, float, float, int]]]:
        """Generate + score in one call.

        :param image: PIL image.
        :param prompt: Text prompt.
        :param categories: Labels to score.
        :returns: ``(generated_text, thinking_text, scored_list)``
        """
        generated_text, thinking_text, thinking_ids = self.generate(
            image, prompt
        )
        scored = self.score_categories(
            image, prompt, categories, thinking_ids=thinking_ids
        )
        return generated_text, thinking_text, scored

    def cleanup(self) -> None:
        """Release GPU/MPS memory."""
        gc.collect()
        if self._device.type == "mps":
            torch.mps.empty_cache()
        elif self._device.type == "cuda":
            torch.cuda.empty_cache()


# -----------------------------------------------------------------------
# Concrete scorers
# -----------------------------------------------------------------------


class QwenVLScorer(VLMScorer):
    """Scorer for Qwen VL models (Qwen3-VL and Qwen3.5).

    Defaults *max_pixels* to ``512 * 28 * 28`` (~400 image tokens)
    when not explicitly set.
    """

    def __init__(
        self,
        model_id: str,
        device: str,
        enable_thinking: bool = True,
        max_thinking_tokens: int = 2000,
        dtype: torch.dtype | None = None,
        max_pixels: int | None = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ) -> None:
        super().__init__(
            model_id,
            device,
            enable_thinking,
            max_thinking_tokens,
            dtype,
            max_pixels=max_pixels if max_pixels is not None else 512 * 28 * 28,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
        )

    def _prepare_inputs(
        self,
        image: Image.Image,
        prompt: str,
        enable_thinking: bool,
    ) -> dict:
        """Build Qwen VL chat-template inputs.

        :param image: PIL image.
        :param prompt: Text prompt.
        :param enable_thinking: Whether to enable thinking.
        :returns: Tokenized inputs dict on device.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=enable_thinking,
        )
        return self._processor(
            text=[text], images=[image], return_tensors="pt"
        ).to(self._device)


# -----------------------------------------------------------------------
# Registry / factory
# -----------------------------------------------------------------------

SCORER_REGISTRY: dict[str, type[VLMScorer]] = {
    "qwen3-vl": QwenVLScorer,
    "qwen3.5": QwenVLScorer,
}


def create_scorer(
    model_id: str,
    device: str,
    enable_thinking: bool = True,
    max_thinking_tokens: int = 2000,
    max_pixels: int | None = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
) -> VLMScorer:
    """Instantiate the right scorer for *model_id*.

    Looks up *model_id* (case-insensitive) against
    :data:`SCORER_REGISTRY` keys.

    :param model_id: HuggingFace model identifier.
    :param device: Torch device string.
    :param enable_thinking: Allow thinking traces.
    :param max_thinking_tokens: Token budget for thinking.
    :param max_pixels: Pixel cap forwarded to the scorer.
    :param load_in_8bit: Load model quantized to 8-bit (bitsandbytes).
    :param load_in_4bit: Load model quantized to 4-bit (bitsandbytes).
    :returns: A concrete :class:`VLMScorer` instance.
    :raises ValueError: If no registry key matches *model_id*.
    """
    model_lower = model_id.lower()
    for key, cls in SCORER_REGISTRY.items():
        if key in model_lower:
            return cls(
                model_id,
                device,
                enable_thinking,
                max_thinking_tokens,
                max_pixels=max_pixels,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
            )
    raise ValueError(
        f"No scorer for model '{model_id}'. "
        f"Known prefixes: {list(SCORER_REGISTRY.keys())}"
    )
