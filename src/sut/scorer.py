"""VLM scorers: teacher-forced log-prob scoring over a closed label set.

Two execution backends share the same scoring code path:

* ``torch`` (default) -- HuggingFace transformers on CUDA/MPS/CPU. Used for
  Mac development (MPS) and CUDA workstations.
* ``openvino`` -- OpenVINO IR on Intel Arc/Xe via :mod:`optimum.intel`.
  ``OVModelForVisualCausalLM`` exposes the same ``(input_ids, past_key_values,
  use_cache)`` forward contract, so :meth:`VLMScorer.score_categories` is
  unchanged across backends.

Public surface:

* :class:`VLMScorer` -- abstract base with backend-agnostic KV-cache scoring.
  Backends plug in via the :meth:`_create_model` / :meth:`_create_processor`
  hooks.
* :class:`QwenVLScorer`, :class:`LlavaScorer` -- per-family torch scorers
  (chat-template + ``encode_text`` differ; loading is shared with the base).
* :class:`OVQwenVLScorer`, :class:`OVLlavaScorer` -- the OV variants. They
  inherit family-specific input prep and override only the model loader.
* :func:`create_scorer` -- factory keyed by ``(family, backend)``.
"""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod

import numpy as np
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
        processor_id: str | None = None,
        ov_device: str = "GPU",
    ) -> None:
        self._device = torch.device(device)
        self._enable_thinking = enable_thinking
        self._max_thinking_tokens = max_thinking_tokens

        self._processor = self._create_processor(
            processor_id=processor_id or model_id,
            max_pixels=max_pixels,
        )
        self._model = self._create_model(
            model_id=model_id,
            dtype=dtype,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            ov_device=ov_device,
        )
        if hasattr(self._model, "eval"):
            self._model.eval()

    # ------------------------------------------------------------------
    # Backend hooks (default: torch via transformers + bitsandbytes).
    # OpenVINO subclasses override these.
    # ------------------------------------------------------------------

    def _create_processor(
        self,
        processor_id: str,
        max_pixels: int | None,
    ) -> AutoProcessor:
        """Default torch impl. Backends share this — :class:`AutoProcessor`
        works against the OV IR repo as long as a ``preprocessor_config.json``
        is shipped alongside (the OpenVINO/* repos do)."""
        proc_kwargs: dict = {}
        if max_pixels is not None:
            proc_kwargs["max_pixels"] = max_pixels
        return AutoProcessor.from_pretrained(processor_id, **proc_kwargs)

    def _create_model(
        self,
        model_id: str,
        dtype: torch.dtype | None,
        load_in_8bit: bool,
        load_in_4bit: bool,
        ov_device: str,  # noqa: ARG002 -- used by OV override
    ) -> AutoModelForImageTextToText:
        """Default torch impl. The ``ov_device`` parameter is unused here
        but kept in the signature so OV overrides plug in without touching
        the call site in :meth:`__init__`."""
        device_str = str(self._device)
        model_kwargs: dict = {}
        if load_in_8bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["device_map"] = device_str
        elif load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            model_kwargs["device_map"] = device_str
        else:
            model_kwargs["torch_dtype"] = dtype or torch.float16
            model_kwargs["device_map"] = device_str
        return AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)

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

    @abstractmethod
    def encode_text(self, texts: list[str]) -> np.ndarray:
        """Return mean-pooled last-hidden-state embeddings for *texts*.

        Text-only — no image conditioning, no chat-template wrapping. Used
        by :class:`~src.sut.text_embedder.TextEmbedder` to measure sentence
        drift in the SUT's own representational space.

        :param texts: Batch of raw sentences.
        :returns: ``float32`` array of shape ``(len(texts), hidden_dim)``.
        """
        ...

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def tokenizer(self):
        """Shortcut to the underlying tokenizer."""
        return self._processor.tokenizer

    @property
    def device(self) -> torch.device:
        """Device the underlying model sits on."""
        return self._device

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

            # FP32 upcast before log_softmax — the model runs in FP16 by
            # default and FP16 log_softmax on near-saturated logits bottoms
            # out at ~1e-4 precision, which is visible as a numerical floor
            # in downstream boundary-distance metrics (see Exp-05 Phase A
            # SMOO run: TgtBal plateau at 5.3e-5 matched the FP16 floor).
            # Upcast costs a handful of extra reads per scoring call and
            # lifts precision to FP32's ~1e-7 relative resolution.
            total_lp = F.log_softmax(
                last_logits.float(), dim=-1,
            )[label_ids[0]].item()

            if n_tokens > 1:
                with torch.no_grad():
                    cont_out = self._model(
                        input_ids=label_ids[:-1].unsqueeze(0),
                        past_key_values=prefix_kvs,
                    )
                for i in range(n_tokens - 1):
                    lp = F.log_softmax(
                        cont_out.logits[0, i, :].float(), dim=-1,
                    )
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
    """Scorer for Qwen VL models (Qwen2/2.5/3-VL and Qwen3.5).

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
        processor_id: str | None = None,
        ov_device: str = "GPU",
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
            processor_id=processor_id,
            ov_device=ov_device,
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

    # Qwen3-VL layout: ForConditionalGeneration -> model (Qwen3VLModel) ->
    # language_model. Qwen3.5 causal wrappers collapse the inner name; we
    # walk defensively.
    def _text_backbone(self) -> torch.nn.Module:
        inner = getattr(self._model, "model", self._model)
        return getattr(inner, "language_model", inner)

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> np.ndarray:
        """Tokenize *texts* raw (no chat template) and mean-pool the last
        hidden state of the text backbone. Image tower is bypassed.
        """
        tok = self._processor.tokenizer
        batch = tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self._device)
        out = self._text_backbone()(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        hidden = out.last_hidden_state                       # (N, T, D)
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return pooled.float().cpu().numpy()


class LlavaScorer(VLMScorer):
    """Scorer for the LLaVA family (LLaVA-1.5 and LLaVA-NeXT/v1.6).

    Both variants use the same processor-driven chat template — concrete
    template differences (Mistral ``[INST]…[/INST]`` vs Vicuna) are baked
    into the processor's ``apply_chat_template`` and don't need scorer-
    level branching.

    LLaVA models do not implement a ``<think>`` trace; ``enable_thinking``
    is silently ignored at chat-template time.
    """

    def _normalize_image(self, image: Image.Image) -> Image.Image:
        """Hook for backend-specific image preprocessing. Default no-op."""
        return image

    def _prepare_inputs(
        self,
        image: Image.Image,
        prompt: str,
        enable_thinking: bool,  # noqa: ARG002 -- LLaVA has no thinking mode
    ) -> dict:
        image = self._normalize_image(image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
        return self._processor(
            text=[text], images=[image], return_tensors="pt",
        ).to(self._device)

    # LLaVA layout: LlavaForConditionalGeneration -> model -> language_model.
    # LLaVA-NeXT: LlavaNextForConditionalGeneration -> model -> language_model.
    def _text_backbone(self) -> torch.nn.Module:
        inner = getattr(self._model, "model", self._model)
        return getattr(inner, "language_model", inner)

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> np.ndarray:
        """Tokenize *texts* raw (no chat template) and mean-pool the last
        hidden state of the language backbone."""
        tok = self._processor.tokenizer
        batch = tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self._device)
        out = self._text_backbone()(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        hidden = out.last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return pooled.float().cpu().numpy()


# -----------------------------------------------------------------------
# OpenVINO backend
# -----------------------------------------------------------------------


class _OpenVINOBackendMixin:
    """Swaps the model loader to use ``OVModelForVisualCausalLM``.

    Designed as a mixin so the family scorers (chat template, encode_text)
    are reused unchanged. The OpenVINO IR is loaded directly — no PyTorch
    weights are touched on the boundary-tester host.

    The OV IR is compiled logits-only, so we cannot mean-pool the SUT's
    own language backbone for ``encode_text`` the way the torch path does.
    Instead we delegate to a lightweight sentence-transformers encoder
    (CPU, ~80 MB). The embedding space differs from the SUT's, but
    :class:`TextEmbeddingDistance` only consumes *cosine distance from
    anchor* — that distance is internally consistent within an experiment,
    which is what the optimizer needs.
    """

    _SENTENCE_ENCODER_ID: str = "sentence-transformers/all-MiniLM-L6-v2"
    _sentence_encoder = None  # lazy, class-level cache (one per process)

    def _create_processor(  # type: ignore[override]
        self,
        processor_id: str,
        max_pixels: int | None,
    ) -> AutoProcessor:
        # Same call as the torch path. The original (FP16) repo carries
        # the processor; the OpenVINO/* IR repo also ships a copy, so
        # either id works.
        proc_kwargs: dict = {}
        if max_pixels is not None:
            proc_kwargs["max_pixels"] = max_pixels
        return AutoProcessor.from_pretrained(processor_id, **proc_kwargs)

    def _create_model(  # type: ignore[override]
        self,
        model_id: str,
        dtype: torch.dtype | None,         # noqa: ARG002 -- OV IR sets this at export
        load_in_8bit: bool,                 # noqa: ARG002 -- OV uses NNCF, not bnb
        load_in_4bit: bool,                 # noqa: ARG002
        ov_device: str,
    ):
        # Late import — keeps optimum-intel optional for torch-only users.
        from optimum.intel import OVModelForVisualCausalLM
        return OVModelForVisualCausalLM.from_pretrained(model_id, device=ov_device)

    @classmethod
    def _get_sentence_encoder(cls):
        if cls._sentence_encoder is None:
            from sentence_transformers import SentenceTransformer
            cls._sentence_encoder = SentenceTransformer(
                cls._SENTENCE_ENCODER_ID, device="cpu",
            )
        return cls._sentence_encoder

    def encode_text(self, texts: list[str]) -> np.ndarray:  # type: ignore[override]
        encoder = self._get_sentence_encoder()
        return encoder.encode(
            texts, convert_to_numpy=True, show_progress_bar=False,
        ).astype(np.float32)

    def cleanup(self) -> None:  # type: ignore[override]
        gc.collect()  # OV manages its own GPU buffers; nothing to free here.


class OVQwenVLScorer(_OpenVINOBackendMixin, QwenVLScorer):
    """Qwen-VL family on OpenVINO. Inherits chat template + max_pixels
    default from :class:`QwenVLScorer`; only model loading is overridden.

    Note: optimum-intel 1.27 supports ``qwen2_vl`` and ``qwen2_5_vl`` —
    ``qwen3_vl`` is not yet covered and will fail at IR load.
    """


class OVLlavaScorer(_OpenVINOBackendMixin, LlavaScorer):
    """LLaVA / LLaVA-NeXT family on OpenVINO.

    Pins images to LLaVA-NeXT's smallest anyres tile (336x336). optimum-
    intel 1.27's anyres adapter has a token-count mismatch with the HF
    processor on certain image sizes -- raises
    ``RuntimeError: Number of elements of source < number of ones in mask``
    inside ``merge_vision_text_embeddings``. Forcing the single-grid case
    sidesteps the bug; remove this override once optimum-intel fixes it.
    """

    _LLAVA_NEXT_BASE_SIZE: tuple[int, int] = (336, 336)

    def _normalize_image(self, image: Image.Image) -> Image.Image:  # type: ignore[override]
        if image.size != self._LLAVA_NEXT_BASE_SIZE:
            return image.resize(self._LLAVA_NEXT_BASE_SIZE, Image.LANCZOS)
        return image


# -----------------------------------------------------------------------
# Registry / factory
# -----------------------------------------------------------------------

# Family detection runs against ``model_id.lower()`` substring matches.
# Order matters: more specific first.
_FAMILY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("llava", "llava"),
    ("qwen",  "qwen-vl"),  # both Qwen-VL and text-only Qwen3.5 use QwenVLScorer
)

# Keyed by (family, backend).
SCORER_REGISTRY: dict[tuple[str, str], type[VLMScorer]] = {
    ("qwen-vl", "torch"):    QwenVLScorer,
    ("qwen-vl", "openvino"): OVQwenVLScorer,
    ("llava",   "torch"):    LlavaScorer,
    ("llava",   "openvino"): OVLlavaScorer,
}


def _detect_family(model_id: str) -> str:
    """Detect the model family from a model id (HF repo)."""
    s = model_id.lower()
    for needle, family in _FAMILY_PATTERNS:
        if needle in s:
            return family
    raise ValueError(
        f"Could not detect family for model '{model_id}'. "
        f"Known families: {sorted({f for _, f in _FAMILY_PATTERNS})}"
    )


def create_scorer(
    model_id: str,
    device: str,
    enable_thinking: bool = True,
    max_thinking_tokens: int = 2000,
    max_pixels: int | None = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    backend: str = "torch",
    processor_id: str | None = None,
    ov_device: str = "GPU",
) -> VLMScorer:
    """Instantiate the right scorer for *model_id* on *backend*.

    Family is detected from *model_id* substring (``"qwen"`` /
    ``"llava"``); backend selects between torch and OpenVINO.

    :param model_id: HuggingFace repo id. For OpenVINO, normally a pre-
        converted IR repo (e.g. ``OpenVINO/Qwen2.5-VL-7B-Instruct-int4-ov``).
    :param device: Torch device string. Ignored on the OpenVINO backend.
    :param backend: ``"torch"`` (default) or ``"openvino"``.
    :param processor_id: Optional override for the processor repo
        (defaults to *model_id*). Useful for OV when the IR repo differs
        from the original FP16 repo.
    :param ov_device: OpenVINO device label (``"GPU"`` for Arc, ``"CPU"``).
    :returns: A concrete :class:`VLMScorer` instance.
    :raises ValueError: If family or backend are not recognized.
    """
    family = _detect_family(model_id) if backend == "torch" else (
        # For OV the model_id is often the IR repo (no "Qwen/"/"llava-hf/"
        # prefix) -- detect via processor_id when set, else model_id.
        _detect_family(processor_id or model_id)
    )
    key = (family, backend)
    cls = SCORER_REGISTRY.get(key)
    if cls is None:
        raise ValueError(
            f"No scorer for {key}. Known: {sorted(SCORER_REGISTRY.keys())}"
        )
    return cls(
        model_id,
        device,
        enable_thinking,
        max_thinking_tokens,
        max_pixels=max_pixels,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        processor_id=processor_id,
        ov_device=ov_device,
    )
