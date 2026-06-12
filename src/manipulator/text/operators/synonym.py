"""MLM-based synonym operator.

Replaces content words with context-aware substitutes proposed by a
masked-language model (default ``answerdotai/ModernBERT-large``), then
filters them through three stages.

When *redis_url* is provided, the raw top-k MLM candidates (per masked
sentence) are cached in Redis under
``mlmcand:sha256(model_name, masked_text, topk_pre)``. Filtering still
runs locally because it depends on the full token context.

1. **Top-k pre-filter** — keep top *topk_pre_filter* MLM logits per
   masked position (default 50).
2. **Fine-grained PoS filter** — substituted into the sentence, the
   candidate must have the same Penn-Treebank tag (``token.tag_``) as
   the original word in its position.
3. **Lemma + morphology reject** — drop candidates whose lemma matches
   the original (``running → run`` is the same lemma) and drop any
   candidate matching the negation-prefix regex
   ``^(non|un|in|im|de|dis|anti|counter)[-_]+`` to prevent the
   artefact-class hits that subword/static embedding pools produce
   ("non-main", "non-subject", "anti-pattern").

The model is loaded lazily on the first :meth:`prepare` call so a
``severity = 0`` profile pays no cost.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..types import CONTENT_POS_TAGS, TokenSequence
from .base import OperatorContext, severity_to_k_max

logger = logging.getLogger(__name__)

# Serialises the lazy MLM/spaCy load across worker threads. The composite
# text manipulator (and hence this operator) is shared by every worker, so
# concurrent first-time prepare() calls would otherwise enter
# ``from_pretrained(...).to(...)`` simultaneously — HF model construction is
# not thread-safe and the loser reads half-materialised meta tensors
# ("Cannot copy out of meta tensor; no data!"). Module-level so it also
# guards distinct instances built per worker.
_LOAD_LOCK = threading.Lock()


# Morphological negation prefixes that indicate an antonym, not a synonym.
# Separator (hyphen / underscore) is required so we don't reject real words
# like "image" (im-) or "incident" (in-) that share the prefix letters.
NEGATION_PREFIX_RE = re.compile(
    r"^(non|un|in|im|de|dis|anti|counter)[-_]+",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SynonymContext(OperatorContext):
    """Synonym context: positions + per-position filtered candidate pools."""

    candidates: tuple[tuple[str, ...], ...]
    original_words: tuple[str, ...]


class SynonymOperator:
    """MLM-based synonym substitution with PoS + lemma + negation filters.

    Genome encoding:

    * ``gene = 0``   → keep original word
    * ``gene = k ≥ 1`` → use ``candidates[i][k - 1]``
    """

    name = "synonym"

    def __init__(
        self,
        severity: float,
        model_name: str = "answerdotai/ModernBERT-large",
        topk_pre_filter: int = 50,
        device: str = "cpu",
        spacy_model: str = "en_core_web_sm",
        content_pos: frozenset[str] = CONTENT_POS_TAGS,
        negation_prefix_re: re.Pattern = NEGATION_PREFIX_RE,
        k_max_override: int | None = None,
        redis_url: str | None = None,
    ) -> None:
        self._severity = float(severity)
        self._k_max = severity_to_k_max(severity, override=k_max_override)
        self._topk_pre = int(topk_pre_filter)
        self._model_name = model_name
        self._device_str = str(device)
        self._spacy_model_name = spacy_model
        self._content_pos = content_pos
        self._negation_re = negation_prefix_re
        self._redis_url = redis_url

        # Lazy resources — populated on first prepare()
        self._model: Any = None
        self._tokenizer: Any = None
        self._nlp: Any = None
        self._device: Any = None
        self._redis: Any = None
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def severity(self) -> float:
        return self._severity

    @property
    def k_max(self) -> int:
        return self._k_max

    @property
    def topk_pre_filter(self) -> int:
        return self._topk_pre

    # ------------------------------------------------------------------
    # Lazy resource loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # Double-checked locking: only one thread builds the MLM/spaCy
        # resources; the rest block here and return on the inner check.
        with _LOAD_LOCK:
            if self._model is not None:
                return
            import torch
            import spacy
            from transformers import AutoModelForMaskedLM, AutoTokenizer

            self._device = torch.device(self._device_str)
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            model = (
                AutoModelForMaskedLM.from_pretrained(self._model_name)
                .to(self._device)
                .eval()
            )
            # Lemmatiser must be enabled for the lemma-reject filter
            self._nlp = spacy.load(
                self._spacy_model_name, disable=["ner", "parser"]
            )

            if self._redis_url:
                try:
                    import redis

                    client = redis.Redis.from_url(
                        self._redis_url, decode_responses=True
                    )
                    client.ping()
                    self._redis = client
                    logger.info(
                        "Synonym MLM candidate cache connected to Redis at %s",
                        self._redis_url,
                    )
                except Exception:  # noqa: BLE001
                    logger.info(
                        "Redis unavailable at %s — synonym MLM running without cache",
                        self._redis_url,
                    )
                    self._redis = None
            # Publish the fully-built model last: other threads gate on
            # ``self._model is not None`` and must never see a partial object.
            self._model = model

    def _set_resources(
        self,
        model: Any,
        tokenizer: Any,
        nlp: Any,
        device: Any = None,
    ) -> None:
        """Inject pre-built resources (used by tests with a stub MLM)."""
        self._model = model
        self._tokenizer = tokenizer
        self._nlp = nlp
        if device is not None:
            self._device = device

    # ------------------------------------------------------------------
    # Two-phase API
    # ------------------------------------------------------------------

    def prepare(
        self,
        tokens: TokenSequence,
        exclude_words: frozenset[str] | None = None,
    ) -> SynonymContext:
        if self._k_max == 0:
            return SynonymContext(
                positions=np.empty(0, dtype=np.intp),
                candidates=(),
                original_words=(),
            )
        self._ensure_loaded()

        excl = {w.lower() for w in exclude_words} if exclude_words else set()

        eligible_positions: list[int] = []
        for i, (tok, pos) in enumerate(zip(tokens.tokens, tokens.pos_tags)):
            if pos not in self._content_pos:
                continue
            if not tok.isalpha():
                continue
            if tok.lower() in excl:
                continue
            eligible_positions.append(i)

        if not eligible_positions:
            return SynonymContext(
                positions=np.empty(0, dtype=np.intp),
                candidates=(),
                original_words=(),
            )

        # Tag the original sentence to read fine-grained PoS + lemma.
        original_text = "".join(t + w for t, w in zip(tokens.tokens, tokens.whitespace))
        orig_doc = self._nlp(original_text)
        if len(orig_doc) == tokens.n_tokens:
            orig_tags = [t.tag_ for t in orig_doc]
            orig_lemmas = [t.lemma_ for t in orig_doc]
        else:
            orig_tags = self._fallback_attribute_lookup(tokens, orig_doc, "tag_")
            orig_lemmas = self._fallback_attribute_lookup(tokens, orig_doc, "lemma_")

        kept_positions: list[int] = []
        kept_candidates: list[tuple[str, ...]] = []
        kept_originals: list[str] = []

        for pos_idx in eligible_positions:
            orig_word = tokens.tokens[pos_idx]
            orig_tag = orig_tags[pos_idx]
            orig_lemma = orig_lemmas[pos_idx]

            raw = self._mlm_candidates(tokens, pos_idx)
            filtered = self._filter_candidates(
                raw,
                tokens=tokens,
                pos_idx=pos_idx,
                orig_word=orig_word,
                orig_tag=orig_tag,
                orig_lemma=orig_lemma,
            )

            if filtered:
                kept_positions.append(pos_idx)
                kept_candidates.append(tuple(filtered[: self._k_max]))
                kept_originals.append(orig_word)

        return SynonymContext(
            positions=np.array(kept_positions, dtype=np.intp),
            candidates=tuple(kept_candidates),
            original_words=tuple(kept_originals),
        )

    def gene_dim(self, ctx: SynonymContext) -> int:
        return ctx.n_positions

    def gene_bounds(self, ctx: SynonymContext) -> NDArray[np.int64]:
        if ctx.n_positions == 0:
            return np.empty(0, dtype=np.int64)
        return np.array([len(c) + 1 for c in ctx.candidates], dtype=np.int64)

    def apply(
        self,
        ctx: SynonymContext,
        genes: NDArray[np.int64],
        current: TokenSequence,
    ) -> TokenSequence:
        if len(genes) != ctx.n_positions:
            raise ValueError(
                f"Synonym: gene length {len(genes)} != positions {ctx.n_positions}"
            )
        if ctx.n_positions == 0:
            return current
        active = np.nonzero(genes)[0]
        if len(active) == 0:
            return current
        positions = ctx.positions[active]
        words = tuple(ctx.candidates[int(i)][int(genes[i]) - 1] for i in active)
        return current.replace(positions, words)

    # ------------------------------------------------------------------
    # MLM scoring
    # ------------------------------------------------------------------

    def _mlm_candidates(
        self,
        tokens: TokenSequence,
        pos_idx: int,
    ) -> list[str]:
        """Mask the word at *pos_idx* and return top-k MLM candidates."""
        import torch

        mask_token = self._tokenizer.mask_token
        masked = list(tokens.tokens)
        masked[pos_idx] = mask_token
        masked_text = "".join(t + w for t, w in zip(masked, tokens.whitespace))

        cache_key = self._cache_key(masked_text)
        if self._redis is not None:
            blob = self._redis.get(cache_key)
            if blob is not None:
                self._cache_hits += 1
                return json.loads(blob)
            self._cache_misses += 1

        inputs = self._tokenizer(masked_text, return_tensors="pt").to(self._device)
        mask_id = self._tokenizer.mask_token_id
        mask_positions = (inputs.input_ids == mask_id).nonzero(as_tuple=True)
        if mask_positions[0].numel() == 0:
            return []

        with torch.no_grad():
            outputs = self._model(**inputs)
        logits = outputs.logits  # (1, seq, vocab)

        row = mask_positions[0][0].item()
        col = mask_positions[1][0].item()
        mask_logits = logits[row, col]  # (vocab,)

        topk = torch.topk(mask_logits, self._topk_pre, dim=-1).indices.tolist()
        cands = self._tokenizer.convert_ids_to_tokens(topk)

        cleaned: list[str] = []
        for c in cands:
            if c.startswith("Ġ"):
                c = c[1:]
            elif c.startswith("▁"):
                c = c[1:]
            cleaned.append(c)

        if self._redis is not None:
            self._redis.set(cache_key, json.dumps(cleaned))
        return cleaned

    def _cache_key(self, masked_text: str) -> str:
        h = hashlib.sha256()
        h.update(self._model_name.encode())
        h.update(b"\x1f")
        h.update(str(self._topk_pre).encode())
        h.update(b"\x1f")
        h.update(masked_text.encode())
        return "mlmcand:" + h.hexdigest()

    @property
    def cache_stats(self) -> dict[str, int]:
        return {"hits": self._cache_hits, "misses": self._cache_misses}

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _filter_candidates(
        self,
        raw_cands: list[str],
        *,
        tokens: TokenSequence,
        pos_idx: int,
        orig_word: str,
        orig_tag: str,
        orig_lemma: str,
    ) -> list[str]:
        kept: list[str] = []
        seen: set[str] = set()

        for cand in raw_cands:
            if not cand:
                continue
            if cand.startswith(("[", "<")):
                continue
            if not cand.isalpha():
                continue
            if cand.lower() == orig_word.lower():
                continue
            if cand.lower() in seen:
                continue
            if self._negation_re.match(cand):
                continue

            cand_doc = self._tag_in_context(tokens, pos_idx, cand)
            if cand_doc is None or pos_idx >= len(cand_doc):
                continue
            cand_token = cand_doc[pos_idx]
            if cand_token.lemma_ == orig_lemma:
                continue
            if cand_token.tag_ != orig_tag:
                continue

            kept.append(cand)
            seen.add(cand.lower())

        return kept

    def _tag_in_context(
        self,
        tokens: TokenSequence,
        pos_idx: int,
        candidate: str,
    ) -> Any | None:
        """Substitute *candidate* into the sentence, run spaCy, return Doc."""
        substituted = list(tokens.tokens)
        substituted[pos_idx] = candidate
        text = "".join(t + w for t, w in zip(substituted, tokens.whitespace))
        doc = self._nlp(text)
        if len(doc) != tokens.n_tokens:
            return None
        return doc

    # ------------------------------------------------------------------
    # Fallback for spaCy tokenisation drift on the original sentence
    # ------------------------------------------------------------------

    def _fallback_attribute_lookup(
        self,
        tokens: TokenSequence,
        doc: Any,
        attr: str,
    ) -> list[str]:
        out: list[str] = []
        cursor = 0
        for tok in tokens.tokens:
            match = None
            for j in range(cursor, len(doc)):
                if doc[j].text == tok:
                    match = doc[j]
                    cursor = j + 1
                    break
            out.append(getattr(match, attr) if match is not None else "X")
        return out
