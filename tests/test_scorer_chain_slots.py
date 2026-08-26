"""Tests for per-slot chain scoring (:meth:`VLMScorer.score_chain_slots`).

Exp-105 steps 5/6: one teacher-forced pass over a multi-slot answer chain,
per-slot length-normalised conditional log-probs under the realised prefix.

Follows the ``tests/test_vlm_sut.py`` style: concrete fakes, no mocks, no
model downloads, no network.  The fake model is a deterministic
full-context scorer -- its logits are a fixed pseudo-random row keyed by a
hash of *every* token consumed so far (KV cache included) -- so a slot's
conditional genuinely moves when the realised prefix changes, and every
expected value can be recomputed independently in the test.
"""

from __future__ import annotations

import math
import warnings
from contextlib import contextmanager

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.sut.scorer import VLMScorer


@contextmanager
def no_warnings():
    """Fail the test if any warning is raised inside the block."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


# ---------------------------------------------------------------------------
# Fake tokenizers
# ---------------------------------------------------------------------------

# Per-character vocabulary covering everything the templates use, so token
# slices can be checked by counting characters.
_VOCAB: tuple[str, ...] = tuple(
    " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,"
)
_CHAR_TO_ID: dict[str, int] = {c: i for i, c in enumerate(_VOCAB)}


class CharTokenizer:
    """Per-character tokenizer.  ``encode`` never merges, so
    ``encode(a) + encode(b) == encode(a + b)`` always holds -- the benign
    case the incremental diff must handle without warning."""

    vocab_size = len(_VOCAB)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:  # noqa: ARG002
        return [_CHAR_TO_ID[c] for c in text]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:  # noqa: ARG002
        return "".join(_VOCAB[int(i)] for i in ids)


class MergingTokenizer(CharTokenizer):
    """Per-character tokenizer with one crafted BPE-style merge: the pair
    ``" a"`` (space + ``a``) collapses into a single token, greedily,
    left to right.

    This is the adversarial case for boundary detection.  A carrier
    ending in a space followed by a slot filler starting with ``a``
    ("Because the applicant was " + "a man") encodes such that the
    carrier's final space token *disappears* -- the extended encoding is
    not an extension of the shorter one, so a naive
    ``len(encode(accum))`` boundary would mis-slice every later slot.
    """

    _MERGE_ID = len(_VOCAB)  # one id past the char vocab
    _MERGE_PAIR = " a"
    vocab_size = len(_VOCAB) + 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:  # noqa: ARG002
        ids: list[int] = []
        i = 0
        while i < len(text):
            if text[i : i + 2] == self._MERGE_PAIR:
                ids.append(self._MERGE_ID)
                i += 2
            else:
                ids.append(_CHAR_TO_ID[text[i]])
                i += 1
        return ids


# ---------------------------------------------------------------------------
# Fake model -- deterministic logits over the full realised context
# ---------------------------------------------------------------------------


class _Out:
    """Minimal stand-in for a transformers ``CausalLMOutputWithPast``."""

    def __init__(self, logits: torch.Tensor, past_key_values) -> None:
        self.logits = logits
        self.past_key_values = past_key_values


class FakeCache:
    """Records the token ids consumed so far, in order.

    Mirrors ``transformers.DynamicCache``: a continuation pass *appends
    to the same object* (verified against transformers 5.3 --
    ``get_seq_length`` grows 3 -> 5 -> 7 across successive updates).
    Keeping that semantics here means the fake reproduces the real
    conditioning, including for ``score_categories``' cache reuse.
    """

    def __init__(self, ids: list[int]) -> None:
        self.ids = list(ids)


class FakeModel:
    """Deterministic causal LM over the fake vocab.

    The logits row emitted after consuming context ``c`` is
    ``table[hash(c)]``: fully context-dependent (not just bigram), so a
    slot conditional changes when anything earlier in the chain changes.
    Records every forward call for call-count / cache assertions.
    """

    _TABLE_ROWS = 4096

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.calls: list[dict] = []
        rng = np.random.default_rng(20260802)
        self._table = torch.tensor(
            rng.normal(0.0, 2.0, size=(self._TABLE_ROWS, vocab_size)),
            dtype=torch.float32,
        )

    def _key(self, ctx_ids) -> int:
        h = 0
        for tok in ctx_ids:
            h = (h * 1000003 + int(tok) + 1) % self._TABLE_ROWS
        return h

    def row(self, ctx_ids) -> torch.Tensor:
        """Logits emitted after having consumed *ctx_ids*."""
        return self._table[self._key(ctx_ids)]

    def logprob(self, ctx_ids, token_id: int) -> float:
        """``log p(token_id | ctx_ids)`` under the fake."""
        return F.log_softmax(self.row(ctx_ids).float(), dim=-1)[token_id].item()

    def eval(self) -> "FakeModel":
        return self

    def __call__(self, input_ids=None, past_key_values=None, use_cache=False, **kw):
        ids = [int(i) for i in input_ids[0].tolist()]
        cache_before = None if past_key_values is None else list(past_key_values.ids)
        self.calls.append(
            {
                "ids": ids,
                "cache_before": cache_before,
                "use_cache": use_cache,
                "extra": sorted(kw),
            }
        )
        if past_key_values is None:
            past_key_values = FakeCache([])
        ctx = list(past_key_values.ids)
        rows = []
        for tok in ids:
            ctx.append(tok)
            rows.append(self.row(ctx))           # causal: position t sees ids[:t+1]
        past_key_values.ids = ctx                 # in-place growth, like DynamicCache
        return _Out(torch.stack(rows).unsqueeze(0), past_key_values)


# ---------------------------------------------------------------------------
# Fake scorer -- real VLMScorer machinery, fake model/tokenizer
# ---------------------------------------------------------------------------

_PREFIX_TEXT = "PROMPT"
_PREFIX_IDS: list[int] = CharTokenizer().encode(_PREFIX_TEXT)


class ChainFakeScorer(VLMScorer):
    """Concrete scorer wired to :class:`FakeModel`.

    Skips ``super().__init__`` (no model load).  ``_prepare_inputs``
    returns fixed prompt-prefix ids and ignores the image -- these tests
    are about token bookkeeping, not vision.
    """

    def __init__(self, tokenizer=None) -> None:
        self._device = torch.device("cpu")
        self._enable_thinking = False
        self._max_thinking_tokens = 0
        self._tokenizer = tokenizer or CharTokenizer()
        self._model = FakeModel(self._tokenizer.vocab_size)
        self._processor = None

    @property
    def tokenizer(self):  # type: ignore[override]
        return self._tokenizer

    def _prepare_inputs(self, image, prompt, enable_thinking):  # type: ignore[override]
        return {"input_ids": torch.tensor([_PREFIX_IDS], device=self._device)}

    def encode_text(self, texts):  # type: ignore[override]
        return np.zeros((len(texts), 1), dtype=np.float32)


def _dummy_image():
    from PIL import Image

    return Image.new("RGB", (8, 8), color=(255, 0, 0))


def _texts(parts) -> list[str]:
    return [p if isinstance(p, str) else p[1] for p in parts]


def _expected_chain_lps(model: FakeModel, chain_ids) -> list[float]:
    """Independently recomputed teacher-forced conditionals for the chain:
    ``log p(chain[j] | PROMPT + chain[:j])``."""
    ctx = list(_PREFIX_IDS)
    out = []
    for tok in chain_ids:
        out.append(model.logprob(ctx, tok))
        ctx.append(tok)
    return out


# Two-slot template in both orders (the Exp-105 step 5/6 stimulus shape).
_PARTS_A = [
    "Because the applicant was ",
    ("G", "a man"),
    ", the application was ",
    ("D", "accepted"),
    ".",
]
_PARTS_B = [
    "The application was ",
    ("D", "accepted"),
    " because the applicant was ",
    ("G", "a man"),
    ".",
]


# =========================================================================
# TestChainTokenSpans -- incremental-diff slicing
# =========================================================================


class TestChainTokenSpans:
    """Slot slices come from the growing chain, not per-part encodes."""

    def test_two_slot_template_order_a(self) -> None:
        scorer = ChainFakeScorer()
        texts = _texts(_PARTS_A)
        spans, chain_ids = scorer._chain_token_spans(texts)

        assert chain_ids == scorer.tokenizer.encode("".join(texts))
        assert spans[0][0] == 0
        assert spans[-1][1] == len(chain_ids)
        for (_, end), (start, _) in zip(spans, spans[1:]):
            assert end == start            # contiguous, non-overlapping
        for text, (start, end) in zip(texts, spans):
            assert scorer.tokenizer.decode(chain_ids[start:end]) == text

    def test_two_slot_template_order_b(self) -> None:
        """Same slots, reversed order: slices follow the realised chain."""
        scorer = ChainFakeScorer()
        texts = _texts(_PARTS_B)
        spans, chain_ids = scorer._chain_token_spans(texts)

        assert scorer.tokenizer.decode(chain_ids[slice(*spans[1])]) == "accepted"
        assert scorer.tokenizer.decode(chain_ids[slice(*spans[3])]) == "a man"
        assert spans[1][0] < spans[3][0]   # D now precedes G

    def test_per_part_encode_would_be_wrong_but_diff_is_right(self) -> None:
        """The merging tokenizer makes ``encode(part)`` disagree with the
        part's realised tokens; the incremental diff still covers the
        chain exactly."""
        scorer = ChainFakeScorer(MergingTokenizer())
        texts = ["The applicant was ", "a man", "."]
        naive = sum(len(scorer.tokenizer.encode(t)) for t in texts)
        with pytest.warns(RuntimeWarning):
            spans, chain_ids = scorer._chain_token_spans(texts)
        assert naive != len(chain_ids)                 # per-part encode is wrong
        assert spans[-1][1] == len(chain_ids)          # diff still covers it

    def test_empty_part_yields_empty_span(self) -> None:
        scorer = ChainFakeScorer()
        spans, chain_ids = scorer._chain_token_spans(["ab", "", "cd"])
        assert spans == [(0, 2), (2, 2), (2, 4)]
        assert len(chain_ids) == 4


# =========================================================================
# TestScoreChainSlots -- the pinned public API
# =========================================================================


class TestScoreChainSlots:
    """Return contract of ``score_chain_slots``."""

    def test_keys_and_types(self) -> None:
        scorer = ChainFakeScorer()
        out = scorer.score_chain_slots(_dummy_image(), "p", _PARTS_A)
        assert list(out) == ["G", "D"]           # slot order of appearance
        for total, norm, n in out.values():
            assert isinstance(total, float)
            assert isinstance(norm, float)
            assert isinstance(n, int)

    def test_n_tokens_matches_slice(self) -> None:
        scorer = ChainFakeScorer()
        for parts in (_PARTS_A, _PARTS_B):
            spans, _ = scorer._chain_token_spans(_texts(parts))
            out = scorer.score_chain_slots(_dummy_image(), "p", parts)
            for part, (start, end) in zip(parts, spans):
                if not isinstance(part, str):
                    assert out[part[0]][2] == end - start
        # Per-character tokenizer -> n_tokens is the filler length.
        out = scorer.score_chain_slots(_dummy_image(), "p", _PARTS_A)
        assert out["G"][2] == len("a man")
        assert out["D"][2] == len("accepted")

    def test_lp_norm_is_total_over_n(self) -> None:
        scorer = ChainFakeScorer()
        for parts in (_PARTS_A, _PARTS_B):
            out = scorer.score_chain_slots(_dummy_image(), "p", parts)
            for total, norm, n in out.values():
                assert norm == pytest.approx(total / n, abs=1e-12)

    @pytest.mark.parametrize("parts", [_PARTS_A, _PARTS_B])
    def test_values_match_independent_conditionals(self, parts) -> None:
        """Per-slot totals equal the sum of the true teacher-forced
        conditionals over exactly that slot's token slice."""
        scorer = ChainFakeScorer()
        spans, chain_ids = scorer._chain_token_spans(_texts(parts))
        expected = _expected_chain_lps(scorer._model, chain_ids)

        out = scorer.score_chain_slots(_dummy_image(), "p", parts)
        for part, (start, end) in zip(parts, spans):
            if isinstance(part, str):
                continue
            assert out[part[0]][0] == pytest.approx(
                sum(expected[start:end]), abs=1e-9
            )

    def test_single_forward_pair_over_the_chain(self) -> None:
        """Exactly one prefix pass + one continuation pass; the
        continuation gets the whole chain minus its last token, on the
        prefix cache."""
        scorer = ChainFakeScorer()
        _, chain_ids = scorer._chain_token_spans(_texts(_PARTS_A))

        scorer.score_chain_slots(_dummy_image(), "p", _PARTS_A)
        calls = scorer._model.calls
        assert len(calls) == 2
        assert calls[0]["ids"] == _PREFIX_IDS
        assert calls[0]["use_cache"] is True
        assert calls[1]["ids"] == chain_ids[:-1]
        assert calls[1]["cache_before"] == _PREFIX_IDS

    def test_empty_slot_filler(self) -> None:
        scorer = ChainFakeScorer()
        out = scorer.score_chain_slots(
            _dummy_image(), "p", ["ab ", ("G", ""), " cd"]
        )
        assert out["G"] == (float("-inf"), float("-inf"), 0)

    def test_fully_empty_chain_skips_model(self) -> None:
        scorer = ChainFakeScorer()
        out = scorer.score_chain_slots(_dummy_image(), "p", ["", ("G", "")])
        assert out == {"G": (float("-inf"), float("-inf"), 0)}
        assert scorer._model.calls == []

    def test_single_token_chain_needs_no_continuation(self) -> None:
        scorer = ChainFakeScorer()
        out = scorer.score_chain_slots(_dummy_image(), "p", [("G", "a")])
        assert out["G"][2] == 1
        assert len(scorer._model.calls) == 1          # prefix only
        assert out["G"][0] == pytest.approx(
            scorer._model.logprob(_PREFIX_IDS, _CHAR_TO_ID["a"]), abs=1e-9
        )

    def test_duplicate_slot_name_raises(self) -> None:
        scorer = ChainFakeScorer()
        with pytest.raises(ValueError, match="Duplicate slot name"):
            scorer.score_chain_slots(
                _dummy_image(), "p", [("G", "a"), " and ", ("G", "b")]
            )

    def test_malformed_part_raises(self) -> None:
        scorer = ChainFakeScorer()
        with pytest.raises(ValueError, match="Chain part must be"):
            scorer.score_chain_slots(_dummy_image(), "p", [("G", "a", "x")])

    def test_all_carrier_no_slots(self) -> None:
        scorer = ChainFakeScorer()
        assert scorer.score_chain_slots(_dummy_image(), "p", ["hello"]) == {}


# =========================================================================
# TestPrefixCancellation -- the property steps 5/6 rely on
# =========================================================================


class TestPrefixCancellation:
    """Chains identical up to the first slot share every conditional
    before it, so the shared carrier cancels when the chains are
    compared."""

    def test_identical_prefix_identical_conditionals(self) -> None:
        scorer = ChainFakeScorer()
        carrier = "Because the applicant was "
        lps_man = _expected_chain_lps(
            scorer._model,
            scorer.tokenizer.encode(carrier + "a man" + ", done."),
        )
        lps_woman = _expected_chain_lps(
            scorer._model,
            scorer.tokenizer.encode(carrier + "a woman" + ", done."),
        )
        n_shared = len(scorer.tokenizer.encode(carrier))
        assert lps_man[:n_shared] == pytest.approx(lps_woman[:n_shared], abs=1e-12)

    def test_first_slot_score_independent_of_later_chain(self) -> None:
        """The G conditional does not move with what follows it -- the
        pass is teacher-forced and causal."""
        scorer = ChainFakeScorer()
        carrier = "Because the applicant was "
        out_1 = scorer.score_chain_slots(
            _dummy_image(), "p", [carrier, ("G", "a man"), ", accepted."]
        )
        out_2 = scorer.score_chain_slots(
            _dummy_image(), "p", [carrier, ("G", "a man"), ", rejected."]
        )
        assert out_1["G"] == pytest.approx(out_2["G"], abs=1e-12)

    def test_second_slot_sees_the_realised_first_slot(self) -> None:
        """Conversely the D conditional *does* move with the G filler --
        that is the edge weight steps 5/6 measure."""
        scorer = ChainFakeScorer()
        tail = ", the application was "
        out_man = scorer.score_chain_slots(
            _dummy_image(), "p",
            ["Because the applicant was ", ("G", "a man"), tail,
             ("D", "accepted"), "."],
        )
        out_woman = scorer.score_chain_slots(
            _dummy_image(), "p",
            ["Because the applicant was ", ("G", "a woman"), tail,
             ("D", "accepted"), "."],
        )
        assert out_man["D"][2] == out_woman["D"][2]          # same tokens
        assert out_man["D"][0] != pytest.approx(out_woman["D"][0], abs=1e-9)

    def test_order_swap_changes_conditionals(self) -> None:
        """Order A vs B: same fillers, different position -> different
        per-slot scores (position vs role effect, step 6)."""
        scorer = ChainFakeScorer()
        out_a = scorer.score_chain_slots(_dummy_image(), "p", _PARTS_A)
        out_b = scorer.score_chain_slots(_dummy_image(), "p", _PARTS_B)
        assert out_a["G"][2] == out_b["G"][2]
        assert out_a["G"][0] != pytest.approx(out_b["G"][0], abs=1e-9)


# =========================================================================
# TestMergeGuard -- the incremental-diff safety net
# =========================================================================


class TestMergeGuard:
    """A tokenizer that merges across a part boundary must warn, not
    silently mis-slice."""

    def test_no_warning_on_clean_boundaries(self) -> None:
        scorer = ChainFakeScorer()
        with no_warnings():
            scorer.score_chain_slots(_dummy_image(), "p", _PARTS_A)
            scorer.score_chain_slots(_dummy_image(), "p", _PARTS_B)

    def test_merge_across_boundary_warns(self) -> None:
        """Carrier ends in a space, filler starts with 'a' -> the crafted
        tokenizer merges them into one token across the boundary."""
        scorer = ChainFakeScorer(MergingTokenizer())
        with pytest.warns(RuntimeWarning, match="merged across chain part"):
            scorer.score_chain_slots(
                _dummy_image(), "p",
                ["Because the applicant was ", ("G", "a man"), "."],
            )

    def test_merge_shifts_boundary_left_and_keeps_spans_sane(self) -> None:
        scorer = ChainFakeScorer(MergingTokenizer())
        texts = ["x ", "abc"]
        # encode("x ")    = [x, ' ']
        # encode("x abc") = [x, MERGE(" a"), b, c]   -> rewrites index 1
        with pytest.warns(RuntimeWarning):
            spans, chain_ids = scorer._chain_token_spans(texts)

        assert chain_ids == scorer.tokenizer.encode("".join(texts))
        assert spans == [(0, 1), (1, 4)]     # carrier clamped, merge -> later part
        assert spans[-1][1] == len(chain_ids)
        for (_, end), (start, _) in zip(spans, spans[1:]):
            assert end == start
        for start, end in spans:
            assert 0 <= start <= end <= len(chain_ids)

    def test_merge_can_empty_a_slot(self) -> None:
        """A slot fully swallowed by a later merge reports zero tokens,
        the same shape score_categories uses for an empty label."""
        scorer = ChainFakeScorer(MergingTokenizer())
        with pytest.warns(RuntimeWarning):
            out = scorer.score_chain_slots(
                _dummy_image(), "p", ["x", ("G", " "), "abc"]
            )
        assert out["G"] == (float("-inf"), float("-inf"), 0)

    def test_scores_still_sum_to_the_chain_after_a_merge(self) -> None:
        """Even in the degraded case the reported slices stay a partition
        of the real chain, so no log-prob mass is invented."""
        scorer = ChainFakeScorer(MergingTokenizer())
        parts = ["The applicant was ", ("G", "a man"), ", ",
                 ("D", "accepted"), "."]
        with pytest.warns(RuntimeWarning):
            spans, chain_ids = scorer._chain_token_spans(_texts(parts))
        with pytest.warns(RuntimeWarning):
            out = scorer.score_chain_slots(_dummy_image(), "p", parts)
        expected = _expected_chain_lps(scorer._model, chain_ids)
        for part, (start, end) in zip(parts, spans):
            if isinstance(part, str):
                continue
            assert out[part[0]][0] == pytest.approx(
                sum(expected[start:end]), abs=1e-9
            )

    def test_char_tokenizer_never_warns_on_long_chain(self) -> None:
        scorer = ChainFakeScorer()
        with no_warnings():
            scorer._chain_token_spans(
                ["The application was ", "rejected",
                 " because the applicant was ", "a woman", "."]
            )


# =========================================================================
# TestScoreCategoriesUnchanged -- the refactor must be behaviour-neutral
# =========================================================================

# Golden values captured by running score_categories from git HEAD (the
# pre-refactor scorer.py) against this same seeded FakeModel + CharTokenizer,
# and verified bit-for-bit identical against the post-refactor code.  Any
# behavioural drift in the shared prefix pass moves these.
#
# They are tied to this exact category LIST, not just to the labels: the
# prefix KV cache is reused across labels and grows in place (FakeCache
# mirrors transformers' DynamicCache), so a label's continuation is
# conditioned on the labels scored before it.  Pre-existing behaviour --
# pinned here, deliberately unchanged by this work.
_GOLDEN_CATEGORIES: list[str] = ["cat", "dog", "bird", ""]
_GOLDEN_SCORE_CATEGORIES: dict[str, tuple[float, float, int]] = {
    "cat": (-19.079808235168457, -6.359936078389485, 3),
    "dog": (-17.800814151763916, -5.933604717254639, 3),
    "bird": (-23.87961769104004, -5.96990442276001, 4),
    "": (float("-inf"), float("-inf"), 0),
}
# Same capture, for the tuple used by the tensor-wrapper test.
_GOLDEN_TENSOR_CATEGORIES: tuple[str, ...] = ("bird", "cat", "dog")
_GOLDEN_TENSOR: list[float] = [
    -8.391491889953613, -5.739893436431885, -5.116476058959961,
]


class TestScoreCategoriesUnchanged:
    """``score_categories`` behaviour is untouched by the ``_prefix_forward``
    extraction."""

    def test_matches_pre_refactor_golden_values(self) -> None:
        scorer = ChainFakeScorer()
        scored = scorer.score_categories(
            _dummy_image(), "p", list(_GOLDEN_CATEGORIES)
        )
        got = {lbl: (lp, norm, n) for lbl, lp, norm, n in scored}
        assert set(got) == set(_GOLDEN_SCORE_CATEGORIES)
        for lbl, (lp, norm, n) in _GOLDEN_SCORE_CATEGORIES.items():
            g_lp, g_norm, g_n = got[lbl]
            assert g_n == n
            if math.isinf(lp):
                assert math.isinf(g_lp) and math.isinf(g_norm)
            else:
                assert g_lp == pytest.approx(lp, abs=1e-6)
                assert g_norm == pytest.approx(norm, abs=1e-6)

    def test_sorted_descending_by_norm(self) -> None:
        scorer = ChainFakeScorer()
        scored = scorer.score_categories(
            _dummy_image(), "p", list(_GOLDEN_CATEGORIES)
        )
        norms = [norm for _, _, norm, _ in scored]
        assert norms == sorted(norms, reverse=True)

    def test_single_token_label_uses_prefix_logits_only(self) -> None:
        """Single-token label: score comes straight from the prefix pass,
        no continuation call -- as before the refactor."""
        scorer = ChainFakeScorer()
        scored = scorer.score_categories(_dummy_image(), "p", ["a"])
        assert len(scorer._model.calls) == 1
        assert scored[0][1] == pytest.approx(
            scorer._model.logprob(_PREFIX_IDS, _CHAR_TO_ID["a"]), abs=1e-9
        )

    def test_one_prefix_pass_shared_by_all_labels(self) -> None:
        """One prefix forward + one continuation per multi-token label."""
        scorer = ChainFakeScorer()
        scorer.score_categories(_dummy_image(), "p", ["cat", "a", "dog"])
        calls = scorer._model.calls
        assert calls[0]["ids"] == _PREFIX_IDS      # single prefix pass
        assert len(calls) == 3                      # + 2 multi-token labels
        assert [c["ids"] for c in calls[1:]] == [
            scorer.tokenizer.encode("ca"),
            scorer.tokenizer.encode("do"),
        ]

    def test_tensor_wrapper_order_preserved(self) -> None:
        scorer = ChainFakeScorer()
        cats = _GOLDEN_TENSOR_CATEGORIES
        tensor = scorer.score_categories_tensor(_dummy_image(), "p", cats)
        assert tensor.shape == (len(cats),)
        assert tensor.tolist() == pytest.approx(_GOLDEN_TENSOR, abs=1e-6)
        # tensor[i] is the norm log-prob of cats[i], not of the sorted order.
        # Tolerance is FP32-sized: the wrapper packs python floats into a
        # float32 tensor, so .item() round-trips through ~1e-7 relative.
        by_label = {
            lbl: norm
            for lbl, _, norm, _ in scorer.score_categories(
                _dummy_image(), "p", cats
            )
        }
        for i, cat in enumerate(cats):
            assert tensor[i].item() == pytest.approx(by_label[cat], abs=1e-5)
