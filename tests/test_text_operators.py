"""Tests for the composite text-manipulator + each operator.

Surface operators (Fragmentation, Character Noise, Saliency) run with
real handcrafted TokenSequences — no spaCy load needed since their
eligibility filters use the PoS tags already on the TokenSequence.

The MLM-based SynonymOperator is exercised via a stub MLM injected
through ``_set_resources``; the real ModernBERT is exercised in a
separate smoke test, not in pytest.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_tokens

from src.manipulator.text.composite import (
    CANONICAL_ORDER,
    CompositeTextManipulator,
)
from src.manipulator.text.operators.base import (
    deterministic_word_rng,
    severity_to_k_max,
)
from src.manipulator.text.operators.character_noise import (
    HOMOGLYPHS,
    ZERO_WIDTH,
    CharacterNoiseOperator,
)
from src.manipulator.text.operators.fragmentation import FragmentationOperator
from src.manipulator.text.operators.saliency import SaliencyOperator
from src.manipulator.text.operators.synonym import (
    NEGATION_PREFIX_RE,
    SynonymOperator,
)
from src.manipulator.text.profiles import (
    OperatorSpec,
    TextProfile,
    build_operators_from_specs,
    load_profile_library,
    resolve_profile,
)
from src.manipulator.text.types import TokenSequence


def _tokens_quick_brown_fox() -> TokenSequence:
    return make_tokens(
        ("The", "DET", " "),
        ("quick", "ADJ", " "),
        ("brown", "ADJ", " "),
        ("fox", "NOUN", ""),
    )


# ===========================================================================
# severity_to_k_max
# ===========================================================================


class TestSeverityToKmax:
    def test_zero_returns_zero(self):
        assert severity_to_k_max(0.0) == 0

    def test_full_returns_five(self):
        assert severity_to_k_max(1.0) == 5

    def test_quarter(self):
        assert severity_to_k_max(0.25) == 2

    def test_half(self):
        assert severity_to_k_max(0.5) == 3

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            severity_to_k_max(-0.1)
        with pytest.raises(ValueError):
            severity_to_k_max(1.1)

    def test_override_bypasses_formula(self):
        assert severity_to_k_max(0.55, override=25) == 25
        assert severity_to_k_max(0.0, override=10) == 10

    def test_override_clipped_at_hard_cap(self):
        assert severity_to_k_max(0.5, override=10_000) == 1024
        assert severity_to_k_max(0.5, override=10_000, hard_cap=50) == 50

    def test_override_negative_raises(self):
        with pytest.raises(ValueError, match="k_max override must be"):
            severity_to_k_max(0.5, override=-1)


# ===========================================================================
# deterministic_word_rng
# ===========================================================================


class TestDeterministicRng:
    def test_same_inputs_same_stream(self):
        a = deterministic_word_rng(42, "fox", 1)
        b = deterministic_word_rng(42, "fox", 1)
        assert a.integers(0, 1_000_000) == b.integers(0, 1_000_000)

    def test_different_word_different_stream(self):
        a = deterministic_word_rng(42, "fox", 1)
        b = deterministic_word_rng(42, "wolf", 1)
        assert a.integers(0, 1_000_000) != b.integers(0, 1_000_000)


# ===========================================================================
# FragmentationOperator
# ===========================================================================


class TestFragmentationOperator:
    def setup_method(self):
        self.tokens = _tokens_quick_brown_fox()

    def test_severity_zero_disables(self):
        op = FragmentationOperator(severity=0.0)
        ctx = op.prepare(self.tokens)
        assert op.gene_dim(ctx) == 0

    def test_eligibility_min_word_len(self):
        op = FragmentationOperator(severity=0.5)
        ctx = op.prepare(self.tokens)
        eligible_words = [self.tokens.tokens[int(p)] for p in ctx.positions]
        assert "quick" in eligible_words
        assert "brown" in eligible_words
        assert "fox" not in eligible_words  # len=3 < 4

    def test_zero_genes_no_op(self):
        op = FragmentationOperator(severity=0.5)
        ctx = op.prepare(self.tokens)
        genes = np.zeros(ctx.n_positions, dtype=np.int64)
        out = op.apply(ctx, genes, self.tokens)
        assert out.text == self.tokens.text

    def test_inserts_spaces(self):
        op = FragmentationOperator(severity=1.0)
        ctx = op.prepare(self.tokens)
        genes = np.full(ctx.n_positions, 2, dtype=np.int64)
        out = op.apply(ctx, genes, self.tokens)
        assert len(out.text) > len(self.tokens.text)
        assert "q" in out.text and "k" in out.text

    def test_deterministic(self):
        op = FragmentationOperator(severity=0.5)
        ctx = op.prepare(self.tokens)
        genes = np.array([1] * ctx.n_positions, dtype=np.int64)
        a = op.apply(ctx, genes, self.tokens).text
        b = op.apply(ctx, genes, self.tokens).text
        assert a == b


# ===========================================================================
# CharacterNoiseOperator
# ===========================================================================


class TestCharacterNoiseOperator:
    def setup_method(self):
        self.tokens = _tokens_quick_brown_fox()

    def test_severity_zero_disables(self):
        op = CharacterNoiseOperator(severity=0.0)
        ctx = op.prepare(self.tokens)
        assert op.gene_dim(ctx) == 0

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="unknown character_noise mode"):
            CharacterNoiseOperator(severity=0.5, modes=("not_a_mode",))

    def test_empty_modes_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            CharacterNoiseOperator(severity=0.5, modes=())

    def test_eligibility_alpha_only(self):
        op = CharacterNoiseOperator(severity=0.5)
        ctx = op.prepare(self.tokens)
        assert ctx.n_positions == 4

    def test_zero_genes_no_op(self):
        op = CharacterNoiseOperator(severity=0.5)
        ctx = op.prepare(self.tokens)
        genes = np.zeros(ctx.n_positions, dtype=np.int64)
        out = op.apply(ctx, genes, self.tokens)
        assert out.text == self.tokens.text

    def test_homoglyph_mode_replaces_chars(self):
        op = CharacterNoiseOperator(severity=1.0, modes=("homoglyph",))
        ctx = op.prepare(self.tokens)
        genes = np.full(ctx.n_positions, 5, dtype=np.int64)
        out = op.apply(ctx, genes, self.tokens)
        homoglyph_chars = set(HOMOGLYPHS.values())
        assert any(c in homoglyph_chars for c in out.text)

    def test_zero_width_mode_inserts_chars(self):
        op = CharacterNoiseOperator(severity=1.0, modes=("zero_width",))
        ctx = op.prepare(self.tokens)
        genes = np.full(ctx.n_positions, 5, dtype=np.int64)
        out = op.apply(ctx, genes, self.tokens)
        zw_chars = set(ZERO_WIDTH)
        assert any(c in zw_chars for c in out.text)

    def test_deterministic(self):
        op = CharacterNoiseOperator(severity=0.5)
        ctx = op.prepare(self.tokens)
        genes = np.array([1] * ctx.n_positions, dtype=np.int64)
        a = op.apply(ctx, genes, self.tokens).text
        b = op.apply(ctx, genes, self.tokens).text
        assert a == b


# ===========================================================================
# SaliencyOperator
# ===========================================================================


class TestSaliencyOperator:
    def setup_method(self):
        self.tokens = _tokens_quick_brown_fox()

    def test_severity_zero_disables(self):
        op = SaliencyOperator(severity=0.0)
        ctx = op.prepare(self.tokens)
        assert op.gene_dim(ctx) == 0

    def test_skips_function_words(self):
        op = SaliencyOperator(severity=0.5)
        ctx = op.prepare(self.tokens)
        eligible = [self.tokens.tokens[int(p)] for p in ctx.positions]
        assert "The" not in eligible  # DET → filtered
        assert "quick" in eligible
        assert "fox" in eligible

    def test_zero_genes_no_op(self):
        op = SaliencyOperator(severity=0.5)
        ctx = op.prepare(self.tokens)
        genes = np.zeros(ctx.n_positions, dtype=np.int64)
        out = op.apply(ctx, genes, self.tokens)
        assert out.text == self.tokens.text

    def test_corruption_changes_text(self):
        op = SaliencyOperator(severity=1.0)
        ctx = op.prepare(self.tokens)
        assert ctx.n_positions > 0
        genes = np.full(ctx.n_positions, 5, dtype=np.int64)
        out = op.apply(ctx, genes, self.tokens)
        assert out.text != self.tokens.text

    def test_first_char_preserved(self):
        op = SaliencyOperator(severity=1.0)
        ctx = op.prepare(self.tokens)
        genes = np.full(ctx.n_positions, 5, dtype=np.int64)
        out = op.apply(ctx, genes, self.tokens)
        for p in ctx.positions:
            original = self.tokens.tokens[int(p)]
            mutated = out.tokens[int(p)]
            assert mutated[0] == original[0]

    def test_short_words_skipped(self):
        tokens = make_tokens(
            ("a", "ADJ", " "),
            ("quick", "ADJ", " "),
            ("fox", "NOUN", ""),
        )
        op = SaliencyOperator(severity=1.0)
        ctx = op.prepare(tokens)
        eligible = [tokens.tokens[int(p)] for p in ctx.positions]
        assert "a" not in eligible

    def test_deterministic(self):
        op = SaliencyOperator(severity=0.5)
        ctx = op.prepare(self.tokens)
        genes = np.array([1] * ctx.n_positions, dtype=np.int64)
        a = op.apply(ctx, genes, self.tokens).text
        b = op.apply(ctx, genes, self.tokens).text
        assert a == b


# ===========================================================================
# Profile loader + builder
# ===========================================================================


class TestProfileLoader:
    def test_load_actual_library(self):
        from pathlib import Path
        lib_path = (
            Path(__file__).resolve().parent.parent
            / "configs" / "templates" / "text_profiles.yaml"
        )
        library = load_profile_library(lib_path)
        assert "full_stack" in library
        assert "domain_expert" in library

    def test_resolve_profile_returns_specs(self):
        lib = {
            "p": TextProfile(
                name="p",
                operators=(
                    OperatorSpec(name="saliency", severity=0.25),
                    OperatorSpec(name="synonym", severity=0.10),
                ),
            ),
        }
        specs = resolve_profile(lib, profile_name="p")
        assert len(specs) == 2

    def test_resolve_with_overrides(self):
        lib = {
            "p": TextProfile(
                name="p",
                operators=(
                    OperatorSpec(name="saliency", severity=0.25),
                    OperatorSpec(name="synonym", severity=0.10),
                ),
            ),
        }
        specs = resolve_profile(
            lib,
            profile_name="p",
            overrides={"saliency": {"severity": 0.50}},
        )
        sal = next(s for s in specs if s.name == "saliency")
        syn = next(s for s in specs if s.name == "synonym")
        assert sal.severity == 0.50
        assert syn.severity == 0.10

    def test_resolve_explicit_operators(self):
        specs = resolve_profile(
            library={},
            profile_name=None,
            operators=[{"name": "fragmentation", "severity": 0.30}],
        )
        assert len(specs) == 1
        assert specs[0].name == "fragmentation"

    def test_unknown_profile_raises(self):
        with pytest.raises(KeyError, match="not in library"):
            resolve_profile({}, profile_name="nonexistent")

    def test_neither_profile_nor_operators_raises(self):
        with pytest.raises(ValueError):
            resolve_profile({}, profile_name=None, operators=None)


class TestBuildOperators:
    def test_severity_zero_skipped(self):
        specs = (
            OperatorSpec(name="synonym", severity=0.0),
            OperatorSpec(name="saliency", severity=0.25),
        )
        ops = build_operators_from_specs(specs)
        assert len(ops) == 1
        assert ops[0].name == "saliency"

    def test_unknown_operator_raises(self):
        specs = (OperatorSpec(name="bogus", severity=0.5),)
        with pytest.raises(ValueError, match="unknown text operator"):
            build_operators_from_specs(specs)

    def test_character_noise_modes_extracted(self):
        specs = (
            OperatorSpec(
                name="character_noise",
                severity=0.5,
                extras={"modes": ["homoglyph", "zero_width"]},
            ),
        )
        ops = build_operators_from_specs(specs)
        assert ops[0].modes == ("homoglyph", "zero_width")

    def test_synonym_constructed_lazy_no_load(self):
        specs = (OperatorSpec(name="synonym", severity=0.5),)
        ops = build_operators_from_specs(specs)
        assert len(ops) == 1
        assert ops[0].name == "synonym"
        # MLM model is lazy — not loaded yet
        assert ops[0]._model is None

    def test_k_max_override_propagates(self):
        specs = (
            OperatorSpec(
                name="synonym",
                severity=0.55,
                extras={"k_max": 25},
            ),
        )
        ops = build_operators_from_specs(specs)
        assert ops[0].k_max == 25

    def test_k_max_override_works_for_surface_operators(self):
        specs = (
            OperatorSpec(name="saliency", severity=0.25, extras={"k_max": 8}),
            OperatorSpec(name="fragmentation", severity=0.10, extras={"k_max": 4}),
            OperatorSpec(
                name="character_noise",
                severity=0.20,
                extras={"k_max": 6, "modes": ["homoglyph"]},
            ),
        )
        ops = build_operators_from_specs(specs)
        kmaxes = {op.name: op.k_max for op in ops}
        assert kmaxes == {"saliency": 8, "fragmentation": 4, "character_noise": 6}


# ===========================================================================
# SynonymOperator (MLM, stubbed)
# ===========================================================================


class _StubTokenizer:
    """Minimal HF-tokenizer stub for MLM-mask scoring."""

    def __init__(self, mask_token="[MASK]", mask_token_id=99, vocab=None):
        self.mask_token = mask_token
        self.mask_token_id = mask_token_id
        self._vocab = vocab or []

    def __call__(self, text, return_tensors="pt"):
        import torch
        words = text.split()
        ids = [self.mask_token_id if w == self.mask_token else 0 for w in words]
        input_ids = torch.tensor([ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        class _Inputs:
            def __init__(self, ids, mask):
                self.input_ids = ids
                self.attention_mask = mask

            def to(self, device):
                return self

            def __iter__(self):
                return iter(["input_ids", "attention_mask"])

            def __getitem__(self, k):
                return getattr(self, k)

            def keys(self):
                return ["input_ids", "attention_mask"]

        return _Inputs(input_ids, attention_mask)

    def convert_ids_to_tokens(self, ids):
        return [self._vocab[i] if i < len(self._vocab) else f"tok_{i}" for i in ids]


class _StubMLM:
    def __init__(self, ranked_token_ids):
        import torch
        self._ranked = list(ranked_token_ids)
        vocab_size = max(self._ranked) + 1 if self._ranked else 1
        scores = torch.full((vocab_size,), -1e9)
        for rank, tid in enumerate(self._ranked):
            scores[tid] = float(len(self._ranked) - rank)
        self._scores = scores

    def __call__(self, **kwargs):
        import torch
        input_ids = kwargs["input_ids"]
        seq_len = input_ids.shape[1]
        vocab = self._scores.shape[0]
        logits = self._scores.unsqueeze(0).unsqueeze(0).expand(1, seq_len, vocab).clone()

        class _Out:
            pass
        out = _Out()
        out.logits = logits
        return out

    def to(self, device):
        return self

    def eval(self):
        return self


class TestSynonymOperator:
    """MLM-Synonym tests via stubbed model. Real ModernBERT is exercised
    in a separate smoke script, not in pytest."""

    def setup_method(self):
        import spacy
        # vocab[i] = token-string at id i; ranking = priority of MLM logits
        vocab = ["[PAD]", "key", "running", "non-main", "quickly", "main", "principal"]
        ranked = [1, 2, 3, 4, 5, 6]
        self.tokenizer = _StubTokenizer(mask_token="[MASK]", mask_token_id=99, vocab=vocab)
        self.model = _StubMLM(ranked_token_ids=ranked)
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        self.op = SynonymOperator(
            severity=1.0, k_max_override=10, topk_pre_filter=len(ranked),
        )
        self.op._set_resources(self.model, self.tokenizer, self.nlp)

    def test_severity_zero_disables(self):
        op = SynonymOperator(severity=0.0)
        ts = make_tokens(("the", "DET", " "), ("main", "ADJ", ""))
        ctx = op.prepare(ts)
        assert ctx.n_positions == 0
        assert op.gene_dim(ctx) == 0

    def test_negation_filter_rejects_non_prefixed(self):
        ts = make_tokens(
            ("What", "PRON", " "),
            ("is", "AUX", " "),
            ("the", "DET", " "),
            ("main", "ADJ", " "),
            ("subject", "NOUN", ""),
        )
        ctx = self.op.prepare(ts)
        flat = [c for cands in ctx.candidates for c in cands]
        assert "non-main" not in flat

    def test_apply_zero_genes_no_op(self):
        ts = make_tokens(
            ("What", "PRON", " "),
            ("is", "AUX", " "),
            ("the", "DET", " "),
            ("main", "ADJ", " "),
            ("subject", "NOUN", ""),
        )
        ctx = self.op.prepare(ts)
        if ctx.n_positions == 0:
            pytest.skip("stub MLM produced no surviving candidates")
        genes = np.zeros(ctx.n_positions, dtype=np.int64)
        assert self.op.apply(ctx, genes, ts).text == ts.text

    def test_negation_regex_pattern(self):
        should_match = (
            "non-main", "non_main", "un-easy", "in-correct", "im-pressed",
            "de-activate", "dis-able", "anti-pattern", "counter-act",
        )
        should_not_match = (
            "main", "subject", "image", "key", "primary", "topic",
            "incident", "imply", "unease", "deduce", "antibody", "nonmain",
        )
        for s in should_match:
            assert NEGATION_PREFIX_RE.match(s) is not None, f"should match: {s}"
        for s in should_not_match:
            assert NEGATION_PREFIX_RE.match(s) is None, f"should NOT match: {s}"

    def test_k_max_override(self):
        op = SynonymOperator(severity=0.5, k_max_override=42)
        assert op.k_max == 42

    def test_default_k_max_from_severity(self):
        op = SynonymOperator(severity=0.5)
        assert op.k_max == 3


# ===========================================================================
# CompositeTextManipulator orchestration (stubbed nlp, no MLM load)
# ===========================================================================


class _StubNLP:
    """Minimal spaCy-API stub matching tokenize() expectations."""

    def __call__(self, text):
        det_set = {"the", "a", "an"}
        adj_set = {"quick", "brown", "rapid", "fast", "lazy"}
        items = []
        for chunk in text.split(" "):
            if not chunk:
                continue
            stripped = chunk.rstrip(".!?,;:")
            trailing_punct = chunk[len(stripped):]
            if stripped.lower() in det_set:
                pos = "DET"
            elif stripped.lower() in adj_set:
                pos = "ADJ"
            else:
                pos = "NOUN"
            items.append((stripped, pos, " "))
            if trailing_punct:
                items.append((trailing_punct, "PUNCT", " "))
        if items:
            last_token, last_pos, _ = items[-1]
            items[-1] = (last_token, last_pos, "")

        class _Tok:
            def __init__(self, text, pos, ws):
                self.text = text
                self.pos_ = pos
                self.whitespace_ = ws

        class _Doc:
            def __init__(self, toks):
                self._toks = toks

            def __iter__(self):
                return iter(self._toks)

        return _Doc([_Tok(t, p, w) for t, p, w in items])


class TestCompositeTextManipulator:
    def setup_method(self):
        self.nlp = _StubNLP()

    def _build(self):
        return CompositeTextManipulator(
            self.nlp,
            operators=[
                FragmentationOperator(severity=0.5),
                SaliencyOperator(severity=0.5),
                CharacterNoiseOperator(severity=0.5),
            ],
        )

    def test_canonical_order_enforced(self):
        cm = CompositeTextManipulator(
            self.nlp,
            operators=[
                SaliencyOperator(severity=0.25),
                FragmentationOperator(severity=0.25),
                CharacterNoiseOperator(severity=0.25),
            ],
        )
        # synonym slot first (absent here), then frag, char, sal in canonical order
        assert cm.operator_names == (
            "fragmentation", "character_noise", "saliency",
        )

    def test_zero_genotype_returns_original(self):
        cm = self._build()
        ctx = cm.prepare("The quick brown fox")
        genome = ctx.zero_genotype()
        assert cm.apply(ctx, genome) == ctx.original_tokens.text

    def test_genotype_dim_is_sum(self):
        cm = self._build()
        ctx = cm.prepare("The quick brown fox")
        assert ctx.genotype_dim == sum(ctx.op_gene_dims)

    def test_gene_bounds_concatenated(self):
        cm = self._build()
        ctx = cm.prepare("The quick brown fox")
        bounds = ctx.gene_bounds
        assert bounds.dtype == np.int64
        assert len(bounds) == ctx.genotype_dim

    def test_random_genotype_within_bounds(self):
        cm = self._build()
        ctx = cm.prepare("The quick brown fox")
        if ctx.genotype_dim == 0:
            pytest.skip("no eligible positions")
        rng = np.random.default_rng(42)
        for _ in range(50):
            g = ctx.random_genotype(rng)
            assert (g >= 0).all()
            assert (g < ctx.gene_bounds).all()

    def test_genotype_length_mismatch_raises(self):
        cm = self._build()
        ctx = cm.prepare("The quick brown fox")
        with pytest.raises(ValueError, match="genotype length"):
            cm.apply(ctx, np.zeros(ctx.genotype_dim + 1, dtype=np.int64))


# ===========================================================================
# Canonical order
# ===========================================================================


class TestCanonicalOrder:
    def test_canonical_order(self):
        assert CANONICAL_ORDER == (
            "synonym",
            "fragmentation",
            "character_noise",
            "saliency",
        )
