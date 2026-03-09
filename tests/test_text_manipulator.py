"""Tests for the text manipulator types, selection, and genotype logic.

Tests exercise pure data structures and functions without requiring
a full embedding model. Embedding-dependent tests use a small
synthetic KeyedVectors.
"""

import numpy as np
import pytest

from src.manipulator.text.types import (
    CONTENT_POS_TAGS,
    ManipulationContext,
    TokenSequence,
    WordSelection,
    _match_case,
)
from src.manipulator.text.selection import (
    select_content_words,
    find_synonym_candidates,
    build_word_selection,
)
from src.manipulator.text.manipulator import apply_genotype


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tokens(*items: tuple[str, str, str]) -> TokenSequence:
    """Build a TokenSequence from (word, pos, whitespace) triples."""
    return TokenSequence(
        tokens=tuple(w for w, _, _ in items),
        pos_tags=tuple(p for _, p, _ in items),
        whitespace=tuple(s for _, _, s in items),
    )


def _make_embeddings(vocab: dict[str, list[float]]):
    """Build a small gensim KeyedVectors from a dict."""
    from gensim.models import KeyedVectors

    dim = len(next(iter(vocab.values())))
    kv = KeyedVectors(vector_size=dim)
    words = list(vocab.keys())
    vectors = np.array(list(vocab.values()), dtype=np.float32)
    kv.add_vectors(words, vectors)
    return kv


# ---------------------------------------------------------------------------
# TokenSequence
# ---------------------------------------------------------------------------


class TestTokenSequence:
    def test_text_reconstruction(self):
        ts = _make_tokens(
            ("The", "DET", " "),
            ("quick", "ADJ", " "),
            ("fox", "NOUN", "."),
        )
        assert ts.text == "The quick fox."

    def test_replace_preserves_case(self):
        ts = _make_tokens(
            ("The", "DET", " "),
            ("Quick", "ADJ", " "),
            ("FOX", "NOUN", ""),
        )
        replaced = ts.replace(
            np.array([1, 2], dtype=np.intp),
            ("fast", "dog"),
        )
        assert replaced.tokens[1] == "Fast"  # capitalized
        assert replaced.tokens[2] == "DOG"   # uppercased

    def test_replace_preserves_whitespace(self):
        ts = _make_tokens(("hello", "NOUN", " "), ("world", "NOUN", "!"))
        replaced = ts.replace(np.array([0], dtype=np.intp), ("greetings",))
        assert replaced.text == "greetings world!"

    def test_validation_pos_length(self):
        with pytest.raises(ValueError, match="pos_tags"):
            TokenSequence(("a",), ("NOUN", "VERB"), (" ",))

    def test_validation_whitespace_length(self):
        with pytest.raises(ValueError, match="whitespace"):
            TokenSequence(("a",), ("NOUN",), (" ", " "))


# ---------------------------------------------------------------------------
# Case matching
# ---------------------------------------------------------------------------


class TestMatchCase:
    def test_lowercase(self):
        assert _match_case("hello", "WORLD") == "world"

    def test_capitalized(self):
        assert _match_case("Hello", "world") == "World"

    def test_uppercase(self):
        assert _match_case("HELLO", "world") == "WORLD"


# ---------------------------------------------------------------------------
# WordSelection
# ---------------------------------------------------------------------------


class TestWordSelection:
    def test_gene_bounds(self):
        sel = WordSelection(
            positions=np.array([0, 2], dtype=np.intp),
            candidates=(("a", "b"), ("c",)),
            original_words=("x", "y"),
        )
        np.testing.assert_array_equal(sel.gene_bounds, [3, 2])

    def test_validation_candidates_length(self):
        with pytest.raises(ValueError, match="candidates"):
            WordSelection(
                positions=np.array([0, 1], dtype=np.intp),
                candidates=(("a",),),  # 1, not 2
                original_words=("x", "y"),
            )

    def test_validation_original_words_length(self):
        with pytest.raises(ValueError, match="original_words"):
            WordSelection(
                positions=np.array([0, 1], dtype=np.intp),
                candidates=(("a",), ("b",)),
                original_words=("x",),  # 1, not 2
            )


# ---------------------------------------------------------------------------
# Content word selection
# ---------------------------------------------------------------------------


class TestSelectContentWords:
    def setup_method(self):
        self.embeddings = _make_embeddings({
            "quick": [1, 0, 0],
            "fox": [0, 1, 0],
            "jumps": [0, 0, 1],
            "lazy": [1, 1, 0],
            "dog": [0, 1, 1],
        })

    def test_selects_content_words(self):
        ts = _make_tokens(
            ("The", "DET", " "),
            ("quick", "ADJ", " "),
            ("fox", "NOUN", " "),
            ("jumps", "VERB", ""),
        )
        pos = select_content_words(ts, self.embeddings)
        np.testing.assert_array_equal(pos, [1, 2, 3])

    def test_skips_oov_words(self):
        ts = _make_tokens(
            ("xylophone", "NOUN", " "),  # not in embeddings
            ("fox", "NOUN", ""),
        )
        pos = select_content_words(ts, self.embeddings)
        np.testing.assert_array_equal(pos, [1])

    def test_skips_function_words(self):
        ts = _make_tokens(
            ("the", "DET", " "),
            ("in", "ADP", " "),
            ("and", "CCONJ", ""),
        )
        pos = select_content_words(ts, self.embeddings)
        assert len(pos) == 0


# ---------------------------------------------------------------------------
# Synonym candidates
# ---------------------------------------------------------------------------


class TestFindSynonymCandidates:
    def setup_method(self):
        self.embeddings = _make_embeddings({
            "dog": [1.0, 0.0, 0.0],
            "puppy": [0.95, 0.05, 0.0],   # very close to dog
            "cat": [0.7, 0.3, 0.0],        # moderate
            "car": [0.0, 0.0, 1.0],        # far
        })

    def test_nearest_first(self):
        cands = find_synonym_candidates("dog", self.embeddings, k=3)
        assert cands[0] == "puppy"  # closest
        assert len(cands) == 3

    def test_oov_returns_empty(self):
        cands = find_synonym_candidates("xyzzy", self.embeddings, k=5)
        assert cands == ()

    def test_k_clamped(self):
        cands = find_synonym_candidates("dog", self.embeddings, k=100)
        assert len(cands) == 3  # only 3 other words


# ---------------------------------------------------------------------------
# Apply genotype
# ---------------------------------------------------------------------------


class TestApplyGenotype:
    def setup_method(self):
        self.tokens = _make_tokens(
            ("The", "DET", " "),
            ("quick", "ADJ", " "),
            ("brown", "ADJ", " "),
            ("fox", "NOUN", ""),
        )
        self.selection = WordSelection(
            positions=np.array([1, 3], dtype=np.intp),
            candidates=(("fast", "rapid"), ("wolf", "dog", "cat")),
            original_words=("quick", "fox"),
        )

    def test_zero_genotype_preserves(self):
        g = np.array([0, 0], dtype=np.int64)
        result = apply_genotype(self.tokens, self.selection, g)
        assert result.text == self.tokens.text

    def test_single_mutation(self):
        g = np.array([1, 0], dtype=np.int64)
        result = apply_genotype(self.tokens, self.selection, g)
        assert result.text == "The fast brown fox"

    def test_full_mutation(self):
        g = np.array([2, 3], dtype=np.int64)
        result = apply_genotype(self.tokens, self.selection, g)
        assert result.text == "The rapid brown cat"

    def test_case_preserved(self):
        tokens = _make_tokens(
            ("Quick", "ADJ", " "),
            ("fox", "NOUN", ""),
        )
        sel = WordSelection(
            positions=np.array([0], dtype=np.intp),
            candidates=(("fast",),),
            original_words=("Quick",),
        )
        g = np.array([1], dtype=np.int64)
        result = apply_genotype(tokens, sel, g)
        assert result.tokens[0] == "Fast"

    def test_original_unchanged(self):
        g = np.array([1, 1], dtype=np.int64)
        apply_genotype(self.tokens, self.selection, g)
        assert self.tokens.tokens[1] == "quick"

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="Genotype length"):
            apply_genotype(self.tokens, self.selection, np.array([1], dtype=np.int64))


# ---------------------------------------------------------------------------
# ManipulationContext
# ---------------------------------------------------------------------------


class TestManipulationContext:
    def test_genotype_properties(self):
        sel = WordSelection(
            positions=np.array([0, 1, 2], dtype=np.intp),
            candidates=(("a", "b"), ("c",), ("d", "e", "f")),
            original_words=("x", "y", "z"),
        )
        tokens = _make_tokens(
            ("x", "NOUN", " "),
            ("y", "VERB", " "),
            ("z", "ADJ", ""),
        )
        ctx = ManipulationContext(original_tokens=tokens, selection=sel)

        assert ctx.genotype_dim == 3
        np.testing.assert_array_equal(ctx.gene_bounds, [3, 2, 4])

    def test_zero_genotype(self):
        sel = WordSelection(
            positions=np.array([0], dtype=np.intp),
            candidates=(("a",),),
            original_words=("x",),
        )
        tokens = _make_tokens(("x", "NOUN", ""),)
        ctx = ManipulationContext(original_tokens=tokens, selection=sel)

        g = ctx.zero_genotype()
        assert len(g) == 1
        assert g[0] == 0

    def test_random_genotype_within_bounds(self):
        sel = WordSelection(
            positions=np.array([0, 1], dtype=np.intp),
            candidates=(("a", "b", "c"), ("d", "e")),
            original_words=("x", "y"),
        )
        tokens = _make_tokens(("x", "NOUN", " "), ("y", "VERB", ""))
        ctx = ManipulationContext(original_tokens=tokens, selection=sel)

        rng = np.random.default_rng(42)
        for _ in range(100):
            g = ctx.random_genotype(rng)
            assert 0 <= g[0] < 4  # 3 candidates + 1
            assert 0 <= g[1] < 3  # 2 candidates + 1


# ---------------------------------------------------------------------------
# build_word_selection integration
# ---------------------------------------------------------------------------


class TestBuildWordSelection:
    def test_end_to_end(self):
        embeddings = _make_embeddings({
            "quick": [1.0, 0.0, 0.0],
            "fast": [0.95, 0.05, 0.0],
            "rapid": [0.9, 0.1, 0.0],
            "fox": [0.0, 1.0, 0.0],
            "wolf": [0.0, 0.9, 0.1],
            "dog": [0.0, 0.8, 0.2],
        })
        tokens = _make_tokens(
            ("The", "DET", " "),
            ("quick", "ADJ", " "),
            ("fox", "NOUN", ""),
        )
        sel = build_word_selection(tokens, embeddings, n_candidates=2)

        assert sel.n_words == 2
        assert sel.original_words == ("quick", "fox")
        assert sel.candidates[0][0] == "fast"  # nearest to quick
        assert sel.candidates[1][0] == "wolf"  # nearest to fox
