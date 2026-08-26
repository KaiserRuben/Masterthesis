"""Tests for the Exp-105 slot-scan runner (``run_slot_scan.py``).

Two layers, neither of which loads a model or touches the network:

1. **Planning** — the shipped ``configs/Exp-105/*_scan_*.yaml`` drafts are
   parsed and expanded (grid product, candidate sets, chain enumeration),
   which doubles as a schema check on those configs.
2. **Execution** — the scoring / chain / free-generation loops run against
   a ``FakeScanScorer`` wired into a ``VLMSUT`` subclass, exactly like
   ``tests/test_vlm_sut.py`` does.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from experiments.runners.run_slot_scan import (
    NULL_IMAGE_ID,
    CandidateSet,
    HexGridSpec,
    ScanInput,
    ScanPlan,
    build_plan,
    chain_text,
    execute_scan,
    format_plan,
    materialise_candidate_sets,
    run_scan,
    template_parts,
)
from src.config import ExperimentConfig, PMIConfig, SUTConfig
from src.sut.scorer import VLMScorer

from tests.test_hex_grid import StubTokenizer

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "Exp-105"


def _load_cfg(filename: str) -> dict:
    with open(_CONFIG_DIR / filename) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _is_null_image(image) -> bool:
    """The PMI gray null image — uniform 128 (see VLMSUT._null_image)."""
    return bool((np.asarray(image) == 128).all())


class FakeScanScorer(VLMScorer):
    """Deterministic scorer: log-probs are a function of the inputs only.

    ``lp_norm = -0.1 * len(candidate) - 0.01 * len(prompt) - 1.0 [null]``

    The constant null-image offset makes the recorded Δ∅ baseline column
    checkable: for a real image, ``lp_norm - lp_norm_null == 1.0``.
    """

    def __init__(self, tokenizer=None) -> None:
        self._device = torch.device("cpu")
        self._enable_thinking = False
        self._max_thinking_tokens = 0
        self._tok = tokenizer if tokenizer is not None else StubTokenizer()
        self.generate_calls: list[tuple[str, bool]] = []
        self.chain_calls: list[tuple[str, tuple]] = []

    # -- plumbing the ABC requires -------------------------------------
    @property
    def tokenizer(self):  # type: ignore[override]
        return self._tok

    def _prepare_inputs(self, image, prompt, enable_thinking):  # type: ignore[override]
        raise NotImplementedError("FakeScanScorer never builds real inputs")

    def encode_text(self, texts):  # type: ignore[override]
        return np.zeros((len(texts), 1), dtype=np.float32)

    # -- scoring --------------------------------------------------------
    def _lp(self, image, prompt: str, text: str) -> float:
        return (
            -0.1 * len(text)
            - 0.01 * len(prompt)
            - (1.0 if _is_null_image(image) else 0.0)
        )

    def score_categories(self, image, prompt, categories, thinking_ids=None):  # type: ignore[override]
        scored = [
            (c, self._lp(image, prompt, c) * 2, self._lp(image, prompt, c), 2)
            for c in categories
        ]
        return sorted(scored, key=lambda x: x[2], reverse=True)

    def score_chain_slots(self, image, prompt, parts):  # type: ignore[override]
        self.chain_calls.append((prompt, tuple(parts)))
        out = {}
        for part in parts:
            if isinstance(part, str):
                continue
            name, filler = part
            n = max(1, len(filler.split()))
            total = self._lp(image, prompt, filler) * n
            out[name] = (total, total / n, n)
        return out

    def generate(self, image, prompt):  # type: ignore[override]
        self.generate_calls.append((prompt, _is_null_image(image)))
        return f"I see something. [{prompt[:12]}]", None, None


def make_fake_sut(
    scorer: FakeScanScorer | None = None,
    pmi: PMIConfig | None = None,
) -> "VLMSUT":  # noqa: F821
    """VLMSUT with a fake scorer injected (same pattern as test_vlm_sut)."""
    from src.sut.vlm_sut import VLMSUT

    inner = scorer if scorer is not None else FakeScanScorer()
    cfg = ExperimentConfig(
        name="fake_scan",
        sut=SUTConfig(model_id="fake/model"),
        pmi=pmi if pmi is not None else PMIConfig(null_image_size=8),
    )

    class FakeSUT(VLMSUT):
        def __init__(self) -> None:
            self._config = cfg
            self._device = torch.device("cpu")
            self._scorer = inner
            self._prompt = "unused"
            self._redis = None
            self._cache_hits = 0
            self._cache_misses = 0
            self._last_call_cached = False
            self._text_embedder = None

    return FakeSUT()


def _write_image(path: Path, colour=(200, 30, 30)) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), colour).save(path)
    return path


# ===========================================================================
# Planning — grid scan (step 0)
# ===========================================================================


class TestGridPlan:
    def setup_method(self) -> None:
        self.cfg = _load_cfg("exp105_step0_dinner_scan_qwen.yaml")
        self.plan = build_plan(self.cfg["scan"], name=self.cfg["name"])

    def test_grid_expands_to_20_prompts(self) -> None:
        """2 budgets × 10 motivation levels → 20 distinct prompts."""
        assert len(self.plan.inputs) == 20
        assert len({inp.prompt for inp in self.plan.inputs}) == 20

    def test_grid_params_recorded_per_input(self) -> None:
        assert self.plan.grid_keys == ("budget", "motivation")
        first = self.plan.inputs[0]
        assert first.grid_params == {"budget": "small", "motivation": 1}
        assert "budget is small" in first.prompt
        assert "is 1 out of 10" in first.prompt
        last = self.plan.inputs[-1]
        assert last.grid_params == {"budget": "medium", "motivation": 10}

    def test_null_image_input(self) -> None:
        """`image: null` → the PMI null image, one input row per prompt."""
        assert all(inp.image_path is None for inp in self.plan.inputs)
        assert self.plan.image_ids == (NULL_IMAGE_ID,)

    def test_single_candidate_tuple_of_three(self) -> None:
        assert len(self.plan.candidate_sets) == 1
        cs = self.plan.candidate_sets[0]
        assert cs.name == "default"
        assert len(cs.candidates) == 3
        assert cs.candidates[0] == "I will cook at home."

    def test_row_and_call_counts(self) -> None:
        assert self.plan.n_rows() == 60
        calls = self.plan.call_counts()
        # The scan already runs on the null image → baselines coincide.
        assert calls == {"main": 20, "baseline": 0, "generate": 0, "total": 20}

    def test_dry_run_summary(self) -> None:
        from experiments.runners.run_boundary_test import load_config

        text = format_plan(self.plan, load_config(self.cfg))
        assert "mode=score" in text
        assert "inputs           : 20" in text
        assert "parquet rows     : 60" in text
        assert "budget(2) × motivation(10)" in text


# ===========================================================================
# Planning — named candidate sets (steps 2/3)
# ===========================================================================


class TestCandidateSetPlan:
    def setup_method(self) -> None:
        self.cfg = _load_cfg("exp105_step23_person_scan_qwen.yaml")
        self.plan = build_plan(self.cfg["scan"], name=self.cfg["name"])

    def test_one_input_per_image(self) -> None:
        assert len(self.plan.inputs) == 6
        assert len(self.plan.image_ids) == 6
        assert len({inp.prompt for inp in self.plan.inputs}) == 1

    def test_named_sets_preserved_in_order(self) -> None:
        names = [cs.name for cs in self.plan.candidate_sets]
        assert names == ["step2_skin", "step2_gender", "step3_gamma0"]
        assert [len(cs.candidates) for cs in self.plan.candidate_sets] == [2, 2, 4]

    def test_counts(self) -> None:
        assert self.plan.n_rows() == 6 * (2 + 2 + 4)
        calls = self.plan.call_counts()
        assert calls["main"] == 18            # 6 images × 3 sets
        assert calls["baseline"] == 3         # one per (prompt, set)
        assert calls["generate"] == 6         # free_generation_probe
        assert calls["total"] == 27

    def test_free_generation_probe_flag(self) -> None:
        assert self.plan.free_generation_probe is True
        assert self.plan.wants_baseline is True


# ===========================================================================
# Planning — chains (steps 5/6)
# ===========================================================================


class TestChainPlan:
    def setup_method(self) -> None:
        self.cfg = _load_cfg("exp105_step56_chain_scan_qwen.yaml")
        self.plan = build_plan(self.cfg["scan"], name=self.cfg["name"])

    def test_mode_and_enumeration(self) -> None:
        assert self.plan.mode == "chain"
        assert self.plan.chain is not None
        assert self.plan.chain.slot_names == ("G", "D")
        # 2 templates × (2 × 2 fillers) = 8 chains per input.
        assert self.plan.n_chains == 8
        assert len(self.plan.inputs) == 3
        assert self.plan.n_rows() == 3 * 8 * 2      # one row per (chain, slot)

    def test_filler_combinations(self) -> None:
        combos = self.plan.chain.filler_combinations()
        assert len(combos) == 4
        assert {"G": "a man", "D": "accepted"} in combos
        assert {"G": "a woman", "D": "rejected"} in combos

    def test_perturbations_captured_not_executed(self) -> None:
        assert self.plan.perturbations == {
            "image": {"n": 8, "severity": "low"},
            "text": {"n": 8, "severity": "low"},
        }

    def test_run_scan_rejects_perturbations_before_loading_a_model(self) -> None:
        with pytest.raises(NotImplementedError, match="not implemented"):
            run_scan(self.cfg, dry_run=False)

    def test_dry_run_reports_chain_api(self) -> None:
        from experiments.runners.run_boundary_test import load_config

        text = format_plan(self.plan, load_config(self.cfg))
        assert "mode=chain" in text
        assert "score_chain_slots" in text
        assert "NOT IMPLEMENTED" in text          # perturbations warning


# ===========================================================================
# Template splitting
# ===========================================================================


class TestTemplateParts:
    def test_split_literals_and_slots(self) -> None:
        parts = template_parts(
            "Because the applicant was {G}, the application was {D}.",
            {"G": "a man", "D": "accepted"},
        )
        assert parts == [
            "Because the applicant was ",
            ("G", "a man"),
            ", the application was ",
            ("D", "accepted"),
            ".",
        ]

    def test_reassembles_to_realised_sentence(self) -> None:
        parts = template_parts(
            "The application was {D} because the applicant was {G}.",
            {"G": "a woman", "D": "rejected"},
        )
        assert chain_text(parts) == (
            "The application was rejected because the applicant was a woman."
        )

    def test_leading_slot_drops_empty_literal(self) -> None:
        parts = template_parts("{G} applied.", {"G": "a man"})
        assert parts == [("G", "a man"), " applied."]

    def test_missing_filler_raises(self) -> None:
        with pytest.raises(KeyError, match="no filler"):
            template_parts("{G} and {D}", {"G": "x"})


# ===========================================================================
# Schema validation
# ===========================================================================


class TestScanValidation:
    def _plan(self, **scan):
        return build_plan(scan)

    def test_empty_block_raises(self) -> None:
        with pytest.raises(ValueError, match="no `scan:` block"):
            build_plan({})

    def test_unknown_key_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown scan keys"):
            self._plan(prompt="p", candidates=["a"], candidatez=["b"])

    def test_prompt_and_carrier_conflict(self) -> None:
        with pytest.raises(ValueError, match="both `prompt` and"):
            self._plan(
                prompt="p", carrier_template="{a}", grid={"a": [1]},
                candidates=["x"],
            )

    def test_missing_prompt_raises(self) -> None:
        with pytest.raises(ValueError, match="either `prompt`"):
            self._plan(candidates=["x"])

    def test_grid_without_carrier_raises(self) -> None:
        with pytest.raises(ValueError, match="only meaningful with"):
            self._plan(prompt="p", grid={"a": [1]}, candidates=["x"])

    def test_grid_key_placeholder_mismatch(self) -> None:
        with pytest.raises(ValueError, match="must match"):
            self._plan(
                carrier_template="{a}", grid={"b": [1]}, candidates=["x"],
            )

    def test_grid_key_column_collision(self) -> None:
        with pytest.raises(ValueError, match="collide with runner-owned"):
            self._plan(
                carrier_template="{candidate}",
                grid={"candidate": [1]},
                candidates=["x"],
            )

    def test_image_and_images_conflict(self) -> None:
        with pytest.raises(ValueError, match="both `image` and `images`"):
            self._plan(image="a.png", images=["b.png"], prompt="p",
                       candidates=["x"])

    def test_candidates_and_candidate_sets_conflict(self) -> None:
        with pytest.raises(ValueError, match="both `candidates` and"):
            self._plan(
                prompt="p", candidates=["x"], candidate_sets={"s": ["y"]},
            )

    def test_no_candidates_raises(self) -> None:
        with pytest.raises(ValueError, match="needs `candidates`"):
            self._plan(prompt="p")

    def test_report_must_include_raw(self) -> None:
        with pytest.raises(ValueError, match="must include `raw`"):
            self._plan(prompt="p", candidates=["x"], report=["pmi_baseline"])

    def test_unknown_report_entry(self) -> None:
        with pytest.raises(ValueError, match="unknown scan.report"):
            self._plan(prompt="p", candidates=["x"], report=["raw", "bogus"])

    def test_report_defaults_to_raw_only(self) -> None:
        plan = self._plan(prompt="p", candidates=["x"])
        assert plan.report == ("raw",)
        assert plan.wants_baseline is False
        assert plan.call_counts()["baseline"] == 0

    def test_chain_requires_per_slot_scoring(self) -> None:
        with pytest.raises(ValueError, match="per_slot_scoring"):
            self._plan(
                prompt="p",
                chain_templates={"a": "{G}"},
                slots={"G": ["x"]},
            )

    def test_chain_template_slot_mismatch(self) -> None:
        with pytest.raises(ValueError, match="every template must realise"):
            self._plan(
                prompt="p",
                chain_templates={"a": "{G}", "b": "{G} {D}"},
                slots={"G": ["x"], "D": ["y"]},
                per_slot_scoring=True,
            )

    def test_chain_and_candidates_conflict(self) -> None:
        with pytest.raises(ValueError, match="cannot be combined"):
            self._plan(
                prompt="p",
                candidates=["x"],
                chain_templates={"a": "{G}"},
                slots={"G": ["x"]},
                per_slot_scoring=True,
            )

    def test_slots_without_chain_templates(self) -> None:
        with pytest.raises(ValueError, match="only applies to a chain scan"):
            self._plan(prompt="p", candidates=["x"], slots={"G": ["y"]})


# ===========================================================================
# Hex-grid candidate sets
# ===========================================================================


class TestHexGridPlan:
    def _plan(self, **hex_kwargs) -> ScanPlan:
        return build_plan({
            "prompt": "Look at the image.",
            "hex_grid": {"template": "The house is {hex}.", **hex_kwargs},
        })

    def test_hex_set_is_instantiated_sentences(self) -> None:
        plan = self._plan(steps=3, hue_start=120.0, hue_end=240.0)
        cs = plan.candidate_sets[0]
        assert cs.name == "hex_grid"
        assert cs.source == "hex_grid"
        assert cs.candidates[0] == "The house is #00FF00."
        assert cs.candidates[-1] == "The house is #0000FF."
        assert cs.labels == ("#00FF00", "#00FFFF", "#0000FF")

    def test_template_needs_hex_placeholder(self) -> None:
        with pytest.raises(ValueError, match=r"\{hex\} placeholder"):
            build_plan({"prompt": "p", "hex_grid": {"template": "no slot"}})

    def test_unknown_hex_key(self) -> None:
        with pytest.raises(ValueError, match="unknown scan.hex_grid keys"):
            build_plan({"prompt": "p", "hex_grid": {"hue": 1}})

    def test_materialise_drops_off_modal_token_counts(self) -> None:
        plan = self._plan(steps=3)
        tok = StubTokenizer({"#00FFFF": 9}, default=4)
        sets = materialise_candidate_sets(plan, tok)
        assert len(sets) == 1
        assert sets[0].labels == ("#00FF00", "#0000FF")
        assert sets[0].candidates == (
            "The house is #00FF00.", "The house is #0000FF.",
        )

    def test_materialise_keeps_all_when_counts_match(self) -> None:
        plan = self._plan(steps=4)
        sets = materialise_candidate_sets(plan, StubTokenizer(default=4))
        assert len(sets[0].candidates) == 4

    def test_materialise_without_hex_needs_no_tokenizer(self) -> None:
        plan = build_plan({"prompt": "p", "candidates": ["a", "b"]})
        assert materialise_candidate_sets(plan, None) == plan.candidate_sets

    def test_require_equal_off_keeps_everything(self) -> None:
        plan = self._plan(steps=3, require_equal_token_count=False)
        tok = StubTokenizer({"#00FFFF": 9}, default=4)
        assert len(materialise_candidate_sets(plan, tok)[0].candidates) == 3


# ===========================================================================
# Execution — score mode
# ===========================================================================


class TestExecuteScoreMode:
    def _run(self, tmp_path: Path, scan: dict, scorer=None):
        import pandas as pd

        plan = build_plan(scan, name="unit_scan")
        sut = make_fake_sut(scorer)
        stats = execute_scan(
            plan, sut._config, sut, tmp_path / "run", raw_cfg={"scan": scan},
        )
        df = pd.read_parquet(tmp_path / "run" / "scan.parquet")
        return plan, stats, df, sut

    def test_writes_all_artifacts(self, tmp_path: Path) -> None:
        img = _write_image(tmp_path / "seed.png")
        _, stats, df, _ = self._run(tmp_path, {
            "images": [str(img)],
            "prompt": "Describe it.",
            "candidates": ["a red house.", "a blue house."],
            "report": ["raw", "pmi_baseline"],
        })
        run_dir = tmp_path / "run"
        assert (run_dir / "scan.parquet").exists()
        assert (run_dir / "config.yaml").exists()
        saved = json.loads((run_dir / "stats.json").read_text())
        assert saved == stats
        assert stats["n_rows"] == 2
        assert stats["model_id"] == "fake/model"
        assert stats["cache_stats"] == {"hits": 0, "misses": 0}
        assert stats["wall_time_sec"] >= 0
        assert len(df) == 2

    def test_row_columns_and_baseline(self, tmp_path: Path) -> None:
        img = _write_image(tmp_path / "seed.png")
        _, _, df, _ = self._run(tmp_path, {
            "images": [str(img)],
            "prompt": "Describe it.",
            "candidates": ["a red house.", "a blue house."],
            "report": ["raw", "pmi_baseline"],
        })
        assert list(df["candidate"]) == ["a red house.", "a blue house."]
        assert list(df["candidate_index"]) == [0, 1]
        assert set(df["candidate_set"]) == {"default"}
        assert not df["image_is_null"].any()
        assert list(df["baseline_prompt"]) == ["Describe it."] * 2
        # FakeScanScorer's null-image offset is exactly -1.0.
        assert (df["lp_norm"] - df["lp_norm_null"]).round(6).tolist() == [1.0, 1.0]

    def test_grid_columns_present(self, tmp_path: Path) -> None:
        _, _, df, _ = self._run(tmp_path, {
            "carrier_template": "Budget {budget}, effort {effort}.",
            "grid": {"budget": ["small", "large"], "effort": [1, 2, 3]},
            "candidates": ["cook.", "order in."],
        })
        assert len(df) == 6 * 2
        assert set(df["budget"]) == {"small", "large"}
        assert set(df["effort"]) == {1, 2, 3}
        assert df["image_is_null"].all()
        assert set(df["image_path"]) == {NULL_IMAGE_ID}

    def test_baseline_calls_are_memoised(self, tmp_path: Path) -> None:
        """Two images, one prompt, one set → 2 main + 1 shared baseline."""
        a = _write_image(tmp_path / "a.png", (10, 200, 10))
        b = _write_image(tmp_path / "b.png", (10, 10, 200))
        _, stats, _, _ = self._run(tmp_path, {
            "images": [str(a), str(b)],
            "prompt": "Describe it.",
            "candidates": ["x.", "y."],
            "report": ["raw", "pmi_baseline"],
        })
        assert stats["n_sut_calls"] == 3

    def test_null_image_scan_reuses_main_call_as_baseline(
        self, tmp_path: Path,
    ) -> None:
        _, stats, df, _ = self._run(tmp_path, {
            "image": None,
            "prompt": "Describe it.",
            "candidates": ["x.", "y."],
            "report": ["raw", "pmi_baseline"],
        })
        assert stats["n_sut_calls"] == 1
        assert (df["lp_norm"] == df["lp_norm_null"]).all()

    def test_baseline_omitted_when_not_reported(self, tmp_path: Path) -> None:
        img = _write_image(tmp_path / "seed.png")
        _, stats, df, _ = self._run(tmp_path, {
            "images": [str(img)],
            "prompt": "Describe it.",
            "candidates": ["x."],
            "report": ["raw"],
        })
        assert stats["n_sut_calls"] == 1
        assert df["lp_norm_null"].isna().all()
        assert df["baseline_prompt"].isna().all()

    def test_pmi_arm_keeps_baseline_raw_and_distinct(
        self, tmp_path: Path,
    ) -> None:
        """With pmi.enabled the main call is corrected, the baseline is not,
        so a null-image scan can no longer reuse one forward pass."""
        import pandas as pd

        scan = {
            "image": None,
            "prompt": "Describe it.",
            "candidates": ["x.", "y."],
            "report": ["raw", "pmi_baseline"],
        }
        plan = build_plan(scan, name="pmi_scan")
        assert plan.call_counts(pmi_enabled=True)["baseline"] == 1
        sut = make_fake_sut(pmi=PMIConfig(enabled=True, null_image_size=8))
        stats = execute_scan(plan, sut._config, sut, tmp_path / "run")
        df = pd.read_parquet(tmp_path / "run" / "scan.parquet")
        assert stats["n_sut_calls"] == 2
        assert df["pmi_enabled"].all()
        # lp_norm_null is the RAW score under the scan prompt …
        raw = -0.1 * len("x.") - 0.01 * len("Describe it.") - 1.0
        assert df["lp_norm_null"].iloc[0] == pytest.approx(raw)
        # … while lp_norm is VLMSUT's corrected value, whose internal
        # baseline uses the canonical prompt — the two differ by design.
        assert df["lp_norm"].iloc[0] != pytest.approx(raw)

    def test_named_sets_scored_separately(self, tmp_path: Path) -> None:
        img = _write_image(tmp_path / "seed.png")
        _, stats, df, _ = self._run(tmp_path, {
            "images": [str(img)],
            "prompt": "p",
            "candidate_sets": {"skin": ["light.", "dark."], "g": ["man.", "woman."]},
        })
        assert stats["n_sut_calls"] == 2
        assert list(df["candidate_set"]) == ["skin", "skin", "g", "g"]
        assert stats["candidate_sets"]["skin"] == ["light.", "dark."]

    def test_missing_image_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="scan image not found"):
            self._run(tmp_path, {
                "images": [str(tmp_path / "nope.png")],
                "prompt": "p",
                "candidates": ["x."],
            })

    def test_free_generation_probe_writes_transcript(
        self, tmp_path: Path,
    ) -> None:
        a = _write_image(tmp_path / "a.png")
        b = _write_image(tmp_path / "b.png", (5, 5, 5))
        scorer = FakeScanScorer()
        _, stats, _, _ = self._run(tmp_path, {
            "images": [str(a), str(b)],
            "prompt": "Complete the statement.",
            "candidates": ["x."],
            "free_generation_probe": True,
        }, scorer=scorer)
        assert stats["n_generate_calls"] == 2
        assert len(scorer.generate_calls) == 2
        transcripts = json.loads(
            (tmp_path / "run" / "free_generation.json").read_text()
        )
        assert [t["prompt"] for t in transcripts] == [
            "Complete the statement."
        ] * 2
        assert transcripts[0]["answer"].startswith("I see something.")

    def test_hex_scan_end_to_end(self, tmp_path: Path) -> None:
        img = _write_image(tmp_path / "house.png")
        scorer = FakeScanScorer(StubTokenizer({"#00FFFF": 9}, default=4))
        _, stats, df, _ = self._run(tmp_path, {
            "images": [str(img)],
            "prompt": "Look at the image.",
            "hex_grid": {"template": "The house is {hex}.", "steps": 3},
        }, scorer=scorer)
        # Off-modal code dropped before any scoring call.
        assert len(df) == 2
        assert list(df["candidate_label"]) == ["#00FF00", "#0000FF"]
        assert stats["n_candidate_sets"] == 1


# ===========================================================================
# Execution — chain mode
# ===========================================================================


class TestExecuteChainMode:
    def _scan(self, images: list[str]) -> dict:
        return {
            "images": images,
            "prompt": "Complete the statement.",
            "chain_templates": {
                "order_a": "Because the applicant was {G}, the application was {D}.",
                "order_b": "The application was {D} because the applicant was {G}.",
            },
            "slots": {"G": ["a man", "a woman"], "D": ["accepted", "rejected"]},
            "per_slot_scoring": True,
            "report": ["raw", "pmi_baseline"],
        }

    def test_one_row_per_chain_and_slot(self, tmp_path: Path) -> None:
        import pandas as pd

        img = _write_image(tmp_path / "face.png")
        plan = build_plan(self._scan([str(img)]), name="chain_unit")
        scorer = FakeScanScorer()
        sut = make_fake_sut(scorer)
        stats = execute_scan(plan, sut._config, sut, tmp_path / "run")
        df = pd.read_parquet(tmp_path / "run" / "scan.parquet")

        assert len(df) == 8 * 2                      # 8 chains × 2 slots
        assert stats["n_chains"] == 8
        assert set(df["template_name"]) == {"order_a", "order_b"}
        assert set(df["slot_name"]) == {"G", "D"}
        # 8 chains on the image + 8 on the null baseline.
        assert stats["n_scorer_calls"] == 16
        assert len(scorer.chain_calls) == 16

    def test_chain_text_and_slot_scores(self, tmp_path: Path) -> None:
        import pandas as pd

        img = _write_image(tmp_path / "face.png")
        plan = build_plan(self._scan([str(img)]), name="chain_unit")
        sut = make_fake_sut()
        execute_scan(plan, sut._config, sut, tmp_path / "run")
        df = pd.read_parquet(tmp_path / "run" / "scan.parquet")

        row = df[
            (df["template_name"] == "order_a")
            & (df["slot_name"] == "G")
            & (df["slot_filler"] == "a man")
            & (df["chain_text"].str.contains("accepted"))
        ]
        assert len(row) == 1
        assert row.iloc[0]["chain_text"] == (
            "Because the applicant was a man, the application was accepted."
        )
        assert row.iloc[0]["n_tokens"] == 2
        assert json.loads(row.iloc[0]["fillers_json"]) == {
            "D": "accepted", "G": "a man",
        }
        # Null baseline recorded per slot; FakeScanScorer offset is -1/token.
        assert row.iloc[0]["lp_norm"] - row.iloc[0]["lp_norm_null"] == pytest.approx(1.0)

    def test_baseline_shared_across_images(self, tmp_path: Path) -> None:
        a = _write_image(tmp_path / "a.png", (10, 200, 10))
        b = _write_image(tmp_path / "b.png", (10, 10, 200))
        plan = build_plan(self._scan([str(a), str(b)]), name="chain_unit")
        sut = make_fake_sut()
        stats = execute_scan(plan, sut._config, sut, tmp_path / "run")
        # 2 images × 8 chains + 8 shared null baselines.
        assert stats["n_scorer_calls"] == 24
        assert stats["n_rows"] == 2 * 8 * 2


# ===========================================================================
# Plan dataclass surface
# ===========================================================================


class TestPlanSurface:
    def test_scan_input_image_id(self) -> None:
        assert ScanInput(0, None, "p").image_id == NULL_IMAGE_ID
        assert ScanInput(0, "a.png", "p").image_id == "a.png"

    def test_candidate_set_defaults_to_literal(self) -> None:
        cs = CandidateSet("n", ("a",), ("a",))
        assert cs.source == "literal"

    def test_hex_spec_fields(self) -> None:
        plan = build_plan({
            "prompt": "p",
            "hex_grid": {"template": "{hex}", "steps": 5, "name": "arc"},
        })
        assert isinstance(plan.hex_grid, HexGridSpec)
        assert plan.hex_grid.name == "arc"
        assert plan.hex_grid.steps == 5
        assert plan.hex_grid.require_equal_token_count is True
        assert plan.candidate_sets[0].name == "arc"
