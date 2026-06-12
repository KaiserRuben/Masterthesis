#!/usr/bin/env python3
"""Validate every HS-GEN-01 YAML parses into a valid ExperimentConfig.

Uses the EXACT loading path of the production runner
(``experiments.runners.run_boundary_test.load_config`` + ``apply_modality``)
— config parsing/validation only, no SUT / model weights / data access.

Enforces the four binding HS-GEN-01 decisions as invariants:
  1. cone ON everywhere (semantic validity for human-eval stimuli);
  2. VQGAN only (no StyleGAN in study-facing generation);
  4. heavy mutation (mutation.prob raised well above the ~1/n_var default).
(Decision 3, human-distinguishability, is enforced by promote_pairs.py, not
here — it is a study-design gate, not a config invariant.)

Run:  conda run -n uni python configs/HS-GEN-01/validate_configs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

import yaml  # noqa: E402

from experiments.runners.run_boundary_test import load_config  # noqa: E402
from src.config import apply_modality  # noqa: E402

HERE = Path(__file__).resolve().parent

failures = 0
n_seen = 0
for yml in sorted(HERE.glob("*.yaml")):
    n_seen += 1
    try:
        with open(yml) as f:
            raw = yaml.safe_load(f)
        exp = load_config(raw)
        exp = apply_modality(exp)

        # HS-GEN-01 invariants
        assert exp.modality == "image_only", f"modality={exp.modality}"
        assert exp.text.composite.profile == "noop", (
            f"text profile not forced to noop: {exp.text.composite.profile}"
        )
        assert exp.sut.backend == "openvino", f"sut.backend={exp.sut.backend}"
        assert "llava" in exp.sut.model_id.lower(), f"SUT={exp.sut.model_id}"
        assert exp.seeds.mode == "gap_filter"
        gf = exp.seeds.gap_filter
        assert (gf.n_per_class, gf.max_logprob_gap) == (3, 3.5), (
            "gap_filter params drifted — roster indices would remap"
        )
        assert exp.n_categories == 50, "n_categories drifted — roster remaps"

        # Decision 2 — VQGAN only.
        assert exp.image.backend == "vqgan_codebook", (
            f"decision 2 (VQGAN only) violated: backend={exp.image.backend}"
        )
        # Decision 1 — cone ON everywhere, α = Exp-26 LLaVA success setting.
        cf = exp.image.cone_filter
        assert cf.enabled is True, "decision 1 (cone ON) violated: cone disabled"
        assert cf.alpha_deg == 20.0, (
            f"cone α drifted from Exp-26 LLaVA setting: {cf.alpha_deg}"
        )

        # Init — dense image-only multitier.
        s = exp.optimizer.sampling
        assert s.mode == "sparse_multitier"
        assert s.zero_anchor_fraction == 0.0
        frac_sum = sum(t.fraction for t in s.tiers)
        assert abs(frac_sum - 1.0) < 1e-9, f"tier fractions sum {frac_sum}"

        # Decision 4 — heavy mutation (per-gene prob well above ~1/n_var).
        mu = exp.optimizer.mutation
        assert mu.prob is not None and mu.prob >= 0.1, (
            f"decision 4 (heavy mutation) violated: mutation.prob={mu.prob}"
        )

        es = exp.optimizer.early_stop
        if es.enable:
            assert es.hypervolume_reference == (1.0, 10.0), (
                f"HV reference {es.hypervolume_reference!r} — plateau trigger "
                "dead without a reference"
            )

        ref = "off" if not es.enable else f"hv_ref={es.hypervolume_reference}"
        cx = exp.optimizer.crossover
        print(
            f"OK   {yml.name}: name={exp.name} modality={exp.modality} "
            f"backend={exp.image.backend} cone={cf.enabled}@{cf.alpha_deg}deg "
            f"gens×pop={exp.generations}×{exp.pop_size} "
            f"filter_indices={exp.seeds.filter_indices or 'ALL'} "
            f"mut(prob={mu.prob},eta={mu.eta}) "
            f"cx(prob={cx.prob},eta={cx.eta}) "
            f"early_stop={ref} workers={exp.parallel.workers}"
        )
    except Exception as e:  # noqa: BLE001 — report every failure, keep going
        failures += 1
        print(f"FAIL {yml.name}: {type(e).__name__}: {e}")

print(f"\n{n_seen} config(s) checked — "
      f"{'ALL CONFIGS VALID' if failures == 0 else f'{failures} FAILURE(S)'}")
sys.exit(1 if failures else 0)
