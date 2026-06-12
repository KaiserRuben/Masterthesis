#!/usr/bin/env python3
"""HS-GEN-01 screen → promote.

Reads the screening sweep outputs (``runs/HS-GEN-01/hs_gen01_screen_seed_*``),
applies the two promotion gates, and emits a full-evolution YAML per promoted
pair (cloned from pair A: VQGAN, cone ON, dense init, annealed heavy mutation).

Two gates, BOTH must pass to auto-emit a config:

  1. Crossability (re-measured under the study-facing cone + heavy-mutation
     regime — the old cone-off floor table is advisory only). Uses the gen-0
     margin predictor: gen-0 minimum fitness_TgtBal.
        gen0_min ≤ CROSS_THRESHOLD (0.3)            → crossable    → promote
        CROSS_THRESHOLD < gen0_min ≤ MARGINAL (1.0) → marginal     → promote
                                                       iff a clearly falling
                                                       4-gen slope
        gen0_min > MARGINAL                         → walled       → skip

  2. Human-distinguishability (laypeople; src.data.taxonomy). Classified from
     the smallest taxonomy level at which the two classes share a cluster:
        common_ancestor_level == 0  → NEEDS ABSTRACTION. The two classes are
            fine siblings (e.g. two shark species); symmetric coarsening
            collapses them to one label, so they cannot be auto-rescued. NOT
            auto-emitted — listed for manual feature-contrast framing (README).
        common_ancestor_level >= 1 or None → LAY-DISTINGUISHABLE. The L0
            cluster labels already differ (e.g. "shark" vs "ray", or distinct
            super-cats); emit, and propose the L0 labels as the clean
            presentation framing.

Usage:
  conda run -n uni python configs/HS-GEN-01/promote_pairs.py            # report + emit
  conda run -n uni python configs/HS-GEN-01/promote_pairs.py --report-only
  conda run -n uni python configs/HS-GEN-01/promote_pairs.py --runs-dir runs/HS-GEN-01

Emitted files:  configs/HS-GEN-01/hs_gen01_promoted_idx<N>_<a>_<b>.yaml
Then: append them to run_hs_gen01_chain.sh CONFIGS and re-run
      configs/HS-GEN-01/validate_configs.py.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from src.data import taxonomy  # noqa: E402

HERE = Path(__file__).resolve().parent

CROSS_THRESHOLD = 0.3      # gen0_min ≤ this → crossable
MARGINAL_THRESHOLD = 1.0   # gen0_min ≤ this (and > CROSS) → marginal
MARGINAL_MIN_SLOPE = 0.20  # marginal promoted only if ≥20% relative drop over 4 gens


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def scan_screen(runs_dir: Path) -> pd.DataFrame:
    """One row per screened roster entry (seed_idx)."""
    rows = []
    for d in sorted(runs_dir.glob("hs_gen01_screen_seed_*")):
        stats_p, trace_p = d / "stats.json", d / "trace.parquet"
        if not (stats_p.exists() and trace_p.exists()):
            continue
        s = json.loads(stats_p.read_text())
        t = pd.read_parquet(trace_p, columns=["generation", "fitness_TgtBal"])
        g0 = t[t.generation == 0].fitness_TgtBal.min()
        glast = t[t.generation == t.generation.max()].fitness_TgtBal.min()
        rel_slope = (g0 - glast) / g0 if g0 > 0 else 0.0
        rows.append({
            "idx": s["seed_idx"],
            "class_a": s["class_a"],
            "class_b": s["class_b"],
            "gen0_min": float(g0),
            "best": float(t.fitness_TgtBal.min()),
            "rel_slope": float(rel_slope),
            "run_dir": d.name,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Multiple screen replicates of the same idx → keep the best (lowest gen0).
    df = df.sort_values("gen0_min").drop_duplicates("idx", keep="first")
    return df.sort_values("best").reset_index(drop=True)


def classify_crossability(r) -> str:
    if r.gen0_min <= CROSS_THRESHOLD:
        return "crossable"
    if r.gen0_min <= MARGINAL_THRESHOLD:
        return "marginal" if r.rel_slope >= MARGINAL_MIN_SLOPE else "marginal_hold"
    return "walled"


def classify_distinguishability(a: str, b: str) -> tuple[str, str]:
    """Return (verdict, presentation_labels)."""
    lvl = taxonomy.common_ancestor_level(a, b)
    if lvl == 0:
        return ("needs_abstraction",
                f"taxonomy coarsening collapses both to "
                f"'{taxonomy.cluster_of(a, 0)}' — manual feature contrast only")
    la = taxonomy.cluster_of(a, 0) or a
    lb = taxonomy.cluster_of(b, 0) or b
    if la != lb:
        return ("distinguishable", f"present as L0: '{la}' vs '{lb}'")
    return ("distinguishable", f"fine labels: '{a}' vs '{b}'")


def emit_config(r, present_labels: str) -> Path:
    a_slug, b_slug = _slug(r.class_a), _slug(r.class_b)
    name = f"hs_gen01_promoted_idx{r.idx}_{a_slug}_{b_slug}"
    out = HERE / f"{name}.yaml"
    body = f"""\
# ═══════════════════════════════════════════════════════════════════════════
# HS-GEN-01 promoted pair — {r.class_a} → {r.class_b} (gap_filter idx {r.idx})
#                    AUTO-EMITTED by promote_pairs.py from the screen.
# ═══════════════════════════════════════════════════════════════════════════
#
# Screen result (cone ON + heavy mutation): gen0_min={r.gen0_min:.4g},
#   best(4 gen)={r.best:.4g}, rel_slope={r.rel_slope:.2f} → {classify_crossability(r)}.
# Distinguishability: {present_labels}.
#
# Clone of hs_gen01_pairA_shark_hammerhead.yaml — VQGAN, cone α=20° ON, dense
# image-only init, annealed heavy mutation (prob=0.1, eta=3.0). Only `name`,
# `filter_indices`, and these comments differ. Verify with validate_configs.py.
# ═══════════════════════════════════════════════════════════════════════════

name: {name}
save_dir: runs/HS-GEN-01

modality: image_only

generations: 150
pop_size: 30

device: cpu

sut:
  model_id: OpenVINO/llava-v1.6-mistral-7b-hf-int8-ov
  processor_id: llava-hf/llava-v1.6-mistral-7b-hf
  backend: openvino
  ov_device: GPU

cache_dirs:
  - /mnt/storage/huggingface/imagenet

n_categories: 50

seeds:
  mode: gap_filter
  filter_indices: [{r.idx}]              # {r.class_a} → {r.class_b}
  gap_filter:
    n_per_class: 3
    max_logprob_gap: 3.5

image:
  backend: vqgan_codebook
  n_candidates: 16383
  knn_cache_path: /mnt/storage/huggingface/vqgan_knn/f8_16384_full.npz
  cone_filter:
    enabled: true
    alpha_deg: 20.0
    target_m: 10

optimizer:
  sampling:
    mode: sparse_multitier
    zero_anchor_fraction: 0.0
    tiers:
      - {{p_active: 0.03, fraction: 0.10}}
      - {{p_active: 0.10, fraction: 0.20}}
      - {{p_active: 0.20, fraction: 0.30}}
      - {{p_active: 0.35, fraction: 0.25}}
      - {{p_active: 0.55, fraction: 0.15}}
  mutation:
    prob: 0.1
    eta: 3.0
  early_stop:
    enable: true
    epsilon_margin: 1.0e-30
    plateau_patience: 50
    no_improvement_warmup: 40
    hypervolume_reference: [1.0, 10.0]
"""
    out.write_text(body)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default=str(REPO / "runs" / "HS-GEN-01"))
    ap.add_argument("--report-only", action="store_true",
                    help="classify + print, but do not write YAMLs")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    df = scan_screen(runs_dir)
    if df.empty:
        print(f"No screen runs found under {runs_dir}/hs_gen01_screen_seed_*")
        print("Run hs_gen01_screen.yaml first (see README, 'Screen → promote').")
        return 0

    print(f"Scanned {len(df)} roster entries from {runs_dir}\n")
    emitted, manual, walled = [], [], []
    print(f"{'idx':>4}  {'gen0':>7}  {'best':>8}  {'slope':>5}  "
          f"{'cross':<13}  {'distinguish':<16}  pair")
    print("-" * 96)
    for r in df.itertuples(index=False):
        cross = classify_crossability(r)
        dist, labels = classify_distinguishability(r.class_a, r.class_b)
        print(f"{r.idx:>4}  {r.gen0_min:>7.4g}  {r.best:>8.4g}  "
              f"{r.rel_slope:>5.2f}  {cross:<13}  {dist:<16}  "
              f"{r.class_a} → {r.class_b}")

        promote = cross in ("crossable", "marginal")
        if not promote:
            walled.append(r)
            continue
        if dist == "needs_abstraction":
            manual.append((r, labels))
            continue
        if not args.report_only:
            p = emit_config(r, labels)
            emitted.append(p)
        else:
            emitted.append(HERE / f"(would emit idx {r.idx})")

    print()
    print(f"EMITTED ({len(emitted)}): crossable+distinguishable")
    for p in emitted:
        print(f"  {p.name if isinstance(p, Path) else p}")
    print(f"\nNEEDS MANUAL ABSTRACTION ({len(manual)}): crossable but fine-sibling")
    for r, labels in manual:
        print(f"  idx {r.idx}: {r.class_a} → {r.class_b}  [{labels}]")
    print(f"\nSKIPPED — walled/marginal-hold ({len(walled)})")

    if not emitted and not manual:
        print("\n*** No crossable pair passed either gate. ***")
        print("Open study-design question (README): the image-heavy strata then "
              "fall back to the calibration pair (idx 1) under its manual "
              "abstraction framing, or the cell is dropped. Re-screen with a "
              "wider roster / higher mutation before declaring image-walled.")

    # Self-validate every emitted YAML through the production loader.
    if emitted and not args.report_only:
        print("\nValidating emitted configs through load_config + apply_modality:")
        from experiments.runners.run_boundary_test import load_config
        from src.config import apply_modality
        import yaml
        bad = 0
        for p in emitted:
            try:
                exp = apply_modality(load_config(yaml.safe_load(p.read_text())))
                assert exp.image.cone_filter.enabled is True
                assert exp.image.backend == "vqgan_codebook"
                print(f"  OK   {p.name}")
            except Exception as e:  # noqa: BLE001
                bad += 1
                print(f"  FAIL {p.name}: {type(e).__name__}: {e}")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
