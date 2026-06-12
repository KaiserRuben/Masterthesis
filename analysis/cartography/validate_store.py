#!/usr/bin/env python3
"""Invariant checks for a cartography store (see README.md).

Run after every (re)build; exits non-zero on violated invariants.

    conda run -n uni python -m analysis.cartography.validate_store \
        --store experiments/analysis/output/cartography/exp100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global FAIL
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAIL += 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", type=Path, required=True)
    args = ap.parse_args()

    print(f"validating {args.store}")

    p = pd.read_parquet(args.store / "points.parquet",
                        columns=["source", "prompt_regime", "pair_margin",
                                 "g_pair", "n_active_img", "n_active_txt",
                                 "hamming_to_anchor", "image_dim", "seed_dir",
                                 "logprobs"])
    print(f"\npoints.parquet: {len(p):,} rows, "
          f"{p.seed_dir.nunique()} seeds")
    by_src = p.source.value_counts().to_dict()
    print(f"  sources: {by_src}")

    check("regime consistent with source",
          ((p.source == "smoo") == (p.prompt_regime == "pair2")).all())
    lp_len = p.logprobs.map(len)
    check("pair2 rows carry 2 logprobs",
          (lp_len[p.prompt_regime == "pair2"] == 2).all())
    check("cat6 rows carry 6 logprobs",
          (lp_len[p.prompt_regime == "cat6"] == 6).all())
    check("g_pair within [-1, 1]",
          bool(p.g_pair.between(-1, 1).all()))
    check("g_pair sign matches pair_margin sign",
          bool((np.sign(p.g_pair.round(9)) ==
                np.sign(p.pair_margin.round(9))).mean() > 0.999),
          "tolerating float rounding at the boundary")
    check("anchors sit at hamming 0",
          bool((p.loc[p.source == "pdq_anchor", "hamming_to_anchor"] == 0).all()))
    # image_dim is a per-seed-image property (VQGAN patch count varies by
    # image); the invariant is per-seed consistency, not a fixed value set.
    check("image_dim consistent within each seed",
          bool((p.groupby("seed_dir")["image_dim"].nunique() == 1).all()))
    check("image_dim plausible (>=200)", bool((p.image_dim >= 200).all()))

    s = pd.read_parquet(args.store / "straddle_pairs.parquet")
    print(f"\nstraddle_pairs.parquet: {len(s):,} rows")
    pm = s[s.boundary_kind == "pair_margin"]
    check("pair_margin straddles flip sign",
          bool(((pm.margin_before > 0) != (pm.margin_after > 0)).all()))
    am = s[s.boundary_kind == "argmax"]
    check("argmax straddles change label",
          bool((am.label_before != am.label_after).all()))
    check("straddle gene modality consistent with gene_idx",
          bool(((s.gene_modality == "txt") ==
                (s.gene_idx >= s.image_dim)).all()))

    t = pd.read_parquet(args.store / "transects.parquet",
                        columns=["seed_dir", "flip_id", "step", "accepted"])
    print(f"\ntransects.parquet: {len(t):,} rows, "
          f"{t.groupby(['seed_dir', 'flip_id']).ngroups} walks")
    steps_sorted = t.groupby(["seed_dir", "flip_id"])["step"].apply(
        lambda x: bool((np.diff(x) > 0).all()))
    check("steps strictly increasing within each walk", bool(steps_sorted.all()))

    print(f"\n{'PASS' if FAIL == 0 else f'{FAIL} FAILURES'}")
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
