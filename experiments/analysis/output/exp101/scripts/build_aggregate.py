#!/usr/bin/env python
"""Exp-101 canonical per-seed and per-cell aggregate builder.

Deterministic and idempotent. Reads every run dir under runs/Exp-101/,
emits exp101_per_seed.csv (one row per run) and exp101_per_cell.csv.

Definitions (verbatim from the Exp-101 context block):
- cell = (anchor_class_concrete, target_class_concrete, level_anchor, level_target)
- probe = median fitness_TgtBal over the 30 generation-0 individuals
- min_tgtbal_at_N = min fitness_TgtBal over individuals in generations 0..N-1
- dex_eroded_at_N = log10(probe / min_tgtbal_at_N)  [median-based, primary]
- dex_eroded_min_at_N = log10(min gen-0 fitness_TgtBal / min_tgtbal_at_N)
- crossed_at_N = any row in generations 0..N-1 with p_class_b > p_class_a
- gen_first_cross = first generation with any individual p_class_b > p_class_a
- stuck = min_tgtbal_at_50 > 0.1
- strata: within_bucket iff common_ancestor_level == 2;
          wall_repl iff cross-bucket and junco is anchor or target;
          cross_breadth iff cross-bucket and junco not involved.
"""
from __future__ import annotations

import glob
import json
import math
import os

import numpy as np
import pandas as pd

REPO = "/Users/kaiser/Projects/Masterarbeit"
RUNS_DIR = os.path.join(REPO, "runs", "Exp-101")
OUT_DIR = os.path.join(REPO, "experiments", "analysis", "output", "exp101")
PER_SEED_CSV = os.path.join(OUT_DIR, "exp101_per_seed.csv")
PER_CELL_CSV = os.path.join(OUT_DIR, "exp101_per_cell.csv")


def stratum_of(common_ancestor_level, anchor, target):
    if common_ancestor_level == 2:
        return "within_bucket"
    # cross-bucket (common_ancestor_level is null/None)
    if anchor == "junco" or target == "junco":
        return "wall_repl"
    return "cross_breadth"


def min_tgtbal_at(df, n):
    sub = df[df["generation"] < n]
    return float(sub["fitness_TgtBal"].min())


def build_rows():
    run_dirs = sorted(glob.glob(os.path.join(RUNS_DIR, "exp101_margin_predictor_seed_*")))
    rows = []
    for rd in run_dirs:
        with open(os.path.join(rd, "stats.json")) as f:
            stats = json.load(f)
        meta = stats["seed_metadata"]
        df = pd.read_parquet(os.path.join(rd, "trace.parquet"))

        gen0 = df[df["generation"] == 0]
        probe = float(gen0["fitness_TgtBal"].median())
        min_gen0 = float(gen0["fitness_TgtBal"].min())

        mt10 = min_tgtbal_at(df, 10)
        mt20 = min_tgtbal_at(df, 20)
        mt50 = min_tgtbal_at(df, 50)

        crossed_mask = df["p_class_b"] > df["p_class_a"]
        crossed_50 = bool(crossed_mask.any())
        n_cross_rows = int(crossed_mask.sum())
        if crossed_50:
            gen_first_cross = int(df.loc[crossed_mask, "generation"].min())
        else:
            gen_first_cross = None

        anchor = meta["anchor_class_concrete"]
        target = meta["target_class_concrete"]
        la = meta["level_anchor"]
        lt = meta["level_target"]
        cal = meta["common_ancestor_level"]

        rows.append({
            "seed_idx": stats["seed_idx"],
            "run_dir": os.path.basename(rd),
            "anchor": anchor,
            "target": target,
            "la": la,
            "lt": lt,
            "anchor_word": meta["anchor_label_in_prompt"],
            "target_word": meta["target_label_in_prompt"],
            "common_ancestor_level": cal,
            "stratum": stratum_of(cal, anchor, target),
            "seed_idx_in_class": meta["seed_idx_in_class"],
            "cell_id": f"{anchor}->{target}({la},{lt})",
            "probe": probe,
            "min_gen0": min_gen0,
            "min_tgtbal_10": mt10,
            "min_tgtbal_20": mt20,
            "min_tgtbal_50": mt50,
            "dex_eroded_10": math.log10(probe / mt10),
            "dex_eroded_20": math.log10(probe / mt20),
            "dex_eroded_50": math.log10(probe / mt50),
            "dex_eroded_min_50": math.log10(min_gen0 / mt50),
            "crossed_50": crossed_50,
            "gen_first_cross": gen_first_cross,
            "n_cross_rows": n_cross_rows,
            "stuck": bool(mt50 > 0.1),
            "pop0_p_class_a_median": float(gen0["p_class_a"].median()),
            "runtime_seconds": stats["runtime_seconds"],
        })
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = build_rows()
    per_seed = pd.DataFrame(rows)

    # n_in_cell: number of runs sharing the same cell_id
    counts = per_seed.groupby("cell_id")["seed_idx"].transform("size")
    per_seed["n_in_cell"] = counts.astype(int)

    # deterministic ordering
    per_seed = per_seed.sort_values(["seed_idx"]).reset_index(drop=True)

    # exact column order from the spec
    col_order = [
        "seed_idx", "run_dir", "anchor", "target", "la", "lt",
        "anchor_word", "target_word", "common_ancestor_level", "stratum",
        "seed_idx_in_class", "cell_id", "n_in_cell", "probe", "min_gen0",
        "min_tgtbal_10", "min_tgtbal_20", "min_tgtbal_50",
        "dex_eroded_10", "dex_eroded_20", "dex_eroded_50", "dex_eroded_min_50",
        "crossed_50", "gen_first_cross", "n_cross_rows", "stuck",
        "pop0_p_class_a_median", "runtime_seconds",
    ]
    per_seed = per_seed[col_order]
    per_seed.to_csv(PER_SEED_CSV, index=False)

    # per-cell aggregate
    grp = per_seed.groupby("cell_id", sort=True)
    cell_rows = []
    for cell_id, g in grp:
        first = g.iloc[0]
        cell_rows.append({
            "cell_id": cell_id,
            "anchor": first["anchor"],
            "target": first["target"],
            "la": first["la"],
            "lt": first["lt"],
            "anchor_word": first["anchor_word"],
            "target_word": first["target_word"],
            "common_ancestor_level": first["common_ancestor_level"],
            "stratum": first["stratum"],
            "n": int(len(g)),
            "probe": float(g["probe"].mean()),
            "min_gen0": float(g["min_gen0"].mean()),
            "min_tgtbal_10": float(g["min_tgtbal_10"].mean()),
            "min_tgtbal_20": float(g["min_tgtbal_20"].mean()),
            "min_tgtbal_50": float(g["min_tgtbal_50"].mean()),
            "dex_eroded_10": float(g["dex_eroded_10"].mean()),
            "dex_eroded_20": float(g["dex_eroded_20"].mean()),
            "dex_eroded_50": float(g["dex_eroded_50"].mean()),
            "dex_eroded_min_50": float(g["dex_eroded_min_50"].mean()),
            "crossed_50_frac": float(g["crossed_50"].mean()),
            "crossed_50_any": bool(g["crossed_50"].any()),
            "gen_first_cross_mean": (
                float(g["gen_first_cross"].dropna().mean())
                if g["gen_first_cross"].notna().any() else None
            ),
            "n_cross_rows": float(g["n_cross_rows"].mean()),
            "stuck_frac": float(g["stuck"].mean()),
            "stuck_any": bool(g["stuck"].any()),
            "pop0_p_class_a_median": float(g["pop0_p_class_a_median"].mean()),
            "runtime_seconds": float(g["runtime_seconds"].mean()),
        })
    per_cell = pd.DataFrame(cell_rows).sort_values("cell_id").reset_index(drop=True)
    per_cell.to_csv(PER_CELL_CSV, index=False)

    print(f"per_seed rows: {len(per_seed)}")
    print(f"per_cell rows: {len(per_cell)}")
    print(f"cells with n=2: {(per_cell['n'] == 2).sum()}")
    print(f"wrote {PER_SEED_CSV}")
    print(f"wrote {PER_CELL_CSV}")


if __name__ == "__main__":
    main()
