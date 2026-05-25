#!/usr/bin/env python3
"""Verify boundary-pair pipeline outputs for a single seed run.

Run after ``run_boundary_pair_test.py`` finishes to assert:

* Expected files exist under ``<seed_dir>/{evolutionary,pdq}/``.
* All six PDQ parquet files exist with the v5 schema.
* archive.parquet rows have ``anchor_source="evolutionary"`` and
  ``pareto_idx`` populated.
* manifest.json aggregates ``n_anchors_evaluated`` ≥ 1 and matches the
  archive row count.
* genotype_anchor ≠ zeros (boundary-pair pipeline never uses zero anchor).

Usage:
    python experiments/runners/verify_combined_outputs.py \\
        runs/Exp-100/smoke_boundary_pair/seed_0000_<ts>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_EVO_FILES = (
    "trace.parquet",
    "convergence.parquet",
    "stats.json",
    "context.json",
    "origin.png",
)

REQUIRED_PDQ_FILES = (
    "sut_calls.parquet",
    "candidates.parquet",
    "stage1_flips.parquet",
    "stage2_trajectories.parquet",
    "archive.parquet",
    "convergence.parquet",
    "config.json",
    "context.json",
)

PDQ_V5_ARCHIVE_COLS = {
    "pareto_idx", "evolutionary_gen", "anchor_source",
}


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)


def _ok(msg: str) -> None:
    print(f"OK:   {msg}")


def verify(seed_dir: Path) -> int:
    """Return 0 on all-OK, 1 on any failure."""
    failed = False

    if not seed_dir.exists():
        _fail(f"seed_dir does not exist: {seed_dir}")
        return 1

    # ---- Top-level files --------------------------------------------------
    for name in ("config.json", "manifest.json"):
        path = seed_dir / name
        if not path.exists():
            _fail(f"missing {name}")
            failed = True
        else:
            _ok(f"top-level {name} present")

    # ---- Evolutionary subdir ---------------------------------------------
    # Boundary-pair runner passes the evo_dir directly to the tester as
    # run_dir, so artifacts land at <seed_dir>/evolutionary/* (no nested
    # ``<name>_seed_<i>_<ts>/`` wrapper).
    evo_dir = seed_dir / "evolutionary"
    if not evo_dir.exists():
        _fail("evolutionary/ subdir missing")
        return 1
    for name in REQUIRED_EVO_FILES:
        if not (evo_dir / name).exists():
            _fail(f"evolutionary/{name} missing")
            failed = True
        else:
            _ok(f"evolutionary/{name}")

    # ---- PDQ subdir ------------------------------------------------------
    pdq_dir = seed_dir / "pdq"
    if not pdq_dir.exists():
        _fail("pdq/ subdir missing")
        return 1
    for name in REQUIRED_PDQ_FILES:
        if not (pdq_dir / name).exists():
            _fail(f"pdq/{name} missing")
            failed = True
        else:
            _ok(f"pdq/{name}")

    # ---- Anchors --------------------------------------------------------
    anchors_dir = pdq_dir / "anchors"
    if not anchors_dir.exists():
        _fail("pdq/anchors/ missing")
        failed = True
    else:
        anchor_pngs = sorted(anchors_dir.glob("anchor_*.png"))
        anchor_jsons = sorted(anchors_dir.glob("anchor_*.json"))
        if not anchor_pngs:
            _fail("no anchor_*.png images")
            failed = True
        else:
            _ok(f"pdq/anchors/ has {len(anchor_pngs)} anchor image(s)")
        if len(anchor_pngs) != len(anchor_jsons):
            _fail(
                f"anchor png/json count mismatch: "
                f"{len(anchor_pngs)} vs {len(anchor_jsons)}"
            )
            failed = True

    # ---- manifest.json schema --------------------------------------------
    manifest_path = seed_dir / "manifest.json"
    manifest: dict = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for key in (
            "pipeline", "seed_idx", "class_a", "class_b",
            "n_pareto", "n_anchors_evaluated", "anchors",
        ):
            if key not in manifest:
                _fail(f"manifest.json missing key: {key}")
                failed = True
        if manifest.get("pipeline") != "boundary_pair":
            _fail(
                f"manifest pipeline={manifest.get('pipeline')!r} != 'boundary_pair'"
            )
            failed = True
        if manifest.get("n_anchors_evaluated", 0) < 1:
            _fail("manifest.n_anchors_evaluated < 1")
            failed = True
        else:
            _ok(
                f"manifest: n_pareto={manifest['n_pareto']}, "
                f"n_anchors_evaluated={manifest['n_anchors_evaluated']}"
            )

    # ---- archive.parquet schema + content -------------------------------
    archive_path = pdq_dir / "archive.parquet"
    if archive_path.exists():
        try:
            df = pd.read_parquet(archive_path)
        except Exception as exc:  # noqa: BLE001
            _fail(f"archive.parquet read failed: {exc!r}")
            return 1

        # v5 columns present.
        missing_cols = PDQ_V5_ARCHIVE_COLS - set(df.columns)
        if missing_cols:
            _fail(f"archive.parquet missing v5 cols: {missing_cols}")
            failed = True
        else:
            _ok("archive.parquet has PDQ v5 provenance columns")

        if len(df) == 0:
            _fail("archive.parquet has 0 rows — no flips minimised")
            failed = True
        else:
            _ok(f"archive.parquet has {len(df)} row(s)")

            # All rows are combined-pipeline rows.
            sources = set(df["anchor_source"].unique().tolist())
            if sources != {"evolutionary"}:
                _fail(
                    f"unexpected anchor_source values: {sources!r} "
                    "(expected only 'evolutionary')"
                )
                failed = True
            else:
                _ok("all archive rows have anchor_source='evolutionary'")

            # pareto_idx populated.
            pareto_idx_nulls = int(df["pareto_idx"].isna().sum())
            if pareto_idx_nulls > 0:
                _fail(f"{pareto_idx_nulls} archive row(s) have null pareto_idx")
                failed = True
            else:
                par_min, par_max = int(df["pareto_idx"].min()), int(
                    df["pareto_idx"].max()
                )
                _ok(f"pareto_idx range: [{par_min}, {par_max}]")

            # genotype_anchor non-zero (combined → never zero).
            sample_anchor = df["genotype_anchor"].iloc[0]
            if isinstance(sample_anchor, (list, np.ndarray)):
                anchor_arr = np.asarray(sample_anchor)
                if int(np.count_nonzero(anchor_arr)) == 0:
                    _fail(
                        "first archive row has all-zero genotype_anchor — "
                        "combined pipeline never uses zero anchor"
                    )
                    failed = True
                else:
                    _ok(
                        f"genotype_anchor non-zero "
                        f"(first row: {int(np.count_nonzero(anchor_arr))} active genes)"
                    )

            # validity should be VV (Stage 2 ran on every flip).
            validities = set(df["validity"].unique().tolist())
            _ok(f"validity classes seen: {validities}")

            # d_i_primary should be in genome-domain (rank_sum_delta).
            d_i_min = float(df["d_i_primary"].min())
            d_i_max = float(df["d_i_primary"].max())
            _ok(f"d_i_primary range: [{d_i_min:.1f}, {d_i_max:.1f}]")

            # archive row count consistency with manifest anchor summaries.
            if manifest and "anchors" in manifest:
                claimed_flips = sum(
                    int(a.get("n_stage2_flips", 0))
                    for a in manifest["anchors"]
                )
                if claimed_flips != len(df):
                    _fail(
                        f"manifest claims {claimed_flips} S2 flips but "
                        f"archive.parquet has {len(df)} rows"
                    )
                    failed = True
                else:
                    _ok(
                        "archive row count matches manifest "
                        f"({claimed_flips} S2 flips)"
                    )

    print()
    if failed:
        print(">>> VERIFY FAILED <<<")
        return 1
    print(">>> VERIFY OK <<<")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify combined-pipeline outputs for one seed run.",
    )
    parser.add_argument("seed_dir", type=Path)
    args = parser.parse_args()
    sys.exit(verify(args.seed_dir))


if __name__ == "__main__":
    main()
