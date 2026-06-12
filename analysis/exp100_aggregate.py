"""Stats-aggregator for Exp-100 (combinatorial class geometry).

Walks ``runs/Exp-100/<run_name>/exp100_*_seed_*_*/`` directories,
loads each ``stats.json`` + ``convergence.parquet``, and produces a
tidy one-row-per-seed DataFrame ready for H1–H5 evaluation.

The aggregator is **defensive**: missing files (run still in progress),
absent ``seed_metadata`` (gap_filter runs accidentally pointing at the
same dir), unparseable parquet (concurrent flush) all log a warning and
skip the seed without aborting the batch. So the same script can be
called repeatedly during a long run to track progress.

Output columns (one row per seed):

Identifying:
    seed_idx, run, seed_dir
    anchor_class_concrete, target_class_concrete   (always L0)
    level_anchor, level_target                     (0/1/2)
    anchor_label_in_prompt, target_label_in_prompt
    common_ancestor_level                          (None/1/2)
    seed_idx_in_class                              (0..N-1)
    anchor_position, target_position               (class_list indices)

Derived bucket / cell labels:
    bucket_anchor, bucket_target                   (L2 super-cat)
    cell_kind                                      ("within" | "across")
    cross_subkind                                  ("animal-animal" |
                                                    "animal-artifact" |
                                                    "artifact-artifact" |
                                                    None when cell_kind=="within")
    is_diagonal                                    (level_anchor == level_target)
    is_symmetric_pair                              (anchor_position < target_position)

Outcome metrics (from convergence + stats):
    min_TgtBal                  smallest pareto_min_TgtBal across all gens
    min_TgtBal_at_gen           gen at which the minimum was reached
    final_pareto_min_TgtBal     pareto_min_TgtBal at the last logged gen
    d_img_at_min_TgtBal         pareto_atbest_TgtBal_MatrixDistance_fro at the same gen
                                (drift of the candidate that achieved best TgtBal,
                                NOT pareto_min_MatrixDistance — the origin sits on
                                the Pareto front so the latter is always 0)
    d_text_at_min_TgtBal        pareto_atbest_TgtBal_TextDist at the same gen
    n_gens_completed            number of generations actually logged
    early_stop_trigger          "flip" | "plateau" | "no_improvement" |
                                "hard_cap" | None
    early_stop_gen              generation at which trigger fired
    runtime_s                   total wall-clock seconds for this seed
    flipped                     bool — early_stop_trigger == "flip"

Usage::

    python -m analysis.exp100_aggregate \\
        --runs-root runs/Exp-100 \\
        --out runs/Exp-100/poc_aggregate.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data.taxonomy import cluster_of  # noqa: E402

logger = logging.getLogger(__name__)


# Animal vs artifact L2 buckets — used to label cross-bucket sub-structure.
# This is hard-coded for the PoC roster (bird, reptile, musical instrument);
# extend when the main run adds more buckets.
ANIMAL_L2 = frozenset({
    "bird", "reptile", "mammal", "fish", "sea animal", "insect", "amphibian",
    "arthropod", "invertebrate", "ungulate", "snake", "dog", "cat",
})
ARTIFACT_L2 = frozenset({
    "musical instrument", "instrument", "tool", "vehicle", "container",
    "structure", "furniture", "clothing", "textile", "machine",
    "sports equipment", "object",
})


def _classify_l2(label: str | None) -> str:
    if label is None:
        return "unknown"
    if label in ANIMAL_L2:
        return "animal"
    if label in ARTIFACT_L2:
        return "artifact"
    return "other"


def _cross_subkind(anchor_l2: str | None, target_l2: str | None) -> str | None:
    """Categorise a cross-bucket pair as animal-animal / animal-artifact / etc."""
    a = _classify_l2(anchor_l2)
    t = _classify_l2(target_l2)
    pair = tuple(sorted([a, t]))
    if pair == ("animal", "animal"):
        return "animal-animal"
    if pair == ("animal", "artifact"):
        return "animal-artifact"
    if pair == ("artifact", "artifact"):
        return "artifact-artifact"
    return f"{pair[0]}-{pair[1]}"


def _safe_load_json(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not parse %s: %s", path, e)
        return None


def _safe_load_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        logger.warning("Could not read %s: %s", path, e)
        return None


def _resolve_evo_dir(seed_dir: Path) -> Path:
    """Locate the evolutionary artefact dir for both supported layouts.

    Flat (original PoC):      <seed_dir>/stats.json
    Boundary-pair (schema 5): <seed_dir>/evolutionary/stats.json
    """
    if (seed_dir / "evolutionary" / "stats.json").exists():
        return seed_dir / "evolutionary"
    return seed_dir


def _seed_row(seed_dir: Path) -> dict[str, Any] | None:
    """Build one tidy row from a single seed directory.

    Returns None and logs when essential files are missing or unparseable.
    """
    evo_dir = _resolve_evo_dir(seed_dir)
    stats = _safe_load_json(evo_dir / "stats.json")
    if stats is None:
        return None  # Run probably still in progress.

    meta = stats.get("seed_metadata")
    if meta is None:
        logger.warning(
            "Seed %s has no seed_metadata — gap_filter run? Skipping.",
            seed_dir.name,
        )
        return None

    anchor = meta["anchor_class_concrete"]
    target = meta["target_class_concrete"]

    bucket_a = cluster_of(anchor, level=2)
    bucket_t = cluster_of(target, level=2)
    cancestor = meta["common_ancestor_level"]
    cell_kind = "within" if cancestor is not None else "across"

    # The tester writes ``stats["early_stop"]`` only when a *non-hard_cap*
    # trigger fires (flip / plateau / no_improvement). Hitting the
    # generation budget is recorded as ``early_stop`` absent — we
    # backfill it here as ``"hard_cap"`` to keep the column complete.
    es = stats.get("early_stop") or {}
    row: dict[str, Any] = {
        # Identifying
        "seed_idx": stats["seed_idx"],
        "run": seed_dir.parent.name,
        "seed_dir": seed_dir.name,
        # Metadata from roster pipeline
        "anchor_class_concrete": anchor,
        "target_class_concrete": target,
        "level_anchor": meta["level_anchor"],
        "level_target": meta["level_target"],
        "anchor_label_in_prompt": meta["anchor_label_in_prompt"],
        "target_label_in_prompt": meta["target_label_in_prompt"],
        "common_ancestor_level": cancestor,
        "seed_idx_in_class": meta["seed_idx_in_class"],
        "anchor_position": meta["anchor_position"],
        "target_position": meta["target_position"],
        # Derived bucket structure
        "bucket_anchor": bucket_a,
        "bucket_target": bucket_t,
        "cell_kind": cell_kind,
        "cross_subkind": _cross_subkind(bucket_a, bucket_t)
                          if cell_kind == "across" else None,
        "is_diagonal": meta["level_anchor"] == meta["level_target"],
        "is_forward": meta["anchor_position"] < meta["target_position"],
        # Run outcome — runtime + early-stop
        "runtime_s": stats.get("runtime_seconds"),
        "early_stop_trigger": es.get("trigger") or "hard_cap",
        "early_stop_gen": es.get("generation"),
    }
    row["flipped"] = row["early_stop_trigger"] == "flip"

    # Boundary-pair pipeline (schema 5) writes a manifest with the PDQ
    # stage summary next to the evolutionary dir — fold the headline
    # numbers in when present so H-evaluation can condition on flips.
    manifest = _safe_load_json(seed_dir / "manifest.json")
    if manifest and manifest.get("pipeline") == "boundary_pair":
        anchors = manifest.get("anchors") or []
        row["n_pareto"] = manifest.get("n_pareto")
        row["n_anchors_evaluated"] = manifest.get("n_anchors_evaluated")
        row["n_stage1_flips_total"] = sum(a.get("n_stage1_flips", 0) for a in anchors)
        row["n_stage2_flips_total"] = sum(a.get("n_stage2_flips", 0) for a in anchors)
        row["n_distinct_targets_max"] = max(
            (a.get("n_distinct_targets", 0) for a in anchors), default=0,
        )
        row["anchor_min_p_gap"] = min(
            (abs(a["p_a"] - a["p_b"]) for a in anchors
             if a.get("p_a") is not None and a.get("p_b") is not None),
            default=float("nan"),
        )

    # Convergence-derived metrics — pull min over all logged generations.
    conv = _safe_load_parquet(evo_dir / "convergence.parquet")
    if conv is not None and len(conv) > 0:
        row["n_gens_completed"] = int(conv["generation"].max() + 1)
        row["final_pareto_min_TgtBal"] = float(conv["pareto_min_TgtBal"].iloc[-1])
        idx_min = int(conv["pareto_min_TgtBal"].idxmin())
        row["min_TgtBal"] = float(conv.loc[idx_min, "pareto_min_TgtBal"])
        row["min_TgtBal_at_gen"] = int(conv.loc[idx_min, "generation"])
        # Drift OF the candidate that achieved best TgtBal at idx_min.
        # `pareto_min_*` is the per-gen min over the whole Pareto front and
        # is identically 0 for both distances (origin = no manipulation
        # always sits on the front), so we read the `atbest_TgtBal_*` columns
        # instead — those track the candidate that scored the current best
        # TgtBal at that generation.
        row["d_img_at_min_TgtBal"] = float("nan")
        for col in conv.columns:
            if col.startswith("pareto_atbest_TgtBal_MatrixDistance"):
                row["d_img_at_min_TgtBal"] = float(conv.loc[idx_min, col])
                break
        row["d_text_at_min_TgtBal"] = float("nan")
        for col in conv.columns:
            if col.startswith("pareto_atbest_TgtBal_Text"):
                row["d_text_at_min_TgtBal"] = float(conv.loc[idx_min, col])
                break
    else:
        row["n_gens_completed"] = 0
        row["final_pareto_min_TgtBal"] = float("nan")
        row["min_TgtBal"] = float("nan")
        row["min_TgtBal_at_gen"] = -1
        row["d_img_at_min_TgtBal"] = float("nan")
        row["d_text_at_min_TgtBal"] = float("nan")

    return row


def aggregate_run(run_dir: Path) -> pd.DataFrame:
    """Aggregate one Exp-100 run directory into a tidy per-seed DataFrame.

    Walks ``run_dir/exp100_*_seed_*_*/`` subdirectories. Tolerates
    seeds whose stats.json is missing (still in progress) — these are
    skipped with a warning, not an error.
    """
    run_dir = Path(run_dir)
    seed_dirs = sorted(
        p for p in run_dir.iterdir()
        if p.is_dir() and ("_seed_" in p.name or p.name.startswith("seed_"))
    )
    rows: list[dict[str, Any]] = []
    for sd in seed_dirs:
        row = _seed_row(sd)
        if row is not None:
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _looks_like_run_dir(path: Path) -> bool:
    """A run dir contains at least one ``*_seed_*`` subdirectory."""
    if not path.is_dir():
        return False
    return any(
        p.is_dir() and ("_seed_" in p.name or p.name.startswith("seed_"))
        for p in path.iterdir()
    )


def aggregate_runs_root(runs_root: Path) -> pd.DataFrame:
    """Aggregate every Exp-100 run directory at or below ``runs_root``.

    Tolerates both layouts:
      runs_root/<seed_dir>/...                      (runs_root IS the run)
      runs_root/<run_dir>/<seed_dir>/...            (one level of nesting)
    """
    runs_root = Path(runs_root)
    if not runs_root.exists():
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    if _looks_like_run_dir(runs_root):
        frames.append(aggregate_run(runs_root))
    else:
        for child in sorted(runs_root.iterdir()):
            if _looks_like_run_dir(child):
                frames.append(aggregate_run(child))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate Exp-100 PoC stats into a tidy DataFrame.",
    )
    parser.add_argument(
        "--runs-root", type=Path,
        default=REPO / "runs" / "Exp-100",
        help="Root directory containing Exp-100 run directories.",
    )
    parser.add_argument(
        "--out", type=Path,
        default=REPO / "runs" / "Exp-100" / "poc_aggregate.parquet",
        help="Output parquet path for the tidy DataFrame.",
    )
    parser.add_argument(
        "--print-summary", action="store_true",
        help="Print a quick coverage summary (n seeds per cell-kind / bucket).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    df = aggregate_runs_root(args.runs_root)
    if df.empty:
        print(f"No Exp-100 seeds found under {args.runs_root}")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"Wrote {len(df)} seed rows → {args.out}")

    if args.print_summary:
        print()
        print("=== Coverage summary ===")
        print(f"Total seeds: {len(df)}")
        print()
        print("By cell_kind:")
        print(df["cell_kind"].value_counts().to_string())
        print()
        print("By cross_subkind (across-bucket only):")
        print(df.loc[df["cell_kind"] == "across", "cross_subkind"]
              .value_counts().to_string())
        print()
        print("By early_stop_trigger:")
        print(df["early_stop_trigger"].value_counts(dropna=False).to_string())
        print()
        print("Flip rate by cell_kind:")
        print(df.groupby("cell_kind")["flipped"].mean().to_string())


if __name__ == "__main__":
    main()
