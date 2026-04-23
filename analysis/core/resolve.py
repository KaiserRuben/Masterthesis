"""Resolve class names to seed directories across all runs.

Usage:
    from analysis.core.resolve import find_seeds

    # Find all seeds where class_a contains "brambling"
    find_seeds(runs_dir, class_a="brambling")

    # Find a specific pair
    find_seeds(runs_dir, class_a="goldfish", class_b="goldfinch")
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def _latest_pdq_seeds(run_dir: Path) -> dict[int, Path]:
    """Map seed index → latest-timestamp directory for PDQ runs."""
    pat = re.compile(r"^seed_(\d{4})_(\d+)$")
    by_idx: dict[int, tuple[int, Path]] = {}
    for d in run_dir.iterdir():
        if not d.is_dir():
            continue
        m = pat.match(d.name)
        if not m:
            continue
        idx, ts = int(m.group(1)), int(m.group(2))
        if idx not in by_idx or ts > by_idx[idx][0]:
            by_idx[idx] = (ts, d)
    return {idx: path for idx, (_, path) in by_idx.items()}


def find_seeds(
    runs_dir: Path,
    class_a: str | None = None,
    class_b: str | None = None,
    pipeline: str | None = None,
) -> list[dict]:
    """Find seed directories matching class name filters.

    Matches are case-insensitive substring matches, so "brambling"
    matches "brambling" and "shark" matches "hammerhead shark".

    Returns list of dicts with keys:
        run, pipeline, seed_dir, class_a, class_b, stats
    """
    runs_dir = Path(runs_dir)
    results = []

    def _match(value: str, pattern: str) -> bool:
        return pattern.lower() in value.lower()

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("."):
            continue

        is_pdq = run_dir.name.startswith("pdq")

        if pipeline and pipeline.lower() == "smoo" and is_pdq:
            continue
        if pipeline and pipeline.lower() == "pdq" and not is_pdq:
            continue

        if is_pdq:
            latest = _latest_pdq_seeds(run_dir)
            for idx, sd in sorted(latest.items()):
                stats_path = sd / "stats.json"
                if not stats_path.exists():
                    continue
                try:
                    with open(stats_path) as f:
                        stats = json.load(f)
                    if not stats:
                        continue
                except (json.JSONDecodeError, ValueError):
                    continue

                ca = stats.get("label_anchor", stats.get("class_a", ""))
                cb = stats.get("class_b", "")

                if class_a and not _match(ca, class_a):
                    continue
                if class_b and not _match(cb, class_b):
                    continue

                results.append({
                    "run": run_dir.name,
                    "pipeline": "pdq",
                    "seed_dir": sd,
                    "class_a": ca,
                    "class_b": cb,
                    "stats": stats,
                })
        else:
            for sd in sorted(run_dir.iterdir()):
                if not sd.is_dir() or not sd.name.startswith("vlm_boundary_seed_"):
                    continue
                stats_path = sd / "stats.json"
                if not stats_path.exists():
                    continue
                with open(stats_path) as f:
                    stats = json.load(f)

                ca = stats.get("class_a", "")
                cb = stats.get("class_b", "")

                if class_a and not _match(ca, class_a):
                    continue
                if class_b and not _match(cb, class_b):
                    continue

                results.append({
                    "run": run_dir.name,
                    "pipeline": "smoo",
                    "seed_dir": sd,
                    "class_a": ca,
                    "class_b": cb,
                    "stats": stats,
                })

    return results


def list_classes(runs_dir: Path) -> None:
    """Print all available class pairs."""
    seeds = find_seeds(runs_dir)
    pairs: dict[str, set[str]] = {}
    for s in seeds:
        key = f"{s['pipeline']:4s}  {s['class_a']} vs {s['class_b']}"
        pairs.setdefault(key, set()).add(s["run"])

    print(f"{'Pipeline':<6} {'class_a':<25} {'class_b':<25} {'runs'}")
    print("-" * 90)
    for key in sorted(pairs):
        runs = ", ".join(sorted(pairs[key]))
        print(f"{key:<58} {runs}")
