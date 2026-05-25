#!/usr/bin/env python3
"""Probe whether a candidate class survives a given SUT's within-roster contrast.

When `RosterSeedGenerator` fails for a class (pool exhaustion via misclass),
the typical fix is a class swap. This tool sanity-checks candidate swaps
*before* committing them to a config: load the SUT once, run N images of
the candidate through the same category-restricted scoring path that
`roster_seeds()` uses, and report whether the SUT can recognize the
candidate inside the roster contrast.

Reports per probe:
  - Top-1 hit rate (GT == argmax within roster classes)
  - Argmax-distribution histogram (where do misclasses go?)
  - gt_logprob distribution (when correct vs when misclass)
  - Suggested verdict (safe / borderline / will-fail)

Usage::

    python experiments/analysis/probe_class.py \\
        --config configs/Exp-100/poc_workstation_cone.yaml \\
        --candidate "snare drum" \\
        --n 20

    # Probe against an explicit roster contrast (overrides config):
    python experiments/analysis/probe_class.py \\
        --config configs/Exp-100/poc_workstation_cone.yaml \\
        --candidate "snare drum" \\
        --roster "junco,ostrich,green iguana,boa constrictor,cello,snare drum" \\
        --n 20

Reads the SUT, prompt_template, answer_format, and cache_dirs from the
config. Reads-only — nothing is written.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402

from experiments.runners.run_boundary_test import load_config  # noqa: E402
from src.data import ImageNetCache  # noqa: E402
from src.sut import VLMSUT  # noqa: E402

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("src").setLevel(logging.INFO)
logger = logging.getLogger("probe_class")
logger.setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config", required=True, type=Path,
        help="YAML config providing SUT + prompt_template + cache_dirs.",
    )
    p.add_argument(
        "--candidate", required=True, type=str, nargs="+",
        help="One or more candidate class names to probe (ImageNet L0 "
             "labels). When multiple are passed, the SUT is loaded once "
             "and each candidate probed sequentially.",
    )
    p.add_argument(
        "--roster", default=None, type=str,
        help="Comma-separated roster classes that form the contrast. "
             "Default: use config.seeds.roster.class_list. The candidate is "
             "added to the roster automatically if missing.",
    )
    p.add_argument(
        "--n", default=20, type=int,
        help="Number of images to probe (default: 20).",
    )
    return p.parse_args()


def _verdict(hit_rate: float, n_examined: int) -> str:
    if n_examined < 5:
        return "INDETERMINATE (n too small)"
    if hit_rate >= 0.5:
        return f"SAFE (≥50%% hit rate — Roster will fill 3 seeds within ~6 examined)"
    if hit_rate >= 0.25:
        return (
            "BORDERLINE (25-50% hit rate — Roster may fill 3 seeds within "
            "~12 examined; risky)"
        )
    return (
        "WILL FAIL (<25% hit rate — Roster will exhaust the ~50-image "
        "ImageNet-val pool before reaching 3 seeds)"
    )


def main() -> None:
    args = parse_args()

    cfg_dict = yaml.safe_load(args.config.read_text())
    cfg = load_config(cfg_dict)

    # Determine roster contrast
    if args.roster is not None:
        roster = tuple(c.strip() for c in args.roster.split(","))
    elif cfg.seeds.roster is not None:
        roster = tuple(cfg.seeds.roster.class_list)
    else:
        raise SystemExit(
            "No roster contrast available: pass --roster or use a config "
            "with seeds.mode=roster."
        )

    candidates = tuple(args.candidate)
    for c in candidates:
        if c not in roster:
            roster = roster + (c,)

    logger.info("Probing %s against roster %s", list(candidates), list(roster))

    # Build prompt — same construction as roster_seeds()
    answer_suffix = cfg.answer_format.format(categories=", ".join(roster))
    full_prompt = cfg.prompt_template + answer_suffix

    # Load SUT — VLMSUT takes the full config; per-call categories override
    # the config-default contrast set via process_input(categories=...).
    logger.info("Loading SUT %s ...", cfg.sut.model_id)
    sut = VLMSUT(cfg)
    logger.info("SUT loaded")

    data_source = ImageNetCache(dirs=cfg.cache_dirs)
    available_labels = set(data_source.labels())

    # Build prompt once — same for all candidates since the roster is fixed.
    answer_suffix = cfg.answer_format.format(categories=", ".join(roster))
    full_prompt = cfg.prompt_template + answer_suffix
    cat_to_idx = {c: i for i, c in enumerate(roster)}

    summary_rows: list[tuple[str, int, int, float, str]] = []

    for candidate in candidates:
        if candidate not in available_labels:
            suggestions = [
                lab for lab in available_labels
                if candidate.lower() in lab.lower()
            ][:5]
            print(
                f"\n[skip] {candidate!r} not in ImageNet labels. "
                f"Closest: {suggestions}"
            )
            continue

        samples = data_source.load_samples(
            categories=[candidate], n_per_class=args.n,
        )
        gt_idx = cat_to_idx[candidate]

        hits = 0
        n_examined = 0
        misclass_targets: Counter[str] = Counter()
        gt_logprobs_when_correct: list[float] = []
        gt_logprobs_when_wrong: list[float] = []

        for sample in samples:
            n_examined += 1
            with torch.no_grad():
                logprobs = sut.process_input(
                    sample.image, text=full_prompt, categories=roster,
                )
            top_idx = int(logprobs.argmax().item())
            gt_logprob = float(logprobs[gt_idx])
            if top_idx == gt_idx:
                hits += 1
                gt_logprobs_when_correct.append(gt_logprob)
            else:
                misclass_targets[roster[top_idx]] += 1
                gt_logprobs_when_wrong.append(gt_logprob)

        hit_rate = hits / max(n_examined, 1)
        verdict = _verdict(hit_rate, n_examined)

        # Per-candidate report
        print()
        print("=" * 70)
        print(f"Class-probe report — {candidate!r}")
        print("=" * 70)
        print(f"SUT:               {cfg.sut.model_id}")
        print(f"Roster contrast:   {list(roster)}")
        print(f"Examined:          {n_examined} images")
        print(f"GT == argmax:      {hits}/{n_examined} ({100 * hit_rate:.0f}%)")
        print()

        if gt_logprobs_when_correct:
            sorted_correct = sorted(gt_logprobs_when_correct)
            print(
                f"gt_logprob when CORRECT:   "
                f"min={min(sorted_correct):.3f}, "
                f"median={sorted_correct[len(sorted_correct) // 2]:.3f}, "
                f"max={max(sorted_correct):.3f}"
            )
        else:
            print("gt_logprob when CORRECT:   — (no hits)")

        if gt_logprobs_when_wrong:
            sorted_wrong = sorted(gt_logprobs_when_wrong)
            print(
                f"gt_logprob when MISCLASS:  "
                f"min={min(sorted_wrong):.3f}, "
                f"median={sorted_wrong[len(sorted_wrong) // 2]:.3f}, "
                f"max={max(sorted_wrong):.3f}"
            )
            print()
            print("Misclass histogram:")
            for lbl, n in misclass_targets.most_common():
                print(f"  {n:>3} × {lbl!r}")
        else:
            print("gt_logprob when MISCLASS:  — (all hits)")

        print()
        print(f"Verdict: {verdict}")
        summary_rows.append((candidate, hits, n_examined, hit_rate, verdict))

    # Multi-candidate summary
    if len(summary_rows) > 1:
        print()
        print("=" * 70)
        print("Multi-candidate summary (sorted by hit rate):")
        print("=" * 70)
        for cand, hits, n, rate, verd in sorted(
            summary_rows, key=lambda r: -r[3]
        ):
            print(f"  {rate*100:>3.0f}%  ({hits:>2}/{n:>2})  {cand!r:<25}  {verd}")
        print("=" * 70)


if __name__ == "__main__":
    main()
