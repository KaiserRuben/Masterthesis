#!/usr/bin/env python3
"""HS-01 — Stage pool candidates from data_raw into the study's NEED shape.

Layer 2 of the data process:
    data_raw/  (provenance-shaped raw extracts, audited)
        -> pool_staging/  (THIS: phase/stratum-shaped candidate sets)
            -> pool freeze (final item selection + assets, separate step)

Writes pool_staging/{pair,image,text}/<stratum>/candidates.parquet plus a
README.md into every stratum that is currently empty (pending data/decisions),
and a STAGING_REPORT.md with counts vs. the design targets.

All stratum thresholds are PROVISIONAL (open decisions, HS-01 spec §9/§10).
Re-run after every data_raw refresh; the script is idempotent (full rebuild).
"""
from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "data_raw"
OUT = HERE / "pool_staging"
REPO = HERE.parent.parent

# ----- PROVISIONAL thresholds (pending pool-freeze decisions) ---------------
TGTBAL_MAX = 1e-2          # qualifying cut (already enforced in data_raw)
IMG_HEAVY_DTEXT_MAX = 0.20  # weak proxy; stratum officially pending HS-GEN runs
TEXT_HEAVY_DIMG_MAX = 0.01
TEXT_HEAVY_DTEXT_MIN = 0.40
BAL_DTEXT = (0.20, 0.60)
BAL_DIMG_MIN = 0.001
TEXT_BANDS = {"low_drift": (0.0, 0.30), "medium_drift": (0.30, 0.55), "high_drift": (0.55, float("inf"))}

# Design targets (HS-01 spec §3)
NEED = {
    ("pair", "baseline"): 8, ("pair", "image_heavy"): 14, ("pair", "text_heavy"): 14, ("pair", "balanced"): 14,
    ("image", "raw"): 6, ("image", "roundtrip"): 6, ("image", "boundary_joint"): 12, ("image", "image_heavy"): 6,
    ("text", "clean"): 6, ("text", "low_drift"): 8, ("text", "medium_drift"): 8, ("text", "high_drift"): 8,
    ("checks", "synthetic"): 2,
}

# Strata still genuinely empty/manual after the HS-GEN-01 + Exp-101q ingest.
# (pair/image_heavy + image/image_heavy are now filled from HS-GEN-01;
#  the Qwen arm is now ingested via Exp-101q — both no longer pending.)
PENDING = {
    ("image", "raw"): "ImageNet source paths were never logged by the pipeline. "
                      "Pending: host-side recovery via seed roster, or drop the raw micro-control.",
    ("checks", "synthetic"): "2 attention-check stimuli (1 nonsense prompt, 1 obvious-class pair) — "
                             "constructed by hand at pool freeze.",
}


def load_raw() -> pd.DataFrame:
    # Both SUTs, every experiment incl. HS-GEN-01 (image-only) and Exp-101q
    # (Qwen). Glob excludes Exp-27's NONQUALIFYING_best3_reference.parquet by
    # name. `sut` / `modality` columns keep the SUT and channel separable.
    frames = []
    for p in sorted(RAW.glob("*/*/boundary_individuals*.parquet")):
        frames.append(pd.read_parquet(p))
    d = pd.concat(frames, ignore_index=True)
    assert (d.tgtbal <= TGTBAL_MAX).all(), "non-qualifying row in data_raw"
    # d_text_embed is NaN for image_only rows (no TextDist); coerce so the
    # mixed-source column stays numeric for the band comparisons below.
    d["d_text_embed"] = pd.to_numeric(d["d_text_embed"], errors="coerce")
    d["png_ready"] = d.pareto_png.notna()
    d["q_le_1e3"] = d.tgtbal <= 1e-3
    return d


def origin_assets(d: pd.DataFrame) -> pd.DataFrame:
    """One row per run: round-trip origin image + clean prompt (baseline / clean / roundtrip strata)."""
    runs = d.drop_duplicates("run_dir")[
        ["experiment", "run_dir", "run_rel", "anchor_class_concrete", "target_class_concrete",
         "anchor_label_in_prompt", "target_label_in_prompt", "level_anchor", "level_target",
         "prompt_template", "answer_format", "sut_model_id", "seed_idx"]
    ].copy()
    paths, exists = [], []
    for rel in runs.run_rel:
        cands = list((REPO / rel).glob("**/origin.png"))
        paths.append(str(cands[0].relative_to(REPO)) if cands else None)
        exists.append(bool(cands))
    runs["origin_png"], runs["origin_png_exists"] = paths, exists
    return runs


def write(df: pd.DataFrame, phase: str, stratum: str, name: str = "candidates.parquet") -> None:
    p = OUT / phase / stratum
    p.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p / name, index=False)


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    d = load_raw()
    runs = origin_assets(d)

    # Strict image-heavy = image_only_drift (0 active text genes). These exist
    # only from the HS-GEN-01 image_only campaign; joint runs never produce them
    # (boundary positioning is text-paid). Feeds BOTH image-heavy strata.
    img_heavy = d[d.drift_class == "image_only_drift"]

    # ---- PAIR ----
    strict_th = d.drift_class == "text_only_drift"
    prag_th = (d.d_img_matrix < TEXT_HEAVY_DIMG_MAX) & (d.d_text_embed > TEXT_HEAVY_DTEXT_MIN)
    text_heavy = d[strict_th | prag_th].assign(strict_text_only=lambda x: x.drift_class == "text_only_drift")
    balanced = d[d.d_text_embed.between(*BAL_DTEXT) & (d.d_img_matrix > BAL_DIMG_MIN) & ~(strict_th | prag_th)]
    write(text_heavy, "pair", "text_heavy")
    write(balanced, "pair", "balanced")
    write(runs, "pair", "baseline")
    write(img_heavy, "pair", "image_heavy")            # strict; supersedes the old weak proxy

    # ---- IMAGE ----
    # boundary_joint = joint-modality boundary images (exclude image_only, which
    # is the image_heavy stratum); prefer rows with rendered PNGs.
    write(d[d.png_ready & (d.modality == "joint")], "image", "boundary_joint")
    write(runs, "image", "roundtrip")
    (OUT / "image" / "raw").mkdir(parents=True, exist_ok=True)
    write(img_heavy, "image", "image_heavy")           # strict image_only_drift

    # ---- TEXT ----
    write(runs, "text", "clean")  # clean prompt = prompt_template per run
    for band, (lo, hi) in TEXT_BANDS.items():
        write(d[(d.d_text_embed >= lo) & (d.d_text_embed < hi)], "text", band)

    # ---- CHECKS ----
    (OUT / "checks" / "synthetic").mkdir(parents=True, exist_ok=True)

    # ---- READMEs for pending strata ----
    for (phase, stratum), note in PENDING.items():
        if phase == "__qwen__":
            (OUT / "README_QWEN_PENDING.md").write_text(note + "\n")
        else:
            (OUT / phase / stratum / "README.md").write_text(
                f"# {phase}/{stratum} — EMPTY / PENDING\n\nTarget: {NEED[(phase, stratum)]} items.\n\n{note}\n")

    # ---- report ----
    lines = [f"# HS-01 Pool Staging Report — {date.today()}", "",
             "Provisional thresholds — see script header. Counts are CANDIDATES, not selected items.",
             "", "| phase/stratum | target | candidates | runs | <=1e-3 | png-ready |", "|---|---|---|---|---|---|"]
    for (phase, stratum), target in NEED.items():
        p = OUT / phase / stratum
        files = list(p.glob("*.parquet"))
        if files:
            c = pd.read_parquet(files[0])
            is_ind = "tgtbal" in c.columns
            lines.append(f"| {phase}/{stratum} | {target} | {len(c)} | "
                         f"{c.run_dir.nunique() if 'run_dir' in c else '—'} | "
                         f"{int(c.q_le_1e3.sum()) if is_ind else '—'} | "
                         f"{int(c.png_ready.sum()) if is_ind else '—'} |"
                         + (" (weak proxy only)" if "weak_proxy" in files[0].name else ""))
        else:
            lines.append(f"| {phase}/{stratum} | {target} | **0 — pending** | — | — | — |")
    sut_str = ", ".join(f"{k}={v}" for k, v in sorted(d.groupby("sut").size().items()))
    n_imgonly = int((d.drift_class == "image_only_drift").sum())
    lines += ["",
              f"Source rows: {len(d)} qualifying individuals across {d.run_dir.nunique()} runs "
              f"and {d.sut.nunique()} SUT(s) ({sut_str}).",
              f"Strict image_only_drift (HS-GEN-01) feeding both image_heavy strata: {n_imgonly} rows.",
              f"origin.png resolved for {int(runs.origin_png_exists.sum())}/{len(runs)} runs."]
    (OUT / "STAGING_REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
