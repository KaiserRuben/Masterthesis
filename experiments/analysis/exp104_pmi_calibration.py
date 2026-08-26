"""Exp-104 Phase A — PMI calibration of the label-walls (diagnostic).

Tests whether the label-walls found in Exp-101/101q/102 are decision geometry
or surface-form-prior artefacts (Holtzman et al. 2021, "Surface Form
Competition"). Recasts the boundary signal from the raw log-prob gap to the
pointwise-mutual-information gap, so the answer-string prior cancels out:

    g_AB(m)      = lp_anchor(m)  - lp_target(m)          (signed, from trace)
    Δ∅           = lp_anchor(∅)  - lp_target(∅)          (null-prompt prior gap)
    g_AB^PMI(m)  = | g_AB(m) - Δ∅ |

∅ is a content-neutral image (gray / black / noise) under the SAME pair prompt.
Δ∅ is constant per class-pair -> two forward passes, cached -> ~free.

Per cell we report the raw floor (best-ever |g_AB| across the archived trace)
and the PMI floor (min |g_AB - Δ∅|). Wall stands under PMI -> decision
geometry; wall collapses (PMI floor -> ~0) -> prior artefact.

Faithful to the tester's scoring path: same prompt construction as
``VLMSUT`` (src/sut/vlm_sut.py), ``score_categories_tensor`` with no thinking,
length-normalized log-probs — the exact quantity stored in ``trace.parquet``
(``logprobs`` column, src/evolutionary/vlm_boundary_tester.py). A zero-genotype
gen-0 individual is re-scored from its origin image as an end-to-end self-check.

Usage (from repo root, `conda run -n uni`):
    python experiments/analysis/exp104_pmi_calibration.py \
        --runs-glob "runs/Exp-101q/exp101q_margin_predictor_qwen_seed_*" \
        --model Qwen/Qwen3.5-4B --device mps

The LLaVA column (Exp-101/102, OpenVINO/Arc) is produced by pointing
``--runs-glob`` at runs/Exp-101 with ``--model``/``--device`` for that host.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.abspath("."))
from src.sut.scorer import create_scorer  # noqa: E402


def build_prompt(prompt_template: str, answer_format: str, a_word: str, t_word: str) -> str:
    """Reproduce VLMSUT's prompt: template + answer_format over ", ".join(pair)."""
    return prompt_template + answer_format.format(categories=", ".join([a_word, t_word]))


def make_nulls(size: tuple[int, int]) -> dict[str, Image.Image]:
    """Content-neutral null images at the origin's size."""
    w, h = size
    rng = np.random.default_rng(0)
    return {
        "gray": Image.new("RGB", (w, h), (128, 128, 128)),
        "black": Image.new("RGB", (w, h), (0, 0, 0)),
        "white": Image.new("RGB", (w, h), (255, 255, 255)),
        "noise": Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), "RGB"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-glob", default="runs/Exp-101q/exp101q_margin_predictor_qwen_seed_*")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--backend", default="torch")
    ap.add_argument("--processor-id", default=None,
                    help="HF processor id (required for OpenVINO LLaVA)")
    ap.add_argument("--ov-device", default="GPU", help="OpenVINO device")
    ap.add_argument("--out", default="experiments/analysis/output/exp104")
    args = ap.parse_args()

    runs = sorted(glob.glob(args.runs_glob))
    if not runs:
        raise SystemExit(f"no runs matched {args.runs_glob!r}")
    os.makedirs(args.out, exist_ok=True)

    print(f"[load] {args.model} on {args.device} ({args.backend}) ...", flush=True)
    scorer = create_scorer(
        model_id=args.model, device=args.device, backend=args.backend,
        processor_id=args.processor_id, ov_device=args.ov_device,
    )
    print("[load] done", flush=True)

    def norm_lp_pair(image: Image.Image, prompt: str, cats: tuple[str, str]) -> np.ndarray:
        return scorer.score_categories_tensor(image, prompt, list(cats)).numpy().astype(np.float64)

    sample_origin = Image.open(os.path.join(runs[0], "origin.png")).convert("RGB")
    nulls = make_nulls(sample_origin.size)
    print(f"[null] origin size {sample_origin.size}", flush=True)

    prior_cache: dict[tuple, dict] = {}
    rows: list[dict] = []
    selfcheck: dict = {}

    for i, d in enumerate(runs):
        st = json.load(open(os.path.join(d, "stats.json")))
        m = st["seed_metadata"]
        a_word, t_word = m["anchor_label_in_prompt"], m["target_label_in_prompt"]
        prompt = build_prompt(st["prompt_template"], st["answer_format"], a_word, t_word)
        cats = (a_word, t_word)

        tr = pd.read_parquet(os.path.join(d, "trace.parquet"))
        lp = np.stack(tr["logprobs"].to_numpy())        # (N,2) [anchor, target]
        gap = lp[:, 0] - lp[:, 1]
        raw_floor = float(np.abs(gap).min())

        ckey = (prompt, cats)
        if ckey not in prior_cache:
            pr = {}
            for name, img in nulls.items():
                v = norm_lp_pair(img, prompt, cats)
                pr[name] = float(v[0] - v[1])
            prior_cache[ckey] = pr
        pr = prior_cache[ckey]

        rec = {
            "seed": st["seed_idx"], "anchor": m["anchor_class_concrete"],
            "target": m["target_class_concrete"], "la": m["level_anchor"],
            "lt": m["level_target"], "a_word": a_word, "t_word": t_word,
            "raw_floor": raw_floor, "gap_med": float(np.median(gap)),
        }
        for name in nulls:
            d0 = pr[name]
            rec[f"d0_{name}"] = d0
            rec[f"pmi_floor_{name}"] = float(np.abs(gap - d0).min())
        rec["d0"] = rec["d0_gray"]
        rec["pmi_floor"] = rec["pmi_floor_gray"]
        rec["explained_frac"] = (
            (raw_floor - rec["pmi_floor"]) / raw_floor if raw_floor > 1e-9 else np.nan
        )
        rows.append(rec)

        if i == 0:
            gen0 = tr[tr["generation"] == 0].copy()
            gen0["nz"] = gen0["genotype"].apply(lambda g: int(np.count_nonzero(np.asarray(g))))
            z = gen0.sort_values("nz").iloc[0]
            origin = Image.open(os.path.join(d, "origin.png")).convert("RGB")
            fresh = norm_lp_pair(origin, prompt, cats)
            selfcheck = {
                "min_nonzero_genes": int(z["nz"]),
                "stored_logprobs": list(map(float, z["logprobs"])),
                "fresh_origin_logprobs": fresh.tolist(),
                "abs_diff": [abs(a - b) for a, b in zip(z["logprobs"], fresh)],
            }
        print(f"[{i+1}/{len(runs)}] seed {st['seed_idx']:>3} {a_word}->{t_word} "
              f"raw={raw_floor:.3f} d0={rec['d0']:+.3f} pmi={rec['pmi_floor']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "exp104_pmi.csv"), index=False)
    json.dump(selfcheck, open(os.path.join(args.out, "exp104_selfcheck.json"), "w"), indent=2)

    print("\n===== SELF-CHECK (faithful scoring path) =====")
    print(json.dumps(selfcheck, indent=2))
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 40)
    print("\n===== boa-constrictor TARGET cells (Qwen wall candidates) =====")
    sub = df[df.target == "boa constrictor"].sort_values(["lt", "anchor"])
    print(sub[["seed", "a_word", "t_word", "lt", "raw_floor", "d0", "pmi_floor",
               "explained_frac", "d0_black", "d0_noise"]].to_string(index=False))
    print("\n===== 'constrictor' as ANCHOR word (slot control: boa->X) =====")
    anc = df[(df.anchor == "boa constrictor") & (df.a_word == "constrictor")]
    print(anc[["seed", "a_word", "t_word", "raw_floor", "d0", "pmi_floor",
               "explained_frac"]].to_string(index=False))
    print("\n===== full table sorted by raw_floor (desc) =====")
    print(df.sort_values("raw_floor", ascending=False)[
        ["seed", "a_word", "t_word", "raw_floor", "d0", "pmi_floor", "explained_frac"]
    ].to_string(index=False))
    print(f"\n[saved] {args.out}/exp104_pmi.csv")


if __name__ == "__main__":
    main()
