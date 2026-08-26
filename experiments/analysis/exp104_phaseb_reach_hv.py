"""Exp-104 Phase B — reach + 2D-Pareto structure harness (E2 analysis).

For each seed-run (a cell × arm × seed), reconstruct BOTH boundary metrics from
the stored trace and score the framework's two promises:

  (R) reach     — does the search drive |g| to the boundary?
  (S) structure — does the (n_active, TgtBal) 2D-Pareto front expand?

Both metrics are derived from the length-normalized ``logprobs`` in
``trace.parquet`` (the signed gap that the stored abs ``fitness_TgtBal`` loses),
so raw and PMI floors are computable for EITHER arm:

    g_raw(m) = lp_a - lp_b                          (raw arm: lp is raw)
    g_pmi(m) = g_raw(m) - Δ∅                        Δ∅ = ℓ(a|∅) - ℓ(b|∅)
  (PMI arm stores corrected lp, so g_raw = g_corrected + Δ∅.)

Δ∅ comes from the PMI arm's ``stats["pmi_baseline"]`` when present, else from the
E0 prior-map CSV (``--e0-csv``, column ``d0_<null>``) keyed by seed_idx.

n_active = count of nonzero IMAGE genes (first ``image_dim`` of the genotype).
2D hypervolume is computed in the unit square (n_active / image_dim,
min(|g|, G_REF) / G_REF), reference point (1, 1) — higher = more of the
sparse-and-balanced region covered, comparable across cells and SUTs.

Usage (validate on archived raw runs; no E2 output needed yet):
    python experiments/analysis/exp104_phaseb_reach_hv.py \
        --runs-glob "runs/Exp-101q/exp101q_margin_predictor_qwen_seed_*" \
        --e0-csv experiments/analysis/output/exp104/exp104_pmi.csv --smoke

Real E2 (once launched):
    python experiments/analysis/exp104_phaseb_reach_hv.py \
        --runs-glob "runs/Exp-104/qwen_*_rep*/exp104_phaseb_qwen_*_seed_*" \
        --e0-csv experiments/analysis/output/exp104/exp104_pmi.csv \
        --out experiments/analysis/output/exp104/phaseb_qwen.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

EPS_REACH = 0.3   # nats; evidence-boundary reach bar (matches spec P2)
G_REF = 6.0       # nats; |g| cap for the HV reference point


def two_d_hv(n_active: np.ndarray, g: np.ndarray, image_dim: int) -> float:
    """Hypervolume of the (n_active, |g|) minimization front in the unit
    square with reference (1, 1). Both axes normalized (n_active/image_dim,
    min(|g|, G_REF)/G_REF); larger HV = more sparse-and-balanced coverage."""
    x = np.clip(n_active / max(image_dim, 1), 0.0, 1.0)
    y = np.clip(g / G_REF, 0.0, 1.0)
    pts = np.column_stack([x, y])
    # Non-dominated set (minimize both).
    order = np.argsort(pts[:, 0], kind="stable")
    pts = pts[order]
    front, best_y = [], np.inf
    for px, py in pts:
        if py < best_y - 1e-12:
            front.append((px, py))
            best_y = py
    # Sweep HV vs ref (1, 1): sum of (x_{i+1}-x_i)*(1-y_i).
    hv, prev_x = 0.0, 0.0
    for px, py in front:
        hv += (px - prev_x) * (1.0 - py)   # dominated strip up to this x
        prev_x = px
    hv += (1.0 - prev_x) * (1.0 - front[-1][1]) if front else 0.0
    return float(hv)


def analyze_run(d: str, e0: pd.DataFrame | None, null: str) -> dict | None:
    sp = os.path.join(d, "stats.json")
    tp = os.path.join(d, "trace.parquet")
    if not (os.path.exists(sp) and os.path.exists(tp)):
        return None
    st = json.load(open(sp))
    m = st.get("seed_metadata", {})
    seed_idx = st["seed_idx"]
    image_dim = int(st["image_dim"])
    pmi_on = bool(st.get("pmi_enabled", False))

    # Δ∅ for this cell (signed anchor-target prior gap, chosen null).
    if pmi_on and st.get("pmi_baseline"):
        b = st["pmi_baseline"]
        d0 = float(b[0] - b[1])
    elif e0 is not None and seed_idx in e0.index:
        d0 = float(e0.loc[seed_idx, f"d0_{null}"])
    else:
        d0 = np.nan

    tr = pd.read_parquet(tp)
    lp = np.stack(tr["logprobs"].to_numpy())          # (N, 2) arm-metric lp
    g_arm = lp[:, 0] - lp[:, 1]                        # signed gap in arm metric
    g_raw = g_arm if not pmi_on else g_arm + d0        # recover raw gap
    g_pmi = g_raw - d0                                 # evidence gap
    genos = tr["genotype"].to_numpy()
    n_active = np.array([int(np.count_nonzero(np.asarray(gt)[:image_dim])) for gt in genos])

    floor_raw = float(np.abs(g_raw).min())
    floor_pmi = float(np.abs(g_pmi).min()) if np.isfinite(d0) else np.nan
    rec = {
        "seed": seed_idx,
        "anchor": m.get("anchor_class_concrete"), "target": m.get("target_class_concrete"),
        "a_word": m.get("anchor_label_in_prompt"), "t_word": m.get("target_label_in_prompt"),
        "arm": "pmi" if pmi_on else "raw",
        "d0": d0, "image_dim": image_dim,
        "floor_own": float(np.abs(g_arm).min()),        # == min fitness_TgtBal
        "floor_raw": floor_raw, "floor_pmi": floor_pmi,
        "reached_evidence": bool(floor_pmi <= EPS_REACH) if np.isfinite(floor_pmi) else None,
        "crossed_raw": bool((g_raw < 0).any()),         # behavioural (deployed argmax)
        "crossed_pmi": bool((g_pmi < 0).any()) if np.isfinite(d0) else None,  # evidence
        "hv_raw": two_d_hv(n_active, np.abs(g_raw), image_dim),
        "hv_pmi": two_d_hv(n_active, np.abs(g_pmi), image_dim) if np.isfinite(d0) else np.nan,
        "n_active_min": int(n_active.min()), "n_active_max": int(n_active.max()),
    }
    # Distance-objective invariance check (scoring-independent objectives).
    for col, key in (("fitness_MatrixDistance_fro", "matdist_min"),
                     ("fitness_TextDist", "textdist_min")):
        if col in tr.columns:
            rec[key] = float(tr[col].min())
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-glob", required=True)
    ap.add_argument("--e0-csv", default="experiments/analysis/output/exp104/exp104_pmi.csv")
    ap.add_argument("--null", default="gray")
    ap.add_argument("--out", default="")
    ap.add_argument("--smoke", action="store_true",
                    help="Validation mode: check floor_own≈floor_raw on raw runs.")
    args = ap.parse_args()

    e0 = None
    if os.path.exists(args.e0_csv):
        e0 = pd.read_csv(args.e0_csv).set_index("seed")

    runs = sorted(glob.glob(args.runs_glob))
    if not runs:
        raise SystemExit(f"no runs matched {args.runs_glob!r}")

    rows = []
    for d in runs:
        rep_m = re.search(r"_rep(\d+)", d)
        rec = analyze_run(d, e0, args.null)
        if rec is None:
            continue
        rec["rep"] = int(rep_m.group(1)) if rep_m else 0
        rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no analyzable runs (missing stats/trace)")

    if args.smoke:
        raw = df[df.arm == "raw"]
        max_dev = float((raw.floor_own - raw.floor_raw).abs().max())
        print(f"[smoke] {len(df)} runs; raw arms={len(raw)}; "
              f"max|floor_own-floor_raw|={max_dev:.2e} (should be ~0)")
        print(df[["seed", "a_word", "t_word", "arm", "d0", "floor_raw",
                  "floor_pmi", "reached_evidence", "hv_raw", "hv_pmi"]]
              .sort_values("d0", ascending=False).to_string(index=False))
        return

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"[saved] {args.out}  ({len(df)} runs)")

    # Per-cell A/B aggregate (mean over reps) + dose-response vs Δ∅.
    keys = ["seed", "a_word", "t_word", "d0"]
    agg = (df.groupby(keys + ["arm"])
             .agg(floor_pmi=("floor_pmi", "mean"), hv_raw=("hv_raw", "mean"),
                  hv_pmi=("hv_pmi", "mean"), reached=("reached_evidence", "mean"))
             .reset_index())
    print("\n===== per-cell × arm (mean over reps), sorted by Δ∅ =====")
    print(agg.sort_values(["d0", "arm"], ascending=[False, True]).to_string(index=False))


if __name__ == "__main__":
    main()
