r"""Exp-104 (PMI calibration) — data loading, recomputation and provenance checks.

Shared by the three thesis figures ``exp104-prior-map``, ``exp104-wall-collapse``
and ``exp104-dose``.  Nothing here plots; it loads the PRIMARY csvs, recomputes
every aggregate the figures use, and asserts that the recomputation agrees with
the archived ``agent*_*.csv`` cross-checks (whose generating scripts are not in
the repo, so no number from them is ever plotted on faith).

PRIMARY sources (produced by scripts that ARE in the repo)
  experiments/analysis/output/exp104/exp104_pmi.csv          Phase A, Qwen, 46 cells
  experiments/analysis/output/exp104_llava/exp104_pmi.csv    Phase A, LLaVA, 46 cells
      <- experiments/analysis/exp104_pmi_calibration.py
  experiments/analysis/output/exp104/phaseb_qwen.csv         Phase B, 48 runs
  experiments/analysis/output/exp104/phaseb_llava.csv        Phase B, 48 runs
      <- experiments/analysis/exp104_phaseb_reach_hv.py

CROSS-CHECK sources (no generating script in the repo -- verified, never plotted)
  experiments/analysis/output/exp104/agentC_master_table.csv
  experiments/analysis/output/exp104/agentC_correlations.json
  experiments/analysis/output/exp104/agentE_null_sensitivity_allcells.csv

Quantities (all in nats of length-normalized log-prob, exp104_pmi_calibration.py)
  g(m)      = lp(anchor word | m) - lp(target word | m)   signed pair gap
  Δ∅        = g(∅) on a content-free null image           the answer-string prior
              (``d0`` == ``d0_gray``; black / white / noise are the sensitivity set)
  raw_floor = min |g(m)| over the run's whole trace       height of the wall
  pmi_floor = min |g(m) - Δ∅|                             what is left after the
                                                          prior is subtracted
  Δ∅ > 0  the prior already prefers the ANCHOR word -> the flip is uphill
  Δ∅ < 0  the prior prefers the TARGET word          -> the flip is downhill

Phase A (post-hoc re-scoring of archived Exp-101/101q traces, 46 cells per SUT)
and Phase B (96 fresh A/B searches, 2 SUTs x raw|pmi x 8 cells x 3 reps) are kept
strictly apart; each figure uses one phase only.

Thresholds are taken from the pipeline, not invented here:
  EPS_REACH = 0.3 nats  the pre-registered evidence-reach bar
                        (``exp104_phaseb_reach_hv.py``: reached_evidence = floor<=0.3)
  WALL      = 1.0 nat   "wall" cut for Phase A.  Sits in a gap that is present in
                        BOTH SUTs (Qwen 0.925 -> 1.622, LLaVA 0.879 -> 1.272) and
                        selects 11/46 Qwen and 14/46 LLaVA cells.

Usage (from the Masterarbeit repo root, conda env `uni`):
    cd ~/Projects/Masterarbeit && PYTHONPATH=$PWD conda run --no-capture-output \
        -n uni python analysis/viz/thesis/exp104_data.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/kaiser/Projects/Masterarbeit")
OUTDIR = REPO / "experiments/analysis/output"
FIGDIR = Path("/Users/kaiser/Desktop/Uni/Masterarbeit/Master Thesis v0.5.0/"
              "figures/results")

# One fixed colour per SUT, used identically in all three Exp-104 figures.
# Blue / orange is the safest deutan-protan axis, and the two also separate in
# print greyscale (relative luminance 0.32 vs 0.56).
SUT_COLOR = {"qwen": "#1F5C99", "llava": "#E07B22"}
SUT_LABEL = {"qwen": "Qwen3.5-4B", "llava": "LLaVA-NeXT-7B"}
SUTS = ("qwen", "llava")

EPS_REACH = 0.3    # nats, pre-registered evidence-reach bar
WALL = 1.0         # nats, Phase A wall cut (see module docstring)
NULL_COLS = ["d0_gray", "d0_black", "d0_white", "d0_noise"]

PHASE_A_CSV = {"qwen": OUTDIR / "exp104/exp104_pmi.csv",
               "llava": OUTDIR / "exp104_llava/exp104_pmi.csv"}
PHASE_B_CSV = {"qwen": OUTDIR / "exp104/phaseb_qwen.csv",
               "llava": OUTDIR / "exp104/phaseb_llava.csv"}


# --------------------------------------------------------------------------
# Phase A -- per-cell post-hoc prior map
# --------------------------------------------------------------------------
def phase_a(sut: str) -> pd.DataFrame:
    """46 cells for one SUT, straight from the primary csv.

    Adds only columns that are a deterministic function of that csv:
      d0_std   spread of Δ∅ across the four null images (gray/black/white/noise),
               sample sd (ddof=1) -- the x error bar.  This is the same number as
               ``agentE_null_sensitivity_allcells.d0_null_std`` (asserted below).
      is_wall  raw_floor >= WALL
    """
    d = pd.read_csv(PHASE_A_CSV[sut]).copy()
    assert len(d) == 46, f"{sut}: expected 46 cells, got {len(d)}"
    assert np.allclose(d.d0, d.d0_gray), f"{sut}: d0 is not d0_gray"
    assert np.allclose(d.pmi_floor, d.pmi_floor_gray)
    d["d0_std"] = d[NULL_COLS].std(axis=1, ddof=1)
    d["is_wall"] = d.raw_floor >= WALL
    d["pair"] = d.a_word + "→" + d.t_word
    d["sut"] = sut
    return d


# --------------------------------------------------------------------------
# Phase B -- live A/B, aggregated to one row per (SUT, cell)
# --------------------------------------------------------------------------
def _regime(row: pd.Series) -> str:
    """Reproducible restatement of the four Phase-B regime labels.

    The label column in ``agentC_master_table.csv`` has no generating script in
    the repo, so it is re-derived here from the primary Phase-B csv only, using
    the pipeline's own reach flag (``reached_evidence`` = pmi floor <= 0.3 nats,
    exp104_phaseb_reach_hv.py) and Δ∅.  Order of the tests matters.

      GEOMETRY        neither arm ever reaches the evidence boundary in 3 reps
                      -- a floor that survives calibration
      PRIOR-ASSISTED  the raw arm never reaches it but the PMI arm does
                      -- the raw arm was riding the prior
      CONTROL         |Δ∅| < 0.5 nats, i.e. no prior to ride.  The cut is
                      anywhere in the empirical gap 0.12 -> 1.02 nats
      EASY            everything else

    Agrees with agentC_master_table.regime on all 16 rows (asserted in check()).
    """
    if row.n_reach_raw == 0 and row.n_reach_pmi == 0:
        return "GEOMETRY"
    if row.n_reach_raw == 0:
        return "PRIOR-ASSISTED"
    if abs(row.d0) < 0.5:
        return "CONTROL"
    return "EASY"


REGIMES = ("EASY", "PRIOR-ASSISTED", "CONTROL", "GEOMETRY")
REGIME_MARKER = {"EASY": "o", "PRIOR-ASSISTED": "^", "CONTROL": "s",
                 "GEOMETRY": "D"}


def phase_b() -> pd.DataFrame:
    """One row per (SUT, cell): 16 rows, each a mean over 3 matched reps.

    Both arms are scored on the SAME PMI scale (``floor_pmi``/``hv_pmi`` columns,
    which exp104_phaseb_reach_hv.py computes for every run regardless of which
    objective that run optimized), so raw and PMI arms are comparable:

      dfloor = floor_pmi(raw arm) - floor_pmi(pmi arm)   >0: calibration helped
      dHV    = hv_pmi(pmi arm)    - hv_pmi(raw arm)      >0: calibration helped
      dose   = max(0, -Δ∅)                               how much prior the raw
                                                         arm could ride

    Δ∅ is averaged over all six runs of a cell.  The two arms read it from
    different places (raw arm from the Phase-A prior map, PMI arm from its own
    ``pmi_baseline`` re-measurement), which on Qwen/MPS differ by up to 0.063
    nats; LLaVA/OpenVINO is bit-identical.  The six-run mean is what
    agentC_master_table and agentC_correlations use, so using it keeps the
    published r values exactly reproducible.
    """
    out = []
    for sut in SUTS:
        d = pd.read_csv(PHASE_B_CSV[sut])
        assert len(d) == 48, f"{sut}: expected 48 Phase-B runs, got {len(d)}"
        assert set(d.arm) == {"raw", "pmi"}
        for seed, g in d.groupby("seed"):
            assert len(g) == 6 and sorted(g.rep) == [1, 1, 2, 2, 3, 3]
            r, p = g[g.arm == "raw"], g[g.arm == "pmi"]
            out.append(dict(
                sut=sut, cell=int(seed),
                pair=f"{g.a_word.iloc[0]}→{g.t_word.iloc[0]}",
                d0=float(g.d0.mean()),
                floor_pmi_raw=float(r.floor_pmi.mean()),
                floor_pmi_raw_sd=float(r.floor_pmi.std(ddof=1)),
                floor_pmi_pmi=float(p.floor_pmi.mean()),
                floor_pmi_pmi_sd=float(p.floor_pmi.std(ddof=1)),
                dfloor=float(r.floor_pmi.mean() - p.floor_pmi.mean()),
                hv_raw=float(r.hv_pmi.mean()),
                hv_pmi=float(p.hv_pmi.mean()),
                dHV=float(p.hv_pmi.mean() - r.hv_pmi.mean()),
                n_reach_raw=int(r.reached_evidence.sum()),
                n_reach_pmi=int(p.reached_evidence.sum()),
            ))
    b = pd.DataFrame(out)
    b["dose"] = np.maximum(0.0, -b.d0)
    b["absd0"] = b.d0.abs()
    b["regime"] = b.apply(_regime, axis=1)
    assert len(b) == 16
    return b


# --------------------------------------------------------------------------
# Provenance checks against the un-scripted agent csvs
# --------------------------------------------------------------------------
def check(verbose: bool = True) -> None:
    from scipy.stats import pearsonr, spearmanr

    def say(*a):
        if verbose:
            print(*a)

    say("[provenance] recomputed-from-primary vs archived agent csvs")

    # --- Phase A: null-image sensitivity (agentE, Qwen only) ---------------
    a_q = phase_a("qwen")
    e = pd.read_csv(OUTDIR / "exp104/agentE_null_sensitivity_allcells.csv"
                    ).set_index("seed")
    m = np.max(np.abs(a_q.set_index("seed").d0_std - e.d0_null_std))
    assert m < 1e-12, f"agentE d0_null_std mismatch {m}"
    m2 = np.max(np.abs(a_q.set_index("seed")[NULL_COLS].mean(axis=1)
                       - e.d0_null_mean))
    assert m2 < 1e-12
    say(f"  agentE_null_sensitivity   d0_null_std/mean  max|diff| "
        f"{max(m, m2):.1e}  (46 cells, ddof=1)  OK")

    # --- Phase B: master table --------------------------------------------
    b = phase_b().set_index(["sut", "cell"])
    mt = pd.read_csv(OUTDIR / "exp104/agentC_master_table.csv"
                     ).set_index(["sut", "cell"]).loc[b.index]
    pairs = [("d0", "d0"), ("floor_pmi_raw", "floor_pmi_raw"),
             ("floor_pmi_raw_sd", "floor_pmi_raw_sd"),
             ("floor_pmi_pmi", "floor_pmi_pmi"),
             ("floor_pmi_pmi_sd", "floor_pmi_pmi_sd"),
             ("dfloor", "dfloor"), ("hv_raw", "hv_evid_raw"),
             ("hv_pmi", "hv_evid_pmi"), ("dHV", "dHV")]
    worst = 0.0
    for mine, theirs in pairs:
        d = float(np.max(np.abs(b[mine].values - mt[theirs].values)))
        worst = max(worst, d)
        assert d < 1e-9, f"agentC_master_table[{theirs}] mismatch {d}"
    say(f"  agentC_master_table       9 columns x 16 rows  max|diff| "
        f"{worst:.1e}  OK")
    bad = (b.regime.values != mt.regime.values).sum()
    assert bad == 0, f"regime rule disagrees on {bad} rows"
    say("  agentC_master_table       regime label 16/16 re-derived  OK")

    # --- Phase B: correlations --------------------------------------------
    cc = json.load(open(OUTDIR / "exp104/agentC_correlations.json"))
    worst = 0.0
    for ycol, ykey in (("dfloor", "dfloor"), ("dHV", "dHV"),
                       ("floor_pmi_raw", "floor_pmi_raw"),
                       ("floor_pmi_pmi", "floor_pmi_pmi")):
        for xcol, xkey in (("dose", "neg_part"), ("absd0", "abs_d0"),
                           ("d0", "d0_signed")):
            r = pearsonr(b[xcol], b[ycol])
            s = spearmanr(b[xcol], b[ycol])
            ref = cc[ykey][xkey]
            worst = max(worst, abs(r.statistic - ref["pearson_r"]),
                        abs(s.statistic - ref["spearman_rho"]))
            assert abs(r.statistic - ref["pearson_r"]) < 1e-9
            assert abs(s.statistic - ref["spearman_rho"]) < 1e-9
    say(f"  agentC_correlations       12 r/rho pairs  max|diff| {worst:.1e}  OK")

    # --- headline numbers --------------------------------------------------
    say("\n[phase A] Spearman rho(Δ∅, floor), 46 cells per SUT")
    for sut in SUTS:
        d = phase_a(sut)
        sr = spearmanr(d.d0, d.raw_floor)
        sp = spearmanr(d.d0, d.pmi_floor)
        say(f"  {sut:5s} raw rho={sr.statistic:+.3f} (p={sr.pvalue:.2g})   "
            f"pmi rho={sp.statistic:+.3f} (p={sp.pvalue:.2g})   "
            f"walls>={WALL}: {int(d.is_wall.sum())}   "
            f"floor<{EPS_REACH}: raw {int((d.raw_floor < EPS_REACH).sum())} -> "
            f"pmi {int((d.pmi_floor < EPS_REACH).sum())}")
    say("\n[phase B] Pearson r, 16 cell means")
    b = b.reset_index()
    for x in ("dose", "absd0"):
        for y in ("dfloor", "dHV"):
            r = pearsonr(b[x], b[y])
            say(f"  r({x:6s}, {y:6s}) = {r.statistic:+.3f}  p={r.pvalue:.2g}")


if __name__ == "__main__":
    check()
