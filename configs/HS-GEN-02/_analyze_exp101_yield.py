#!/usr/bin/env python3
"""Scan Exp-101 runs: per-cell crossing/yield, grouped by cell kind.

For HS-GEN-02 we want the HIGH-YIELD cell family. "Yield" for the human
study = a run produced boundary individuals that actually CROSSED (some
gen had an individual with p(target) > p(anchor)) and pushed TgtBal low.
We measure, per run:
  * min TgtBal reached (convergence.parquet pop_min_TgtBal / pareto)
  * crossed = min TgtBal < some small eps (boundary essentially reached)
and group by (within-bucket vs cross-bucket, level-diagonal (0,0) vs other,
direction forward/reverse), using stats.json metadata.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

RUNS = Path("runs/Exp-101")
EPS_CROSS = 0.01   # TgtBal below this = essentially at/over the boundary
EPS_DEEP = 1e-3


def main() -> None:
    rows = []
    for d in sorted(RUNS.glob("exp101_margin_predictor_seed_*")):
        sj = d / "stats.json"
        cv = d / "convergence.parquet"
        if not sj.exists() or not cv.exists():
            continue
        st = json.loads(sj.read_text())
        meta = st.get("seed_metadata") or st.get("metadata") or {}
        # metadata fields may live at top-level too
        def g(k, default=None):
            return meta.get(k, st.get(k, default))
        la = g("level_anchor")
        lt = g("level_target")
        anchor = g("anchor_class_concrete")
        target = g("target_class_concrete")
        ap = g("anchor_position")
        tp = g("target_position")
        cancestor = g("common_ancestor_level")
        df = pd.read_parquet(cv)
        # find a min-TgtBal column
        cols = [c for c in df.columns if "TgtBal" in c or "tgtbal" in c.lower()]
        min_cols = [c for c in cols if "min" in c.lower()]
        use = min_cols[0] if min_cols else (cols[0] if cols else None)
        min_tgt = float(df[use].min()) if use else float("nan")
        gen0 = None
        if use and "generation" in df.columns:
            g0 = df[df["generation"] == df["generation"].min()][use]
            gen0 = float(g0.min()) if len(g0) else None
        within = (cancestor is not None)
        direction = "fwd" if (ap is not None and tp is not None and ap < tp) else "rev"
        diag00 = (la == 0 and lt == 0)
        rows.append(dict(
            dir=d.name.split("_seed_")[1].split("_")[0],
            anchor=anchor, target=target, la=la, lt=lt,
            bucket=("within" if within else "cross"),
            direction=direction, diag00=diag00,
            min_tgt=min_tgt,
            crossed=(min_tgt < EPS_CROSS),
            deep=(min_tgt < EPS_DEEP),
            cols=",".join(cols),
        ))
    R = pd.DataFrame(rows)
    if R.empty:
        print("no runs parsed")
        return
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 100)
    print("=== columns seen in convergence.parquet (first run) ===")
    print(R["cols"].iloc[0])
    print()
    print("=== per-run table ===")
    print(R.drop(columns=["cols"]).to_string(index=False))
    print()
    print("=== yield (crossed: min TgtBal < %.3g) by group ===" % EPS_CROSS)
    for keys in [["bucket"], ["bucket", "diag00"], ["bucket", "direction"],
                 ["bucket", "diag00", "direction"]]:
        grp = R.groupby(keys).agg(
            n=("crossed", "size"),
            crossed=("crossed", "sum"),
            crossed_rate=("crossed", "mean"),
            deep=("deep", "sum"),
            median_min_tgt=("min_tgt", "median"),
        )
        print()
        print(grp.to_string())


if __name__ == "__main__":
    sys.exit(main())
