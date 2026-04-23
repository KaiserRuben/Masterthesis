#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# EXP-10b Overnight Batch — Phase-1 sparse init generalization
# ═══════════════════════════════════════════════════════════════════════════
#
# 3 runs, ~6 h MPS @ 1.86 s/call. Tests whether the Phase-1 sparse-init
# result generalizes beyond shark to junco-chickadee (unreachable pair),
# junco-leatherback (far cross-cluster), and stingray-eray (second reachable).
#
# Each pair has a direct Exp-09 n=16383 uniform-init counterpart for A/B
# comparison on the (L0, TgtBal) 2D Pareto.
#
# Not halting on failure (';' rather than '&&'), so a crash in one run
# doesn't block the remainder.
# ═══════════════════════════════════════════════════════════════════════════

set -u -o pipefail
cd /Users/kaiser/Projects/Masterarbeit

LOGDIR=runs/exp10/logs
mkdir -p "$LOGDIR"

echo "[$(date +%H:%M:%S)] EXP-10b batch starting"

# ── 1. junco → chickadee (unreachable) ──────────────────────────────────
echo "[$(date +%H:%M:%S)] run 1/3 — junco_chickadee"
python experiments/run_boundary_test.py \
  configs/EXP-10/phase1_junco_chickadee_n16383.yaml \
  2>&1 | tee "$LOGDIR/run1_junco_chickadee.log"

# ── 2. junco → leatherback (far cross-cluster) ──────────────────────────
echo "[$(date +%H:%M:%S)] run 2/3 — junco_leatherback"
python experiments/run_boundary_test.py \
  configs/EXP-10/phase1_junco_leatherback_n16383.yaml \
  2>&1 | tee "$LOGDIR/run2_junco_leatherback.log"

# ── 3. stingray → electric_ray (second reachable) ───────────────────────
echo "[$(date +%H:%M:%S)] run 3/3 — stingray_eray"
python experiments/run_boundary_test.py \
  configs/EXP-10/phase1_stingray_eray_n16383.yaml \
  2>&1 | tee "$LOGDIR/run3_stingray_eray.log"

echo "[$(date +%H:%M:%S)] EXP-10b batch complete"
