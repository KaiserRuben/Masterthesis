#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
# Exp-104 Phase B — E2·Q : raw-vs-PMI live search A/B on Qwen3.5-4B / MPS.
#
# 8 dose-response cells × 2 arms × 3 matched seeds = 48 seed-runs (30×50 each).
# Ordered rep-1 BOTH arms first so a complete paired A/B lands early, then
# reps 2 and 3 extend the noise bands. Sequential (single MPS device).
#
# Matched --seed makes raw and PMI arms share an identical initial population
# and GA RNG stream; pmi.apply_to_seedgen=false makes the anchor photos
# identical too. Net: the ONLY difference between arms is the search objective.
#
# Usage:  bash configs/Exp-104/launch_qwen_ab.sh
# Resumable: each invocation supports --resume against its own rep save-dir.
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail

REPO="/Users/kaiser/Projects/Masterarbeit"
cd "$REPO"

RUN="conda run -n uni python experiments/runners/run_boundary_test.py"
RAW="configs/Exp-104/exp104_phaseb_qwen_raw.yaml"
PMI="configs/Exp-104/exp104_phaseb_qwen_pmi.yaml"
LOGDIR="runs/Exp-104/logs"
mkdir -p "$LOGDIR"

launch () {   # $1=config  $2=arm  $3=seed
  local cfg="$1" arm="$2" seed="$3"
  local sd="runs/Exp-104/qwen_${arm}_rep${seed}"
  local log="${LOGDIR}/qwen_${arm}_rep${seed}.log"
  echo "=== $(date '+%F %T')  E2·Q·${arm}  rep/seed=${seed}  → ${sd} ==="
  # --resume: skip seeds already completed in this rep's save-dir (recovers
  # from a mid-run crash without redoing finished seeds). --clean-partials:
  # drop incomplete seed dirs before restarting. Roster mode → resume is
  # metadata-matched (safe).
  $RUN "$cfg" --seed "$seed" --save-dir "$sd" --resume --clean-partials >"$log" 2>&1
  echo "=== $(date '+%F %T')  done ${arm} rep${seed} (exit $?) ==="
}

for seed in 1 2 3; do
  launch "$RAW" raw "$seed"
  launch "$PMI" pmi "$seed"
done

echo "ALL E2·Q RUNS COMPLETE $(date '+%F %T')"
