#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
# Exp-104 Phase B — E2·L : raw-vs-PMI live search A/B on LLaVA-OV-INT8 / Arc.
# Run ON THE WORKSTATION (ssh efedora). Portable repo-root cd (no Mac paths).
#
# 8 cells × 2 arms × 3 matched seeds = 48 seed-runs (30×50). rep-1 both arms
# first. workers=2 (OV-GPU SUT + CPU/VQGAN). Matched --seed + pmi.apply_to_
# seedgen=false → only the search objective differs between arms.
#
# Usage (on efedora, in a tmux/screen so it survives the ssh drop —
#   `ssh efedora` (LAN) can die while the box is up; use `ssh efedora`/VPN):
#     conda activate uni    # or the workstation's env
#     bash configs/Exp-104/launch_llava_ab.sh
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1        # repo root, host-agnostic

PY="python experiments/runners/run_boundary_test.py"   # env already activated
RAW="configs/Exp-104/exp104_phaseb_llava_raw.yaml"
PMI="configs/Exp-104/exp104_phaseb_llava_pmi.yaml"
LOGDIR="runs/Exp-104/logs"
mkdir -p "$LOGDIR"

launch () {   # $1=config  $2=arm  $3=seed
  local cfg="$1" arm="$2" seed="$3"
  local sd="runs/Exp-104/llava_${arm}_rep${seed}"
  echo "=== $(date '+%F %T')  E2·L·${arm}  rep/seed=${seed}  → ${sd} ==="
  # --resume/--clean-partials: skip seeds already completed in this rep's
  # save-dir and drop incomplete dirs — survives a box reboot mid-run.
  # Roster mode → resume is metadata-matched (safe).
  $PY "$cfg" --seed "$seed" --save-dir "$sd" --resume --clean-partials >"${LOGDIR}/llava_${arm}_rep${seed}.log" 2>&1
  echo "=== $(date '+%F %T')  done ${arm} rep${seed} (exit $?) ==="
}

for seed in 1 2 3; do
  launch "$RAW" raw "$seed"
  launch "$PMI" pmi "$seed"
done
echo "ALL E2·L RUNS COMPLETE $(date '+%F %T')"
