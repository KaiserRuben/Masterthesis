#!/usr/bin/env bash
# Exp-105 Step 1b — Qwen-Arme im Kampagnen-Budget 30x50 (Ruben, 2026-08-03):
# {raw, pmi} × Seeds 1..3, sequenziell (eine MPS-GPU). raw/pmi-Paare teilen
# den --seed (identische Init-Population, nur Scoring differiert).
# Detached starten, damit Session-Stopps den Lauf nicht killen:
#   setsid nohup bash configs/Exp-105/launch_step1_qwen.sh \
#       > runs/Exp-105/logs/launch_qwen_step1b.log 2>&1 < /dev/null &
# Erstpass 24x25 liegt unter runs/Exp-105/step1_qwen_* (archiviert).
set -uo pipefail
cd "$(dirname "$0")/../.."

mkdir -p runs/Exp-105/logs
for rep in 1 2 3; do
  for arm in raw pmi; do
    tag="step1b_qwen_${arm}_rep${rep}"
    echo "=== $(date '+%F %T') launching ${tag} ==="
    conda run -n uni python experiments/runners/run_boundary_test.py \
      "configs/Exp-105/exp105_step1_house_qwen_${arm}.yaml" \
      --seed "${rep}" \
      --save-dir "runs/Exp-105/${tag}" \
      > "runs/Exp-105/logs/${tag}.log" 2>&1
    rc=$?
    echo "=== $(date '+%F %T') ${tag} exit=${rc} ==="
  done
done
echo "ALL STEP-1B QWEN ARMS DONE $(date '+%F %T')"
