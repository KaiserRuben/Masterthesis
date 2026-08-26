#!/usr/bin/env bash
# Exp-105 Step 1b — LLaVA-OV-INT8-Arme im Kampagnen-Budget 30x50
# (Ruben, 2026-08-03): {raw, pmi} × Seeds 1..3, sequenziell auf der Arc.
# Aufruf über fedora_chain.sh (setzt LD_PRELOAD + HF_HOME) oder direkt:
#   PYTHON=~/miniconda3/envs/uni/bin/python bash configs/Exp-105/launch_step1_llava.sh
# Erstpass 24x25 liegt unter runs/Exp-105/step1_llava_* (rep1 komplett,
# rep2/rep3 partiell — gestoppt beim Budget-Wechsel, nicht wertbar).
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

PY="${PYTHON:-python}"
mkdir -p runs/Exp-105/logs
for rep in 1 2 3; do
  for arm in raw pmi; do
    tag="step1b_llava_${arm}_rep${rep}"
    echo "=== $(date '+%F %T') launching ${tag} ==="
    "$PY" experiments/runners/run_boundary_test.py \
      "configs/Exp-105/exp105_step1_house_llava_${arm}.yaml" \
      --seed "${rep}" \
      --save-dir "runs/Exp-105/${tag}" \
      > "runs/Exp-105/logs/${tag}.log" 2>&1
    rc=$?
    echo "=== $(date '+%F %T') ${tag} exit=${rc} ==="
  done
done
echo "ALL STEP-1B LLAVA ARMS DONE $(date '+%F %T')"
