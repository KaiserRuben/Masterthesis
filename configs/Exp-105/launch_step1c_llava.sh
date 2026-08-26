#!/usr/bin/env bash
# Exp-105 Step 1c — LLaVA raw-deep (60x100 = 4x Step-1b-Budget), Seeds 1..3.
# Zweck siehe exp105_step1_house_llava_raw_deep.yaml (Wall-Zertifizierung).
# Aufruf über fedora_step1c_queue.sh (wartet auf Step-1b-Kette) oder direkt:
#   PYTHON=~/miniconda3/envs/uni/bin/python bash configs/Exp-105/launch_step1c_llava.sh
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

PY="${PYTHON:-python}"
mkdir -p runs/Exp-105/logs
for rep in 1 2 3; do
  tag="step1c_llava_rawdeep_rep${rep}"
  echo "=== $(date '+%F %T') launching ${tag} ==="
  "$PY" experiments/runners/run_boundary_test.py \
    "configs/Exp-105/exp105_step1_house_llava_raw_deep.yaml" \
    --seed "${rep}" \
    --save-dir "runs/Exp-105/${tag}" \
    > "runs/Exp-105/logs/${tag}.log" 2>&1
  rc=$?
  echo "=== $(date '+%F %T') ${tag} exit=${rc} ==="
done
echo "ALL STEP-1C LLAVA ARMS DONE $(date '+%F %T')"
