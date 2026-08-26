#!/usr/bin/env bash
# Exp-105 auf der Workstation: Step-1b-LLaVA-Arme (30x50). Der Step-0-Scan
# ist bereits gelaufen (beide SUTs) und nicht mehr Teil der Kette.
# Detached starten:
#   ssh fedora 'setsid nohup bash ~/Projects/Masterarbeit/configs/Exp-105/fedora_chain.sh \
#       > ~/Projects/Masterarbeit/runs/Exp-105/logs/fedora_chain.log 2>&1 < /dev/null &'
set -u
cd "$(dirname "$0")/../.." || exit 1
export HF_HOME=/mnt/storage/huggingface
# Arc-GPU-Fix (2026-08-03): conda-libstdc++ (6.0.34) ist älter als das, was
# intel-opencl 26.22 (System-Update) braucht — OV sieht die GPU nur mit der
# System-libstdc++. Scoped Preload statt Env-Mutation; dauerhafte Alternative:
# conda install -c conda-forge libstdcxx-ng (Rubens Entscheid).
export LD_PRELOAD=/usr/lib64/libstdc++.so.6

echo "=== $(date '+%F %T') fedora step1b chain start ==="
PYTHON="$HOME/miniconda3/envs/uni/bin/python" bash configs/Exp-105/launch_step1_llava.sh
echo "=== $(date '+%F %T') FEDORA CHAIN DONE ==="
