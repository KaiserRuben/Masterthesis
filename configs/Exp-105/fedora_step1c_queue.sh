#!/usr/bin/env bash
# Exp-105: Step-1c-Queue auf der Workstation. Wartet auf das Ende der
# Step-1b-Kette (DONE-Marker ODER Kette tot) und startet dann die
# raw-deep-Arme. Detached starten:
#   ssh fedora 'setsid nohup bash ~/Projects/Masterarbeit/configs/Exp-105/fedora_step1c_queue.sh \
#       > ~/Projects/Masterarbeit/runs/Exp-105/logs/fedora_step1c_queue.log 2>&1 < /dev/null &'
set -u
cd "$(dirname "$0")/../.." || exit 1
export HF_HOME=/mnt/storage/huggingface
# Arc-GPU-Fix (2026-08-03): conda-libstdc++ (6.0.34) aelter als Bedarf von
# intel-opencl 26.22 — GPU nur mit System-libstdc++ sichtbar.
export LD_PRELOAD=/usr/lib64/libstdc++.so.6

CHAIN_LOG=runs/Exp-105/logs/fedora_chain.log
echo "=== $(date '+%F %T') step1c queue armed, warte auf Step-1b-Kette ==="
while true; do
  if grep -q "FEDORA CHAIN DONE" "$CHAIN_LOG" 2>/dev/null; then
    echo "=== $(date '+%F %T') Step-1b DONE-Marker gefunden ==="
    break
  fi
  # Bracket-Trick: Muster matcht nicht die eigene Kommandozeile.
  if ! pgrep -f "launch_step1[_]llava.sh" >/dev/null; then
    echo "=== $(date '+%F %T') Step-1b-Kette laeuft nicht mehr (kein DONE-Marker — pruefen!) ==="
    break
  fi
  sleep 60
done

echo "=== $(date '+%F %T') starte Step 1c (raw-deep 60x100, Seeds 1..3) ==="
PYTHON="$HOME/miniconda3/envs/uni/bin/python" bash configs/Exp-105/launch_step1c_llava.sh
echo "=== $(date '+%F %T') FEDORA STEP1C QUEUE DONE ==="
