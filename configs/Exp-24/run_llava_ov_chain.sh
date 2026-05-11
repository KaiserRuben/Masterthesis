#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# Exp-24 LLaVA-NeXT INT8 OV chain — sequential modality sweep on Arc A770.
#
# Runs joint → image_only → text_only with combined stdout/stderr captured
# to runs/Exp-24/_logs/exp24_llava_ov_chain.log. Detaches via nohup so the
# chain survives terminal close.
#
# Usage:
#   bash configs/Exp-24/run_llava_ov_chain.sh           # detached, returns immediately
#   tail -f runs/Exp-24/_logs/exp24_llava_ov_chain.log  # follow live
#
# Stop early:
#   pkill -f 'run_boundary_test.py configs/Exp-24/llava_ov'
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

# Re-exec under bash if invoked via `sh script.sh` on a non-bash sh.
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="runs/Exp-24/_logs"
LOG_FILE="$LOG_DIR/exp24_llava_ov_chain.log"
mkdir -p "$LOG_DIR"

# Detach: nohup runs an inline bash invocation that holds the full chain.
# The HEREDOC body must be self-contained — no outer-shell variables,
# arrays, or functions are inherited by the detached process.
nohup bash <<'CHAIN_EOF' >"$LOG_FILE" 2>&1 &
set -u

# Inherits CWD from launcher (already cd'd to REPO_ROOT).
CONFIGS=(
  "configs/Exp-24/llava_ov_joint.yaml"
  "configs/Exp-24/llava_ov_image_only.yaml"
  "configs/Exp-24/llava_ov_text_only.yaml"
)

echo "=== START $(date) ==="
echo "host: $(hostname)  pwd: $(pwd)"
echo "n_configs: ${#CONFIGS[@]}"

for i in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$i]}"
  label="$(basename "$cfg" .yaml)"
  extra=()
  if [ "$i" -eq 0 ]; then
    extra+=("--preflight")
  fi
  echo
  echo "=== [$((i+1))/${#CONFIGS[@]}] $label  $(date) ==="
  if ! python experiments/runners/run_boundary_test.py "$cfg" "${extra[@]}"; then
    echo "=== FAIL on $label  $(date) ==="
    echo "Continuing to next config; check earlier output for cause."
  fi
done

echo
echo "=== END $(date) ==="
CHAIN_EOF

PID=$!
disown "$PID" 2>/dev/null || true

echo "started chain pid=$PID  log=$LOG_FILE"
echo "follow: tail -f $LOG_FILE"
