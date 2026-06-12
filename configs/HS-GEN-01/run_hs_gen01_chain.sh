#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# HS-GEN-01 chain — image-only boundary-item generation for HS-01 (LLaVA arm).
# VQGAN + cone ON everywhere; screen-then-promote workflow.
#
# Sequential runs on the workstation Arc A770, detached via nohup, combined
# stdout/stderr to runs/HS-GEN-01/_logs/hs_gen01_chain.log.
#
# WORKFLOW (decision 4 — screen, then promote):
#   Stage 1 (this script, HS_GEN_WITH_SCREEN=1): broad full-roster crossability
#           screen — VQGAN, cone ON, HEAVY mutation, 4 gen × 30 pop per entry.
#   --- then OFFLINE, between stages ---
#           conda run -n uni python configs/HS-GEN-01/promote_pairs.py
#           # emits hs_gen01_promoted_idx<N>_*.yaml for pairs passing BOTH the
#           # crossability screen AND the human-distinguishability gate.
#           conda run -n uni python configs/HS-GEN-01/validate_configs.py
#           # append the emitted YAMLs to PROMOTED below, re-run this script.
#   Stage 2 (this script): full-evolution runs — the calibration pair A plus
#           every promoted pair.
#
# Pair A (great white shark → hammerhead, idx 1) is the REFERENCE/CALIBRATION
# pair: proven crossable, but NEEDS ABSTRACTION for study eligibility (two
# shark species — see README). It always runs; it is the fallback strata
# source and the pipeline sanity check.
#
# Usage:
#   HS_GEN_ONLY_SCREEN=1 bash configs/HS-GEN-01/run_hs_gen01_chain.sh   # screen only
#   bash configs/HS-GEN-01/run_hs_gen01_chain.sh                        # full runs (A + promoted)
#   HS_GEN_WITH_SCREEN=1 bash configs/HS-GEN-01/run_hs_gen01_chain.sh   # screen THEN full runs
#   tail -f runs/HS-GEN-01/_logs/hs_gen01_chain.log                     # follow live
#
# Stop early:
#   pkill -f 'run_boundary_test.py configs/HS-GEN-01/'
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

# Re-exec under bash if invoked via `sh script.sh` on a non-bash sh.
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="runs/HS-GEN-01/_logs"
LOG_FILE="$LOG_DIR/hs_gen01_chain.log"
mkdir -p "$LOG_DIR"

# Detach: nohup runs an inline bash invocation that holds the full chain.
# The HEREDOC body is quoted (no outer-shell expansion); the toggles are
# passed through the environment via `env`.
nohup env \
  HS_GEN_WITH_SCREEN="${HS_GEN_WITH_SCREEN:-0}" \
  HS_GEN_ONLY_SCREEN="${HS_GEN_ONLY_SCREEN:-0}" \
  bash <<'CHAIN_EOF' >"$LOG_FILE" 2>&1 &
set -u

# Inherits CWD from launcher (already cd'd to REPO_ROOT).
CONFIGS=()

if [ "${HS_GEN_WITH_SCREEN:-0}" = "1" ] || [ "${HS_GEN_ONLY_SCREEN:-0}" = "1" ]; then
  CONFIGS+=("configs/HS-GEN-01/hs_gen01_screen.yaml")
fi

if [ "${HS_GEN_ONLY_SCREEN:-0}" != "1" ]; then
  # --- full-evolution runs ---
  CONFIGS+=(
    "configs/HS-GEN-01/hs_gen01_pairA_shark_hammerhead.yaml"   # calibration / reference
    "configs/HS-GEN-01/hs_gen01_pairA_shark_hammerhead.yaml"   # ×2 for yield
  )
  # ── screen-promoted pair configs (run promote_pairs.py, then list here) ──
  # PROMOTED=(configs/HS-GEN-01/hs_gen01_promoted_idx<N>_<a>_<b>.yaml ...)
  PROMOTED=()
  CONFIGS+=("${PROMOTED[@]}")
fi

echo "=== START $(date) ==="
echo "host: $(hostname)  pwd: $(pwd)"
echo "with_screen=${HS_GEN_WITH_SCREEN:-0}  only_screen=${HS_GEN_ONLY_SCREEN:-0}"
echo "n_runs: ${#CONFIGS[@]}"

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
if [ "${HS_GEN_ONLY_SCREEN:-0}" = "1" ]; then
  echo "Screen complete. Next:"
  echo "  conda run -n uni python configs/HS-GEN-01/promote_pairs.py"
  echo "  conda run -n uni python configs/HS-GEN-01/validate_configs.py"
  echo "  # append emitted YAMLs to PROMOTED, then re-run without HS_GEN_ONLY_SCREEN"
fi
echo "=== END $(date) ==="
CHAIN_EOF

PID=$!
disown "$PID" 2>/dev/null || true

echo "started chain pid=$PID  log=$LOG_FILE"
echo "follow: tail -f $LOG_FILE"
