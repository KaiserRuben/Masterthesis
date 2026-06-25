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
#   Stage 2 (this script): full-evolution runs — promoted pairs ONLY.
#
# No pair runs without passing BOTH gates (crossability + lay-
# distinguishability). idx 1 (gw shark → hammerhead) is covered by the screen
# like every roster entry, but as a fine-sibling pair it is never auto-
# promoted; its old calibration config is archived under
# configs/Archive/HS-GEN-01-pairA-calibration/ (study-owner decision required
# to resurrect it under an abstraction framing).
#
# Usage:
#   HS_GEN_ONLY_SCREEN=1 bash configs/HS-GEN-01/run_hs_gen01_chain.sh   # screen only
#   bash configs/HS-GEN-01/run_hs_gen01_chain.sh                        # full runs (promoted)
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
  # ── screen-promoted pair configs (run promote_pairs.py, then list here) ──
  # Selected from the 1024-pair screen (06-15): 6 distinct-anchor pairs spanning
  # amphibian/bird/fish/reptile, deepest crossings, lay-distinguishable. Full
  # promotable set (176) archived under configs/Archive/HS-GEN-01-screen-emitted/.
  PROMOTED=(
    configs/HS-GEN-01/hs_gen01_promoted_idx564_fire_salamander_american_bullfrog.yaml   # fire salamander → American bullfrog     (best 1.0e-5)
    configs/HS-GEN-01/hs_gen01_promoted_idx249_cock_ostrich.yaml                         # cock → ostrich                          (best 5.5e-5)
    configs/HS-GEN-01/hs_gen01_promoted_idx71_hammerhead_shark_nile_crocodile.yaml       # hammerhead shark → Nile crocodile       (best 3.2e-4)
    configs/HS-GEN-01/hs_gen01_promoted_idx639_loggerhead_sea_turtle_carolina_anole.yaml # loggerhead sea turtle → Carolina anole  (best 1.3e-4)
    configs/HS-GEN-01/hs_gen01_promoted_idx442_bald_eagle_american_robin.yaml            # bald eagle → American robin             (best 3.7e-4)
    configs/HS-GEN-01/hs_gen01_promoted_idx137_stingray_indigo_bunting.yaml              # stingray → indigo bunting               (best 9.1e-4)
  )
  if [ "${#PROMOTED[@]}" -eq 0 ]; then
    echo "No promoted configs listed. Run the screen, then promote_pairs.py,"
    echo "then append the emitted YAMLs to PROMOTED in this script."
  else
    CONFIGS+=("${PROMOTED[@]}")
  fi
fi

if [ "${#CONFIGS[@]}" -eq 0 ]; then
  echo "Nothing to run — exiting."
  exit 0
fi

echo "=== START $(date) ==="
echo "host: $(hostname)  pwd: $(pwd)"
echo "with_screen=${HS_GEN_WITH_SCREEN:-0}  only_screen=${HS_GEN_ONLY_SCREEN:-0}"
echo "n_runs: ${#CONFIGS[@]}"

for i in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$i]}"
  label="$(basename "$cfg" .yaml)"
  # NOTE: --resume is NOT wired here. src/common/resume.py keys completion on a
  # `seed_metadata` block that only roster-mode runs write; gap_filter stats.json
  # (this campaign) has none, so --resume mis-detects every finished seed as
  # unfinished and re-runs the whole chain. Until resume.py supports gap_filter,
  # restart a stopped chain by running only the missing pair config directly.
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
