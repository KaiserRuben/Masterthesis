#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# Exp-22 Seed-1 Replication — full 5-condition suite + 3 score computations
#
# Replicates the seed-83 results on a different seed (filter_indices: [1])
# to confirm pattern-score wins generic vs pair-specific.
#
# Sequence:
#   1. Exp-22 sparse default        (~3.3h)
#   2. Exp-22b multi-tier           (~3.3h)
#   3. Pattern score-eval (~15min) → main run (~3.3h)
#   4. Ablation score-eval (~55min) → main run (~3.3h)
#   5. Sobol score-eval (~25min)   → main run (~3.4h)
#
# Total ≈ 17–18h. Each step writes its own log under logs/.
#
# Usage:
#   nohup bash experiments/runners/run_seed1_replication.sh \
#       > logs/seed1_replication_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   disown
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

cd "$(dirname "$0")/../.."

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs runs/Exp-22/scores

# Score-eval config (provides SUT/seed/pipeline; score-method-agnostic).
EVAL_CFG="configs/Exp-22/mlm_composite_seed1_multitier.yaml"

run_main() {
    local LABEL="$1"
    local CFG="$2"
    local LOG="logs/seed1_${LABEL}_main_${TS}.log"
    echo "============================================================"
    echo "[seed1/${LABEL}] Main run"
    echo "  cfg: ${CFG}"
    echo "  log: ${LOG}"
    echo "============================================================"
    python experiments/runners/run_boundary_test.py "${CFG}" > "${LOG}" 2>&1
    echo "[seed1/${LABEL}] Main run done."
}

run_score() {
    local METHOD="$1"
    local OUT="runs/Exp-22/scores/${METHOD}_seed1.npy"
    local LOG="logs/seed1_${METHOD}_score_${TS}.log"

    local EXTRA=""
    case "${METHOD}" in
        pattern)  EXTRA="--pattern-pop 30 --pattern-gens 20" ;;
        ablation) EXTRA="--ablation-backgrounds 10" ;;
        sobol)    EXTRA="--sobol-n-base 20 --sobol-p-active 0.10" ;;
    esac

    echo "============================================================"
    echo "[seed1/${METHOD}] Score computation → ${OUT}"
    echo "  log: ${LOG}"
    echo "============================================================"
    python experiments/runners/compute_position_score.py \
        "${EVAL_CFG}" --method "${METHOD}" --out "${OUT}" ${EXTRA} \
        > "${LOG}" 2>&1

    if [[ ! -f "${OUT}" ]]; then
        echo "ERROR: Score file not produced for ${METHOD}; aborting." >&2
        exit 1
    fi
    echo "[seed1/${METHOD}] Score done."
}

# 1. Sparse default baseline
run_main "sparse"    "configs/Exp-22/mlm_composite_seed1.yaml"

# 2. Multi-tier (Phase-1 baseline)
run_main "multitier" "configs/Exp-22/mlm_composite_seed1_multitier.yaml"

# 3. Pattern: score → main
run_score "pattern"
run_main  "pattern"  "configs/Exp-22/mlm_composite_seed1_score_pattern.yaml"

# 4. Ablation: score → main
run_score "ablation"
run_main  "ablation" "configs/Exp-22/mlm_composite_seed1_score_ablation.yaml"

# 5. Sobol: score → main
run_score "sobol"
run_main  "sobol"    "configs/Exp-22/mlm_composite_seed1_score_sobol.yaml"

echo "============================================================"
echo "Seed-1 replication suite complete."
echo "============================================================"
