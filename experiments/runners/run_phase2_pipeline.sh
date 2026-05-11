#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# Exp-22c Phase-2 Pipeline — sequential score-compute + main-run for 3 methods.
#
# Total ~10-12h sequential. Each step writes its own log under logs/.
# Configs: configs/Exp-22/mlm_composite_junco_chickadee_score_{method}.yaml
# Score outputs: runs/Exp-22/scores/{method}_seed83.npy
#
# Usage:
#   bash experiments/runners/run_phase2_pipeline.sh
#
# Or run a single method:
#   bash experiments/runners/run_phase2_pipeline.sh pattern
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ $# -gt 0 ]]; then
    METHODS=("$@")
else
    METHODS=(pattern ablation sobol)
fi
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs runs/Exp-22/scores

for METHOD in "${METHODS[@]}"; do
    SCORE_LOG="logs/exp22c_${METHOD}_score_${TS}.log"
    MAIN_LOG="logs/exp22c_${METHOD}_main_${TS}.log"
    SCORE_OUT="runs/Exp-22/scores/${METHOD}_seed83.npy"

    EVAL_CFG="configs/Exp-22/mlm_composite_junco_chickadee_multitier.yaml"
    MAIN_CFG="configs/Exp-22/mlm_composite_junco_chickadee_score_${METHOD}.yaml"

    EXTRA_ARGS=""
    case "${METHOD}" in
        pattern)  EXTRA_ARGS="--pattern-pop 30 --pattern-gens 20" ;;
        ablation) EXTRA_ARGS="--ablation-backgrounds 10" ;;
        sobol)    EXTRA_ARGS="--sobol-n-base 20 --sobol-p-active 0.10" ;;
    esac

    echo "============================================================"
    echo "[${METHOD}] Score computation → ${SCORE_OUT}"
    echo "  log: ${SCORE_LOG}"
    echo "============================================================"
    python experiments/runners/compute_position_score.py \
        "${EVAL_CFG}" \
        --method "${METHOD}" \
        --out "${SCORE_OUT}" \
        ${EXTRA_ARGS} \
        > "${SCORE_LOG}" 2>&1

    if [[ ! -f "${SCORE_OUT}" ]]; then
        echo "ERROR: Score file not produced for method ${METHOD}; aborting." >&2
        exit 1
    fi

    echo "[${METHOD}] Score done. Launching main run."
    echo "  log: ${MAIN_LOG}"
    python experiments/runners/run_boundary_test.py \
        "${MAIN_CFG}" \
        > "${MAIN_LOG}" 2>&1

    echo "[${METHOD}] Main run done."
done

echo "============================================================"
echo "Phase-2 pipeline complete. Methods: ${METHODS[*]}"
echo "============================================================"
