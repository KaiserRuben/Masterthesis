#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# Exp-23 Pipeline — VLM-SUT Comparison II (Qwen2.5-VL int4 + LLaVA-NeXT int8)
#
# Sequential run of both OpenVINO/Arc configs. Per-attempt timestamped logs
# (no overwrite). Continues to next config even if the first one crashes —
# the boundary tester writes parquet incrementally so partial data is
# salvageable per run dir.
#
# Wall-time estimate: ~4-6h per run on Arc A770 (400 gen × 50 pop × OV-int4/8).
# Total ~8-12h sequential.
#
# Run as nohup background job:
#   nohup bash experiments/runners/run_exp23_pipeline.sh \
#       > runs/Exp-23/orchestrator_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   disown
#
# Single config:
#   bash experiments/runners/run_exp23_pipeline.sh qwen25vl_int4_seed1
# ═══════════════════════════════════════════════════════════════════════════

set -uo pipefail   # NOTE: no -e — we want to continue on per-config crashes

cd "$(dirname "$0")/../.."

# Workstation defaults — override via env if running elsewhere.
export HF_HOME="${HF_HOME:-/mnt/storage/huggingface}"

if [[ $# -gt 0 ]]; then
    CONFIGS=("$@")
else
    CONFIGS=(qwen25vl_int4_seed1 llava_next_int8_seed1)
fi

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p runs/Exp-23

ORCH_LOG="runs/Exp-23/orchestrator_${TS}.log"

{
    echo "=== Exp-23 orchestrator start $(date -Iseconds) ==="
    echo "configs: ${CONFIGS[*]}"
    echo "HF_HOME: ${HF_HOME}"
    echo
} | tee -a "${ORCH_LOG}"

run_config() {
    local NAME="$1"
    local CFG="configs/Exp-23/${NAME}.yaml"
    local LOG="runs/Exp-23/${NAME}_${TS}.log"

    if [[ ! -f "${CFG}" ]]; then
        echo "[$(date -Iseconds)] ERROR: ${CFG} not found, skip" | tee -a "${ORCH_LOG}"
        return 1
    fi

    {
        echo "[$(date -Iseconds)] >>> ${NAME}"
        echo "[$(date -Iseconds)] log: ${LOG}"
    } | tee -a "${ORCH_LOG}"

    python experiments/runners/run_boundary_test.py "${CFG}" \
        > "${LOG}" 2>&1
    local RC=$?

    {
        if [[ ${RC} -eq 0 ]]; then
            echo "[$(date -Iseconds)] <<< ${NAME} done (rc=0)"
        else
            echo "[$(date -Iseconds)] <<< ${NAME} crashed (rc=${RC}) — see ${LOG}"
        fi
        echo
    } | tee -a "${ORCH_LOG}"

    return ${RC}
}

for NAME in "${CONFIGS[@]}"; do
    run_config "${NAME}" || true   # don't abort orchestrator on single-config crash
done

{
    echo "=== Exp-23 orchestrator complete $(date -Iseconds) ==="
} | tee -a "${ORCH_LOG}"
