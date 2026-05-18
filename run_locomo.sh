#!/usr/bin/env bash
# =============================================================================
# run_locomo.sh
#
# End-to-end script:
#   1. install deps (vLLM server-side, openai+tqdm client-side)
#   2. download LoCoMo dataset (data/locomo10.json) if missing
#   3. launch a vLLM OpenAI-compatible server with Qwen2.5-3B-Instruct
#   4. wait for it to be ready
#   5. run the LoCoMo QA eval with the original prompts
#   6. shut the server down
#
# Edit the variables in the CONFIG block below if you want a different model,
# port, GPU memory fraction, context length, etc.
#
# Usage:
#   chmod +x run_locomo.sh
#   ./run_locomo.sh                 # full run on all 7,512 questions
#   ./run_locomo.sh --smoke         # quick 20-question smoke test
#   ./run_locomo.sh --no-install    # skip pip installs
#   ./run_locomo.sh --no-server     # assume vLLM is already running on $PORT
# =============================================================================
set -euo pipefail

# -------------------------- CONFIG -------------------------------------------
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-30000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TEMPERATURE="${TEMPERATURE:-0.0}"

DATA_DIR="data"
DATA_FILE="${DATA_DIR}/locomo10.json"
DATA_URL="https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"

OUT_DIR="outputs"
OUT_FILE="${OUT_DIR}/$(echo "${MODEL}" | tr '/' '_')_locomo.json"

SERVER_LOG="vllm_server.log"
PYTHON="${PYTHON:-python}"
# ----------------------------------------------------------------------------

# -------------------------- flag parsing -------------------------------------
DO_INSTALL=1
START_SERVER=1
SMOKE=0
EXTRA_EVAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke)        SMOKE=1; shift ;;
        --no-install)   DO_INSTALL=0; shift ;;
        --no-server)    START_SERVER=0; shift ;;
        --model)        MODEL="$2"; shift 2 ;;
        --port)         PORT="$2"; shift 2 ;;
        --out-file)     OUT_FILE="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,25p' "$0"; exit 0 ;;
        *)
            EXTRA_EVAL_ARGS+=("$1"); shift ;;
    esac
done

if [[ "${SMOKE}" -eq 1 ]]; then
    EXTRA_EVAL_ARGS+=(--max-samples 20)
    OUT_FILE="${OUT_DIR}/smoke_test.json"
fi

mkdir -p "${DATA_DIR}" "${OUT_DIR}"

# # -------------------------- 1. install ---------------------------------------
# if [[ "${DO_INSTALL}" -eq 1 ]]; then
#     echo "[1/5] installing dependencies ..."
#     ${PYTHON} -m pip install --quiet --upgrade pip
#     ${PYTHON} -m pip install --quiet "vllm>=0.6.0" "openai>=1.40.0" "tqdm>=4.65.0"
# else
#     echo "[1/5] skipping pip install (--no-install)"
# fi

# # -------------------------- 2. download data ---------------------------------
# if [[ ! -f "${DATA_FILE}" ]]; then
#     echo "[2/5] downloading ${DATA_URL} -> ${DATA_FILE}"
#     if command -v curl >/dev/null 2>&1; then
#         curl -fsSL "${DATA_URL}" -o "${DATA_FILE}"
#     else
#         wget -q "${DATA_URL}" -O "${DATA_FILE}"
#     fi
#     echo "      $(du -h "${DATA_FILE}" | cut -f1) downloaded."
# else
#     echo "[2/5] dataset already at ${DATA_FILE}"
# fi

# -------------------------- 3. start vLLM ------------------------------------
SERVER_PID=""
cleanup() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo
        echo "[cleanup] stopping vLLM server (pid=${SERVER_PID}) ..."
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

BASE_URL="http://${HOST}:${PORT}/v1"

if [[ "${START_SERVER}" -eq 1 ]]; then
    echo "[3/5] launching vLLM server (logs -> ${SERVER_LOG}) ..."
    echo "      model=${MODEL}  port=${PORT}  max_model_len=${MAX_MODEL_LEN}"
    ${PYTHON} -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" \
        --host "${HOST}" \
        --port "${PORT}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${GPU_MEM_UTIL}" \
        > "${SERVER_LOG}" 2>&1 &
    SERVER_PID=$!

    echo "      waiting for ${BASE_URL}/models to come up (pid=${SERVER_PID}) ..."
    for i in $(seq 1 600); do
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "      vLLM died. Last 50 lines of ${SERVER_LOG}:"
            tail -n 50 "${SERVER_LOG}" || true
            exit 1
        fi
        if curl -fsS "${BASE_URL}/models" >/dev/null 2>&1; then
            echo "      ready after ${i}s."
            break
        fi
        sleep 1
        if [[ "${i}" -eq 600 ]]; then
            echo "      timeout waiting for vLLM. Last 50 lines of ${SERVER_LOG}:"
            tail -n 50 "${SERVER_LOG}" || true
            exit 1
        fi
    done
else
    echo "[3/5] --no-server: assuming vLLM already serving at ${BASE_URL}"
fi

# -------------------------- 4. run the eval ----------------------------------
echo "[4/5] running LoCoMo eval ..."
${PYTHON} run_locomo.py \
    --data "${DATA_FILE}" \
    --model "${MODEL}" \
    --base-url "${BASE_URL}" \
    --api-key EMPTY \
    --out-file "${OUT_FILE}" \
    --max-context-tokens "${MAX_CONTEXT_TOKENS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    "${EXTRA_EVAL_ARGS[@]}"

# -------------------------- 5. done ------------------------------------------
echo "[5/5] done. Predictions saved to ${OUT_FILE}"
