#!/usr/bin/env bash
# =============================================================================
# run_longmemeval.sh
#
# Run SimpleMem on LongMemEval.  Mirrors run_locomo.sh:
#   1. (optional) launch a vLLM OpenAI-compatible server
#   2. wait for it to be ready
#   3. ensure the LongMemEval JSON is in data/
#   4. run eval/run_longmemeval.py
#   5. (optional) tear down the server
#
# Examples
# --------
#   # Baseline (current LLM_MODEL in config.py)
#   ./run_longmemeval.sh
#
#   # Distilled LoCoMo checkpoint (already merged to a HF dir, served by vLLM)
#   ./run_longmemeval.sh \
#       --model train/checkpoints/qwen25-3b-distill-merged \
#       --variant oracle \
#       --out-tag distill
#
#   # Server already running externally:
#   ./run_longmemeval.sh --no-server --model qwen25-3b-distill
#
#   # Small smoke test (first 5 questions)
#   ./run_longmemeval.sh --smoke
# =============================================================================
set -euo pipefail

# -------------------------- CONFIG -------------------------------------------
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"

# Which LongMemEval file to evaluate on.
#   oracle: only evidence sessions (smallest, fastest sanity check)
#   s     : ~115k-token haystack, ~40 sessions  (standard setting)
#   m     : ~1.5M-token haystack, ~500 sessions (the hard one)
VARIANT="${VARIANT:-s}"

# Tag appended to the output paths so baseline / distill runs don't clobber
# each other. e.g. `--out-tag distill` ⇒ results/lme_oracle_distill.json
OUT_TAG="${OUT_TAG:-baseline}"

DATA_DIR="data"
RESULT_DIR="results"
SERVER_LOG="vllm_lme.log"
PYTHON="${PYTHON:-python}"

# Optional: path / HF id of a contrastive-finetuned SentenceTransformer
# (Phase 2 checkpoint from train/train_contrastive.py). Leave empty to use
# config.EMBEDDING_MODEL.
EMBEDDING="${EMBEDDING:-}"
# ----------------------------------------------------------------------------

# -------------------------- flag parsing -------------------------------------
DO_INSTALL=0
START_SERVER=1
SMOKE=0
LLM_JUDGE=0
EXTRA_EVAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke)        SMOKE=1; shift ;;
        --install)      DO_INSTALL=1; shift ;;
        --no-server)    START_SERVER=0; shift ;;
        --model)        MODEL="$2"; shift 2 ;;
        --embedding)    EMBEDDING="$2"; shift 2 ;;
        --port)         PORT="$2"; shift 2 ;;
        --variant)      VARIANT="$2"; shift 2 ;;
        --out-tag)      OUT_TAG="$2"; shift 2 ;;
        --llm-judge)    LLM_JUDGE=1; shift ;;
        -h|--help)
            sed -n '2,30p' "$0"; exit 0 ;;
        *)
            EXTRA_EVAL_ARGS+=("$1"); shift ;;
    esac
done

# Map variant → file name (the cleaned versions are the post-Sep-2025 release).
case "${VARIANT}" in
    oracle) DATA_FILE="${DATA_DIR}/longmemeval_oracle.json" ;;
    s)      DATA_FILE="${DATA_DIR}/longmemeval_s_cleaned.json" ;;
    m)      DATA_FILE="${DATA_DIR}/longmemeval_m_cleaned.json" ;;
    *)
        echo "[error] unknown --variant '${VARIANT}'. Use oracle|s|m." >&2
        exit 1 ;;
esac

RESULT_FILE="${RESULT_DIR}/lme_${VARIANT}_${OUT_TAG}.json"
HYP_FILE="${RESULT_DIR}/lme_${VARIANT}_${OUT_TAG}.hyp.jsonl"

if [[ "${SMOKE}" -eq 1 ]]; then
    EXTRA_EVAL_ARGS+=(--num-samples 5)
    RESULT_FILE="${RESULT_DIR}/lme_${VARIANT}_${OUT_TAG}_smoke.json"
    HYP_FILE="${RESULT_DIR}/lme_${VARIANT}_${OUT_TAG}_smoke.hyp.jsonl"
fi

if [[ "${LLM_JUDGE}" -eq 1 ]]; then
    EXTRA_EVAL_ARGS+=(--llm-judge)
fi

mkdir -p "${DATA_DIR}" "${RESULT_DIR}"

# -------------------------- 1. (optional) install ----------------------------
if [[ "${DO_INSTALL}" -eq 1 ]]; then
    echo "[install] pip dependencies ..."
    ${PYTHON} -m pip install --quiet --upgrade pip
    ${PYTHON} -m pip install --quiet "vllm>=0.6.0" "openai>=1.40.0" "tqdm>=4.65.0"
fi

# -------------------------- 2. ensure data -----------------------------------
DATA_URL_BASE="https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"
case "${VARIANT}" in
    oracle) DATA_URL="${DATA_URL_BASE}/longmemeval_oracle.json" ;;
    s)      DATA_URL="${DATA_URL_BASE}/longmemeval_s_cleaned.json" ;;
    m)      DATA_URL="${DATA_URL_BASE}/longmemeval_m_cleaned.json" ;;
esac

if [[ ! -f "${DATA_FILE}" ]]; then
    echo "[data] downloading ${DATA_URL} -> ${DATA_FILE}"
    if command -v curl >/dev/null 2>&1; then
        curl -fSL "${DATA_URL}" -o "${DATA_FILE}"
    else
        wget -q "${DATA_URL}" -O "${DATA_FILE}"
    fi
    echo "       $(du -h "${DATA_FILE}" | cut -f1) downloaded."
else
    echo "[data] using existing ${DATA_FILE}"
fi

# -------------------------- 3. (optional) start vLLM -------------------------
SERVER_PID=""
cleanup() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo
        echo "[cleanup] stopping vLLM (pid=${SERVER_PID}) ..."
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

BASE_URL="http://${HOST}:${PORT}/v1"

if [[ "${START_SERVER}" -eq 1 ]]; then
    echo "[server] launching vLLM (logs -> ${SERVER_LOG}) ..."
    echo "         model=${MODEL}  port=${PORT}  max_model_len=${MAX_MODEL_LEN}"
    ${PYTHON} -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" \
        --host "${HOST}" \
        --port "${PORT}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${GPU_MEM_UTIL}" \
        > "${SERVER_LOG}" 2>&1 &
    SERVER_PID=$!

    echo "         waiting for ${BASE_URL}/models ..."
    for i in $(seq 1 600); do
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "         vLLM died. Last 50 lines of ${SERVER_LOG}:"
            tail -n 50 "${SERVER_LOG}" || true
            exit 1
        fi
        if curl -fsS "${BASE_URL}/models" >/dev/null 2>&1; then
            echo "         ready after ${i}s."
            break
        fi
        sleep 1
        if [[ "${i}" -eq 600 ]]; then
            echo "         timeout. Last 50 lines of ${SERVER_LOG}:"
            tail -n 50 "${SERVER_LOG}" || true
            exit 1
        fi
    done
else
    echo "[server] --no-server: assuming vLLM serves at ${BASE_URL}"
fi

# -------------------------- 4. run eval --------------------------------------
echo "[eval] LongMemEval (${VARIANT}) — model=${MODEL}"
if [[ -n "${EMBEDDING}" ]]; then
    echo "                              embedding=${EMBEDDING}"
    EXTRA_EVAL_ARGS+=(--embedding-override "${EMBEDDING}")
fi
echo "       → ${RESULT_FILE}"
echo "       → ${HYP_FILE}"

${PYTHON} eval/run_longmemeval.py \
    --dataset         "${DATA_FILE}" \
    --result-file     "${RESULT_FILE}" \
    --hypothesis-file "${HYP_FILE}" \
    --model-override  "${MODEL}" \
    --base-url-override "${BASE_URL}" \
    "${EXTRA_EVAL_ARGS[@]}"

echo
echo "[done] Detailed results : ${RESULT_FILE}"
echo "       Hypotheses (jsonl): ${HYP_FILE}"
echo
echo "To get the official LongMemEval GPT-4o accuracy on these predictions:"
echo "  cd <LongMemEval-repo>/src/evaluation"
echo "  export OPENAI_API_KEY=..."
echo "  python evaluate_qa.py gpt-4o ${HYP_FILE} ${DATA_FILE}"
