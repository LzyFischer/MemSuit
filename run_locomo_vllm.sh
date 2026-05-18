#!/usr/bin/env bash
# =============================================================================
# Full pipeline: install deps -> launch vLLM server -> run LoCoMo QA eval
# Default model: Qwen/Qwen2.5-3B-Instruct
# Usage:
#   bash run_locomo.sh                    # full run (all 10 conversations)
#   bash run_locomo.sh --smoke            # 1 conversation, single-hop only
#   bash run_locomo.sh --skip-install     # skip pip installs
#   bash run_locomo.sh --skip-server      # use an already-running server
#   MODEL=Qwen/Qwen2.5-7B-Instruct bash run_locomo.sh
# =============================================================================
set -euo pipefail

# ----------------------------- config ---------------------------------------
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-28000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
DTYPE="${DTYPE:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
LOG_DIR="${LOG_DIR:-logs}"
PY_SCRIPT="${PY_SCRIPT:-run_locomo_vllm.py}"
SERVER_LOG="${LOG_DIR}/vllm_server.log"
SERVER_PID_FILE="${LOG_DIR}/vllm_server.pid"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-600}"   # max seconds to wait for /v1/models

SMOKE=0
SKIP_INSTALL=0
SKIP_SERVER=0
USE_LLM_JUDGE=0
EXTRA_PY_ARGS=()

# ----------------------------- arg parsing ----------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)        SMOKE=1; shift ;;
    --skip-install) SKIP_INSTALL=1; shift ;;
    --skip-server)  SKIP_SERVER=1; shift ;;
    --judge)        USE_LLM_JUDGE=1; shift ;;
    --model)        MODEL="$2"; shift 2 ;;
    --port)         PORT="$2"; shift 2 ;;
    --tp)           TENSOR_PARALLEL="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --max-ctx)      MAX_CONTEXT_TOKENS="$2"; shift 2 ;;
    --)             shift; EXTRA_PY_ARGS+=("$@"); break ;;
    *)              EXTRA_PY_ARGS+=("$1"); shift ;;
  esac
done

mkdir -p "$LOG_DIR" "$OUTPUT_DIR" data

echo "================================================================"
echo "  LoCoMo eval pipeline"
echo "  model       : $MODEL"
echo "  port        : $PORT"
echo "  TP / dtype  : $TENSOR_PARALLEL / $DTYPE"
echo "  max-model-len / max-ctx : $MAX_MODEL_LEN / $MAX_CONTEXT_TOKENS"
echo "  output dir  : $OUTPUT_DIR"
echo "  smoke test  : $SMOKE"
echo "  skip install/server : $SKIP_INSTALL / $SKIP_SERVER"
echo "================================================================"

# # ----------------------------- 1. install -----------------------------------
# if [[ "$SKIP_INSTALL" -eq 0 ]]; then
#   echo "[1/4] installing dependencies ..."
#   python -m pip install --upgrade pip >/dev/null
#   # vllm pulls torch; if you already have a matched torch+cuda, this is a no-op
#   python -m pip install "vllm>=0.6.0"
#   python -m pip install "openai>=1.30" tqdm requests transformers
# else
#   echo "[1/4] skipped install"
# fi

# ----------------------------- 2. start vLLM --------------------------------
cleanup() {
  if [[ -f "$SERVER_PID_FILE" ]]; then
    pid=$(cat "$SERVER_PID_FILE" 2>/dev/null || true)
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "[cleanup] stopping vLLM server (pid=$pid)"
      kill "$pid" 2>/dev/null || true
      # give it a moment, then SIGKILL if needed
      for _ in 1 2 3 4 5; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
      done
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$SERVER_PID_FILE"
  fi
}

if [[ "$SKIP_SERVER" -eq 0 ]]; then
  echo "[2/4] launching vLLM server ..."

  # if something is already on the port, bail out instead of clobbering
  if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "  port $PORT already serves /v1/models -- reusing it (treating as --skip-server)"
  else
    # only register cleanup when we actually started the server ourselves
    trap cleanup EXIT INT TERM

    nohup vllm serve "$MODEL" \
      --host "$HOST" \
      --port "$PORT" \
      --max-model-len "$MAX_MODEL_LEN" \
      --tensor-parallel-size "$TENSOR_PARALLEL" \
      --gpu-memory-utilization "$GPU_MEM_UTIL" \
      --dtype "$DTYPE" \
      --served-model-name "$MODEL" \
      > "$SERVER_LOG" 2>&1 &
    echo $! > "$SERVER_PID_FILE"
    echo "  pid=$(cat "$SERVER_PID_FILE")  log=$SERVER_LOG"

    echo -n "  waiting for http://localhost:${PORT}/v1/models "
    SECONDS=0
    while true; do
      if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
        echo " ready (${SECONDS}s)"
        break
      fi
      # detect early death
      pid=$(cat "$SERVER_PID_FILE")
      if ! kill -0 "$pid" 2>/dev/null; then
        echo
        echo "[fatal] vLLM server died before becoming healthy. Tail of log:"
        tail -n 60 "$SERVER_LOG" || true
        exit 1
      fi
      if [[ "$SECONDS" -gt "$HEALTH_TIMEOUT" ]]; then
        echo
        echo "[fatal] timeout (>${HEALTH_TIMEOUT}s). Tail of log:"
        tail -n 60 "$SERVER_LOG" || true
        exit 1
      fi
      echo -n "."
      sleep 3
    done
  fi
else
  echo "[2/4] skipped server launch"
  if ! curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "[fatal] no server reachable at http://localhost:${PORT}/v1/models"
    exit 1
  fi
fi

# ----------------------------- 3. quick smoke -------------------------------
echo "[3/4] sanity-checking the API ..."
curl -sf "http://localhost:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Reply with the single word: pong\"}],
    \"max_tokens\": 8,
    \"temperature\": 0
  }" | python -c "import sys, json; r=json.load(sys.stdin); print('  ->', r['choices'][0]['message']['content'].strip())" \
  || { echo "[fatal] sanity check failed"; exit 1; }

# ----------------------------- 4. run eval ----------------------------------
echo "[4/4] running LoCoMo evaluation ..."

PY_ARGS=(
  --model "$MODEL"
  --base_url "http://localhost:${PORT}/v1"
  --output_dir "$OUTPUT_DIR"
  --max_context_tokens "$MAX_CONTEXT_TOKENS"
)
if [[ "$SMOKE" -eq 1 ]]; then
  PY_ARGS+=(--num_samples 1 --category 1)
fi
if [[ "$USE_LLM_JUDGE" -eq 1 ]]; then
  PY_ARGS+=(--use_llm_judge)
fi
if [[ "${#EXTRA_PY_ARGS[@]}" -gt 0 ]]; then
  PY_ARGS+=("${EXTRA_PY_ARGS[@]}")
fi

echo "  python $PY_SCRIPT ${PY_ARGS[*]}"
python "$PY_SCRIPT" "${PY_ARGS[@]}"

echo
echo "[done] results in $OUTPUT_DIR/"
ls -lh "$OUTPUT_DIR/"