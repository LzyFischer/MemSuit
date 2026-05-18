#!/usr/bin/env bash
# ==============================================================================
# Preliminary Study 1 — vanilla vs utility-aware summarization on LoCoMo.
#
# What this script does:
#   1. Launches a vLLM server (Qwen2.5-3B-Instruct by default) on $VLLM_PORT.
#   2. Patches memsuit/config.py so OPENAI_BASE_URL points at that server.
#   3. Runs eval/prelim_study1.py with --num-queries / --workers as given.
#   4. Tears down vLLM cleanly on exit (success, failure, or Ctrl-C).
#
# Usage:
#   chmod +x run_prelim_study1.sh
#   ./run_prelim_study1.sh                       # 100 queries, 8 workers, default model
#   ./run_prelim_study1.sh 200 16                # 200 queries, 16 workers
#   NUM_QUERIES=50 WORKERS=4 ./run_prelim_study1.sh
#
# Environment overrides (all optional):
#   NUM_QUERIES   number of queries to evaluate (default: 100)
#   WORKERS       parallel LLM-call workers     (default: 8)
#   SEED          random seed                    (default: 42)
#   BASE_LLM      HF model id or local path      (default: Qwen/Qwen2.5-3B-Instruct)
#   VLLM_PORT     port for vLLM                  (default: 11434)
#   VLLM_GPU_UTIL gpu memory util fraction       (default: 0.85)
#   VLLM_MAX_LEN  max model context length       (default: 8192)
#   DATA          dataset path                    (default: data/locomo10.json)
#   OUTPUT        result JSON path               (default: prelim_study1_results.json)
#   SKIP_VLLM=1   skip launching vLLM (assume already running on $VLLM_PORT)
# ==============================================================================
set -euo pipefail

# ---------- defaults ----------
NUM_QUERIES="${NUM_QUERIES:-${1:-1000}}"
WORKERS="${WORKERS:-${2:-8}}"
SEED="${SEED:-42}"

BASE_LLM="${BASE_LLM:-Qwen/Qwen2.5-3B-Instruct}"
SERVED_NAME="${SERVED_NAME:-Qwen/Qwen2.5-3B-Instruct}"
VLLM_PORT="${VLLM_PORT:-11434}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"
VLLM_MAX_LEN="${VLLM_MAX_LEN:-8192}"
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-600}"

DATA="${DATA:-data/locomo10.json}"
OUTPUT="${OUTPUT:-prelim_study1_results.json}"
SKIP_VLLM="${SKIP_VLLM:-0}"
# ------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO_ROOT/logs/prelim1_$TS"
mkdir -p "$LOG_DIR"
VLLM_LOG="$LOG_DIR/vllm.log"
RUN_LOG="$LOG_DIR/prelim_study1.log"

# pretty print
RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; BLUE=$'\033[0;34m'; NC=$'\033[0m'
log()  { echo "${BLUE}[$(date +%H:%M:%S)]${NC} $*"; }
ok()   { echo "${GREEN}[OK]${NC} $*"; }
warn() { echo "${YELLOW}[WARN]${NC} $*"; }
err()  { echo "${RED}[ERR]${NC} $*" >&2; }

# ============================================================
# vLLM lifecycle
# ============================================================
VLLM_PID=""

vllm_is_up() {
    curl -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1
}

start_vllm() {
    if vllm_is_up; then
        warn "vLLM already responding on :${VLLM_PORT} — killing it for a clean state"
        stop_vllm
        sleep 3
    fi
    log "Launching vLLM: model=${BASE_LLM}  served=${SERVED_NAME}  port=${VLLM_PORT}"
    log "  log -> $VLLM_LOG"
    nohup vllm serve "$BASE_LLM" \
        --port "$VLLM_PORT" \
        --served-model-name "$SERVED_NAME" \
        --gpu-memory-utilization "$VLLM_GPU_UTIL" \
        --max-model-len "$VLLM_MAX_LEN" \
        > "$VLLM_LOG" 2>&1 &
    VLLM_PID=$!
    log "vLLM PID=$VLLM_PID — waiting for /v1/models (up to ${VLLM_STARTUP_TIMEOUT}s)…"

    local waited=0
    while [ "$waited" -lt "$VLLM_STARTUP_TIMEOUT" ]; do
        if vllm_is_up; then
            ok "vLLM ready on :${VLLM_PORT} after ${waited}s"
            return 0
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            err "vLLM process died — see $VLLM_LOG"
            tail -60 "$VLLM_LOG" >&2 || true
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
        if [ $((waited % 30)) -eq 0 ]; then
            log "  still waiting … (${waited}s/${VLLM_STARTUP_TIMEOUT}s)"
        fi
    done
    err "vLLM did not become ready within ${VLLM_STARTUP_TIMEOUT}s"
    tail -60 "$VLLM_LOG" >&2 || true
    return 1
}

stop_vllm() {
    if [ "$SKIP_VLLM" = "1" ]; then return 0; fi
    log "Stopping vLLM…"
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID" 2>/dev/null || true
        sleep 2
        kill -9 "$VLLM_PID" 2>/dev/null || true
    fi
    pkill -f "vllm serve"          2>/dev/null || true
    pkill -f "vllm.entrypoints"    2>/dev/null || true
    sleep 2
    if vllm_is_up; then
        warn "vLLM still on :${VLLM_PORT} — SIGKILL anyone holding the port"
        fuser -k -9 "${VLLM_PORT}/tcp" 2>/dev/null || true
        sleep 2
    fi
    VLLM_PID=""
    ok "vLLM stopped"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null || true
}

trap 'stop_vllm 2>/dev/null || true' EXIT INT TERM

# ============================================================
# Patch config.py so prelim_study1.py talks to our vLLM instance
# ============================================================
patch_config() {
    log "Patching config.py: OPENAI_BASE_URL=:${VLLM_PORT}, LLM_MODEL=${SERVED_NAME}"
    python - <<PY
import re
from pathlib import Path
p = Path("config.py")
src = p.read_text()
def sub(pat, repl, s):
    new, n = re.subn(pat, repl, s, count=1, flags=re.MULTILINE)
    if n == 0:
        raise SystemExit(f"FAILED to patch: {pat}")
    return new
src = sub(r'^OPENAI_BASE_URL\s*=.*$',
          f'OPENAI_BASE_URL = "http://localhost:${VLLM_PORT}/v1"', src)
src = sub(r'^LLM_MODEL\s*=.*$',
          f'LLM_MODEL = "${SERVED_NAME}"', src)
p.write_text(src)
print("config.py patched.")
PY
}

# ============================================================
# Run the experiment
# ============================================================
run_study() {
    log "Running preliminary study 1"
    log "  data        : $DATA"
    log "  num_queries : $NUM_QUERIES"
    log "  workers     : $WORKERS"
    log "  seed        : $SEED"
    log "  output      : $OUTPUT"
    log "  log         : $RUN_LOG"

    python eval/prelim_study1.py \
        --data "$DATA" \
        --num-queries "$NUM_QUERIES" \
        --workers "$WORKERS" \
        --seed "$SEED" \
        --output "$OUTPUT" \
        2>&1 | tee "$RUN_LOG"
}

# ============================================================
# Main
# ============================================================
log "==== preliminary study 1 ===="
log "log dir: $LOG_DIR"

if [ ! -f "$DATA" ]; then
    err "dataset not found: $DATA"
    exit 1
fi
if [ ! -f "eval/prelim_study1.py" ]; then
    err "eval/prelim_study1.py not found — drop the script into memsuit/eval/ first"
    exit 1
fi

# 1) vLLM
if [ "$SKIP_VLLM" = "1" ]; then
    if ! vllm_is_up; then
        err "SKIP_VLLM=1 but no server responding on :${VLLM_PORT}"
        exit 1
    fi
    ok "Reusing existing vLLM on :${VLLM_PORT}"
else
    start_vllm
fi

# 2) config.py
patch_config

# 3) experiment
run_study

ok "Done. Results: $OUTPUT  |  Logs: $LOG_DIR"
