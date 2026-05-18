#!/usr/bin/env bash
# ==============================================================================
# Preliminary Study 2 — In-context transfer of utility-aware summarization
#
# Flow:
#   1. Start vLLM with the base model
#   2. Run all three demo-type conditions (temporal / open / mixed) sequentially
#      — for each condition the teacher is called exactly k=6 times to build
#        demos from the train split, then eval runs on the test split
#   3. Print the 4-row x 2-col F1 table
#
# Usage:
#   chmod +x run_prelim_study2.sh
#   ./run_prelim_study2.sh           # run everything
#   ./run_prelim_study2.sh eval      # skip vLLM start, assume already up
#   ./run_prelim_study2.sh table     # just reprint the table (no vLLM needed)
# ==============================================================================
set -euo pipefail

# ── knobs ──────────────────────────────────────────────────────────────────────
BASE_LLM="Qwen/Qwen2.5-3B-Instruct"
DATASET="data/locomo10.json"
WINDOW_SIZE=5
N_EVAL=500
K_DEMOS=10
SEED=42

VLLM_PORT=11434
VLLM_GPU_UTIL=0.85
VLLM_MAX_MODEL_LEN=8192
VLLM_STARTUP_TIMEOUT=600

OUT_DIR="results/prelim_study2"
# ───────────────────────────────────────────────────────────────────────────────

STAGE="${1:-all}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO_ROOT/logs/prelim_study2_$TS"
mkdir -p "$LOG_DIR" "$OUT_DIR"

# ── colours ────────────────────────────────────────────────────────────────────
RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'
BLUE=$'\033[0;34m'; NC=$'\033[0m'
log()  { echo "${BLUE}[$(date +%H:%M:%S)]${NC} $*"; }
ok()   { echo "${GREEN}[OK]${NC} $*"; }
warn() { echo "${YELLOW}[WARN]${NC} $*"; }
err()  { echo "${RED}[ERR]${NC} $*" >&2; }

# ==============================================================================
# vLLM helpers
# ==============================================================================
VLLM_PID=""

vllm_is_up() {
    curl -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1
}

start_vllm() {
    local logfile="$1"

    if vllm_is_up; then
        warn "vLLM already up on :$VLLM_PORT — killing first"
        stop_vllm
        sleep 3
    fi

    log "Starting vLLM: model=$BASE_LLM  port=$VLLM_PORT"
    nohup vllm serve "$BASE_LLM" \
        --port                   "$VLLM_PORT" \
        --served-model-name      "$BASE_LLM" \
        --gpu-memory-utilization "$VLLM_GPU_UTIL" \
        --max-model-len          "$VLLM_MAX_MODEL_LEN" \
        > "$logfile" 2>&1 &
    VLLM_PID=$!
    log "vLLM PID=$VLLM_PID — waiting for /v1/models ..."

    local waited=0
    while [ "$waited" -lt "$VLLM_STARTUP_TIMEOUT" ]; do
        if vllm_is_up; then
            ok "vLLM ready on :$VLLM_PORT after ${waited}s"
            return 0
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            err "vLLM process died — see $logfile"
            tail -50 "$logfile" >&2 || true
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
        [ $((waited % 60)) -eq 0 ] && log "  still waiting ... (${waited}s)"
    done
    err "vLLM did not become ready within ${VLLM_STARTUP_TIMEOUT}s"
    tail -50 "$logfile" >&2 || true
    return 1
}

stop_vllm() {
    log "Stopping vLLM ..."
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID" 2>/dev/null || true
        sleep 2
        kill -9 "$VLLM_PID" 2>/dev/null || true
    fi
    pkill -f "vllm serve"       2>/dev/null || true
    pkill -f "vllm.entrypoints" 2>/dev/null || true
    sleep 3
    VLLM_PID=""
    if vllm_is_up; then
        warn "vLLM still up — force-killing by port"
        fuser -k -9 "${VLLM_PORT}/tcp" 2>/dev/null || true
        sleep 2
    fi
    ok "vLLM stopped"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null || true
}

trap 'stop_vllm 2>/dev/null || true' EXIT INT TERM

# Patch config.py to point at our vLLM instance
patch_config() {
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
          'OPENAI_BASE_URL = "http://localhost:${VLLM_PORT}/v1"', src)
src = sub(r'^LLM_MODEL\s*=.*$',
          'LLM_MODEL = "${BASE_LLM}"', src)
src = sub(r'^JUDGE_BASE_URL\s*=.*$',
          'JUDGE_BASE_URL  = "http://localhost:${VLLM_PORT}/v1"', src)
src = sub(r'^JUDGE_MODEL\s*=.*$',
          'JUDGE_MODEL = "${BASE_LLM}"', src)
p.write_text(src)
print("config.py patched: port=${VLLM_PORT}, model=${BASE_LLM}")
PY
}

# ==============================================================================
# Stages
# ==============================================================================

stage_eval() {
    log "=== Running 3 demo-type conditions (temporal / open / mixed) ==="

    python eval/run_prelim_study2.py \
        --dataset     "$DATASET" \
        --out-dir     "$OUT_DIR" \
        --window-size "$WINDOW_SIZE" \
        --n-eval      "$N_EVAL" \
        --k-demos     "$K_DEMOS" \
        --demo-type   all \
        --seed        "$SEED" \
        2>&1 | tee "$LOG_DIR/study2_eval.log"

    ok "Eval done — results in $OUT_DIR"
}

stage_table() {
    log "=== Summary table ==="
    python eval/run_prelim_study2.py \
        --table-only \
        --out-dir "$OUT_DIR" \
        --seed    "$SEED"
}

# ==============================================================================
# Driver
# ==============================================================================
log "Prelim Study 2 | log dir: $LOG_DIR | results: $OUT_DIR"

case "$STAGE" in
    eval)
        patch_config
        stage_eval
        stage_table
        ;;
    table)
        stage_table
        ;;
    all)
        patch_config
        start_vllm "$LOG_DIR/vllm.log"
        stage_eval
        stop_vllm
        stage_table
        ;;
    *)
        err "Unknown stage: $STAGE"
        echo "Usage: $0 [all|eval|table]" >&2
        exit 2
        ;;
esac

ok "Done.  Results: $OUT_DIR"