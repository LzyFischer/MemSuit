#!/usr/bin/env bash
# ==============================================================================
# Granularity Ablation — fix the memory builder to emit exactly 1 entry per
# window, sweep WINDOW_SIZE ∈ {1, 5, 10, 20}.
#
# The whole pipeline (SimpleMem store → HybridRetriever → AnswerGenerator)
# runs unchanged — only the extraction prompt switches.
#
# Usage:
#   chmod +x run_granularity_ablation.sh
#   ./run_granularity_ablation.sh            # vLLM + 4 evals + plot
#   ./run_granularity_ablation.sh eval       # assume vLLM already up
#   ./run_granularity_ablation.sh plot       # just (re)draw the bar plot
#   ./run_granularity_ablation.sh table      # just reprint the table
# ==============================================================================
set -euo pipefail

# ── knobs ──────────────────────────────────────────────────────────────────────
BASE_LLM="Qwen/Qwen2.5-3B-Instruct"
DATASET="data/locomo10.json"
TURN_COUNTS=(5 10)
NUM_SAMPLES=""           # empty = full benchmark; e.g. NUM_SAMPLES="3" for smoke test
LLM_JUDGE="false"        # set to "true" to enable LLM-as-judge
PARALLEL_QUESTIONS="true"
TEST_WORKERS=8

VLLM_PORT=11434
VLLM_GPU_UTIL=0.85
VLLM_MAX_MODEL_LEN=30000
VLLM_STARTUP_TIMEOUT=600

OUT_DIR="results/granularity_ablation"
FIG_OUT="figures/granularity_ablation"
# ───────────────────────────────────────────────────────────────────────────────

STAGE="${1:-all}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO_ROOT/logs/granularity_ablation_$TS"
mkdir -p "$LOG_DIR" "$OUT_DIR"

# ── colours ────────────────────────────────────────────────────────────────────
RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'
BLUE=$'\033[0;34m'; NC=$'\033[0m'
log()  { echo "${BLUE}[$(date +%H:%M:%S)]${NC} $*"; }
ok()   { echo "${GREEN}[OK]${NC} $*"; }
warn() { echo "${YELLOW}[WARN]${NC} $*"; }
err()  { echo "${RED}[ERR]${NC} $*" >&2; }

# ==============================================================================
# vLLM helpers (copied from run_prelim_study2.sh — same conventions)
# ==============================================================================
VLLM_PID=""
vllm_is_up() { curl -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; }

start_vllm() {
    local logfile="$1"
    if vllm_is_up; then warn "vLLM already up on :$VLLM_PORT — killing first"; stop_vllm; sleep 3; fi
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
        if vllm_is_up; then ok "vLLM ready on :$VLLM_PORT after ${waited}s"; return 0; fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            err "vLLM process died — see $logfile"; tail -50 "$logfile" >&2 || true; return 1
        fi
        sleep 5; waited=$((waited + 5))
        [ $((waited % 60)) -eq 0 ] && log "  still waiting ... (${waited}s)"
    done
    err "vLLM did not become ready within ${VLLM_STARTUP_TIMEOUT}s"
    tail -50 "$logfile" >&2 || true; return 1
}

stop_vllm() {
    log "Stopping vLLM ..."
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID" 2>/dev/null || true; sleep 2
        kill -9 "$VLLM_PID" 2>/dev/null || true
    fi
    pkill -f "vllm serve"       2>/dev/null || true
    pkill -f "vllm.entrypoints" 2>/dev/null || true
    sleep 3; VLLM_PID=""
    if vllm_is_up; then
        warn "vLLM still up — force-killing by port"
        fuser -k -9 "${VLLM_PORT}/tcp" 2>/dev/null || true; sleep 2
    fi
    ok "vLLM stopped"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null || true
}
trap 'stop_vllm 2>/dev/null || true' EXIT INT TERM

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
build_eval_args() {
    local args=(--dataset "$DATASET"
                --out-dir "$OUT_DIR"
                --turn-counts ${TURN_COUNTS[@]}
                --test-workers "$TEST_WORKERS")
    [ -n "$NUM_SAMPLES" ]            && args+=(--num-samples "$NUM_SAMPLES")
    [ "$LLM_JUDGE" = "true" ]        && args+=(--llm-judge)
    [ "$PARALLEL_QUESTIONS" = "true" ] && args+=(--parallel-questions)
    echo "${args[@]}"
}

stage_eval() {
    log "=== Granularity ablation eval: T = ${TURN_COUNTS[*]} ==="
    # shellcheck disable=SC2046
    python eval/run_granularity_ablation.py $(build_eval_args) \
        2>&1 | tee "$LOG_DIR/eval.log"
    ok "Eval done — per-T summaries in $OUT_DIR"
}

stage_table() {
    log "=== Summary table ==="
    python eval/run_granularity_ablation.py \
        --table-only \
        --out-dir     "$OUT_DIR" \
        --turn-counts ${TURN_COUNTS[@]}
}

stage_plot() {
    log "=== Bar plot ==="
    python plot_granularity_ablation.py \
        --summary "$OUT_DIR/summary_combined.json" \
        --out     "$FIG_OUT" \
        --metric  f1 \
        --per-category
    ok "Plot saved: ${FIG_OUT}.pdf / .png (+ _by_category)"
}

# ==============================================================================
# Driver
# ==============================================================================
log "Granularity Ablation | log dir: $LOG_DIR | results: $OUT_DIR"

case "$STAGE" in
    eval)  patch_config; stage_eval; stage_table; stage_plot ;;
    table) stage_table ;;
    plot)  stage_plot ;;
    all)
        patch_config
        start_vllm "$LOG_DIR/vllm.log"
        stage_eval
        stop_vllm
        stage_table
        stage_plot
        ;;
    *)
        err "Unknown stage: $STAGE"
        echo "Usage: $0 [all|eval|table|plot]" >&2
        exit 2
        ;;
esac

ok "Done.  Results: $OUT_DIR  Figure: ${FIG_OUT}.pdf"
