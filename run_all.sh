#!/usr/bin/env bash
# ==============================================================================
# SimpleMem-Distill — End-to-end pipeline (single A100)
#   Phase 1: self-distillation of memory-builder LLM (LoRA SFT on Qwen2.5-3B)
#   Phase 2: contrastive fine-tune of embedder (all-MiniLM-L6-v2)
#   Eval:    baseline vs. distilled, on the same 1307-question held-out split
#
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh [STAGE]
#
#   STAGE (optional, default = "all"):
#     all              run everything from scratch (setup + data + train + eval)
#     setup            install deps only
#     data             build train/val/test splits + contrastive pairs (needs vLLM)
#     train            SFT distill + merge LoRA + contrastive train
#     eval             run baseline + distilled evaluation (needs vLLM)
#     eval_ablation    run the two ablation evals (distill-only, contrastive-only)
#     eval_all         run baseline + distill-only + contrastive-only + full
#                      (the complete 4-cell ablation matrix; reuses vLLM
#                       instances so it's faster than running each separately)
#
#   You can also resume from any stage; later stages assume earlier outputs exist.
#
# Ablation matrix (relative to baseline):
#                   |   LLM      |  Embedder         |  result file
#   ----------------+------------+-------------------+--------------------------
#   baseline        |   base     |  base             |  results/baseline.json
#   distill_only    |   distilled|  base             |  results/distill_only.json
#   contrastive_only|   base     |  fine-tuned       |  results/contrastive_only.json
#   full (distilled)|   distilled|  fine-tuned       |  results/distilled.json
#
# Logs:
#   logs/<timestamp>/{vllm.log,build_dataset.log,sft.log,contrastive.log,
#                     eval_*.log}
# ==============================================================================
set -euo pipefail

# ---------- knobs you might want to change ----------
BASE_LLM="Qwen/Qwen2.5-3B-Instruct"
BASE_EMB="sentence-transformers/all-MiniLM-L6-v2"
DATASET="data/locomo10.json"
WINDOW_SIZE=20
SPLIT_COUNTS=(152 81 1307)

VLLM_PORT=11434
VLLM_MODEL_NAME_BASE="${BASE_LLM}"               # served-model-name for baseline
VLLM_MODEL_NAME_DISTILL="qwen25-3b-distill"      # served-model-name for distilled
# VLLM_MODEL_NAME_DISTILL="llama31-8b-distill"      # served-model-name for distilled
VLLM_GPU_UTIL=0.85
VLLM_MAX_MODEL_LEN=8192
VLLM_STARTUP_TIMEOUT=600                         # seconds to wait for /v1/models

DISTILL_CKPT_DIR="train/checkpoints/qwen25-3b-distill"
DISTILL_MERGED_DIR="train/checkpoints/qwen25-3b-distill-merged"
EMBED_CKPT_DIR="train/checkpoints/qwen25-embed-contrastive"

EVAL_PARALLEL_WORKERS=4
EVAL_LLM_JUDGE=0                                  # set to 0 to skip judge (faster)
# ----------------------------------------------------

STAGE="${1:-all}"
# STAGE="train"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO_ROOT/logs/$TS"
mkdir -p "$LOG_DIR" train/data train/checkpoints train/results

# ----- pretty print -----
RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; BLUE=$'\033[0;34m'; NC=$'\033[0m'
log()  { echo "${BLUE}[$(date +%H:%M:%S)]${NC} $*"; }
ok()   { echo "${GREEN}[OK]${NC} $*"; }
warn() { echo "${YELLOW}[WARN]${NC} $*"; }
err()  { echo "${RED}[ERR]${NC} $*" >&2; }

# ============================================================
# vLLM lifecycle helpers
# ============================================================
VLLM_PID=""

vllm_is_up() {
    local port="$1"
    curl -sf "http://localhost:${port}/v1/models" >/dev/null 2>&1
}

start_vllm() {
    local model_path="$1"
    local served_name="$2"
    local port="$3"
    local logfile="$4"

    if vllm_is_up "$port"; then
        warn "vLLM already up on :$port — killing it first to ensure clean state"
        stop_vllm
        sleep 3
    fi

    log "Starting vLLM: model=$model_path  name=$served_name  port=$port"
    log "  logs -> $logfile"
    nohup vllm serve "$model_path" \
        --port "$port" \
        --served-model-name "$served_name" \
        --gpu-memory-utilization "$VLLM_GPU_UTIL" \
        --max-model-len "$VLLM_MAX_MODEL_LEN" \
        > "$logfile" 2>&1 &
    VLLM_PID=$!
    log "vLLM PID=$VLLM_PID — waiting for /v1/models …"

    local waited=0
    while [ "$waited" -lt "$VLLM_STARTUP_TIMEOUT" ]; do
        if vllm_is_up "$port"; then
            ok "vLLM ready on :$port after ${waited}s"
            return 0
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            err "vLLM process died — see $logfile"
            tail -50 "$logfile" >&2 || true
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
        if [ $((waited % 30)) -eq 0 ]; then
            log "  still waiting … (${waited}s/${VLLM_STARTUP_TIMEOUT}s)"
        fi
    done
    err "vLLM did not become ready within ${VLLM_STARTUP_TIMEOUT}s"
    tail -50 "$logfile" >&2 || true
    return 1
}

stop_vllm() {
    log "Stopping vLLM …"
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID" 2>/dev/null || true
        sleep 2
        kill -9 "$VLLM_PID" 2>/dev/null || true
    fi
    # Catch any orphaned vllm/python workers on our port
    pkill -f "vllm serve" 2>/dev/null || true
    pkill -f "vllm.entrypoints" 2>/dev/null || true
    sleep 3
    VLLM_PID=""
    if vllm_is_up "$VLLM_PORT"; then
        warn "vLLM still responding on :$VLLM_PORT — sending SIGKILL to anyone holding it"
        fuser -k -9 "${VLLM_PORT}/tcp" 2>/dev/null || true
        sleep 2
    fi
    ok "vLLM stopped — GPU free"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null || true
}

# Always kill vLLM if the script dies/exits
trap 'stop_vllm 2>/dev/null || true' EXIT INT TERM

# ============================================================
# Sanity: make sure config.py points at the same port we use
# and judge shares the same vLLM instance.
# We patch config.py in-place once at the top so it's consistent.
# ============================================================
patch_config_for_pipeline() {
    log "Patching config.py to use single vLLM instance on :${VLLM_PORT}"
    python - <<PY
import re
from pathlib import Path
p = Path("config.py")
src = p.read_text()
def sub(pat, repl, s):
    new, n = re.subn(pat, repl, s, count=1, flags=re.MULTILINE)
    if n == 0:
        raise SystemExit(f"FAILED to patch pattern: {pat}")
    return new
src = sub(r'^OPENAI_BASE_URL\s*=.*$',
          f'OPENAI_BASE_URL = "http://localhost:${VLLM_PORT}/v1"', src)
src = sub(r'^JUDGE_BASE_URL\s*=.*$',
          f'JUDGE_BASE_URL  = "http://localhost:${VLLM_PORT}/v1"', src)
p.write_text(src)
print("config.py: OPENAI_BASE_URL and JUDGE_BASE_URL set to :${VLLM_PORT}")
PY
}

# Set EMBEDDING_MODEL in config.py to either default or fine-tuned path
set_embedding_model() {
    local emb_path="$1"
    log "Setting config.EMBEDDING_MODEL = $emb_path"
    python - <<PY
import re
from pathlib import Path
p = Path("config.py")
src = p.read_text()
new, n = re.subn(r'^EMBEDDING_MODEL\s*=.*$',
                 f'EMBEDDING_MODEL = "${emb_path}"', src,
                 count=1, flags=re.MULTILINE)
if n == 0:
    raise SystemExit("FAILED to patch EMBEDDING_MODEL")
p.write_text(new)
PY
}

# ============================================================
# Stages
# ============================================================

stage_setup() {
    log "=== STAGE: setup ==="
    pip install -q -r requirements.txt
    pip install -q transformers datasets peft accelerate trl bitsandbytes
    pip install -q vllm
    ok "deps installed"
}

stage_data() {
    log "=== STAGE: data (build train/val/test + contrastive pairs) ==="
    patch_config_for_pipeline
    set_embedding_model "$BASE_EMB"
    start_vllm "$BASE_LLM" "$BASE_LLM" "$VLLM_PORT" "$LOG_DIR/vllm_teacher.log"

    log "Step 1/3: build self-distillation dataset (teacher pass over train+val)"
    python train/build_dataset.py \
        --dataset "$DATASET" \
        --out-dir train/data \
        --window-size "$WINDOW_SIZE" \
        --split-counts "${SPLIT_COUNTS[@]}" \
        --seed 42 \
        --save-teacher-failures \
        2>&1 | tee "$LOG_DIR/build_dataset.log"
    ok "train/data/{train,val,test}.jsonl written"

    log "Step 2/3: build contrastive pairs (train split)"
    python train/build_contrastive_pairs.py \
        --train-file train/data/train.jsonl \
        --out-file   train/data/contrastive_pairs.jsonl \
        --save-failures \
        2>&1 | tee "$LOG_DIR/contrastive_pairs_train.log"

    log "Step 3/3: build contrastive pairs (val split — for recall@k)"
    python train/build_contrastive_pairs.py \
        --train-file train/data/val.jsonl \
        --out-file   train/data/contrastive_pairs_val.jsonl \
        --save-failures \
        2>&1 | tee "$LOG_DIR/contrastive_pairs_val.log"

    stop_vllm
    ok "data stage done"
}

stage_train() {
    log "=== STAGE: train (SFT distill + merge + contrastive) ==="

    # Make sure no vLLM is hogging the GPU
    if vllm_is_up "$VLLM_PORT"; then
        warn "vLLM still up — stopping before training"
        stop_vllm
    fi

    # ---- Phase 1: SFT distill ----
    log "Step 1/3: LoRA SFT on (no-hint prompt → teacher output) pairs"
    python train/sft_distill.py \
        --train-file train/data/train.jsonl \
        --val-file   train/data/val.jsonl \
        --base-model "$BASE_LLM" \
        --output-dir "$DISTILL_CKPT_DIR" \
        --epochs 3 \
        --batch-size 2 --grad-accum 8 --lr 2e-5 \
        --bf16 --gradient-checkpointing \
        2>&1 | tee "$LOG_DIR/sft.log"
    ok "LoRA adapter saved to $DISTILL_CKPT_DIR"

    # ---- Phase 1: merge LoRA into base for vLLM serving ----
    log "Step 2/3: merge LoRA → standalone HF model for vLLM"
    python - <<PY 2>&1 | tee "$LOG_DIR/merge.log"
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

print("Loading base model in bf16 …")
base = AutoModelForCausalLM.from_pretrained(
    "${BASE_LLM}", torch_dtype=torch.bfloat16
)
print("Attaching LoRA adapter from ${DISTILL_CKPT_DIR} …")
m = PeftModel.from_pretrained(base, "${DISTILL_CKPT_DIR}")
print("Merging …")
merged = m.merge_and_unload()
merged.save_pretrained("${DISTILL_MERGED_DIR}")
AutoTokenizer.from_pretrained("${BASE_LLM}").save_pretrained("${DISTILL_MERGED_DIR}")
print("Merged model saved to ${DISTILL_MERGED_DIR}")
PY
    ok "merged model at $DISTILL_MERGED_DIR"

    # ---- Phase 2: contrastive embedder ----
    log "Step 3/3: contrastive fine-tune the embedder"
    python train/train_contrastive.py \
        --pairs-file   train/data/contrastive_pairs.jsonl \
        --val-pairs    train/data/contrastive_pairs_val.jsonl \
        --base-model   "$BASE_EMB" \
        --output-dir   "$EMBED_CKPT_DIR" \
        --epochs 3 --batch-size 32 --lr 2e-5 --temperature 0.05 \
        --bf16 \
        2>&1 | tee "$LOG_DIR/contrastive.log"
    ok "fine-tuned embedder at $EMBED_CKPT_DIR"

    ok "train stage done"
}

stage_eval() {
    log "=== STAGE: eval (baseline + distilled) ==="
    patch_config_for_pipeline

    local judge_flag=""
    if [ "$EVAL_LLM_JUDGE" -eq 1 ]; then
        judge_flag="--llm-judge"
    fi

    # ---- Eval 1: baseline (base LLM + base embedder) ----
    log "Eval 1/2: BASELINE — base LLM + base embedder"
    set_embedding_model "$BASE_EMB"
    start_vllm "$BASE_LLM" "$BASE_LLM" "$VLLM_PORT" "$LOG_DIR/vllm_baseline.log"

    python train/eval_on_split.py \
        --dataset "$DATASET" \
        --split-file train/data/test.jsonl \
        --result-file train/results/baseline.json \
        $judge_flag \
        --parallel-questions --test-workers "$EVAL_PARALLEL_WORKERS" \
        2>&1 | tee "$LOG_DIR/eval_baseline.log"
    ok "baseline results -> train/results/baseline.json"

    stop_vllm

    # ---- Eval 2: distilled (distilled LLM + fine-tuned embedder) ----
    log "Eval 2/2: DISTILLED — distilled LLM + fine-tuned embedder"
    set_embedding_model "$EMBED_CKPT_DIR"
    start_vllm "$DISTILL_MERGED_DIR" "$VLLM_MODEL_NAME_DISTILL" "$VLLM_PORT" "$LOG_DIR/vllm_distilled.log"

    python train/eval_on_split.py \
        --dataset "$DATASET" \
        --split-file train/data/test.jsonl \
        --model-override "$VLLM_MODEL_NAME_DISTILL" \
        --result-file train/results/distilled.json \
        $judge_flag \
        --parallel-questions --test-workers "$EVAL_PARALLEL_WORKERS" \
        2>&1 | tee "$LOG_DIR/eval_distilled.log"
    ok "distilled results -> train/results/distilled.json"

    stop_vllm

    # Restore embedder config to base for any later manual runs
    set_embedding_model "$BASE_EMB"

    print_results_summary baseline distilled
    ok "eval stage done"
}

# ------------------------------------------------------------
# Ablation: factored helper that runs ONE eval cell.
# Prereq: the desired vLLM model is already up on $VLLM_PORT, and
# config.EMBEDDING_MODEL has already been pointed at the desired embedder.
# ------------------------------------------------------------
_run_eval_cell() {
    local result_name="$1"   # e.g. "distill_only"
    local model_arg="$2"     # value to pass to --model-override, or "" for default
    local judge_flag=""
    if [ "$EVAL_LLM_JUDGE" -eq 1 ]; then
        judge_flag="--llm-judge"
    fi

    local override=""
    if [ -n "$model_arg" ]; then
        override="--model-override $model_arg"
    fi

    log "Running eval cell: $result_name (model_override='${model_arg:-<default>}')"
    python train/eval_on_split.py \
        --dataset "$DATASET" \
        --split-file train/data/test.jsonl \
        $override \
        --result-file "train/results/${result_name}.json" \
        $judge_flag \
        --parallel-questions --test-workers "$EVAL_PARALLEL_WORKERS" \
        2>&1 | tee "$LOG_DIR/eval_${result_name}.log"
    ok "${result_name} results -> train/results/${result_name}.json"
}

# Distill-only ablation: distilled LLM + base embedder.
# Tells you how much LLM SFT alone contributes vs. baseline.
stage_eval_distill_only() {
    log "=== STAGE: eval ablation — DISTILL ONLY (base embedder + distilled LLM) ==="
    patch_config_for_pipeline

    if [ ! -d "$DISTILL_MERGED_DIR" ]; then
        err "$DISTILL_MERGED_DIR does not exist — run 'train' stage first."
        exit 1
    fi

    set_embedding_model "$BASE_EMB"
    start_vllm "$DISTILL_MERGED_DIR" "$VLLM_MODEL_NAME_DISTILL" "$VLLM_PORT" \
               "$LOG_DIR/vllm_distill_only.log"
    _run_eval_cell "distill_only" "$VLLM_MODEL_NAME_DISTILL"
    stop_vllm

    print_results_summary baseline distill_only
    ok "distill-only ablation done"
}

# Contrastive-only ablation: base LLM + fine-tuned embedder.
# Tells you how much embedder fine-tuning alone contributes vs. baseline.
stage_eval_contrastive_only() {
    log "=== STAGE: eval ablation — CONTRASTIVE ONLY (fine-tuned embedder + base LLM) ==="
    patch_config_for_pipeline

    if [ ! -d "$EMBED_CKPT_DIR" ]; then
        err "$EMBED_CKPT_DIR does not exist — run 'train' stage first."
        exit 1
    fi

    set_embedding_model "$EMBED_CKPT_DIR"
    start_vllm "$BASE_LLM" "$BASE_LLM" "$VLLM_PORT" \
               "$LOG_DIR/vllm_contrastive_only.log"
    _run_eval_cell "contrastive_only" ""
    stop_vllm

    # Restore embedder config to base
    set_embedding_model "$BASE_EMB"

    print_results_summary baseline contrastive_only
    ok "contrastive-only ablation done"
}

# Run the two ablation cells back-to-back. Reuses vLLM where possible:
# we group cells by which LLM they need so we only start vLLM twice
# (once for base LLM, once for distilled LLM) instead of four times.
stage_eval_ablation() {
    log "=== STAGE: eval ablation (distill_only + contrastive_only) ==="
    patch_config_for_pipeline

    if [ ! -d "$DISTILL_MERGED_DIR" ]; then
        err "$DISTILL_MERGED_DIR does not exist — run 'train' stage first."
        exit 1
    fi
    if [ ! -d "$EMBED_CKPT_DIR" ]; then
        err "$EMBED_CKPT_DIR does not exist — run 'train' stage first."
        exit 1
    fi

    # ---- Group A: base LLM is up → run contrastive_only ----
    set_embedding_model "$EMBED_CKPT_DIR"
    start_vllm "$BASE_LLM" "$BASE_LLM" "$VLLM_PORT" \
               "$LOG_DIR/vllm_ablation_base_llm.log"
    _run_eval_cell "contrastive_only" ""
    stop_vllm

    # ---- Group B: distilled LLM is up → run distill_only ----
    set_embedding_model "$BASE_EMB"
    start_vllm "$DISTILL_MERGED_DIR" "$VLLM_MODEL_NAME_DISTILL" "$VLLM_PORT" \
               "$LOG_DIR/vllm_ablation_distill_llm.log"
    _run_eval_cell "distill_only" "$VLLM_MODEL_NAME_DISTILL"
    stop_vllm

    set_embedding_model "$BASE_EMB"
    print_results_summary baseline distill_only contrastive_only
    ok "ablation evals done"
}

# Run the FULL 4-cell ablation matrix in one shot.
# Order chosen to launch vLLM exactly twice:
#   vLLM=base_LLM      → baseline (base emb)        + contrastive_only (FT emb)
#   vLLM=distilled_LLM → distill_only (base emb)    + distilled (FT emb)
stage_eval_all() {
    log "=== STAGE: eval_all (full 4-cell ablation matrix) ==="
    patch_config_for_pipeline

    if [ ! -d "$DISTILL_MERGED_DIR" ]; then
        err "$DISTILL_MERGED_DIR does not exist — run 'train' stage first."
        exit 1
    fi
    if [ ! -d "$EMBED_CKPT_DIR" ]; then
        err "$EMBED_CKPT_DIR does not exist — run 'train' stage first."
        exit 1
    fi

    # ---- Group A: vLLM serves the BASE LLM ----
    log "Group A/2: base LLM up → baseline + contrastive_only"
    start_vllm "$BASE_LLM" "$BASE_LLM" "$VLLM_PORT" \
               "$LOG_DIR/vllm_eval_all_base_llm.log"

    set_embedding_model "$BASE_EMB"
    _run_eval_cell "baseline" ""

    set_embedding_model "$EMBED_CKPT_DIR"
    _run_eval_cell "contrastive_only" ""

    stop_vllm

    # ---- Group B: vLLM serves the DISTILLED LLM ----
    log "Group B/2: distilled LLM up → distill_only + distilled (full)"
    start_vllm "$DISTILL_MERGED_DIR" "$VLLM_MODEL_NAME_DISTILL" "$VLLM_PORT" \
               "$LOG_DIR/vllm_eval_all_distill_llm.log"

    set_embedding_model "$BASE_EMB"
    _run_eval_cell "distill_only" "$VLLM_MODEL_NAME_DISTILL"

    set_embedding_model "$EMBED_CKPT_DIR"
    _run_eval_cell "distilled" "$VLLM_MODEL_NAME_DISTILL"

    stop_vllm

    # Restore embedder config
    set_embedding_model "$BASE_EMB"

    print_results_summary baseline distill_only contrastive_only distilled
    ok "eval_all done"
}

# ------------------------------------------------------------
# Pretty-print summary across an arbitrary list of result names.
# Usage: print_results_summary baseline distilled distill_only contrastive_only
# ------------------------------------------------------------
print_results_summary() {
    log "=== Summary ==="
    NAMES="$*" python - <<'PY'
import json
import os
from pathlib import Path
names = os.environ.get("NAMES", "").split()
for name in names:
    p = Path(f"train/results/{name}.json")
    if not p.exists():
        print(f"  {name}: missing")
        continue
    d = json.loads(p.read_text())
    s = d.get("summary", {})
    agg = d.get("aggregated_metrics", {})
    print(f"\n  === {name.upper()} ===")
    print(f"    n_samples   : {s.get('num_samples')}")
    print(f"    n_questions : {s.get('num_questions')}")
    print(f"    model       : {s.get('model')}")
    overall = agg.get("overall", agg)
    for k in ("f1", "rouge_l", "bleu", "bertscore_f1", "sbert", "llm_judge"):
        v = overall.get(k)
        if v is not None:
            if isinstance(v, (int, float)):
                print(f"    {k:14s}: {v:.4f}")
            else:
                print(f"    {k:14s}: {v}")
PY
}

# ============================================================
# Driver
# ============================================================
log "Pipeline log dir: $LOG_DIR"
case "$STAGE" in
    setup)
        stage_setup
        ;;
    data)
        stage_data
        ;;
    train)
        stage_train
        ;;
    eval)
        stage_eval
        ;;
    eval_distill_only)
        stage_eval_distill_only
        ;;
    eval_contrastive_only)
        stage_eval_contrastive_only
        ;;
    eval_ablation)
        stage_eval_ablation
        ;;
    eval_all)
        stage_eval_all
        ;;
    all)
        stage_setup
        stage_data
        stage_train
        stage_eval_all
        ;;
    *)
        err "Unknown stage: $STAGE"
        echo "Usage: $0 [all|setup|data|train|eval|eval_distill_only|eval_contrastive_only|eval_ablation|eval_all]" >&2
        exit 2
        ;;
esac

ok "ALL DONE."
log "Logs: $LOG_DIR"
log "Results dir: train/results/  (baseline.json, distilled.json, distill_only.json, contrastive_only.json — whichever stages you ran)"