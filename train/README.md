# Training pipelines for SimpleMem

Two-phase training on the LoCoMo training split:

- **Phase 1 — Self-distillation of the memory-build LLM** (sections below).
  Teaches the summarizer to produce question-relevant facts even without
  a hint at inference time.
- **Phase 2 — Contrastive fine-tuning of the embedder** (later in this
  file). Reuses the same training set; teaches the retriever to put a
  query close to the memory entry that grounds its answer.

You can run Phase 1 alone, Phase 2 alone (using the teacher entries from
a previous Phase-1 run), or both — each phase improves a different
component of the pipeline.

---

# Phase 1 — Self-distillation of the memory builder

Answer-aware self-distillation for the SimpleMem memory builder,
evaluated on LoCoMo with a Memory-R1 style split.

## Idea

The standard memory-build prompt asks the LLM to summarize a dialogue
window into structured memory entries without knowing what will be asked
later. We bias a **teacher** copy of the same model toward better summaries
by injecting the downstream question and gold answer into its prompt, then
**distill** that behavior back into the original model by SFT on
`(no-hint prompt → teacher output)` pairs.

At inference time, we use only the original prompt — the model has learned
to produce summaries that preserve question-relevant facts even without a
hint.

### One QA → multiple training examples

A LoCoMo question is paired with one or more evidence dia_ids. We expand
each `(question, [evidence_1, evidence_2, ...])` into one training example
per evidence:

```
Q: "What activities does Melanie partake in?"
evidence: ['D5:4', 'D9:1', 'D1:12', 'D1:18']

  -> 4 training examples:
     1. (Q, D5:4)  -> 20-turn window centered on D5:4
     2. (Q, D9:1)  -> 20-turn window centered on D9:1
     3. (Q, D1:12) -> 20-turn window centered on D1:12
     4. (Q, D1:18) -> 20-turn window centered on D1:18
```

This avoids the "evidence spans 60+ turns" problem (windows always equal
the inference-time `WINDOW_SIZE`) and naturally captures the paper's intent:
each evidence-bearing window should learn to preserve the facts grounding
that question.

To prevent leakage, splits are made at the **QA level** — all expanded rows
of a given question go to the same split, so the same question never appears
in both train and test (even with different evidence).

## Files

| File | Purpose |
|---|---|
| `build_dataset.py` | Generate self-distillation pairs from LoCoMo evidence |
| `sft_distill.py` | LoRA SFT on the (prompt, teacher_summary) pairs |
| `eval_on_split.py` | Run the existing intra-session eval on the held-out test split |
| `_smoke_build.py` | Offline test for the data-prep pipeline |
| `_smoke_sft.py` | Offline test for the SFT label-masking algorithm |

## Pipeline

```
data/locomo10_full.json
        │
        │  (1) build_dataset.py   ← teacher LLM in the loop
        ▼
train/data/{train,val,test}.jsonl
        │
        │  (2) sft_distill.py     ← LoRA SFT on student
        ▼
train/checkpoints/<ckpt>
        │
        │  (3) merge LoRA → serve via Ollama/vLLM
        ▼
config.LLM_MODEL points at distilled model
        │
        │  (4) eval_on_split.py   ← intra-session eval, test split only
        ▼
train/results/distilled.json
```

## Step 1 — Build the distillation dataset

```bash
# Make sure your teacher LLM is reachable per config.py
# (Ollama, vLLM, OpenAI-compatible endpoint).

python train/build_dataset.py \
    --dataset data/locomo10.json \
    --out-dir train/data \
    --window-size 20 \
    --split-counts 152 81 1307 \
    --seed 42 \
    --save-teacher-failures
```

Output:

```
train/data/
├── train.jsonl          # 152 rows
├── val.jsonl            # 81 rows
├── test.jsonl           # 1307 rows
├── teacher_failures.jsonl
└── build_meta.json
```

Each row of `train.jsonl` has:
- `student_prompt`: the no-hint prompt the fine-tuned model will see at inference
- `teacher_output`: the answer-aware summary, serialized as a `\`\`\`json … \`\`\`` block
- `teacher_entries`: same content, deserialized — each entry contains only
  `{"lossless_restatement": "..."}`

`val.jsonl` and `test.jsonl` carry the **same row schema minus the teacher
fields** (no `teacher_output`, no `teacher_entries`). The teacher LLM is
**only invoked for the train split** — generating teacher summaries for the
held-out val/test rows would be wasted compute, since those splits are only
used to (a) sanity-check training (val) and (b) define the held-out question
set for `eval_on_split.py` (test). Both of those uses care about which
questions are in the split, not about teacher labels.

### A note on the split

You said "1:1:8 train/val/test (152/81/1307)" but those numbers are not a
1:1:8 ratio (they're closer to 2:1:17, ≈10% / 5% / 85%). I prioritized the
exact counts you cited, since those match Memory-R1's reported LoCoMo split
and let you compare numbers directly. To use a literal 1:1:8 ratio instead,
pass `--split-counts 0 0 0` (which falls through to `--split-ratios 1 1 8`).

Splits are at the **QA level** — `--split-counts 152 81 1307` means 152
*questions* in train, not 152 examples. After per-evidence expansion the
example counts will be larger (≈1.4–1.7×, since many QAs have 2+ evidences).
The `build_meta.json` file records both numbers for traceability.

### A note on excluding adversarial (cat 5)

Cat-5 questions have answer "Not mentioned…" — there is no positive
evidence to anchor the teacher hint on. Following Memory-R1, we exclude
them from training. They're also implicitly excluded from `eval_on_split.py`
because they don't appear in the test split.

## Step 2 — Train

```bash
pip install transformers datasets peft accelerate trl bitsandbytes

python train/sft_distill.py \
    --train-file train/data/train.jsonl \
    --val-file   train/data/val.jsonl \
    --base-model Qwen/Qwen2.5-3B-Instruct \
    --output-dir train/checkpoints/qwen25-3b-distill \
    --epochs 3 \
    --batch-size 2 --grad-accum 8 --lr 2e-4 \
    --bf16 --gradient-checkpointing
```

Resource notes:
- 3B model + LoRA (r=16) + bf16 + grad-ckpt fits in ~16GB GPU RAM at
  `max_length=4096`.
- For 7B add `--load-in-4bit` (QLoRA) to fit in 24GB.
- Training is small (152 examples × 3 epochs ≈ 460 updates at batch 2/grad-accum 8) — 
  expect 15–30 min on a single A100/4090.

Loss is masked NLL on the assistant turn only; the prompt is correctly
masked out (`_smoke_sft.py` verifies this).

## Step 3 — Serve the distilled model

LoRA adapters need to be merged for serving via Ollama. Quick path:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
m = PeftModel.from_pretrained(base, "train/checkpoints/qwen25-3b-distill")
merged = m.merge_and_unload()
merged.save_pretrained("train/checkpoints/qwen25-3b-distill-merged")
AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct").save_pretrained(
    "train/checkpoints/qwen25-3b-distill-merged"
)
```

Then either:
1. Convert to GGUF via `llama.cpp` and `ollama create`, or
2. Serve via vLLM:
   ```bash
   vllm serve train/checkpoints/qwen25-3b-distill-merged \
       --port 11434 --served-model-name qwen25-3b-distill
   ```

## Step 4 — Evaluate

```bash
# Baseline (current LLM_MODEL in config.py)
python train/eval_on_split.py \
    --split-file train/data/test.jsonl \
    --result-file train/results/baseline.json \
    --llm-judge

# Distilled
python train/eval_on_split.py \
    --split-file train/data/test.jsonl \
    --model-override qwen25-3b-distill \
    --result-file train/results/distilled.json \
    --llm-judge
```

Both runs use the **same 1307 test questions**, the **same retrieval
pipeline**, and the **same answer generator** — only the memory-build LLM
changes. F1 / ROUGE-L / BERTScore / LLM-judge deltas isolate the impact of
self-distillation on memory quality.

## What this implementation deliberately does NOT do

- **No data leakage at eval time.** The student prompt is identical to the
  inference prompt — no question is fed to the memory builder at eval.
- **No retrieval/answering changes.** Only the memory-build step is
  fine-tuned, so any improvement is attributable to better summaries.
- **No teacher caching across windows.** Each (window, question) pair gets
  one teacher call. If you want to reuse teacher outputs when the same
  window covers multiple questions, that's a follow-up optimization.

---