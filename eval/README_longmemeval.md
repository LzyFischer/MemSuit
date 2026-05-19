# LongMemEval evaluation for SimpleMem

This adds LongMemEval (Wu et al., ICLR 2025) as a second evaluation
benchmark alongside LoCoMo. The training pipeline is unchanged — we still
train on LoCoMo's training split (per `train/README.md`), then evaluate
the resulting checkpoints on LongMemEval.

## What was added / changed

| File                            | Status | Role                                                                  |
| ------------------------------- | ------ | --------------------------------------------------------------------- |
| `eval/longmemeval_dataset.py`   | NEW    | Parser for LongMemEval JSON → `LongMemEvalSample` (same schema as `LoCoMoSample`, so the rest of the pipeline works untouched). |
| `eval/run_longmemeval.py`       | NEW    | Subclass of `LoCoMoEvaluator` with: user/assistant speakers, abstention branch for `*_abs` ids, per-question-type reporting, and an official-format hypothesis dump. |
| `run_longmemeval.sh`            | NEW    | End-to-end runner: launches vLLM, fetches the dataset, runs eval, prints summary. Mirrors `run_locomo.sh`. |
| `main.py`                       | EDIT   | `SimpleMemSystem.__init__` now accepts `embedding_model_name=` and threads it to `EmbeddingModel`. Two-line change. Required so a Phase-2 contrastive checkpoint can be swapped in without touching `config.py`. |

Nothing in `core/`, `database/`, `models/`, `utils/`, or `config.py`
needed to change. Memory builder, retriever, answer generator, and the
embedder wrapper itself are benchmark-agnostic — `EmbeddingModel` already
accepted `model_name`; the new arg in `SimpleMemSystem` just exposes it.

## Data layout

LongMemEval ships three variants. Pick based on compute budget:

| Variant         | Sessions / Q | Tokens / Q | Use when                       |
| --------------- | ------------ | ---------- | ------------------------------ |
| `oracle`        | ~1–6         | small      | quickest sanity check; only evidence sessions in haystack |
| `s` (cleaned)   | ~40          | ~115k      | the standard reported setting  |
| `m` (cleaned)   | ~500         | ~1.5M      | long-context stress test       |

The runner downloads from the *cleaned* HF dataset (post-Sep 2025) by
default since the original release had noisy distractor sessions that
the maintainers later removed.

## Quick start

```bash
# Baseline run on the oracle subset (fast, ~500 questions, evidence-only).
./run_longmemeval.sh --variant oracle --out-tag baseline

# Same but with the in-repo LLM-judge enabled.
./run_longmemeval.sh --variant oracle --out-tag baseline --llm-judge

# Phase-1 only: distilled memory-builder LLM, default embedder.
./run_longmemeval.sh \
    --variant s \
    --model train/checkpoints/qwen25-3b-distill-merged \
    --out-tag distill_llm \
    --llm-judge

# Full pipeline: distilled LLM + contrastive embedder.
# (--embedding is a local path to the SentenceTransformer dir saved by
#  train/train_contrastive.py, or an HF id if you pushed it.)
./run_longmemeval.sh \
    --variant s \
    --model     train/checkpoints/qwen25-3b-distill-merged \
    --embedding train/checkpoints/embed-contrastive \
    --out-tag   distill_full \
    --llm-judge

# Smoke test: 5 questions, server already running.
./run_longmemeval.sh --variant oracle --no-server --smoke
```

Outputs go to:

```
results/lme_<variant>_<tag>.json        # detailed per-question results + aggregates
results/lme_<variant>_<tag>.hyp.jsonl   # {question_id, hypothesis} for the official judge
```

Suggested tags for your run plan: `baseline`, `distill_llm`, `distill_embed`,
`distill_full`. The four together give the ablation table you'd want in
the paper (Δ from LLM-only, Δ from embedder-only, full Δ).

## Running the *official* LongMemEval LLM-judge

The in-repo LLM-judge uses whatever model `config.JUDGE_MODEL` points at
(by default, the same vLLM server hosting the answerer). For numbers
directly comparable to the paper, run the official `evaluate_qa.py` with
GPT-4o as the judge against our hypothesis dump:

```bash
git clone https://github.com/xiaowu0162/LongMemEval
cd LongMemEval && pip install -r requirements-lite.txt
export OPENAI_API_KEY=sk-...
python src/evaluation/evaluate_qa.py \
    gpt-4o \
    <abs path to>/results/lme_<variant>_<tag>.hyp.jsonl \
    <abs path to>/data/longmemeval_<variant>.json
```

## Question-type ↔ category-int mapping

`eval/metrics.py:aggregate_metrics` groups by integer `category`, so we
assign a stable mapping:

| `question_type`             | int cat |
| --------------------------- | ------- |
| single-session-user         | 1       |
| single-session-assistant    | 2       |
| single-session-preference   | 3       |
| multi-session               | 4       |
| temporal-reasoning          | 5       |
| knowledge-update            | 6       |
| *abstention* (`*_abs`)      | 7       |

The output JSON also includes `aggregated_by_question_type`, keyed by the
original strings, so you don't have to memorize the integers.

## Behavioural notes

- **Abstention questions** (`question_id` ending in `_abs`) take the same
  branch as LoCoMo cat-5: the answerer is given an A/B choice between
  "Not mentioned in the conversation" and the proposed-but-unsupported
  claim from the `answer` field. Reflection is disabled to avoid the
  retriever looping on a fact that isn't there.
- **Intra-history evaluation**: the vector store is cleared before each
  question. The official benchmark conflates "question" and "haystack"
  one-to-one, so this is the right setting; you build memory from the
  haystack once, then answer one question.
- **Wall-clock**: oracle is ~minutes per 500 Qs; *s* takes hours; *m*
  is overnight. For *s* specifically (which is what you said you'd run):
  500 questions × ~40 sessions × ~30 turns = ~600k turns total go through
  the memory builder. The memory-build phase dominates. Practical tips:
  - bump `MAX_PARALLEL_WORKERS` in `config.py` (vLLM continuous-batches well)
  - keep `ENABLE_REFLECTION = False` (default) — re-querying for missing
    facts adds another LLM round-trip per Q and rarely helps when the
    haystack is this big
  - run `--variant oracle` first as a smoke test (~10–20× faster) to
    confirm the distilled checkpoint actually loaded correctly before
    burning the full *s* budget
