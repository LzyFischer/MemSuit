# Phase 2 — Contrastive Retrieval Fine-Tuning

This phase complements Phase 1 (memory-builder distillation, see `README.md`).
Phase 1 makes the **memory entries** higher quality. Phase 2 makes the
**retrieval encoder** better at matching queries to those entries, by
fine-tuning the embedding model with in-batch InfoNCE on
(query, teacher-generated memory entry) pairs.

The two phases are orthogonal: Phase 1 trains the LLM that builds memory,
Phase 2 trains the embedding model that retrieves memory. You can run them
independently, but using both compounds.

## Idea

After Phase 1, every TRAIN row in `train/data/train.jsonl` carries:
- `question` — a downstream LoCoMo query
- `evidence_dia_id` — the dialogue turn that grounds its answer
- `teacher_entries` — 5–15 memory entries the teacher LLM produced for that
  window (one of them, usually, is the entry that grounds this question's
  answer)

We turn each row into one **(query, positive_entry)** pair for contrastive
retrieval training. Within a mini-batch of B such pairs, every other pair's
positive serves as a hard negative for the current query — those negatives
are real teacher-generated entries from *different* (question, evidence)
windows, so they're fluent and plausibly retrievable, just not for *this*
query. That's the signal we want the encoder to learn.

```
batch of size B:
  q_1 ─── positive ───► p_1     ◄── negative for q_2 .. q_B
  q_2 ─── positive ───► p_2     ◄── negative for q_1, q_3..q_B
  ...
  q_B ─── positive ───► p_B
```

Loss: symmetric InfoNCE over the B×B cosine-similarity matrix
(see `train_contrastive.py:info_nce_loss`).

## Files

| File | Purpose |
|---|---|
| `build_contrastive_pairs.py` | LLM-agent picks the positive entry per train row |
| `train_contrastive.py` | InfoNCE fine-tuning of the SentenceTransformer encoder |
| `_smoke_contrastive.py` | Offline tests for the sampler and InfoNCE math |

## Pipeline

```
train/data/train.jsonl                ← Phase 1 output (already exists)
        │
        │  (1) build_contrastive_pairs.py  ← LLM agent picks 1 of N entries
        ▼
train/data/contrastive_pairs.jsonl    ← (query, positive) pairs
        │
        │  (2) train_contrastive.py       ← in-batch InfoNCE on the encoder
        ▼
train/checkpoints/embed-contrastive/  ← fine-tuned SentenceTransformer
        │
        │  (3) point config.EMBEDDING_MODEL at the checkpoint dir
        ▼
        re-run eval/run_eval.py — same retrieval pipeline,
        better encoder → better recall on the test split.
```

## Step 1 — Pick the positive entry per train and val row

`build_dataset.py` now runs the teacher on **both** train and val splits (use
`--no-teacher-on-val` to restore the old TRAIN-only behavior). That means
both `train.jsonl` and `val.jsonl` carry `teacher_entries`, so we can build
contrastive pairs from each independently:

```bash
python train/build_contrastive_pairs.py \
    --train-file   train/data/train.jsonl \
    --out-file     train/data/contrastive_pairs.jsonl \
    --save-failures

python train/build_contrastive_pairs.py \
    --train-file   train/data/val.jsonl \
    --out-file     train/data/contrastive_pairs_val.jsonl \
    --save-failures
```

(The script's `--train-file` flag is named for the common case, but it
accepts any jsonl with the train-row schema, including the new val.jsonl.)

**What the agent decides.** Given the question, the gold answer, the
evidence turn, and the numbered list of teacher entries, the agent emits
`{"index": k}` — the 0-based index of the entry that best answers the
question. We do not let it write rationales (saves tokens and prevents
drift); the entry's text is used directly as the positive.

**Why an LLM and not a heuristic.** Phase 1 explicitly instructs the
teacher *not* to leak the answer string, so token-overlap matching is
unreliable on principle. Plus, multi-evidence questions often have several
entries that mention answer-tokens, but only one that actually grounds the
question. The agent is more reliable, and we only call it ~150–300 times
per split, so cost is negligible.

**Fallback.** If JSON parsing fails after 2 attempts, we pick the entry with
the highest non-stopword token overlap with the gold answer. Rows where
even the fallback fails (zero overlap, or no teacher entries) are dropped
and recorded in `*_failures.jsonl`.

Output schema (one row per pair):
```json
{
  "sample_id": "...",
  "qa_idx": 7,
  "evidence_dia_id": "D5:4",
  "query": "What activities does Melanie partake in?",
  "positive_text": "Melanie regularly hikes in the Marin Headlands ...",
  "positive_index": 3,
  "all_entries": ["...", "...", "..."],
  "selection_method": "llm_agent"
}
```

The `all_entries` field is preserved for debugging and for any later
extension that wants to mine "in-window negatives" (entries from the same
window that are *not* the positive — those are even harder negatives than
in-batch ones and could be used as explicit negatives in a triplet loss).

### A note on val coming from the same distribution

Train and val pairs are produced by the same teacher with the same prompt
on disjoint QA-level subsets of LoCoMo, so the val recall@k signal is a
fair internal validation metric for early stopping. The held-out **test**
split is intentionally label-free — it never sees the answer-aware teacher
hint, and is what `eval_on_split.py` runs on for final numbers.

## Step 2 — Contrastive training

```bash
pip install sentence-transformers torch

	
```

Hyperparameters worth knowing:

- `--batch-size`: also sets the **negative pool size** (B-1 negatives per
  query). 32 is the sweet spot on a single GPU; bigger is better up to the
  point where you start sampling false negatives. Our `NoQACollisionBatchSampler`
  already filters out within-batch QA collisions.
- `--temperature 0.05`: SimCSE default. Smaller temperatures sharpen the
  softmax and amplify the contrast — typical range is 0.02–0.10.
- `--lr 2e-5`: standard for sentence-transformer fine-tuning. `all-MiniLM`
  is small (22M params), so this trains in minutes on a single GPU.

The script reports:
- training loss + in-batch top-1 accuracy per step
- recall@{1,5,10} on the val set after every epoch (if `--val-pairs` given)

It writes a `train_log.jsonl` with all per-step metrics for plotting.

### A note on sample efficiency

LoCoMo gives us ~150–250 contrastive pairs after Phase-1 splits and per-
evidence expansion. That's small for contrastive learning, but in-batch
InfoNCE is sample-efficient because each pair contributes B-1 negatives.
At batch=32, 150 pairs × 31 negatives = ~4.6k effective contrastive
comparisons per epoch.

If you want more pairs, the cleanest extension is to **mine in-window
negatives**: for each row in `contrastive_pairs.jsonl`, the entries in
`all_entries` that are NOT the positive are very hard negatives (same
window, same conversational context, but not what the question is asking
about). Adding them as explicit negatives in a triplet loss or MultipleNeg
loss roughly doubles the contrastive signal.

## Step 3 — Use the fine-tuned encoder

The output directory is a standard sentence-transformers checkpoint, so
you can either:

1. Edit `config.py`:
   ```python
   EMBEDDING_MODEL = "train/checkpoints/embed-contrastive"
   ```
   `utils/embedding.py:_init_standard` accepts local paths transparently.

2. Or pass it explicitly to the eval harness:
   ```bash
   EMBEDDING_MODEL=train/checkpoints/embed-contrastive \
   python train/eval_on_split.py \
       --split-file train/data/test.jsonl \
       --result-file train/results/contrastive.json \
       --llm-judge
   ```

The retrieval pipeline (`HybridRetriever`) uses the encoder via
`VectorStore` — no code changes are needed downstream of the encoder swap.

## What this implementation deliberately does NOT do

- **No false-negative leakage.** The custom batch sampler guarantees no
  two pairs in the same batch share `(sample_id, qa_idx)`, so different
  evidences of the same question can never be served as negatives for each
  other.
- **No retrieval-pipeline changes.** Same `HybridRetriever`, same answerer.
  Any quality delta is attributable to the encoder.
- **No mining of explicit hard negatives** — relies entirely on in-batch
  negatives. See "sample efficiency" note above.

## Smoke test

```bash
python train/_smoke_contrastive.py
```

Verifies (without needing real data or a real encoder):
- the no-collision sampler is correct, makes progress, drops singletons,
  and terminates on pathological inputs
- InfoNCE math: perfect alignment → ~0 loss + 100% top-1 acc; random
  embeddings → ~1/B top-1 acc
- a tiny 50-step training loop on synthetic pairs reduces loss

## Combining with Phase 1

Run Phase 1 first (produces a distilled memory-builder LLM), then Phase 2
on the same `train.jsonl`. At eval time, `config.LLM_MODEL` points at the
distilled LLM and `config.EMBEDDING_MODEL` points at the contrastively
trained encoder. Both improvements stack:

- Better memory entries (Phase 1) means each entry covers more answer-
  relevant facts.
- Better encoder (Phase 2) means a query is more likely to retrieve the
  right entry from the store.
