# SimpleMem — Research Repo

Clean implementation of the SimpleMem memory pipeline for intra-session evaluation experiments.

**Paper:** [SimpleMem: Efficient Lifelong Memory for LLM Agents](https://arxiv.org/abs/2601.02553)

---

## Repo structure

```
simplemem_research/
├── config.py               # All hyperparameters — edit this first
├── main.py                 # SimpleMemSystem main class
│
├── models/
│   └── memory_entry.py     # MemoryEntry + Dialogue data structures
│
├── utils/
│   ├── llm_client.py       # OpenAI-compatible LLM wrapper
│   └── embedding.py        # SentenceTransformer wrapper
│
├── database/
│   └── vector_store.py     # LanceDB: semantic + lexical + symbolic index
│
├── core/
│   ├── memory_builder.py   # Stage 1+2: compression + intra-session synthesis
│   ├── hybrid_retriever.py # Stage 3: intent-aware retrieval
│   └── answer_generator.py # Final answer synthesis
│
├── eval/
│   ├── dataset.py          # LoCoMo data structures + loader
│   ├── metrics.py          # F1, ROUGE, BLEU, BERTScore, SBERT, LLM-judge
│   └── run_eval.py         # Main evaluation script
│
├── data/
│   └── locomo10.json       # LoCoMo-10 benchmark (10 samples)
│
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt

# Edit config.py with your API key and model
cp config.py config.py.bak   # keep a backup
```

---

## Run evaluation

```bash
# Quick test (3 samples)
python eval/run_eval.py --num-samples 3

# Full LoCoMo-10 benchmark
python eval/run_eval.py

# With LLM-as-judge scoring
python eval/run_eval.py --llm-judge

# Parallel question processing (faster)
python eval/run_eval.py --parallel-questions --test-workers 4

# Custom output file
python eval/run_eval.py --result-file experiments/exp1.json
```

---

## Quick API usage

```python
from main import SimpleMemSystem

system = SimpleMemSystem(clear_db=True)

# Feed dialogue turns
system.add_dialogue("Alice", "Let's meet at Starbucks tomorrow at 2pm", "2025-11-15T14:30:00")
system.add_dialogue("Bob", "Sure, I'll bring the report", "2025-11-15T14:31:00")
system.finalize()

# Query
answer = system.ask("When and where will Alice and Bob meet?")
```

---

## Ablation knobs (in `config.py`)

| Parameter | Effect |
|---|---|
| `ENABLE_PLANNING` | Turn off Stage 3 intent-aware query planning |
| `ENABLE_REFLECTION` | Turn off iterative reflection loop |
| `WINDOW_SIZE` | Memory granularity (Stage 1 compression) |
| `OVERLAP_SIZE` | Context continuity between windows |
| `SEMANTIC_TOP_K` | Retrieval depth (top-k cosine-similarity hits) |

You can also override any of these per-experiment by passing kwargs directly to `SimpleMemSystem(...)`.

> **Note on retrieval.** Retrieval is embedding-similarity only; the lexical
> (BM25) and symbolic (metadata-filter) layers were removed. Each
> `MemoryEntry` carries a single field, `lossless_restatement`, which is what
> gets embedded and what the answerer sees. All disambiguation (full names,
> absolute timestamps, locations) is inlined into that text by the memory
> builder.

---

## QA categories (LoCoMo)

| Cat | Type | Notes |
|---|---|---|
| 1 | Single-hop | Direct fact lookup |
| 2 | Multi-hop | Requires combining multiple memories |
| 3 | Temporal | Time-anchored questions |
| 4 | Commonsense | Requires inference |
| 5 | Adversarial | Answer is "Not mentioned in the conversation" |
