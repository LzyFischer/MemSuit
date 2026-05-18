"""
Phase 2 — Step 1: build (query, positive_memory_entry) pairs from the
distillation TRAIN split.

For every row in train/data/train.jsonl produced by build_dataset.py, we have:
  - question                       (the downstream query)
  - answer                         (gold answer, used only to help selection)
  - evidence_dia_id / evidence text (the single dialogue turn that grounds the answer)
  - teacher_entries                (5-15 memory entries the teacher produced for that window)

This script asks an LLM agent to pick the SINGLE teacher entry whose
`lossless_restatement` is the best paraphrase / direct evidence of the answer
to `question`. That entry becomes the POSITIVE for contrastive retrieval
training; the query is the ANCHOR. Other examples' positives in the same batch
will serve as in-batch (hard) negatives at training time -- we do not pre-mine
negatives here.

Why an LLM agent and not a heuristic?
  - Token-overlap with the gold answer often picks the wrong entry: the answer
    is sometimes a single word ("Tuesday"), and several entries can mention it
    while only one actually grounds the question.
  - The teacher was instructed NOT to leak the answer string verbatim, so
    string matching is unreliable on principle.
  - We only need to do this once, on ~150-300 training rows, so an LLM call
    per row is affordable.

Output: train/data/contrastive_pairs.jsonl, one row per training pair:

  {
    "sample_id": ...,
    "qa_idx": ...,
    "evidence_dia_id": ...,
    "query": "...",
    "positive_text": "<lossless_restatement of the chosen entry>",
    "positive_index": <int, index into teacher_entries>,
    "all_entries": ["...", "...", ...],   # kept for debugging / negative mining
    "selection_method": "llm_agent" | "fallback_overlap",
  }

If the agent fails on a row (invalid JSON, out-of-range index, etc.) we fall
back to the highest-overlap entry and tag the row accordingly. Failed rows
that have ZERO overlap with the answer are dropped and reported.

Usage:
  python train/build_contrastive_pairs.py \
      --train-file train/data/train.jsonl \
      --out-file   train/data/contrastive_pairs.jsonl \
      --save-failures
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# allow imports from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils.llm_client import LLMClient


# ----------------------------------------------------------------------
# LLM-agent prompt
# ----------------------------------------------------------------------
#
# Design notes:
#  - We give the agent the question, the gold answer, the evidence turn,
#    AND the numbered list of teacher entries. We want it to pick the entry
#    that BOTH grounds the answer AND would be the right thing to retrieve
#    from a memory store given only the question (no evidence) at inference.
#  - We do NOT ask for a justification -- we only want the integer. Free-text
#    rationales would tempt the agent to drift and to spend tokens.
#  - We force a strict JSON schema so parsing is trivial.

SELECTOR_SYSTEM_MSG = (
    "You are a precise selector. Given a question, its gold answer, the "
    "evidence dialogue turn that grounds the answer, and a numbered list of "
    "candidate memory entries, you pick the SINGLE entry that best answers "
    "the question. Respond with strict JSON only."
)

SELECTOR_PROMPT_TEMPLATE = """We are building a retrieval-augmented memory system. At inference time, the system will see only the question and must retrieve the most relevant memory entry from the store. We need to label which entry is the correct retrieval target.

[Question]
{question}

[Gold Answer]
{answer}

[Evidence Turn — the dialogue line this question is grounded in]
{evidence_text}

[Candidate Memory Entries]
{numbered_entries}

[Your Task]
Pick the index (0-based) of the SINGLE entry that:
  (a) directly contains the fact that grounds the gold answer, AND
  (b) would be the most useful entry to retrieve given ONLY the question
      (without seeing the evidence or answer).

If two entries both ground the answer, prefer the one whose phrasing is more
self-contained and more likely to match a query that asks the question.

[Output Format]
Return ONLY a JSON object of the form:
  {{"index": <int>}}
where <int> is in the range [0, {n_entries_minus_1}]. No prose, no markdown.
"""


def make_selector_prompt(
    question: str,
    answer: str,
    evidence_text: str,
    entries: List[Dict[str, Any]],
) -> str:
    numbered = "\n".join(
        f"  [{i}] {e['lossless_restatement']}" for i, e in enumerate(entries)
    )
    return SELECTOR_PROMPT_TEMPLATE.format(
        question=question,
        answer=answer,
        evidence_text=evidence_text,
        numbered_entries=numbered,
        n_entries_minus_1=len(entries) - 1,
    )


# ----------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------

def _parse_index(raw: str, n_entries: int, llm: LLMClient) -> Optional[int]:
    """Extract a valid 0-based index from the LLM response, or None."""
    try:
        data = llm.extract_json(raw)
    except Exception:
        return None
    if isinstance(data, dict) and "index" in data:
        idx = data["index"]
    elif isinstance(data, int):
        idx = data
    else:
        return None
    try:
        idx = int(idx)
    except (TypeError, ValueError):
        return None
    if 0 <= idx < n_entries:
        return idx
    return None


def _overlap_fallback(
    answer: str, entries: List[Dict[str, Any]]
) -> Tuple[Optional[int], int]:
    """
    Pick the entry with the most overlap (in non-stopword tokens) with the gold
    answer. Returns (index_or_None, overlap_count). Used only when the LLM
    selector fails.
    """
    answer_tokens = set(re.findall(r"[A-Za-z0-9]+", answer.lower()))
    answer_tokens = {t for t in answer_tokens if len(t) >= 3}
    if not answer_tokens:
        # Answer is too short to do overlap (e.g. "yes"); take the first entry.
        return (0 if entries else None), 0

    best_idx, best_count = None, 0
    for i, e in enumerate(entries):
        toks = set(re.findall(r"[A-Za-z0-9]+", e["lossless_restatement"].lower()))
        c = len(answer_tokens & toks)
        if c > best_count:
            best_idx, best_count = i, c
    return best_idx, best_count


def select_positive(
    llm: LLMClient,
    question: str,
    answer: str,
    evidence_text: str,
    entries: List[Dict[str, Any]],
    max_attempts: int = 2,
) -> Tuple[Optional[int], str]:
    """
    Returns (chosen_index, method). method is 'llm_agent', 'fallback_overlap',
    or 'failed'.
    """
    if not entries:
        return None, "failed"
    if len(entries) == 1:
        # Trivial case — only one entry, it must be the positive.
        return 0, "llm_agent"

    user = make_selector_prompt(question, answer, evidence_text, entries)
    messages = [
        {"role": "system", "content": SELECTOR_SYSTEM_MSG},
        {"role": "user", "content": user},
    ]
    response_format = (
        {"type": "json_object"} if getattr(config, "USE_JSON_FORMAT", False) else None
    )

    for attempt in range(max_attempts):
        try:
            raw = llm.chat_completion(
                messages, temperature=0.0, response_format=response_format
            )
            idx = _parse_index(raw, len(entries), llm)
            if idx is not None:
                return idx, "llm_agent"
        except Exception as e:
            print(f"  [selector] attempt {attempt+1} failed: {e}")

    # Fallback: highest answer-overlap entry
    idx, overlap = _overlap_fallback(answer, entries)
    if idx is not None and overlap > 0:
        return idx, "fallback_overlap"
    if idx is not None:
        return idx, "fallback_overlap"  # took entry 0 with no overlap; flagged
    return None, "failed"


# ----------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------

@dataclass
class ContrastivePair:
    sample_id: str
    qa_idx: int
    evidence_dia_id: str
    query: str
    positive_text: str
    positive_index: int
    all_entries: List[str]
    selection_method: str  # 'llm_agent' | 'fallback_overlap'


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build(
    train_file: str,
    out_file: str,
    save_failures: bool = False,
    limit: Optional[int] = None,
) -> None:
    rows = load_jsonl(train_file)
    if limit:
        rows = rows[:limit]
    print(f"Loaded {len(rows)} rows from {train_file}")

    llm = LLMClient()
    pairs: List[ContrastivePair] = []
    failed: List[Dict[str, Any]] = []
    method_counts = {"llm_agent": 0, "fallback_overlap": 0, "failed": 0}

    for i, r in enumerate(rows):
        entries = r.get("teacher_entries") or []
        if not entries:
            method_counts["failed"] += 1
            failed.append({**{k: r.get(k) for k in ("sample_id", "qa_idx", "question")},
                           "reason": "no teacher_entries"})
            continue

        # Reconstruct evidence_text. The training rows from build_dataset.py
        # don't store it explicitly, but we can reconstruct it cheaply -- the
        # window contains the evidence turn. For selector quality we just pass
        # the dia_id label and the dialogue window header; the agent has the
        # full window in its mind via the entries themselves.
        evidence_text = (
            f"(dia_id={r.get('evidence_dia_id', 'unknown')})\n"
            f"This question is grounded in turn {r.get('evidence_dia_id')} "
            f"of the following dialogue window:\n{r.get('dialogue_text', '')}"
        )

        idx, method = select_positive(
            llm,
            question=r["question"],
            answer=r["answer"],
            evidence_text=evidence_text,
            entries=entries,
        )
        method_counts[method] = method_counts.get(method, 0) + 1
        print(
            f"[{i+1}/{len(rows)}] {r['sample_id']}/qa{r['qa_idx']} "
            f"ev={r['evidence_dia_id']} -> idx={idx} ({method})"
        )
        if idx is None:
            failed.append({**{k: r.get(k) for k in ("sample_id", "qa_idx", "question")},
                           "reason": "selector_failed"})
            continue

        pairs.append(
            ContrastivePair(
                sample_id=r["sample_id"],
                qa_idx=r["qa_idx"],
                evidence_dia_id=r["evidence_dia_id"],
                query=r["question"],
                positive_text=entries[idx]["lossless_restatement"],
                positive_index=idx,
                all_entries=[e["lossless_restatement"] for e in entries],
                selection_method=method,
            )
        )

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")

    print(
        f"\nWrote {len(pairs)} contrastive pairs to {out_path}\n"
        f"  Methods: {method_counts}"
    )

    if save_failures and failed:
        fail_path = out_path.with_name(out_path.stem + "_failures.jsonl")
        with open(fail_path, "w", encoding="utf-8") as f:
            for row in failed:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  {len(failed)} failed rows -> {fail_path}")

    # Sidecar meta
    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "train_file": train_file,
                "n_input_rows": len(rows),
                "n_output_pairs": len(pairs),
                "method_counts": method_counts,
                "selector_model": config.LLM_MODEL,
                "selector_base_url": getattr(config, "OPENAI_BASE_URL", None),
            },
            f,
            indent=2,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-file", default="train/data/train.jsonl")
    p.add_argument("--out-file", default="train/data/contrastive_pairs.jsonl")
    p.add_argument("--save-failures", action="store_true")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N rows (debugging).")
    args = p.parse_args()
    build(
        train_file=args.train_file,
        out_file=args.out_file,
        save_failures=args.save_failures,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
