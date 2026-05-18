"""
Build the self-distillation dataset for memory summarization.

Pipeline (per LoCoMo QA pair, excluding category-5 adversarial):
  1. Locate the evidence dialogues (e.g. "D2:8" -> session 2, dia_id 'D2:8').
  2. Slice a window of WINDOW_SIZE turns around the evidence (centered),
     mirroring the inference-time sliding window.
  3. Expand each (question, evidence_i) into one "candidate" example.
  4. Split candidates at the SAMPLE (conversation) level into train/val/test:
       - 1st & 2nd samples (in dataset order) -> train
       - 3rd sample                           -> val
       - all remaining samples                -> test
  5. Call the teacher LLM with the (question, gold answer) hint on the TRAIN
     and VAL splits to produce teacher_output. The test split is left label-
     free — it's the held-out evaluation set for `eval_on_split.py`, where
     running the teacher would leak the answer-aware hint.

     Val gets teacher labels so that the Phase-2 contrastive training script
     (`train_contrastive.py`, see README_contrastive.md) can extract real
     (query, positive_entry) pairs from val rows and report recall@k as an
     early-stopping signal. Use `--no-teacher-on-val` to restore the old
     TRAIN-only behavior if you don't intend to run Phase 2.
  6. Save jsonl files per split. Additionally, the test split is also
     dumped as `test.json` in the ORIGINAL LoCoMo format (list of full
     sample dicts) so it can be fed directly to any locomo-aware tool.

Memory entries produced by the teacher contain a single field:
    {"lossless_restatement": "..."}
matching the simplified MemoryEntry schema.

Usage:
  python train/build_dataset.py \
      --dataset data/locomo10_full.json \
      --out-dir train/data \
      --window-size 20 \
      --seed 42
"""
import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# allow imports from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from eval.dataset import LoCoMoSample, Turn, load_locomo
from utils.llm_client import LLMClient


# ----------------------------------------------------------------------
# Window slicing
# ----------------------------------------------------------------------

def _flatten_turns(sample: LoCoMoSample) -> List[Tuple[Turn, str, str]]:
    """
    Return a flat list of (turn, session_date_time, dia_id) over all sessions
    in chronological order.
    """
    flat: List[Tuple[Turn, str, str]] = []
    for sid in sorted(sample.conversation.sessions):
        sess = sample.conversation.sessions[sid]
        for turn in sess.turns:
            flat.append((turn, sess.date_time, turn.dia_id))
    return flat


def _window_around(
    flat: List[Tuple[Turn, str, str]],
    evidence_idx: int,
    window_size: int,
) -> Tuple[List[Tuple[Turn, str, str]], int, int]:
    """
    Build a window of EXACTLY `window_size` turns centered on a single
    evidence index. Returns (window, start, end_exclusive).
    """
    n = len(flat)
    if window_size >= n:
        return flat[:n], 0, n

    half = window_size // 2
    start = max(0, evidence_idx - half)
    end = min(n, start + window_size)
    # If we hit the right edge, slide left to keep the window full size
    if end - start < window_size:
        start = max(0, end - window_size)
    return flat[start:end], start, end


def _format_window_text(window: List[Tuple[Turn, str, str]]) -> str:
    """Render the window in the same '[ts] speaker: text' format the builder uses."""
    lines = []
    for turn, ts, _dia_id in window:
        prefix = f"[{ts}] " if ts else ""
        lines.append(f"{prefix}{turn.speaker}: {turn.text}")
    return "\n".join(lines)


def _format_evidence_text(
    flat: List[Tuple[Turn, str, str]], evidence_idx: int
) -> str:
    """Just the single evidence turn, used to anchor the teacher prompt."""
    turn, ts, dia_id = flat[evidence_idx]
    prefix = f"[{ts}] " if ts else ""
    return f"({dia_id}) {prefix}{turn.speaker}: {turn.text}"


# ----------------------------------------------------------------------
# Prompt templates
# ----------------------------------------------------------------------

# This is a verbatim copy of MemoryBuilder._extraction_prompt's body, with
# {context} stripped (we don't pass cross-window context for training pairs --
# each training example is self-contained). Keeping it identical to inference
# is critical: the student we fine-tune will be queried with this exact prompt
# at memory-build time.

STUDENT_PROMPT_TEMPLATE = """Your task is to extract all valuable information from the following dialogues and convert them into structured memory entries.

[Current Window Dialogues]
{dialogue_text}

[Requirements]
1. **Complete Coverage**: Generate enough memory entries to ensure ALL information in the dialogues is captured
2. **Force Disambiguation**: Absolutely PROHIBIT using pronouns (he, she, it, they, this, that) and relative time (yesterday, today, last week, tomorrow). Use full names and absolute ISO 8601 timestamps inline.
3. **Lossless Information**: Each entry's lossless_restatement must be a complete, independent, understandable sentence that includes all relevant subjects, objects, time, and location inline.

[Output Format]
Return a JSON array. Each element is a memory entry with a single field:

```json
[
  {{
    "lossless_restatement": "Complete unambiguous restatement (must include all subjects, objects, time, location, etc.)"
  }},
  ...
]
```

Now process the above dialogues. Return ONLY the JSON array, no other explanations.
"""

# Teacher prompt = student prompt + an "answer hint" block. The hint tells the
# teacher LLM that one specific question must be answerable from the resulting
# memory entries. This biases the teacher toward producing summaries that
# preserve the relevant facts -- without leaking the answer string itself into
# the entries (we explicitly instruct the teacher not to mention the question).

TEACHER_HINT_BLOCK = """
[Internal Coverage Check — do not mention in output, do not let it shape your focus]
A downstream system will later be asked the following question against your memory entries. This block exists ONLY to verify coverage after you have already drafted a comprehensive set of entries. It is NOT the topic of this extraction and must NOT cause you to drop, shorten, merge, reorder, or de-prioritize any other entry.

  Question: {question}
  Gold answer: {answer}
  Evidence utterance (must be preserved verbatim in at least one entry):
    {evidence_text}

Procedure:
1. First, draft your memory entries normally, covering ALL information in the dialogue at the level of detail required by the main Requirements above (≈ one entry per dialogue turn). Do this as if this block did not exist.
2. Then, verify that the evidence utterance above appears, with its original wording preserved (key nouns, verbs, and named entities unchanged), inside the lossless_restatement of at least one entry — and that the gold answer is recoverable from that entry using only the original words from the dialogue.
3. If the check already passes, change nothing. If it does not, add or minimally revise exactly ONE entry to satisfy it, and leave every other entry untouched.

Hard constraints:
- Do NOT mention the question, the gold answer, the evidence, or this check in your output.
- Do NOT phrase any entry as an answer to the question; entries are statements about the dialogue, not responses.
- Do NOT use the gold answer's wording if it does not appear in the dialogue — use the original word(s) from the raw text.
- Coverage of unrelated topics (small talk, emotions, plans, image descriptions, etc.) must remain identical to what you would produce without this block. Fewer entries than the dialogue's richness warrants is a failure of this task.
"""


def make_student_prompt(dialogue_text: str) -> str:
    return STUDENT_PROMPT_TEMPLATE.format(dialogue_text=dialogue_text)


def make_teacher_prompt(
    dialogue_text: str, question: str, answer: str, evidence_text: str
) -> str:
    student = STUDENT_PROMPT_TEMPLATE.format(dialogue_text=dialogue_text)
    hint = TEACHER_HINT_BLOCK.format(
        question=question, answer=answer, evidence_text=evidence_text
    )
    # Insert the hint block right after the dialogues, before the requirements.
    # Easiest: append to the prompt -- the LLM treats the whole user message as
    # the task and our explicit instructions still apply.
    return hint + "\n" + student


SYSTEM_MSG = (
    "You are a professional information extraction assistant. "
    "Extract structured, unambiguous facts from conversations. "
    "Output valid JSON only."
)


# ----------------------------------------------------------------------
# Teacher generation
# ----------------------------------------------------------------------

def _validate_entries(data: Any) -> Optional[List[Dict[str, Any]]]:
    """Light schema check. Returns None if invalid."""
    if not isinstance(data, list) or not data:
        return None
    cleaned = []
    for item in data:
        # Accept either {"lossless_restatement": "..."} or a bare string.
        if isinstance(item, dict):
            text = item.get("lossless_restatement")
        elif isinstance(item, str):
            text = item
        else:
            return None
        if not isinstance(text, str) or not text.strip():
            return None
        cleaned.append({"lossless_restatement": text.strip()})
    return cleaned


def _answer_covered(entries: List[Dict[str, Any]], answer: str) -> bool:
    """
    Heuristic: does at least one entry's lossless_restatement contain at least
    one non-stopword token from the gold answer? Used to filter teacher outputs.
    """
    if not answer:
        return True
    answer_tokens = re.findall(r"[A-Za-z0-9]+", answer.lower())
    answer_tokens = [t for t in answer_tokens if len(t) >= 3]
    if not answer_tokens:
        return True
    blob = " ".join(e["lossless_restatement"].lower() for e in entries)
    return any(tok in blob for tok in answer_tokens)


def run_teacher(
    llm: LLMClient,
    dialogue_text: str,
    question: str,
    answer: str,
    evidence_text: str,
    max_attempts: int = 3,
) -> Optional[List[Dict[str, Any]]]:
    """Call the teacher LLM until we get valid JSON entries that mention the answer."""
    user = make_teacher_prompt(dialogue_text, question, answer, evidence_text)
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": user},
    ]
    response_format = (
        {"type": "json_object"} if getattr(config, "USE_JSON_FORMAT", False) else None
    )

    for attempt in range(max_attempts):
        try:
            raw = llm.chat_completion(
                messages, temperature=0.1, response_format=response_format
            )
            data = llm.extract_json(raw)
            entries = _validate_entries(data)
            if entries and _answer_covered(entries, answer):
                return entries
            if entries:
                # Valid JSON but doesn't cover the answer -- try once more
                # with a stronger nudge, then fall back to None.
                if attempt == max_attempts - 1:
                    return entries  # accept best-effort on final attempt
        except Exception as e:
            print(f"  [teacher] attempt {attempt+1} failed: {e}")
    return None


# ----------------------------------------------------------------------
# Splits
# ----------------------------------------------------------------------

@dataclass
class Candidate:
    """
    One (question, single-evidence) pair before any teacher call.

    We materialize all candidates first, then split them at the QA level,
    then only run the teacher on the candidates that landed in the train split.
    """
    sample_id: str
    qa_idx: int
    category: int
    question: str
    answer: str
    evidence_dia_id: str
    evidence_idx_in_qa: int
    qa_all_evidence: List[str]
    window_start: int
    window_end: int
    dialogue_text: str
    evidence_text: str
    student_prompt: str


@dataclass
class TrainExample:
    """One self-distillation example: (query, single evidence) -> teacher summary."""
    sample_id: str
    qa_idx: int
    category: int
    question: str
    answer: str
    evidence_dia_id: str         # the single evidence dia_id this example pairs with
    evidence_idx_in_qa: int      # which evidence-of-this-QA we picked (0..len(qa.evidence)-1)
    qa_all_evidence: List[str]   # the full evidence list of the source QA (for traceability)
    window_start: int
    window_end: int
    dialogue_text: str
    student_prompt: str          # input to fine-tune on
    teacher_entries: List[Dict[str, Any]]  # target (deserialized JSON)
    teacher_output: str          # target as a string (what the model emits)


@dataclass
class EvalExample:
    """
    One held-out (val/test) row. Carries everything needed to identify the
    question and its window slice, but NO teacher labels — we don't fine-tune
    on these.
    """
    sample_id: str
    qa_idx: int
    category: int
    question: str
    answer: str
    evidence_dia_id: str
    evidence_idx_in_qa: int
    qa_all_evidence: List[str]
    window_start: int
    window_end: int
    dialogue_text: str
    student_prompt: str


def split_candidates(
    candidates: List[Candidate],
    counts: Optional[Tuple[int, int, int]] = None,
    ratios: Tuple[int, int, int] = (1, 1, 8),
    seed: int = 42,
) -> Dict[str, List[Candidate]]:
    """
    Split candidates into train/val/test at the SAMPLE LEVEL (i.e., by conversation).

    Concretely:
      - The first sample (in dataset order) -> train
      - The second sample                   -> train
      - The third sample                    -> val
      - All remaining samples               -> test

    All candidates from a given conversation go to the same split, so QA pairs
    and evidence windows from the same conversation never leak across splits.

    The `counts`, `ratios`, and `seed` arguments are kept for API compatibility
    but are ignored under this fixed sample-level scheme.
    """
    # Preserve first-seen sample order from the candidate stream (which itself
    # follows dataset order in build()), so the split is deterministic and
    # reflects the conversation ordering in the source JSON.
    seen: List[str] = []
    seen_set: set = set()
    for c in candidates:
        if c.sample_id not in seen_set:
            seen.append(c.sample_id)
            seen_set.add(c.sample_id)

    n_samples = len(seen)
    if n_samples < 4:
        print(
            f"  [warn] sample-level split expects >=4 samples (2 train + 1 val + >=1 test); "
            f"got {n_samples}. Splitting as best as possible."
        )

    train_samples = set(seen[:2])
    val_samples = set(seen[2:3])
    test_samples = set(seen[3:])

    splits: Dict[str, List[Candidate]] = {"train": [], "val": [], "test": []}
    for c in candidates:
        if c.sample_id in train_samples:
            splits["train"].append(c)
        elif c.sample_id in val_samples:
            splits["val"].append(c)
        elif c.sample_id in test_samples:
            splits["test"].append(c)

    print(
        f"  Sample-level split: train={sorted(train_samples)}, "
        f"val={sorted(val_samples)}, test={sorted(test_samples)}"
    )
    print(
        f"  Candidate-level: train={len(splits['train'])}, "
        f"val={len(splits['val'])}, test={len(splits['test'])}"
    )
    return splits


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------

def serialize_teacher_output(entries: List[Dict[str, Any]]) -> str:
    """The exact string the student should learn to emit."""
    # Wrap in ```json ... ``` to match the inference-time format the parser
    # already handles, and give the model clean tokens to imitate.
    body = json.dumps(entries, ensure_ascii=False, indent=2)
    return f"```json\n{body}\n```"


def candidate_to_eval(c: Candidate) -> EvalExample:
    return EvalExample(
        sample_id=c.sample_id,
        qa_idx=c.qa_idx,
        category=c.category,
        question=c.question,
        answer=c.answer,
        evidence_dia_id=c.evidence_dia_id,
        evidence_idx_in_qa=c.evidence_idx_in_qa,
        qa_all_evidence=list(c.qa_all_evidence),
        window_start=c.window_start,
        window_end=c.window_end,
        dialogue_text=c.dialogue_text,
        student_prompt=c.student_prompt,
    )


def build(
    dataset_path: str,
    out_dir: str,
    window_size: int,
    seed: int,
    limit_qa: Optional[int],
    save_teacher_failures: bool,
    split_counts: Optional[Tuple[int, int, int]],
    split_ratios: Tuple[int, int, int],
    teacher_on_val: bool = True,
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    samples = load_locomo(dataset_path)

    # Also load the RAW JSON. `load_locomo` overrides `sample_id` with the
    # integer index of the sample in the raw list (str(idx)), so raw_samples[i]
    # corresponds to the parsed sample whose sample_id == str(i). We use this
    # to emit the test split in the original LoCoMo format (see Phase 4).
    with open(dataset_path, encoding="utf-8") as _f:
        raw_samples = json.load(_f)

    # ---------- Phase 1: enumerate ALL candidates (no teacher calls) ----------
    candidates: List[Candidate] = []
    n_qa_seen = 0

    for sample in samples:
        flat = _flatten_turns(sample)
        for qa_idx, qa in enumerate(sample.qa):
            if qa.category == 5:
                continue  # exclude adversarial
            if not qa.final_answer:
                continue
            n_qa_seen += 1
            if limit_qa is not None and n_qa_seen > limit_qa:
                break

            answer = str(qa.final_answer)

            # Map evidence dia_ids to flat indices.
            id_to_idx = {dia_id: i for i, (_, _, dia_id) in enumerate(flat)}
            located: List[Tuple[int, str, int]] = []  # (in_qa_pos, dia_id, flat_idx)
            for j, ev in enumerate(qa.evidence):
                if isinstance(ev, str) and ev in id_to_idx:
                    located.append((j, ev, id_to_idx[ev]))

            if not located:
                print(
                    f"[{sample.sample_id}/{qa_idx}] cat={qa.category} "
                    f"NO LOCATABLE EVIDENCE in {qa.evidence}, skipping"
                )
                continue

            # Expand: one candidate per evidence
            for in_qa_pos, dia_id, ev_flat_idx in located:
                window, start, end = _window_around(flat, ev_flat_idx, window_size)
                dialogue_text = _format_window_text(window)
                evidence_text = _format_evidence_text(flat, ev_flat_idx)

                candidates.append(
                    Candidate(
                        sample_id=sample.sample_id,
                        qa_idx=qa_idx,
                        category=qa.category or 0,
                        question=qa.question,
                        answer=answer,
                        evidence_dia_id=dia_id,
                        evidence_idx_in_qa=in_qa_pos,
                        qa_all_evidence=list(qa.evidence),
                        window_start=start,
                        window_end=end,
                        dialogue_text=dialogue_text,
                        evidence_text=evidence_text,
                        student_prompt=make_student_prompt(dialogue_text),
                    )
                )
        if limit_qa is not None and n_qa_seen >= limit_qa:
            break

    n_unique_qas = len({(c.sample_id, c.qa_idx) for c in candidates})
    print(
        f"\nEnumerated {len(candidates)} candidates from {n_unique_qas} unique QAs"
    )

    # ---------- Phase 2: split candidates at QA level ----------
    splits = split_candidates(
        candidates,
        counts=split_counts,
        ratios=split_ratios,
        seed=seed,
    )

    # ---------- Phase 3: teacher generation — TRAIN and (by default) VAL ----------
    #
    # We run the teacher on both train and val so that downstream contrastive
    # training (Phase 2 of the pipeline, see README_contrastive.md) can build
    # (query, positive_entry) pairs from val rows for an honest recall@k early-
    # stopping signal. Test stays label-free — it's the held-out evaluation set
    # for the final intra-session eval, where running the teacher would leak
    # the answer-aware hint into the held-out set.
    splits_for_teacher = ["train"]
    if teacher_on_val:
        splits_for_teacher.append("val")

    n_teacher_total = sum(len(splits[s]) for s in splits_for_teacher)
    print(
        f"\nRunning teacher on {n_teacher_total} candidates "
        f"({', '.join(f'{s}={len(splits[s])}' for s in splits_for_teacher)}). "
        "Test split is left label-free for evaluation."
    )
    llm = LLMClient()  # uses config.LLM_MODEL as the teacher

    examples_by_split: Dict[str, List[TrainExample]] = {}
    failed_by_split: Dict[str, List[Dict[str, Any]]] = {}

    def _run_teacher_on_split(name: str, cands: List[Candidate]) -> None:
        out: List[TrainExample] = []
        fails: List[Dict[str, Any]] = []
        for c in cands:
            print(
                f"[{name}] [{c.sample_id}/{c.qa_idx}] cat={c.category} "
                f"ev={c.evidence_dia_id} -> window [{c.window_start},{c.window_end}) "
                f"Q: {c.question[:60]}"
            )
            entries = run_teacher(
                llm, c.dialogue_text, c.question, c.answer, c.evidence_text
            )
            if entries is None:
                print(f"  -> teacher failed, skipping")
                if save_teacher_failures:
                    fails.append(
                        {
                            "split": name,
                            "sample_id": c.sample_id,
                            "qa_idx": c.qa_idx,
                            "evidence_dia_id": c.evidence_dia_id,
                            "question": c.question,
                            "answer": c.answer,
                        }
                    )
                continue
            teacher_str = serialize_teacher_output(entries)
            out.append(
                TrainExample(
                    sample_id=c.sample_id,
                    qa_idx=c.qa_idx,
                    category=c.category,
                    question=c.question,
                    answer=c.answer,
                    evidence_dia_id=c.evidence_dia_id,
                    evidence_idx_in_qa=c.evidence_idx_in_qa,
                    qa_all_evidence=list(c.qa_all_evidence),
                    window_start=c.window_start,
                    window_end=c.window_end,
                    dialogue_text=c.dialogue_text,
                    student_prompt=c.student_prompt,
                    teacher_entries=entries,
                    teacher_output=teacher_str,
                )
            )
        examples_by_split[name] = out
        failed_by_split[name] = fails

    for s in splits_for_teacher:
        _run_teacher_on_split(s, splits[s])

    train_examples = examples_by_split["train"]
    if teacher_on_val:
        val_examples: List[Any] = examples_by_split["val"]
    else:
        val_examples = [candidate_to_eval(c) for c in splits["val"]]
    test_examples = [candidate_to_eval(c) for c in splits["test"]]

    n_train_failed = len(failed_by_split.get("train", []))
    n_val_failed = len(failed_by_split.get("val", []))
    failed_all = failed_by_split.get("train", []) + failed_by_split.get("val", [])

    val_label_status = "with teacher labels" if teacher_on_val else "no teacher labels"
    print(
        f"\nFinal: train={len(train_examples)} (with teacher labels) "
        f"val={len(val_examples)} ({val_label_status}) "
        f"test={len(test_examples)} (no teacher labels) "
        f"| teacher failures: train={n_train_failed} val={n_val_failed}"
    )

    # ---------- Phase 4: save ----------
    def _dump(rows: List[Any], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
        print(f"  wrote {path} ({len(rows)} rows)")

    _dump(train_examples, out_path / "train.jsonl")
    _dump(val_examples,   out_path / "val.jsonl")
    _dump(test_examples,  out_path / "test.jsonl")

    # Additionally dump the test split in the ORIGINAL LoCoMo JSON format
    # (a list of full sample dicts with `qa`, `conversation`, etc.), so that
    # tools that consume locomo10.json directly can use this file as a
    # drop-in held-out set.
    test_sample_ids = sorted(
        {ex.sample_id for ex in test_examples}, key=lambda s: int(s) if s.isdigit() else s
    )
    locomo_test: List[Dict[str, Any]] = []
    for sid in test_sample_ids:
        try:
            idx = int(sid)
        except ValueError:
            # Fallback: locate by matching the raw sample_id field if present
            idx = next(
                (i for i, rs in enumerate(raw_samples) if str(rs.get("sample_id")) == sid),
                None,
            )
        if idx is None or idx >= len(raw_samples):
            print(f"  [warn] could not locate raw sample for sample_id={sid}, skipping in test.json")
            continue
        # Deep-copy via json round-trip so we don't mutate the loaded raw data
        locomo_test.append(json.loads(json.dumps(raw_samples[idx])))

    locomo_test_path = out_path / "test.json"
    with open(locomo_test_path, "w", encoding="utf-8") as f:
        json.dump(locomo_test, f, ensure_ascii=False, indent=2)
    print(f"  wrote {locomo_test_path} ({len(locomo_test)} samples in original LoCoMo format)")

    if save_teacher_failures and failed_all:
        with open(out_path / "teacher_failures.jsonl", "w") as f:
            for row in failed_all:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Save a small "meta" file capturing the build config for reproducibility
    with open(out_path / "build_meta.json", "w") as f:
        json.dump(
            {
                "dataset_path": dataset_path,
                "window_size": window_size,
                "seed": seed,
                "split_counts": list(split_counts) if split_counts else None,
                "split_ratios": list(split_ratios),
                "n_candidates_total": len(candidates),
                "n_train_examples": len(train_examples),
                "n_val_examples": len(val_examples),
                "n_test_examples": len(test_examples),
                "n_train_teacher_failed": n_train_failed,
                "n_val_teacher_failed": n_val_failed,
                "teacher_on_val": teacher_on_val,
                "teacher_model": config.LLM_MODEL,
                "teacher_base_url": getattr(config, "OPENAI_BASE_URL", None),
                "note": (
                    "train and val both have teacher labels; test is "
                    "label-free for evaluation"
                    if teacher_on_val
                    else "only train has teacher labels (val/test label-free)"
                ),
            },
            f,
            indent=2,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/locomo10_full.json")
    p.add_argument("--out-dir", default="train/data")
    p.add_argument("--window-size", type=int, default=config.WINDOW_SIZE)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-qa", type=int, default=None,
                   help="Stop after this many QA pairs (debugging).")
    p.add_argument("--save-teacher-failures", action="store_true")
    p.add_argument(
        "--no-teacher-on-val",
        action="store_true",
        help="Skip teacher generation on the val split (restores the old "
             "TRAIN-only behavior). By default the teacher runs on val too "
             "so that Phase 2 contrastive training can build a real recall@k "
             "validation signal.",
    )
    # Split mode
    p.add_argument("--split-counts", nargs=3, type=int, metavar=("TRAIN", "VAL", "TEST"),
                   default=[152, 81, 1307],
                   help="Exact train/val/test counts (default: 152 81 1307, "
                        "matching Memory-R1's LoCoMo split).")
    p.add_argument("--split-ratios", nargs=3, type=int, metavar=("A", "B", "C"),
                   default=[1, 1, 8],
                   help="Used only when --split-counts is set to '0 0 0'.")
    args = p.parse_args()

    counts = tuple(args.split_counts) if any(args.split_counts) else None
    build(
        dataset_path=args.dataset,
        out_dir=args.out_dir,
        window_size=args.window_size,
        seed=args.seed,
        limit_qa=args.limit_qa,
        save_teacher_failures=args.save_teacher_failures,
        split_counts=counts,
        split_ratios=tuple(args.split_ratios),
        teacher_on_val=not args.no_teacher_on_val,
    )


if __name__ == "__main__":
    main()