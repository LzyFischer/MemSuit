"""
Preliminary Study 2: Can utility-aware summarization behavior transfer via demonstration?

Setup (no retrieval):
  Each evaluation unit is (block, question, gold_answer) where the block is a
  WINDOW_SIZE-turn window centered on the evidence turn. We summarize the block
  under two conditions, feed ALL resulting entries directly to the answer
  generator (no retrieval / no top-k), and score token-F1 against the gold.

Conditions (paired on every eval candidate):
  A. Vanilla    — plain student prompt, no query shown, no demonstrations.
  B. With-demo  — same student prompt preceded by k=6 demonstrations, each of
                  the form (block, query, gold_answer, teacher_summary).
                  The current candidate's query is NOT shown.

Demo sourcing:
  Demos come from the TRAIN split (first 2 conversations in dataset order).
  The teacher is called exactly k=6 times — once per demo candidate — using
  the query+answer hint, producing utility-aware summaries.
  Eval candidates come from the TEST split (conversations 4 onward).
  Val conversation (3rd) is excluded from both, matching the main pipeline.

Demo-type conditions (one run each):
  temporal  — 6 demos drawn from category-3 (temporal) train candidates
  open      — 6 demos drawn from category-1/2 train candidates
  mixed     — 3 temporal + 3 open train candidates

Eval-type breakdown:
  Results are reported separately for temporal (cat 3) and open (cat 1+2) eval
  candidates, giving the 4-row x 2-col table in the paper.

Usage:
  # run all three demo-type conditions in one shot (called by the .sh)
  python eval/run_prelim_study2.py \\
      --dataset data/locomo10.json \\
      --out-dir results/prelim_study2 \\
      --demo-type all \\
      --n-eval 100 --k-demos 6 --seed 42

  # single condition
  python eval/run_prelim_study2.py --demo-type temporal ...

  # print the summary table from existing result files
  python eval/run_prelim_study2.py --table-only \\
      --out-dir results/prelim_study2 --seed 42
"""

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from core.answer_generator import AnswerGenerator
from eval.dataset import load_locomo, LoCoMoSample
from eval.metrics import token_f1
from models.memory_entry import MemoryEntry
from utils.llm_client import LLMClient

from train.build_dataset import (
    STUDENT_PROMPT_TEMPLATE,
    SYSTEM_MSG,
    _flatten_turns,
    _format_evidence_text,
    _format_window_text,
    _window_around,
    _validate_entries,
    make_student_prompt,
    run_teacher,
)


# ─────────────────────────────────────────────────────────────────────────────
# Category groups
# ─────────────────────────────────────────────────────────────────────────────

# LoCoMo category definitions:
#   cat 1 = single-hop open-domain
#   cat 2 = temporal
#   cat 3 = multi-hop open-domain
#   cat 4 = adversarial (excluded)
TEMPORAL_CATS = {2}
OPEN_CATS     = {3}
EVAL_CATS     = TEMPORAL_CATS | OPEN_CATS   # cat 1/4 (single-hop/multi-session) and cat 5 (adversarial) excluded


def cat_group(cat: int) -> Optional[str]:
    if cat in TEMPORAL_CATS:
        return "temporal"
    if cat in OPEN_CATS:
        return "open"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Candidate dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalCandidate:
    sample_id:       str
    qa_idx:          int
    category:        int
    group:           str        # "temporal" | "open"
    question:        str
    answer:          str
    evidence_dia_id: str
    dialogue_text:   str
    evidence_text:   str


# ─────────────────────────────────────────────────────────────────────────────
# Enumerate candidates filtered to a sample-id set
# ─────────────────────────────────────────────────────────────────────────────

def enumerate_candidates(
    samples:            List[LoCoMoSample],
    window_size:        int,
    allowed_sample_ids: set,
) -> List[EvalCandidate]:
    out: List[EvalCandidate] = []
    for sample in samples:
        if sample.sample_id not in allowed_sample_ids:
            continue
        flat      = _flatten_turns(sample)
        id_to_idx = {dia_id: i for i, (_, _, dia_id) in enumerate(flat)}

        for qa_idx, qa in enumerate(sample.qa):
            if qa.category not in EVAL_CATS:
                continue
            if not qa.final_answer:
                continue
            grp = cat_group(qa.category)

            # anchor on first locatable evidence turn
            anchor_idx, anchor_dia_id = None, None
            for ev in qa.evidence:
                if isinstance(ev, str) and ev in id_to_idx:
                    anchor_idx    = id_to_idx[ev]
                    anchor_dia_id = ev
                    break
            if anchor_idx is None:
                continue

            window, _, _ = _window_around(flat, anchor_idx, window_size)
            out.append(EvalCandidate(
                sample_id       = sample.sample_id,
                qa_idx          = qa_idx,
                category        = qa.category,
                group           = grp,
                question        = qa.question,
                answer          = str(qa.final_answer),
                evidence_dia_id = anchor_dia_id,
                dialogue_text   = _format_window_text(window),
                evidence_text   = _format_evidence_text(flat, anchor_idx),
            ))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Split  (mirrors build_dataset.split_candidates)
# ─────────────────────────────────────────────────────────────────────────────

def get_split_sample_ids(
    samples: List[LoCoMoSample],
) -> Tuple[set, set, set]:
    """
    train = first 2 conversations
    val   = 3rd conversation
    test  = conversations 4+
    """
    ordered   = [s.sample_id for s in samples]
    train_ids = set(ordered[:2])
    val_ids   = set(ordered[2:3])
    test_ids  = set(ordered[3:])
    return train_ids, val_ids, test_ids


# ─────────────────────────────────────────────────────────────────────────────
# Demo construction  (teacher called exactly k times)
# ─────────────────────────────────────────────────────────────────────────────

def build_demos(
    train_cands: List[EvalCandidate],
    demo_type:   str,
    k:           int,
    rng:         random.Random,
    llm:         LLMClient,
) -> List[Dict[str, Any]]:
    """
    Sample k candidates from the train split by demo_type, call the teacher
    once per candidate (with query+answer hint), return list of demo dicts:
      {dialogue_text, question, answer, teacher_entries}
    """
    by_group: Dict[str, List[EvalCandidate]] = {
        "temporal": [c for c in train_cands if c.group == "temporal"],
        "open":     [c for c in train_cands if c.group == "open"],
    }

    if demo_type == "temporal":
        pool = by_group["temporal"]
        assert len(pool) >= k, \
            f"Need {k} temporal train candidates, only {len(pool)} available"
        selected = rng.sample(pool, k)

    elif demo_type == "open":
        pool = by_group["open"]
        assert len(pool) >= k, \
            f"Need {k} open train candidates, only {len(pool)} available"
        selected = rng.sample(pool, k)

    elif demo_type == "mixed":
        half = k // 2
        assert len(by_group["temporal"]) >= half and len(by_group["open"]) >= half, \
            f"Need {half} of each group; have " \
            f"temporal={len(by_group['temporal'])}, open={len(by_group['open'])}"
        selected = (rng.sample(by_group["temporal"], half) +
                    rng.sample(by_group["open"],     half))

    else:
        raise ValueError(f"Unknown demo_type: {demo_type}")

    demos: List[Dict[str, Any]] = []
    for i, c in enumerate(selected):
        print(f"  [demo {i+1}/{k}] teacher on {c.sample_id}/{c.qa_idx} "
              f"cat={c.category} Q: {c.question[:60]}")
        entries = run_teacher(
            llm,
            c.dialogue_text,
            c.question,
            c.answer,
            c.evidence_text,
        )
        if entries is None:
            print("    -> teacher failed, skipping")
            continue
        demos.append({
            "dialogue_text":   c.dialogue_text,
            "question":        c.question,
            "answer":          c.answer,
            "teacher_entries": entries,
        })

    if len(demos) < k:
        print(f"  [warn] only {len(demos)}/{k} demos built (some teacher calls failed)")
    return demos


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────────────

# Max chars kept from each demo's dialogue_text.
# A 20-turn block can be up to ~4000 chars; 6 demos × 4000 = 24000 chars
# which blows past the 8192-token limit. We truncate each demo's dialogue to
# ~1000 chars (~250 tokens) — enough to show the extraction style without
# overwhelming the context window.
#   Budget breakdown (chars → tokens ÷4):
#     6 demos × (1000 dialogue + 200 Q/A + 600 entries) ≈ 2700 tokens
#     student prompt (current block, up to ~4000 chars)  ≈ 1000 tokens
#     system message                                      ≈  200 tokens
#     output budget                                       ≈  512 tokens
#     Total                                               ≈ 4412 tokens  < 8192
DEMO_DIALOGUE_MAX_CHARS = 1000


def format_demo_block(demos: List[Dict[str, Any]]) -> str:
    """
    Prefix block: each demo shows (block, query, answer, teacher_summary).
    The demo's dialogue is truncated to DEMO_DIALOGUE_MAX_CHARS to stay within
    the model's context window. The current block's query is intentionally withheld.
    """
    parts = []
    for i, d in enumerate(demos, 1):
        teacher_json = json.dumps(d["teacher_entries"], ensure_ascii=False, indent=2)
        dialogue = d["dialogue_text"]
        if len(dialogue) > DEMO_DIALOGUE_MAX_CHARS:
            dialogue = dialogue[:DEMO_DIALOGUE_MAX_CHARS] + "\n[... truncated ...]"
        parts.append(
            f"### Example {i}\n"
            f"[Example Dialogues]\n{dialogue}\n\n"
            f"[Example Question]\n{d['question']}\n"
            f"[Example Answer]\n{d['answer']}\n\n"
            f"[Example Memory Entries]\n```json\n{teacher_json}\n```"
        )
    header = (
        "The following examples show how to extract memory entries from a "
        "dialogue when you know a specific question that must be answerable "
        "from those entries. Study what details each summary preserves "
        "(named entities, numerals, dates, specific events) relative to the "
        "question. Then apply the same extraction quality to the new dialogue "
        "below — the question for the new dialogue is NOT provided, so "
        "anticipate any plausible future question and preserve details "
        "accordingly.\n\n"
    )
    return header + "\n\n---\n\n".join(parts)


def make_with_demo_prompt(dialogue_text: str, demos: List[Dict[str, Any]]) -> str:
    demo_block   = format_demo_block(demos)
    student_part = STUDENT_PROMPT_TEMPLATE.format(dialogue_text=dialogue_text)
    return (
        f"{demo_block}\n\n"
        f"---\n\n"
        f"### New dialogue - apply the same extraction style\n\n"
        f"{student_part}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Summarizer call
# ─────────────────────────────────────────────────────────────────────────────

def call_summarizer(llm: LLMClient, user_prompt: str) -> List[MemoryEntry]:
    total_chars = len(SYSTEM_MSG) + len(user_prompt)
    approx_tokens = total_chars // 4
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",   "content": user_prompt},
    ]
    response_format = (
        {"type": "json_object"}
        if getattr(config, "USE_JSON_FORMAT", False) else None
    )
    for attempt in range(3):
        try:
            raw     = llm.chat_completion(messages, temperature=0.1,
                                          response_format=response_format)
            data    = llm.extract_json(raw)
            entries = _validate_entries(data)
            if entries:
                return [MemoryEntry(lossless_restatement=e["lossless_restatement"])
                        for e in entries]
            # _validate_entries returned empty — log raw output for debugging
            print(f"  [summarizer] valid JSON but no entries extracted. "
                  f"raw[:200]={str(raw)[:200]}")
        except Exception as e:
            print(f"  [summarizer] attempt {attempt+1}/3 failed "
                  f"(~{approx_tokens} tokens): {e}")
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_eval(
    eval_cands: List[EvalCandidate],
    demos:      List[Dict[str, Any]],
    llm:        LLMClient,
    answerer:   AnswerGenerator,
    out_jsonl:  Path,
) -> List[Dict[str, Any]]:
    """
    For each eval candidate run BOTH conditions (paired), write records to
    out_jsonl, return list of records.
    """
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []

    with out_jsonl.open("w", encoding="utf-8") as fh:
        for i, c in enumerate(eval_cands):
            print(f"\n[eval {i+1}/{len(eval_cands)}] "
                  f"{c.sample_id}/{c.qa_idx} cat={c.category} ({c.group}) "
                  f"Q: {c.question[:60]}")

            # ── Condition A: vanilla ──────────────────────────────────────
            vanilla_entries = call_summarizer(
                llm, make_student_prompt(c.dialogue_text)
            )
            vanilla_pred = (answerer.generate_answer(c.question, vanilla_entries)
                            if vanilla_entries else "")
            vanilla_f1   = token_f1(vanilla_pred, c.answer)

            # ── Condition B: with demo ────────────────────────────────────
            demo_entries = call_summarizer(
                llm, make_with_demo_prompt(c.dialogue_text, demos)
            )
            demo_pred = (answerer.generate_answer(c.question, demo_entries)
                         if demo_entries else "")
            demo_f1   = token_f1(demo_pred, c.answer)

            print(f"  vanilla f1={vanilla_f1:.3f} | "
                  f"with-demo f1={demo_f1:.3f} | gold: {c.answer[:50]}")

            rec = {
                "sample_id":   c.sample_id,
                "qa_idx":      c.qa_idx,
                "category":    c.category,
                "group":       c.group,
                "question":    c.question,
                "gold_answer": c.answer,
                "vanilla":   {"n_entries": len(vanilla_entries),
                              "pred": vanilla_pred, "f1": vanilla_f1},
                "with_demo": {"n_entries": len(demo_entries),
                              "pred": demo_pred,    "f1": demo_f1},
            }
            records.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _mean(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    result: Dict[str, Any] = {}
    for grp in ("temporal", "open", "all"):
        sub = records if grp == "all" else [r for r in records if r["group"] == grp]
        result[grp] = {
            "n":            len(sub),
            "vanilla_f1":   _mean([r["vanilla"]["f1"]   for r in sub]),
            "with_demo_f1": _mean([r["with_demo"]["f1"] for r in sub]),
            "delta":        _mean([r["with_demo"]["f1"] - r["vanilla"]["f1"]
                                   for r in sub]),
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Table printer
# ─────────────────────────────────────────────────────────────────────────────

def print_table(out_dir: Path, seed: int) -> None:
    def load(dtype: str) -> Optional[List[Dict[str, Any]]]:
        p = out_dir / f"{dtype}_seed{seed}.jsonl"
        if not p.exists():
            return None
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

    def mean_f1(records: List[Dict], group: str, cond: str) -> float:
        sub  = records if group == "all" else [r for r in records if r["group"] == group]
        vals = [r[cond]["f1"] for r in sub]
        return sum(vals) / len(vals) if vals else float("nan")

    # vanilla from temporal run (condition A is identical across all runs)
    ref = load("temporal")
    rows = []
    if ref:
        rows.append(("Vanilla (no demo)",
                     mean_f1(ref, "temporal", "vanilla"),
                     mean_f1(ref, "open",     "vanilla")))
    else:
        rows.append(("Vanilla (no demo)", float("nan"), float("nan")))

    for dtype in ("temporal", "open", "mixed"):
        data = load(dtype)
        if data is None:
            rows.append((f"Demo: {dtype}", float("nan"), float("nan")))
        else:
            rows.append((f"Demo: {dtype}",
                         mean_f1(data, "temporal", "with_demo"),
                         mean_f1(data, "open",     "with_demo")))

    vanilla_t, vanilla_o = rows[0][1], rows[0][2]

    hdr = (f"{'Condition':<24} {'Eval: temporal':>16} "
           f"{'Eval: open-domain':>18} {'Avg':>8}")
    sep = "-" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")
    for label, t, o in rows:
        avg = (t + o) / 2
        ft  = " *" if (not label.startswith("Vanilla") and t > vanilla_t) else "  "
        fo  = " *" if (not label.startswith("Vanilla") and o > vanilla_o) else "  "
        print(f"{label:<24} {t:>14.3f}{ft} {o:>16.3f}{fo} {avg:>8.3f}")
    print(sep)
    print("* = above vanilla baseline\n")

    for dtype in ("temporal", "open", "mixed"):
        data = load(dtype)
        if data:
            nt = sum(1 for r in data if r["group"] == "temporal")
            no = sum(1 for r in data if r["group"] == "open")
            print(f"  n ({dtype}): temporal={nt}, open={no}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",     default="data/locomo10.json")
    p.add_argument("--out-dir",     default="results/prelim_study2")
    p.add_argument("--window-size", type=int, default=config.WINDOW_SIZE)
    p.add_argument("--n-eval",      type=int, default=100,
                   help="eval candidates to sample (balanced across groups)")
    p.add_argument("--k-demos",     type=int, default=6)
    p.add_argument("--demo-type",
                   choices=["temporal", "open", "mixed", "all"],
                   default="all",
                   help="'all' runs temporal, open, mixed sequentially")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--table-only",  action="store_true",
                   help="skip eval, just print table from existing result files")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.table_only:
        print_table(out_dir, args.seed)
        return

    # ── load dataset + split ─────────────────────────────────────────────────
    samples                      = load_locomo(args.dataset)
    train_ids, val_ids, test_ids = get_split_sample_ids(samples)
    print(f"Split  train={sorted(train_ids)}  "
          f"val={sorted(val_ids)}  test={sorted(test_ids)}")

    train_cands = enumerate_candidates(samples, args.window_size, train_ids)
    test_cands  = enumerate_candidates(samples, args.window_size, test_ids)
    print(f"Train candidates: "
          f"temporal={sum(1 for c in train_cands if c.group=='temporal')}, "
          f"open={sum(1 for c in train_cands if c.group=='open')}")
    print(f"Test  candidates: "
          f"temporal={sum(1 for c in test_cands  if c.group=='temporal')}, "
          f"open={sum(1 for c in test_cands  if c.group=='open')}")

    # ── sample eval candidates (balanced across groups) ──────────────────────
    rng      = random.Random(args.seed)
    half     = args.n_eval // 2
    by_group = {
        "temporal": [c for c in test_cands if c.group == "temporal"],
        "open":     [c for c in test_cands if c.group == "open"],
    }
    n_t = min(half, len(by_group["temporal"]))
    n_o = min(args.n_eval - n_t, len(by_group["open"]))
    if n_t + n_o < args.n_eval:
        print(f"  [warn] requested {args.n_eval}, only {n_t+n_o} available")
    eval_cands = (rng.sample(by_group["temporal"], n_t) +
                  rng.sample(by_group["open"],     n_o))
    rng.shuffle(eval_cands)
    print(f"Eval candidates sampled: temporal={n_t}, open={n_o} "
          f"(total={len(eval_cands)})")

    # ── shared LLM + answerer (one vLLM connection throughout) ───────────────
    llm      = LLMClient()
    answerer = AnswerGenerator(LLMClient())

    # ── demo-type loop ────────────────────────────────────────────────────────
    demo_types = (["temporal", "open", "mixed"]
                  if args.demo_type == "all" else [args.demo_type])

    for dtype in demo_types:
        tag       = f"{dtype}_seed{args.seed}"
        out_jsonl = out_dir / f"{tag}.jsonl"

        if out_jsonl.exists():
            n = sum(1 for l in out_jsonl.read_text().splitlines() if l.strip())
            print(f"\n[{dtype}] {out_jsonl.name} already exists ({n} records) — skipping")
            print(f"  Delete the file to re-run this condition.")
            continue

        print(f"\n{'='*60}")
        print(f"Demo-type: {dtype}  "
              f"(building {args.k_demos} demos from train split)")
        print(f"{'='*60}")

        demos = build_demos(train_cands, dtype, args.k_demos, rng, llm)
        if not demos:
            print(f"[{dtype}] No demos built — skipping condition.")
            continue

        records = run_eval(eval_cands, demos, llm, answerer, out_jsonl)
        agg     = aggregate(records)

        print(f"\n[{dtype}] quick results:")
        for grp in ("temporal", "open", "all"):
            s = agg[grp]
            print(f"  {grp:<10} n={s['n']}  "
                  f"vanilla={s['vanilla_f1']:.3f}  "
                  f"with_demo={s['with_demo_f1']:.3f}  "
                  f"delta={s['delta']:+.3f}")

        summary = {
            "demo_type":  dtype,
            "seed":       args.seed,
            "k_demos":    args.k_demos,
            "demos_used": len(demos),
            "results":    agg,
        }
        sp = out_dir / f"{tag}_summary.json"
        sp.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                      encoding="utf-8")
        print(f"Wrote {out_jsonl}")
        print(f"Wrote {sp}")

    # ── final table (when running multiple or all conditions) ─────────────────
    if len(demo_types) > 1:
        print_table(out_dir, args.seed)


if __name__ == "__main__":
    main()