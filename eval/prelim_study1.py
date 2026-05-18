"""
Preliminary Study 1 — Does utility improve summary quality?
============================================================

Implements the exact comparison described in §Preliminary Study, Study 1:

    For each (conversation, query, gold_answer) triple in LoCoMo:

      (a) Identify the 20-turn block that contains the query's evidence.
      (b) VANILLA condition:
            summarize the block in a query-AGNOSTIC way (no query, no answer)
            → answer the query from the summary alone
      (c) UTILITY-AWARE condition:
            summarize the block while looking at the query AND its gold answer
            (instructed to retain only what is needed to support the answer)
            → answer the query from the summary alone

    Both summaries are written by the SAME LLM; both answers are produced by
    the SAME reader. NO retrieval is performed. The reader sees ONLY the
    summary (NOT the gold answer, NOT the raw turns) so the only thing that
    differs is what the summarizer chose to keep.

    F1 is computed against the gold answer for each condition and averaged
    over a random sample of N queries (default 100), reported overall and
    per QA category.

Usage
-----
    # quick smoke test
    python eval/prelim_study1.py --num-queries 10

    # full study (matches paper text)
    python eval/prelim_study1.py --num-queries 100 --workers 8

    # custom dataset / output
    python eval/prelim_study1.py --data data/locomo10.json \
        --output prelim_study1_results.json
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# allow imports from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.dataset import LoCoMoSample, Session, Turn, load_locomo
from eval.metrics import token_f1
from utils.llm_client import LLMClient


# ============================================================
# Block selection: which 20 turns belong to this query?
# ============================================================

WINDOW = 20


def _parse_dia_id(dia_id: str) -> Optional[Tuple[int, int]]:
    """'D2:8' -> (2, 8). Returns None on malformed input."""
    if not isinstance(dia_id, str) or not dia_id.startswith("D"):
        return None
    try:
        sess_str, turn_str = dia_id[1:].split(":", 1)
        return int(sess_str), int(turn_str)
    except (ValueError, IndexError):
        return None


def select_block_for_query(
    sample: LoCoMoSample, evidence: List[str], window: int = WINDOW
) -> Optional[Tuple[Session, List[Turn]]]:
    """
    Choose the 20-turn block that contains the query's evidence.

    Strategy:
      1. Parse all evidence dia_ids -> {session_id: [turn_idx, ...]}.
      2. Pick the session that contains the MOST evidence turns
         (deterministic tiebreak: smallest session_id).
      3. Within that session, slide a `window`-turn window so that as many
         evidence turns as possible fall inside, anchored on the median
         evidence turn. If the session has <= window turns, return all turns.

    Returns (session, list_of_turns) or None if no usable evidence.
    """
    if not evidence:
        return None

    by_session: Dict[int, List[int]] = defaultdict(list)
    for ev in evidence:
        parsed = _parse_dia_id(ev)
        if parsed is None:
            continue
        sid, tidx = parsed
        by_session[sid].append(tidx)

    if not by_session:
        return None

    # session with most evidence (tiebreak: smallest id)
    best_sid = sorted(by_session, key=lambda s: (-len(by_session[s]), s))[0]
    session = sample.conversation.sessions.get(best_sid)
    if session is None or not session.turns:
        return None

    turns = session.turns
    n = len(turns)
    if n <= window:
        return session, turns

    # Build a dia_id -> position map for this session
    pos_of: Dict[str, int] = {t.dia_id: i for i, t in enumerate(turns)}
    ev_positions = [
        pos_of[f"D{best_sid}:{tidx}"]
        for tidx in by_session[best_sid]
        if f"D{best_sid}:{tidx}" in pos_of
    ]
    if not ev_positions:
        # Fallback: first window turns
        return session, turns[:window]

    # Center the window so evidence fits; clamp to session bounds.
    lo_ev, hi_ev = min(ev_positions), max(ev_positions)
    span = hi_ev - lo_ev + 1
    if span >= window:
        # evidence wider than window; take the first `window` turns starting at lo_ev
        start = min(lo_ev, n - window)
    else:
        center = (lo_ev + hi_ev) // 2
        start = center - window // 2
        start = max(0, min(start, n - window))
        # nudge so all evidence is inside if possible
        if hi_ev >= start + window:
            start = hi_ev - window + 1
        if lo_ev < start:
            start = lo_ev
        start = max(0, min(start, n - window))
    return session, turns[start : start + window]


def format_block(session: Session, turns: List[Turn]) -> str:
    """Render the selected block as a flat dialogue string."""
    header = f"[Session {session.session_id} · {session.date_time}]"
    body = "\n".join(f"{t.speaker}: {t.text}" for t in turns)
    return f"{header}\n{body}"


# ============================================================
# Summarization prompts
# ============================================================

SYSTEM_SUMMARIZER = (
    "You are a careful summarizer. Produce concise plain-text summaries. "
    "Do NOT add commentary or preamble; output only the summary text."
)


def vanilla_prompt(block_text: str) -> str:
    """Query-agnostic summarization (the standard practice baseline)."""
    return f"""Summarize the following conversation excerpt.

Capture the important information so that a reader who has not seen the
original conversation can understand what was discussed. Use absolute dates
and full names where possible. Output the summary as a few short paragraphs
or a list of self-contained statements. No preamble, no commentary.

[Conversation excerpt]
{block_text}

[Summary]"""


def utility_aware_prompt(block_text: str, query: str, gold_answer: str) -> str:
    """Query-conditioned summarization: keep only what supports the answer."""
    return f"""Summarize the following conversation excerpt so that the
resulting summary contains every fact required to answer the target question
below. The reference answer is provided so you can verify which information
is essential; you must NOT copy the reference answer verbatim into the
summary, and you must NOT mention that you were given the answer.

Retain only the information necessary to support the reference answer.
Discard chit-chat, tangential remarks, and any facts unrelated to the
question. Use absolute dates and full names where possible. Output the
summary as a few short statements.

[Target question]
{query}

[Reference answer — for your eyes only, used to decide what to keep]
{gold_answer}

[Conversation excerpt]
{block_text}

[Summary]"""


def summarize(
    llm: LLMClient,
    block_text: str,
    query: Optional[str] = None,
    gold_answer: Optional[str] = None,
) -> str:
    if query is None:
        prompt = vanilla_prompt(block_text)
    else:
        assert gold_answer is not None
        prompt = utility_aware_prompt(block_text, query, gold_answer)
    messages = [
        {"role": "system", "content": SYSTEM_SUMMARIZER},
        {"role": "user", "content": prompt},
    ]
    return llm.chat_completion(messages, temperature=0.1).strip()


# ============================================================
# Reader: answer the query from the summary alone
# ============================================================

SYSTEM_READER = (
    "You are a question-answering assistant. Answer using ONLY the provided "
    "summary. If the summary does not contain the answer, say "
    "'Not mentioned in the conversation'. Be concise."
)


def answer_from_summary(llm: LLMClient, summary: str, query: str) -> str:
    prompt = f"""Answer the question using ONLY the summary below.
Be concise — a short phrase or single sentence is best. Do not invent
information that is not in the summary.

[Summary]
{summary}

[Question]
{query}

[Answer]"""
    messages = [
        {"role": "system", "content": SYSTEM_READER},
        {"role": "user", "content": prompt},
    ]
    return llm.chat_completion(messages, temperature=0.0).strip()


# ============================================================
# Per-query pipeline
# ============================================================

def run_one_query(
    llm: LLMClient,
    sample: LoCoMoSample,
    qa_idx: int,
) -> Optional[Dict[str, Any]]:
    qa = sample.qa[qa_idx]
    gold = qa.final_answer
    if not gold or not qa.evidence:
        return None
    selection = select_block_for_query(sample, qa.evidence, window=WINDOW)
    if selection is None:
        return None
    session, turns = selection
    block_text = format_block(session, turns)

    # --- VANILLA ---
    s_vanilla = summarize(llm, block_text, query=None)
    pred_vanilla = answer_from_summary(llm, s_vanilla, qa.question)
    f1_vanilla = token_f1(pred_vanilla, gold)

    # --- UTILITY-AWARE ---
    s_util = summarize(llm, block_text, query=qa.question, gold_answer=gold)
    pred_util = answer_from_summary(llm, s_util, qa.question)
    f1_util = token_f1(pred_util, gold)

    return {
        "sample_id": sample.sample_id,
        "qa_idx": qa_idx,
        "category": qa.category,
        "question": qa.question,
        "gold": gold,
        "evidence": qa.evidence,
        "session_id": session.session_id,
        "block_size": len(turns),
        "summary_vanilla": s_vanilla,
        "summary_utility": s_util,
        "pred_vanilla": pred_vanilla,
        "pred_utility": pred_util,
        "f1_vanilla": f1_vanilla,
        "f1_utility": f1_util,
    }


# ============================================================
# Aggregation
# ============================================================

def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _stats(vals: List[float]) -> Dict[str, float]:
        if not vals:
            return {"mean": 0.0, "std": 0.0, "n": 0}
        return {
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "n": len(vals),
        }

    out: Dict[str, Any] = {}
    for tag in ("vanilla", "utility"):
        out[tag] = {"overall": _stats([r[f"f1_{tag}"] for r in rows])}
        by_cat: Dict[int, List[float]] = defaultdict(list)
        for r in rows:
            by_cat[r["category"]].append(r[f"f1_{tag}"])
        for cat in sorted(by_cat):
            out[tag][f"cat_{cat}"] = _stats(by_cat[cat])

    # Delta
    out["delta"] = {
        "overall": out["utility"]["overall"]["mean"]
        - out["vanilla"]["overall"]["mean"]
    }
    for cat_key in [k for k in out["vanilla"] if k.startswith("cat_")]:
        out["delta"][cat_key] = (
            out["utility"][cat_key]["mean"] - out["vanilla"][cat_key]["mean"]
        )
    return out


# ============================================================
# Main
# ============================================================

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/locomo10.json")
    p.add_argument("--num-queries", type=int, default=100,
                   help="Random sample of QA pairs to evaluate")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel worker threads (LLM calls)")
    p.add_argument("--output", default="prelim_study1_results.json")
    p.add_argument("--limit-samples", type=int, default=None,
                   help="Use only the first K LoCoMo samples (debugging)")
    args = p.parse_args()

    random.seed(args.seed)

    samples = load_locomo(args.data, limit=args.limit_samples)

    # Build the global pool of (sample, qa_idx) pairs that have valid evidence.
    pool: List[Tuple[LoCoMoSample, int]] = []
    for s in samples:
        for j, qa in enumerate(s.qa):
            if qa.final_answer and qa.evidence:
                # quick check: at least one evidence id must resolve
                ok = False
                for ev in qa.evidence:
                    parsed = _parse_dia_id(ev)
                    if parsed and parsed[0] in s.conversation.sessions:
                        ok = True
                        break
                if ok:
                    pool.append((s, j))

    print(f"Eligible QA pool: {len(pool)}")
    if args.num_queries < len(pool):
        chosen = random.sample(pool, args.num_queries)
    else:
        chosen = pool
    print(f"Selected {len(chosen)} queries for evaluation")

    llm = LLMClient()

    # Run in parallel
    results: List[Dict[str, Any]] = []
    failures = 0
    t0 = time.time()
    if args.workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(run_one_query, llm, s, j): (s.sample_id, j)
                for s, j in chosen
            }
            for k, fut in enumerate(concurrent.futures.as_completed(futures), 1):
                tag = futures[fut]
                try:
                    r = fut.result()
                    if r is not None:
                        results.append(r)
                    else:
                        failures += 1
                except Exception as e:
                    print(f"  [{tag}] failed: {e}")
                    failures += 1
                if k % 10 == 0:
                    elapsed = time.time() - t0
                    print(
                        f"  [{k}/{len(chosen)}] elapsed {elapsed:.1f}s, "
                        f"running mean f1 vanilla="
                        f"{statistics.mean([r['f1_vanilla'] for r in results]):.3f} "
                        f"utility="
                        f"{statistics.mean([r['f1_utility'] for r in results]):.3f}"
                    )
    else:
        for k, (s, j) in enumerate(chosen, 1):
            try:
                r = run_one_query(llm, s, j)
                if r is not None:
                    results.append(r)
                else:
                    failures += 1
            except Exception as e:
                print(f"  [{s.sample_id}, {j}] failed: {e}")
                failures += 1
            if k % 10 == 0:
                print(f"  [{k}/{len(chosen)}]")

    print(f"\nDone in {time.time() - t0:.1f}s — {len(results)} ok, {failures} failed")

    summary = aggregate(results)

    # Pretty-print headline
    print("\n" + "=" * 60)
    print("Study 1 — F1 (token-level) on LoCoMo")
    print("=" * 60)
    v = summary["vanilla"]["overall"]["mean"]
    u = summary["utility"]["overall"]["mean"]
    n = summary["vanilla"]["overall"]["n"]
    print(f"  Vanilla        : {v:.3f}   (n={n})")
    print(f"  Utility-aware  : {u:.3f}   (n={n})")
    print(f"  Δ (utility-vanilla): {u - v:+.3f}")
    print()
    print(" Per category (mean F1):")
    print(f"  {'cat':>5}  {'n':>4}  {'vanilla':>8}  {'utility':>8}  {'delta':>7}")
    for k in sorted(k for k in summary["vanilla"] if k.startswith("cat_")):
        nk = summary["vanilla"][k]["n"]
        vv = summary["vanilla"][k]["mean"]
        uu = summary["utility"][k]["mean"]
        print(f"  {k:>5}  {nk:>4d}  {vv:>8.3f}  {uu:>8.3f}  {uu - vv:>+7.3f}")
    print("=" * 60)

    out_path = Path(args.output)
    out_path.write_text(
        json.dumps(
            {
                "config": {
                    "data": args.data,
                    "num_queries_requested": args.num_queries,
                    "num_queries_evaluated": len(results),
                    "window": WINDOW,
                    "seed": args.seed,
                    "model": llm.model,
                },
                "summary": summary,
                "rows": results,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"\nFull results written to: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
