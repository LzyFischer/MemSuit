"""
LoCoMo Benchmark - QA evaluation using a local vLLM-served model
(default: Qwen/Qwen2.5-3B-Instruct).

What this script does
---------------------
1. Downloads the official LoCoMo dataset (data/locomo10.json) from
   snap-research/locomo if it's not already present.
2. For every QA pair in every conversation it:
     - flattens the entire multi-session dialog into a single transcript
       (the same "full conversation context" setting used in the paper)
     - asks the model to answer the question, given the transcript
3. Saves predictions and computes:
     - Exact Match
     - Token-level F1  (the standard SQuAD-style metric used by LoCoMo)
     - per-category breakdown (single-hop / multi-hop / temporal /
       open-domain / adversarial)
4. Optional: also run an LLM-as-Judge accuracy using the same vLLM server
   (--use_llm_judge).

How to use
----------
Step 1 (in another terminal): start a vLLM OpenAI-compatible server
    pip install vllm
    vllm serve Qwen/Qwen2.5-3B-Instruct \
        --host 0.0.0.0 --port 8000 \
        --max-model-len 32768 \
        --dtype auto

Step 2: run this script
    pip install openai tqdm requests
    python run_locomo_vllm.py \
        --model Qwen/Qwen2.5-3B-Instruct \
        --base_url http://localhost:8000/v1 \
        --output_dir outputs \
        --max_context_tokens 28000

Optional flags:
    --num_samples N        only evaluate first N conversations (debug)
    --use_llm_judge        also score with LLM-as-Judge
    --category 1           only run a single QA category (1..5)
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests
from openai import OpenAI
from tqdm import tqdm

# -------------------------------------------------------------------- #
# Constants - mirror the official LoCoMo repo
# -------------------------------------------------------------------- #
LOCOMO_DATA_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
)

# QA category labels used in the paper.  Category 5 = adversarial
# (answer should be "no information available" / refuse to answer).
CATEGORY_NAMES = {
    1: "single_hop",
    2: "multi_hop",
    3: "temporal",
    4: "open_domain",  # commonsense / world-knowledge
    5: "adversarial",
}

# ── Official prompt family ────────────────────────────────────────────────────
# These match the prompts used in HiGMem / A-Mem / EverMemOS full-context
# baseline (the de-facto community standard for comparable LoCoMo numbers).
# Source: ZeroLoss-Lab/HiGMem full_context_test.py  build_category_prompt()
# Key differences from a "nice" prompt:
#   - No system message (pure user turn only)
#   - Per-category wording (especially cat 2 = temporal)
#   - "Answer with exact words from the context whenever possible"
#   - Category 5 is a forced-choice between the trap answer and the refusal
#
# DO NOT "improve" these prompts if you want numbers comparable to published
# papers.  A better prompt will give higher scores but they won't be comparable.
# ─────────────────────────────────────────────────────────────────────────────

# No system prompt — single user message only (matches official baseline).
QA_SYSTEM_PROMPT = ""   # kept for API compat; passed as system="" (ignored by vLLM)

def build_qa_prompt(
    context: str,
    question: str,
    category: int,
    adversarial_answer: str = "",
) -> str:
    """Return the official per-category QA prompt (no system message)."""
    if category == 5:
        import random as _rnd
        choices = ["Not mentioned in the conversation", adversarial_answer]
        if _rnd.random() < 0.5:
            choices = choices[::-1]
        return (
            f"Based on the context: {context}, answer the following question. "
            f"{question}\n"
            f"Select the correct answer: {choices[0]} or {choices[1]} Short answer:"
        ).strip()
    elif category == 2:
        return (
            f"Based on the context: {context}, answer the following question.\n"
            # f"Use DATE of CONVERSATION to answer with an approximate date.\n"
            # f"Please generate the shortest possible answer, using words from the "
            # f"conversation where possible, and avoid using any subjects.\n"
            f"Question: {question} Short answer:"
        ).strip()
    else:
        # categories 1, 3, 4
        return (
            f"Based on the context: {context}, write an answer.\n"
            # f"short phrase for the following question. "
            # f"Answer with exact words from the context whenever possible.\n"
            f"Question: {question} Short answer:"
        ).strip()


# Distractor-mode variant: prepend a short instruction before the same template.
# We keep it minimal — the key context label already says TARGET / DISTRACTOR.
def build_qa_prompt_distractor(
    context: str,
    question: str,
    category: int,
    speaker_a: str,
    speaker_b: str,
    adversarial_answer: str = "",
) -> str:
    """Official prompt with a one-line distractor reminder prepended."""
    base = build_qa_prompt(context, question, category, adversarial_answer)
    header = (
        f"The context below contains sessions from multiple conversations. "
        f"Use ONLY the TARGET sessions (between {speaker_a} and {speaker_b}) "
        f"to answer.\n\n"
    )
    return header + base

JUDGE_SYSTEM_PROMPT = (
    "You are a strict grader. Given a question, a gold reference answer "
    "and a candidate answer, decide if the candidate is semantically "
    "correct. Respond with ONLY 'YES' or 'NO'."
)

JUDGE_USER_TEMPLATE = """Question: {question}
Gold answer: {gold}
Candidate answer: {pred}

Is the candidate answer correct? Reply with YES or NO only."""


# -------------------------------------------------------------------- #
# Data loading
# -------------------------------------------------------------------- #
def download_locomo(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print(f"[data] using cached dataset at {path}")
        return
    print(f"[data] downloading LoCoMo from {LOCOMO_DATA_URL} ...")
    r = requests.get(LOCOMO_DATA_URL, timeout=120)
    r.raise_for_status()
    path.write_bytes(r.content)
    print(f"[data] saved to {path} ({path.stat().st_size/1024:.1f} KiB)")


def _sorted_session_keys(conv: dict) -> list[str]:
    return sorted(
        [k for k in conv if re.fullmatch(r"session_\d+", k)],
        key=lambda k: int(k.split("_")[1]),
    )


def _session_num(sk: str) -> int:
    return int(sk.split("_")[1])


def _evidence_session_nums(qa: dict) -> set[int]:
    """Return session numbers that contain the evidence turns.
    dia_id formats seen in the wild: 'D5:3', 'D12:7', '5:3'"""
    nums: set[int] = set()
    for dia_id in qa.get("evidence", []):
        m = re.match(r"D?(\d+):", str(dia_id))
        if m:
            nums.add(int(m.group(1)))
    return nums


def flatten_conversation(
    conv: dict,
    mode: str = "full",
    qa: dict | None = None,
    masked_text: str = "[REDACTED]",
) -> str:
    """Build a plain-text transcript from a conversation dict.

    mode='full'  (default, original behaviour)
        The entire conversation, turn by turn.  The model can find the
        answer with a literal keyword search -- the easiest setting.

    mode='evidence_mask'
        All sessions are included, but the specific turns listed in
        qa['evidence'] are replaced with [REDACTED].  The model must
        reason from surrounding context rather than copy-pasting the
        answer verbatim.  Requires qa to be passed in.

    mode='session_cutoff'
        Only sessions whose index is STRICTLY LESS THAN the earliest
        session that contains an evidence turn are kept.  The model
        cannot see the answer at all -- it must reason from earlier
        context.  This is the hardest / most realistic memory setting.
        Falls back to 'full' when evidence is missing or the answer
        lives in session 1 (nothing left to cut off).
    """
    speaker_a = conv["speaker_a"]
    speaker_b = conv["speaker_b"]
    session_keys = _sorted_session_keys(conv)

    # Resolve masking / cutoff parameters
    evidence_dia_ids: set[str] = set()
    cutoff_session: int | None = None

    if mode in ("evidence_mask", "session_cutoff") and qa is not None:
        ev_sessions = _evidence_session_nums(qa)
        evidence_dia_ids = {str(d) for d in qa.get("evidence", [])}
        if ev_sessions:
            if mode == "session_cutoff":
                cutoff_session = min(ev_sessions)
        else:
            mode = "full"   # no evidence annotation → safe fallback

    out_lines = [f"Speakers: {speaker_a} and {speaker_b}", ""]

    for sk in session_keys:
        snum = _session_num(sk)

        if mode == "session_cutoff" and cutoff_session is not None:
            if snum >= cutoff_session:
                break   # sessions are in order, safe to stop here

        date = conv.get(f"{sk}_date_time", "")
        out_lines.append(f"--- {sk.upper()} (DATE/TIME: {date}) ---")

        for turn in conv[sk]:
            spk    = turn.get("speaker", "")
            txt    = turn.get("text", "")
            dia_id = turn.get("dia_id", "")
            cap    = turn.get("blip_caption")

            if mode == "evidence_mask" and dia_id in evidence_dia_ids:
                out_lines.append(
                    f"talk start time:{date}"
                    f"memory content: Speaker {spk}says : [REDACTED]"
                    f"memory context: memory keywords: []memory tags: []"
                )
            else:
                text_with_cap = txt + (f" [image: {cap}]" if cap else "")
                out_lines.append(
                    f"talk start time:{date}"
                    f"memory content: Speaker {spk}says : {text_with_cap}"
                    f"memory context: "
                    f"memory keywords: []"
                    f"memory tags: []"
                )

        out_lines.append("")

    if mode == "session_cutoff" and cutoff_session is not None:
        out_lines.append(
            f"[Note: only sessions prior to session {cutoff_session} "
            f"are shown. Answer the question from memory.]"
        )

    return "\n".join(out_lines)


def load_locomo(path: Path) -> list[dict]:
    """Return a list of conversation samples."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# -------------------------------------------------------------------- #
# Cross-conversation distractor context
# -------------------------------------------------------------------- #


def build_distractor_transcript(
    target_sample: dict,
    all_samples: list[dict],
    *,
    tokenizer=None,
    max_context_tokens: int = 28000,
) -> str:
    """Build a long context with ALL sessions from ALL conversations.

    Layout (deterministic, no shuffling)
    ─────────────────────────────────────
    [preamble]

    ── All DISTRACTOR sessions, in dataset order ──
       (every session of every conversation that is NOT the target,
        preserving within-conversation chronological order)

    ── All TARGET sessions, in chronological order ──
       (the conversation being evaluated)

    The target is placed LAST so that front-truncation (when the combined
    text exceeds max_context_tokens) always preserves the full target
    conversation and drops the oldest distractor content first.

    With LoCoMo-10 (~35 sessions × 10 conversations) the raw context is
    ~270 K tokens before truncation.  Set --max_context_tokens as high as
    your vLLM --max-model-len allows (e.g. 120000 for a 128 K model).
    """
    target_conv  = target_sample["conversation"]
    target_sid   = target_sample["sample_id"]
    target_spk_a = target_conv["speaker_a"]
    target_spk_b = target_conv["speaker_b"]

    def _render_session(conv: dict, sid: str, sk: str, label: str) -> str:
        date  = conv.get(f"{sk}_date_time", "")
        spk_a = conv["speaker_a"]
        spk_b = conv["speaker_b"]
        lines = [
            f"--- {label} | conv:{sid} | speakers:{spk_a}&{spk_b} | "
            f"{sk.upper()} (DATE/TIME: {date}) ---"
        ]
        for turn in conv[sk]:
            spk    = turn.get("speaker", "")
            txt    = turn.get("text", "")
            dia_id = turn.get("dia_id", "")
            cap    = turn.get("blip_caption")
            text_with_cap = txt + (f" [image: {cap}]" if cap else "")
            lines.append(
                f"talk start time:{date}"
                f"memory content: Speaker {spk}says : {text_with_cap}"
                f"memory context: "
                f"memory keywords: []"
                f"memory tags: []"
            )
    distractor_parts: list[str] = []
    n_distractor_sessions = 0
    for ds in all_samples:
        if ds["sample_id"] == target_sid:
            continue
        dc = ds["conversation"]
        for sk in _sorted_session_keys(dc):
            distractor_parts.append(
                _render_session(dc, ds["sample_id"], sk, "DISTRACTOR")
            )
            n_distractor_sessions += 1

    # ── 2. all TARGET sessions (chronological order) ──────────────────────
    target_parts: list[str] = []
    for sk in _sorted_session_keys(target_conv):
        target_parts.append(
            _render_session(target_conv, target_sid, sk, "TARGET")
        )
    n_target_sessions = len(target_parts)

    preamble = (
        f"Below is a long context containing sessions from ALL {len(all_samples)} "
        f"conversations in the dataset.\n"
        f"The first {n_distractor_sessions} sessions (labelled DISTRACTOR) are from "
        f"{len(all_samples) - 1} unrelated conversations — ignore them.\n"
        f"The last {n_target_sessions} sessions (labelled TARGET) belong to the "
        f"conversation between {target_spk_a} and {target_spk_b} "
        f"(conv:{target_sid}) — answer using ONLY these.\n\n"
    )

    full_text = preamble + "\n".join(distractor_parts + target_parts)

    raw_chars = len(full_text)
    print(
        f"  [distractor] {n_distractor_sessions} distractor sessions + "
        f"{n_target_sessions} target sessions = {raw_chars:,} chars raw; "
        f"truncating to {max_context_tokens} tokens"
    )

    # ── 3. token-budget truncation: drop from FRONT → target always intact ─
    return truncate_to_tokens(full_text, max_context_tokens, tokenizer)


# -------------------------------------------------------------------- #
# Metrics
# -------------------------------------------------------------------- #
# These functions are copied to match exactly the implementation used by
# mem0 / A-Mem / Memobase (which is the de-facto community standard for
# LoCoMo numbers), so results are directly comparable to published tables.
# Source: https://github.com/mem0ai/mem0/blob/main/evaluation/metrics/utils.py
# Original attribution: https://github.com/WujiangXu/AgenticMemory/blob/main/utils.py
#
# Key things to NOT change (we got bitten by these):
#   1) Tokenization is `lower().replace('.',' ')...split()` - it does NOT
#      strip stopwords, articles, or non-.!,? punctuation.
#   2) F1 is computed on the *set* of tokens (deduplicated), not on a
#      multiset / Counter. Using a Counter gives systematically higher
#      F1 numbers and won't match the published baselines.
#   3) EM is a raw lowercase string compare, not over normalized tokens.

def _simple_tokenize(text: str) -> list[str]:
    text = str(text)
    return (
        text.lower()
        .replace(".", " ")
        .replace(",", " ")
        .replace("!", " ")
        .replace("?", " ")
        .split()
    )


def exact_match(pred: str, gold: str) -> float:
    """Official EM: lowercase string equality (no tokenization)."""
    return float(str(pred).strip().lower() == str(gold).strip().lower())


def f1_score(pred: str, gold: str) -> float:
    """Token-set F1 - the metric reported by mem0 / A-Mem / Memobase."""
    pred_tokens = set(_simple_tokenize(pred))
    gold_tokens = set(_simple_tokenize(gold))
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = pred_tokens & gold_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# BLEU-1 via NLTK (smoothing method1 to avoid zero-grams blowing up).
# Kept in a closure so we lazy-import nltk and only download data once.
_bleu_smooth = None
_bleu_word_tokenize = None


def _ensure_bleu():
    global _bleu_smooth, _bleu_word_tokenize
    if _bleu_smooth is not None:
        return
    import nltk  # type: ignore
    from nltk.translate.bleu_score import SmoothingFunction  # type: ignore

    # Try to use word_tokenize (needs 'punkt' / 'punkt_tab').  If the data
    # isn't available we fall back to .split() which is what most other
    # LoCoMo evaluators do anyway and the BLEU-1 numbers come out within
    # ~1 percentage point of NLTK's word-level tokenizer.
    tok_fn = None
    for pkg in ("punkt_tab", "punkt"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:  # noqa: BLE001
                pass
    try:
        from nltk.tokenize import word_tokenize  # type: ignore
        word_tokenize("test ok")  # raises if punkt is missing
        tok_fn = word_tokenize
    except Exception:  # noqa: BLE001
        tok_fn = lambda s: str(s).split()  # noqa: E731

    _bleu_smooth = SmoothingFunction().method1
    _bleu_word_tokenize = tok_fn


def bleu1_score(pred: str, gold: str) -> float:
    """BLEU-1 = unigram precision with brevity penalty + smoothing method1.
    Same call signature mem0 uses: sentence_bleu([ref_tokens], pred_tokens,
    weights=(1,0,0,0), smoothing_function=method1).
    """
    _ensure_bleu()
    from nltk.translate.bleu_score import sentence_bleu  # type: ignore

    pred_tokens = _bleu_word_tokenize(str(pred).lower())
    gold_tokens = _bleu_word_tokenize(str(gold).lower())
    if not pred_tokens or not gold_tokens:
        return 0.0
    try:
        return float(
            sentence_bleu(
                [gold_tokens],
                pred_tokens,
                weights=(1, 0, 0, 0),
                smoothing_function=_bleu_smooth,
            )
        )
    except Exception:  # noqa: BLE001
        return 0.0


# -------------------------------------------------------------------- #
# Context truncation - keep the most recent tokens, like in the paper
# -------------------------------------------------------------------- #
def truncate_to_tokens(text: str, max_tokens: int, tokenizer) -> str:
    """Truncate FROM THE FRONT so the most recent dialog is kept."""
    if tokenizer is None:
        # ~4 chars per token heuristic
        max_chars = max_tokens * 4
        return text if len(text) <= max_chars else "...[earlier sessions truncated]...\n" + text[-max_chars:]
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    keep = ids[-max_tokens:]
    return "...[earlier sessions truncated]...\n" + tokenizer.decode(keep)


# -------------------------------------------------------------------- #
# Inference helpers
# -------------------------------------------------------------------- #
def call_chat(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 128,
    temperature: float = 0.0,
    retries: int = 3,
) -> str:
    last_err = None
    for attempt in range(retries):
        try:
            messages = []
            if system:   # omit system message when empty (official baseline)
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": user})
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait = 2 ** attempt
            print(f"[warn] API error ({e}); retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"API failed after {retries} retries: {last_err}")


# -------------------------------------------------------------------- #
# Main eval loop
# -------------------------------------------------------------------- #
def evaluate(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data_file)
    download_locomo(data_path)
    samples = load_locomo(data_path)
    if args.num_samples:
        samples = samples[: args.num_samples]
    print(f"[data] loaded {len(samples)} conversations")

    # Optional tokenizer for accurate truncation
    tokenizer = None
    try:
        from transformers import AutoTokenizer  # type: ignore

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        print(f"[tok] loaded tokenizer for {args.model}")
    except Exception as e:  # noqa: BLE001
        print(f"[tok] could not load HF tokenizer ({e}); using char heuristic")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    pred_path = out_dir / f"{args.model.replace('/', '_')}_predictions.jsonl"
    # resume: skip ids we already answered
    done_ids: set[str] = set()
    if pred_path.exists() and not args.overwrite:
        with pred_path.open() as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:  # noqa: BLE001
                    pass
        print(f"[resume] {len(done_ids)} predictions already on disk")

    fout = pred_path.open("a", encoding="utf-8")

    total_qa = 0
    for s in samples:
        conv = s["conversation"]
        sid = s["sample_id"]

        # Pre-build the full transcript once per conversation (used by
        # 'full' mode and also as a cache key for evidence_mask /
        # session_cutoff so we don't re-flatten for every QA).
        if args.context_mode == "cross_conv_distractor":
            # build once per conversation; all_samples used as-is (ordered)
            base_transcript = build_distractor_transcript(
                s,
                samples,
                tokenizer=tokenizer,
                max_context_tokens=args.max_context_tokens,
            )
        else:
            base_transcript = flatten_conversation(conv, mode="full")

        for j, qa in enumerate(s.get("qa", [])):
            cat = qa.get("category", -1)
            if args.category and cat != args.category:
                continue
            if args.skip_adversarial and cat == 5:
                continue
            # Official baseline skips QA with no evidence annotation
            if not qa.get("evidence") and not args.include_no_evidence:
                continue
            qid = f"{sid}::qa{j}"
            if qid in done_ids:
                continue

            question = qa["question"]
            gold = qa.get("answer", "")
            if isinstance(gold, list):
                gold = gold[0] if gold else ""
            gold = str(gold)
            adversarial_answer = qa.get("adversarial_answer", "")

            # Build the context for this specific QA
            if args.context_mode in ("full", "cross_conv_distractor"):
                transcript = truncate_to_tokens(
                    base_transcript, args.max_context_tokens, tokenizer
                )
            else:
                transcript = flatten_conversation(
                    conv, mode=args.context_mode, qa=qa
                )
                transcript = truncate_to_tokens(
                    transcript, args.max_context_tokens, tokenizer
                )

            # Build official per-category prompt (no system message)
            if args.context_mode == "cross_conv_distractor":
                user_prompt = build_qa_prompt_distractor(
                    context=transcript,
                    question=question,
                    category=cat,
                    speaker_a=conv["speaker_a"],
                    speaker_b=conv["speaker_b"],
                    adversarial_answer=adversarial_answer,
                )
            else:
                user_prompt = build_qa_prompt(
                    context=transcript,
                    question=question,
                    category=cat,
                    adversarial_answer=adversarial_answer,
                )

            try:
                # No system prompt — single user message, matching official baseline
                pred = call_chat(
                    client,
                    args.model,
                    system="",        # empty system prompt
                    user=user_prompt,
                    max_tokens=args.max_new_tokens,
                    temperature=0.0,
                )
            except Exception as e:  # noqa: BLE001
                print(f"[err] {qid}: {e}")
                pred = ""

            rec = {
                "id": qid,
                "sample_id": sid,
                "category": cat,
                "category_name": CATEGORY_NAMES.get(cat, "unknown"),
                "question": question,
                "gold": gold,
                "pred": pred,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            total_qa += 1
            if total_qa % 25 == 0:
                print(f"[run] answered {total_qa} QA so far  (last category={cat})")
    fout.close()
    print(f"[run] finished. predictions at {pred_path}")

    # ---------- score ----------
    score(pred_path, out_dir, args, client, tokenizer)


def score(
    pred_path: Path,
    out_dir: Path,
    args: argparse.Namespace,
    client: OpenAI,
    tokenizer,
) -> None:
    records = [json.loads(l) for l in pred_path.open()]
    print(f"[score] scoring {len(records)} predictions")

    # F1 / EM / BLEU-1
    by_cat: dict[str, list[dict[str, float]]] = {}
    for r in records:
        em = exact_match(r["pred"], r["gold"])
        f1 = f1_score(r["pred"], r["gold"])
        b1 = bleu1_score(r["pred"], r["gold"])
        r["em"] = em
        r["f1"] = f1
        r["bleu1"] = b1
        by_cat.setdefault(r["category_name"], []).append(
            {"em": em, "f1": f1, "bleu1": b1}
        )

    # Optional LLM-as-judge
    if args.use_llm_judge:
        print("[judge] running LLM-as-Judge ...")
        for r in tqdm(records):
            user = JUDGE_USER_TEMPLATE.format(
                question=r["question"], gold=r["gold"], pred=r["pred"]
            )
            try:
                ans = call_chat(
                    client,
                    args.model,
                    JUDGE_SYSTEM_PROMPT,
                    user,
                    max_tokens=4,
                    temperature=0.0,
                )
            except Exception as e:  # noqa: BLE001
                print(f"[judge-err] {e}")
                ans = ""
            ok = ans.strip().upper().startswith("Y")
            r["judge_ok"] = float(ok)
        # rebuild by_cat with judge_ok included
        by_cat = {}
        for r in records:
            by_cat.setdefault(r["category_name"], []).append(
                {"em": r["em"], "f1": r["f1"], "bleu1": r["bleu1"],
                 "judge_ok": r.get("judge_ok", 0.0)}
            )

    # aggregate
    overall = {"em": 0.0, "f1": 0.0, "bleu1": 0.0, "judge_ok": 0.0, "n": 0}
    cat_summary: dict[str, dict[str, float]] = {}
    for cat, lst in by_cat.items():
        n = len(lst)
        if n == 0:
            continue
        em = sum(x["em"] for x in lst) / n
        f1 = sum(x["f1"] for x in lst) / n
        b1 = sum(x["bleu1"] for x in lst) / n
        d = {"n": n, "em": round(em, 4), "f1": round(f1, 4),
             "bleu1": round(b1, 4)}
        if args.use_llm_judge:
            d["judge_acc"] = round(sum(x.get("judge_ok", 0.0) for x in lst) / n, 4)
        cat_summary[cat] = d
        overall["em"] += em * n
        overall["f1"] += f1 * n
        overall["bleu1"] += b1 * n
        if args.use_llm_judge:
            overall["judge_ok"] += sum(x.get("judge_ok", 0.0) for x in lst)
        overall["n"] += n

    if overall["n"]:
        overall["em"] = round(overall["em"] / overall["n"], 4)
        overall["f1"] = round(overall["f1"] / overall["n"], 4)
        overall["bleu1"] = round(overall["bleu1"] / overall["n"], 4)
        if args.use_llm_judge:
            overall["judge_acc"] = round(overall["judge_ok"] / overall["n"], 4)
        del overall["judge_ok"]

    summary = {
        "model": args.model,
        "num_predictions": len(records),
        "overall": overall,
        "per_category": cat_summary,
    }

    sum_path = out_dir / f"{args.model.replace('/', '_')}_summary.json"
    sum_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    scored_path = out_dir / f"{args.model.replace('/', '_')}_scored.jsonl"
    with scored_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n========== LoCoMo QA results ==========")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[saved] {sum_path}\n[saved] {scored_path}")


# -------------------------------------------------------------------- #
# CLI
# -------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--base_url", default="http://localhost:8000/v1")
    p.add_argument("--api_key", default="EMPTY",
                   help="vLLM ignores the key but openai-python requires one")
    p.add_argument("--data_file", default="data/locomo10.json")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--max_context_tokens", type=int, default=28000,
                   help="Truncate the conversation transcript to at most "
                        "this many tokens. Leave headroom under the model's "
                        "max-model-len for the prompt template + answer.")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--num_samples", type=int, default=0,
                   help="Only run on the first N conversations (debug). "
                        "0 = all.")
    p.add_argument("--category", type=int, default=0,
                   help="Only score a single category 1..5. 0 = all.")
    p.add_argument(
        "--context_mode",
        default="full",
        choices=["full", "evidence_mask", "session_cutoff", "cross_conv_distractor"],
        help=(
            "How to build the conversation context for each QA.\n"
            "  full                   – entire target conversation (default)\n"
            "  evidence_mask          – full but evidence turns are [REDACTED]\n"
            "  session_cutoff         – only sessions before the answer session\n"
            "  cross_conv_distractor  – ALL sessions from ALL 10 conversations;\n"
            "                           distractors first (in dataset order),\n"
            "                           target last. Front-truncated to fit\n"
            "                           --max_context_tokens so target is\n"
            "                           always fully preserved."
        ),
    )
    p.add_argument("--skip_adversarial", action="store_true",
                   help="Skip category 5 (adversarial). On by default in our "
                        "bash launcher because the 'No information available' "
                        "gold breaks F1/EM in a misleading way for small models.")
    p.add_argument("--include_no_evidence", action="store_true",
                   help="Also evaluate QA pairs that have no 'evidence' annotation. "
                        "The official baseline SKIPS these (default). Only enable "
                        "if you want to score all questions regardless.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-run even if predictions file exists.")
    p.add_argument("--use_llm_judge", action="store_true",
                   help="Also compute LLM-as-Judge accuracy using the same "
                        "vLLM server.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.overwrite:
        out_path = Path(args.output_dir) / f"{args.model.replace('/', '_')}_predictions.jsonl"
        if out_path.exists():
            out_path.unlink()
            print(f"[reset] removed {out_path}")
    evaluate(args)