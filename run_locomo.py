"""
Run LoCoMo QA benchmark with a local open-source model served via vLLM
(OpenAI-compatible API). Uses the ORIGINAL snap-research/locomo prompts
verbatim, in the "Base" setting (whole conversation in context, truncated
from the front if it exceeds the model's window).

Reference repo:    https://github.com/snap-research/locomo
Reference paper:   https://arxiv.org/abs/2402.17753

Usage (single command, end-to-end)
----------------------------------
1. In one terminal start vLLM:

   pip install "vllm>=0.6.0"
   python -m vllm.entrypoints.openai.api_server \
       --model Qwen/Qwen2.5-3B-Instruct \
       --port 8000 \
       --max-model-len 32768 \
       --gpu-memory-utilization 0.9

2. In another terminal (this script):

   pip install openai tqdm scikit-learn
   python run_locomo.py \
       --data data/locomo10.json \
       --model Qwen/Qwen2.5-3B-Instruct \
       --base-url http://localhost:8000/v1 \
       --out-file outputs/qwen25_3b_locomo.json \
       --max-context-tokens 30000

It will write per-question predictions to --out-file and print a
per-category F1 summary at the end.

The dataset file (data/locomo10.json) must be downloaded from
https://github.com/snap-research/locomo/blob/main/data/locomo10.json
(CC BY-NC 4.0).
"""

import argparse
import collections
import json
import os
import re
import string
import sys
import time
from typing import Any, Dict, List, Tuple

from openai import OpenAI
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Prompts. Lifted from snap-research/locomo task_eval / paper Appendix C.
# I have kept them in their original short, GPT-tuned form on purpose so the
# evaluation is a faithful reproduction of the paper's "Base LLM" setting.
# -----------------------------------------------------------------------------

# Category-conditioned answer instructions, mirroring the original repo
# (single-hop / multi-hop / temporal / open-domain / adversarial).
# 1: single-hop, 2: multi-hop, 3: temporal, 4: open-domain, 5: adversarial
ANSWER_PROMPT = {
    1: (
        "Based on the above conversations, write short answers for each of the "
        "following questions in a few words. Write the answers in the form of a "
        "short phrase for each question. Answer with exact words from the "
        "conversations whenever possible."
    ),
    2: (
        "Based on the above conversations, write short answers for each of the "
        "following questions in a few words. Write the answers in the form of a "
        "short phrase for each question. Answer with exact words from the "
        "conversations whenever possible."
    ),
    3: (
        "Based on the above conversations, write short answers for each of the "
        "following questions using DATE of CONVERSATION for reference. Write the "
        "answer in the form of a short phrase. The answers need to be "
        "grounded in the dates of the conversations. Answer with exact words "
        "from the conversations whenever possible."
    ),
    4: (
        "Based on the above conversations, answer the following question. Use "
        "DATE of CONVERSATION to answer with an approximate date. Answer with "
        "exact words from the conversation whenever possible."
    ),
    5: (
        "Based on the above conversations, answer the following question. "
        "Write the answer as \"Not mentioned in the conversation\" if the "
        "information is not present in the conversation. Otherwise write a "
        "short phrase as the answer."
    ),
}

# Default for any unknown category. Same as single-hop.
DEFAULT_ANSWER_PROMPT = ANSWER_PROMPT[1]


# -----------------------------------------------------------------------------
# Conversation formatting (mirrors how the snap-research/locomo Base setting
# stringifies a multi-session dialogue: each session prefaced with its date,
# each turn as "<speaker> said, \"<text>\"". Images appear as their BLIP-2
# caption since the QA/event-summarization tasks are run on text-only inputs.)
# -----------------------------------------------------------------------------

def format_conversation(sample: Dict[str, Any]) -> str:
    """Flatten a LoCoMo conversation dict into a single string prompt."""
    conv = sample["conversation"]
    speaker_a = conv.get("speaker_a", "Speaker A")
    speaker_b = conv.get("speaker_b", "Speaker B")

    # Sessions are keyed as session_1, session_2, ... in chronological order.
    session_keys = sorted(
        [k for k in conv.keys() if re.fullmatch(r"session_\d+", k)],
        key=lambda x: int(x.split("_")[1]),
    )

    parts: List[str] = []
    parts.append(f"The following is a conversation between {speaker_a} and {speaker_b}.\n")
    for sk in session_keys:
        dt_key = f"{sk}_date_time"
        date_str = conv.get(dt_key, "")
        parts.append(f"\nDATE: {date_str}")
        parts.append(f"CONVERSATION:")
        for turn in conv[sk]:
            speaker = turn.get("speaker", "")
            text = turn.get("text", "")
            # If the turn has an image, append the BLIP caption as in the paper.
            caption = turn.get("blip_caption") or turn.get("caption")
            if caption:
                text = f'{text} [shares a photo of {caption}]'
            parts.append(f'{speaker} said, "{text}"')
    return "\n".join(parts)


# -----------------------------------------------------------------------------
# Context truncation: if the formatted conversation is too long for the model's
# window, the original paper truncates the EARLIEST sessions (keep the most
# recent context). We do the same. We use a simple character-based budget
# (~4 chars / token) as a cheap proxy and let vLLM error out if we still go
# over — the user can lower --max-context-tokens.
# -----------------------------------------------------------------------------

def truncate_context(context: str, max_chars: int) -> str:
    if len(context) <= max_chars:
        return context
    # Drop characters from the front (oldest sessions), keep the tail.
    cut = len(context) - max_chars
    truncated = context[cut:]
    # Try to align to the start of a "DATE:" line so a session isn't half-cut.
    m = re.search(r"\nDATE:", truncated)
    if m:
        truncated = truncated[m.start() + 1 :]
    return "[... earlier sessions truncated ...]\n" + truncated


# -----------------------------------------------------------------------------
# F1 scoring. Same normalization as SQuAD / the snap-research/locomo repo:
# lowercase, strip punctuation, drop articles, collapse whitespace.
# -----------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        # Both empty -> 1, one empty -> 0 (SQuAD convention).
        return float(pred_tokens == gt_tokens)
    common = collections.Counter(pred_tokens) & collections.Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def bleu1_score(prediction: str, ground_truth: str) -> float:
    """Unigram BLEU (BLEU-1) with brevity penalty.

    Matches the nltk sentence_bleu(weights=(1,0,0,0)) calculation used in the
    original locomo repo and the Mem0 evaluation paper.
    """
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens:
        return 0.0
    if not gt_tokens:
        return 0.0

    # Clipped unigram precision
    pred_counts = collections.Counter(pred_tokens)
    gt_counts = collections.Counter(gt_tokens)
    clipped = sum((pred_counts & gt_counts).values())
    precision = clipped / len(pred_tokens) if pred_tokens else 0.0

    # Brevity penalty: BP = 1 if len(pred) > len(ref), else exp(1 - ref/pred)
    if len(pred_tokens) >= len(gt_tokens):
        bp = 1.0
    else:
        import math
        bp = math.exp(1.0 - len(gt_tokens) / len(pred_tokens))

    return bp * precision


# -----------------------------------------------------------------------------
# Main eval loop.
# -----------------------------------------------------------------------------

def build_user_prompt(context: str, question: str, category: int) -> str:
    instr = ANSWER_PROMPT.get(category, DEFAULT_ANSWER_PROMPT)
    return (
        f"{context}\n\n"
        f"{instr}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )


def call_model(client: OpenAI, model: str, user_prompt: str,
               max_tokens: int = 128, temperature: float = 0.0) -> str:
    """One synchronous call to the OpenAI-compatible vLLM server."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1.0,
    )
    return (resp.choices[0].message.content or "").strip()


def run(args):
    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)

    # Resume support: if out_file already has predictions, skip those questions.
    done = {}
    if os.path.exists(args.out_file) and not args.overwrite:
        try:
            with open(args.out_file, "r", encoding="utf-8") as f:
                done = {entry["uid"]: entry for entry in json.load(f)}
            print(f"[resume] Found {len(done)} existing predictions, skipping those.")
        except Exception:
            done = {}

    results: List[Dict[str, Any]] = list(done.values())

    # Char budget. Approx 4 chars per token is a coarse but conservative estimate.
    char_budget = args.max_context_tokens * 4

    # The QA list lives at the top level of each sample as the "qa" key.
    # Each entry has: question, answer, category (int 1-5), evidence (list of dia_ids).
    # For category 5 (adversarial), some entries have "adversarial_answer" — we
    # treat the gold as "Not mentioned in the conversation" as per the original
    # eval (the model should refuse).
    all_qas: List[Tuple[int, Dict, Dict]] = []
    for sample_idx, sample in enumerate(data):
        for qa in sample.get("qa", []):
            all_qas.append((sample_idx, sample, qa))

    if args.max_samples is not None:
        all_qas = all_qas[: args.max_samples]

    print(f"[eval] {len(all_qas)} QA pairs across {len(data)} conversations.")

    # Cache formatted context per sample so we don't re-stringify N times.
    context_cache: Dict[int, str] = {}

    pbar = tqdm(all_qas, ncols=100)
    for sample_idx, sample, qa in pbar:
        uid = f"{sample.get('sample_id', sample_idx)}::{qa.get('question', '')[:80]}"
        if uid in done:
            continue

        if sample_idx not in context_cache:
            context_cache[sample_idx] = format_conversation(sample)
        context = truncate_context(context_cache[sample_idx], char_budget)

        category = int(qa.get("category", 1))
        question = qa["question"]
        gold = qa.get("answer", "")
        if category == 5 and not gold:
            gold = "Not mentioned in the conversation"

        user_prompt = build_user_prompt(context, question, category)

        try:
            pred = call_model(
                client,
                args.model,
                user_prompt,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        except Exception as e:
            pred = f"[error] {e}"

        f1 = f1_score(pred, str(gold))
        bleu1 = bleu1_score(pred, str(gold))
        entry = {
            "uid": uid,
            "sample_id": sample.get("sample_id", sample_idx),
            "category": category,
            "question": question,
            "gold": gold,
            "prediction": pred,
            "f1": f1,
            "bleu1": bleu1,
        }
        results.append(entry)

        # Save every N to be resilient to crashes.
        if len(results) % args.save_every == 0:
            with open(args.out_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        pbar.set_postfix(f1=f"{f1:.2f}", bleu1=f"{bleu1:.2f}", cat=category)

    # Final save.
    with open(args.out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ---------------------------------------------------------------------
    # Summary: per-category mean F1 / BLEU-1 and overall means.
    # ---------------------------------------------------------------------
    by_cat_f1:    Dict[int, List[float]] = collections.defaultdict(list)
    by_cat_bleu1: Dict[int, List[float]] = collections.defaultdict(list)
    for r in results:
        by_cat_f1[r["category"]].append(r["f1"])
        by_cat_bleu1[r["category"]].append(r.get("bleu1", 0.0))

    cat_names = {
        1: "single-hop",
        2: "multi-hop",
        3: "temporal",
        4: "open-domain",
        5: "adversarial",
    }
    print("\n=== LoCoMo QA results ===")
    print(f"  {'category':<22s}  {'n':>5s}  {'F1':>7s}  {'BLEU-1':>7s}")
    print(f"  {'-'*22}  {'-'*5}  {'-'*7}  {'-'*7}")
    overall_f1, overall_bleu1 = [], []
    for c in sorted(by_cat_f1):
        f1s    = by_cat_f1[c]
        bleu1s = by_cat_bleu1[c]
        mf1    = sum(f1s)    / max(1, len(f1s))
        mb1    = sum(bleu1s) / max(1, len(bleu1s))
        overall_f1.extend(f1s)
        overall_bleu1.extend(bleu1s)
        label = f"cat {c} ({cat_names.get(c, '?')})"
        print(f"  {label:<22s}  {len(f1s):>5d}  {mf1*100:>6.2f}%  {mb1*100:>6.2f}%")
    if overall_f1:
        print(f"  {'overall':<22s}  {len(overall_f1):>5d}  "
              f"{sum(overall_f1)/len(overall_f1)*100:>6.2f}%  "
              f"{sum(overall_bleu1)/len(overall_bleu1)*100:>6.2f}%")
    print(f"\nSaved predictions to {args.out_file}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/locomo10.json",
                   help="Path to the locomo10.json file.")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct",
                   help="Model name as registered with vLLM.")
    p.add_argument("--base-url", default="http://localhost:8000/v1",
                   help="OpenAI-compatible base URL of the vLLM server.")
    p.add_argument("--api-key", default="EMPTY",
                   help="API key (vLLM ignores it, but openai client requires one).")
    p.add_argument("--out-file", default="outputs/locomo_predictions.json")
    p.add_argument("--max-context-tokens", type=int, default=30000,
                   help="Approximate token budget for the conversation context. "
                        "Should be < vLLM --max-model-len minus generation room.")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-samples", type=int, default=None,
                   help="If set, only evaluate the first N QA pairs (for smoke test).")
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--overwrite", action="store_true",
                   help="Ignore existing out-file and re-run from scratch.")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()