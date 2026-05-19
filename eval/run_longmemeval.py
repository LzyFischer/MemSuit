"""
Run SimpleMem on LongMemEval (oracle / s / m).

Pipeline per question (same as `eval/run_eval.py`, but on LongMemEval
data and with LongMemEval-specific metrics + abstention handling):

  1. Clear vector store      (intra-history evaluation — fresh memory per Q)
  2. Feed all haystack sessions into SimpleMem
  3. Retrieve context for the question
  4. Generate an answer (abstention branch for *_abs ids)
  5. Score against the gold answer with the existing metric suite

Why is this not just `eval/run_eval.py`?
- LongMemEval's "speakers" are `user` / `assistant`, not named individuals.
- One sample = one question (vs. LoCoMo's ~140), so per-sample memory build
  cost dominates the wall clock. We process samples sequentially but allow
  larger memory-builder parallelism.
- Question types are different (single-session-user, multi-session,
  knowledge-update, temporal-reasoning, single-session-assistant,
  single-session-preference, abstention). We report per-type breakdowns.
- The benchmark ships an *official* LLM-judge eval script (uses GPT-4o).
  We also dump a `{question_id, hypothesis}` jsonl so you can run
  the official `evaluate_qa.py` against our predictions for the
  apples-to-apples comparison in your paper.

Usage
-----

  # Baseline
  python eval/run_longmemeval.py \
      --dataset data/longmemeval_oracle.json \
      --result-file results/lme_oracle_baseline.json \
      --hypothesis-file results/lme_oracle_baseline.hyp.jsonl

  # Distilled memory builder (LoCoMo-trained checkpoint, served via vLLM)
  python eval/run_longmemeval.py \
      --dataset data/longmemeval_oracle.json \
      --model-override qwen25-3b-distill \
      --result-file results/lme_oracle_distill.json \
      --hypothesis-file results/lme_oracle_distill.hyp.jsonl \
      --llm-judge --parallel-questions
"""
import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# allow imports from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.longmemeval_dataset import (
    ABSTENTION_CAT,
    CAT_TO_QTYPE,
    LongMemEvalSample,
    load_longmemeval,
)
from eval.metrics import aggregate_metrics, compute_metrics
from eval.run_eval import LoCoMoEvaluator, _cat5_answer
from main import SimpleMemSystem
from models.memory_entry import Dialogue


# ------------------------------------------------------------------
# LongMemEval-specific evaluator
# ------------------------------------------------------------------

class LongMemEvalEvaluator(LoCoMoEvaluator):
    """
    Inherits from LoCoMoEvaluator and overrides:
      - `_sample_to_dialogues`: route through user/assistant speakers
      - `_process_question`:    use abstention branch on *_abs question ids
      - `run`:                  dump official-format hypothesis jsonl, per-type
                                breakdowns, no random LoCoMo-isms
    """

    def __init__(self, *args, hypothesis_file: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.hypothesis_file = hypothesis_file
        # Track question_id ↔ prediction for the official-format dump
        self.hypotheses: List[Dict[str, str]] = []
        # Track question_type strings parallel to all_metrics for grouping
        self.all_qtypes: List[str] = []

    # ----- override: build dialogues from a LongMemEval sample -----

    def _sample_to_dialogues(self, sample: LongMemEvalSample) -> List[Dialogue]:
        dialogues = []
        did = 1
        for sid in sorted(sample.conversation.sessions):
            session = sample.conversation.sessions[sid]
            for turn in session.turns:
                dialogues.append(
                    Dialogue(
                        dialogue_id=did,
                        speaker=turn.speaker,            # "user" or "assistant"
                        content=turn.text,
                        timestamp=session.date_time,
                    )
                )
                did += 1
        return dialogues

    # ----- override: per-question processing with abstention branch -----

    def _process_question(
        self, sample: LongMemEvalSample, qa_idx: int
    ) -> Dict[str, Any]:
        qa = sample.qa[qa_idx]
        question = qa.question
        category = qa.category or 0
        qtype = qa.question_type or "unknown"
        ref = qa.final_answer or ""

        print(f"\n  [Q] type={qtype} cat={category}  abstain={qa.is_abstention}")
        print(f"      {question}")

        t0 = time.time()
        # Abstention: the model is supposed to say it doesn't know.
        # We disable reflection there (otherwise the retriever keeps
        # re-querying for a fact that doesn't exist).
        contexts = self.system.hybrid_retriever.retrieve(
            question,
            enable_reflection=(False if qa.is_abstention else None),
        )
        t_retrieval = time.time() - t0

        t1 = time.time()
        if qa.is_abstention:
            # Reuse LoCoMo cat-5 logic, but the "adversarial answer" is the
            # gold answer field of the abstention question (often a plausible-
            # but-unsupported claim). Falls back to "Unknown" if absent.
            adv = qa.answer or "Unknown"
            answer = _cat5_answer(question, contexts, adv, self.system)
        else:
            answer = self.system.answer_generator.generate_answer(question, contexts)
        t_answer = time.time() - t1

        metrics = compute_metrics(
            pred=answer,
            ref=ref,
            question=question,
            judge_client=self.judge_client,
            use_llm_judge=self.use_llm_judge,
        ) if ref else {}

        # accumulate
        self.retrieval_times.append(t_retrieval)
        self.answer_times.append(t_answer)
        if metrics:
            self.all_metrics.append(metrics)
            self.all_categories.append(category)
            self.all_qtypes.append(qtype)

        # Record for the official hypothesis dump
        if qa.question_id is not None:
            self.hypotheses.append({
                "question_id": qa.question_id,
                "hypothesis": answer,
            })

        print(f"      retrieved={len(contexts)}  "
              f"ret={t_retrieval:.2f}s  ans={t_answer:.2f}s")
        print(f"      answer: {answer}")
        print(f"      ref:    {ref}")
        if metrics:
            print(
                f"      F1={metrics.get('f1', 0):.3f}  "
                f"BLEU-1={metrics.get('bleu1', 0):.3f}  "
                f"ROUGE-L={metrics.get('rougeL_f', 0):.3f}  "
                f"BERT={metrics.get('bert_f1', 0):.3f}"
                + (f"  LLM-judge={metrics.get('llm_judge', 0):.3f}"
                   if self.use_llm_judge else "")
            )

        return {
            "question_id": qa.question_id,
            "question_type": qtype,
            "is_abstention": qa.is_abstention,
            "question": question,
            "answer": answer,
            "reference": ref,
            "category": category,
            "retrieval_time": t_retrieval,
            "answer_time": t_answer,
            "num_retrieved": len(contexts),
            "metrics": metrics,
        }

    # ----- override: full run with LongMemEval reporting + hyp dump -----

    def run(
        self,
        num_samples: Optional[int] = None,
        result_file: str = "lme_results.json",
        save_results: bool = True,
        parallel_questions: bool = False,
    ) -> List[Dict[str, Any]]:
        print("\n" + "=" * 70)
        print("  SimpleMem — LongMemEval evaluation")
        print("=" * 70)

        samples = load_longmemeval(self.dataset_path, limit=num_samples)
        all_results: List[Dict[str, Any]] = []

        t_eval_start = time.time()
        for idx, sample in enumerate(samples):
            self.system.vector_store.clear()
            results = self._evaluate_sample(sample, idx, parallel_questions)
            all_results.extend(results)

            # Persist hypotheses incrementally — these runs are long, and the
            # official evaluator only needs the jsonl, so we don't want to
            # lose them if the whole run crashes near the end.
            if self.hypothesis_file:
                self._dump_hypotheses(self.hypothesis_file)

        wall = time.time() - t_eval_start
        print(f"\nTotal eval wall time: {wall:.1f}s")

        self._print_summary_lme()

        if save_results:
            aggregated = aggregate_metrics(self.all_metrics, self.all_categories)
            # Add question-type-string-keyed aggregation as well (the int
            # categories are an internal artifact; reporting will lean on
            # qtypes).
            by_qtype = self._aggregate_by_qtype()
            output = {
                "summary": {
                    "dataset": str(self.dataset_path),
                    "num_samples": len(samples),
                    "num_questions": len(all_results),
                    "wall_time_sec": wall,
                    "avg_retrieval_time": (
                        sum(self.retrieval_times) / len(self.retrieval_times)
                        if self.retrieval_times else 0
                    ),
                    "avg_answer_time": (
                        sum(self.answer_times) / len(self.answer_times)
                        if self.answer_times else 0
                    ),
                },
                "aggregated_metrics": aggregated,
                "aggregated_by_question_type": by_qtype,
                "detailed_results": all_results,
            }
            Path(result_file).parent.mkdir(parents=True, exist_ok=True)
            with open(result_file, "w") as f:
                json.dump(output, f, indent=2)
            print(f"\nDetailed results saved to {result_file}")

        if self.hypothesis_file:
            self._dump_hypotheses(self.hypothesis_file)
            print(f"Official-format hypotheses saved to {self.hypothesis_file}")
            print("\nTo run the official LongMemEval evaluator (GPT-4o judge):")
            print("  cd <LongMemEval repo>/src/evaluation")
            print(f"  python evaluate_qa.py gpt-4o {self.hypothesis_file} \\")
            print(f"      {self.dataset_path}")

        return all_results

    # ----- helpers -----

    def _aggregate_by_qtype(self) -> Dict[str, Any]:
        """Same stats as aggregate_metrics, but grouped by the string question_type."""
        buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        cats: Dict[str, List[int]] = defaultdict(list)
        for m, cat, qt in zip(self.all_metrics, self.all_categories, self.all_qtypes):
            buckets[qt].append(m)
            cats[qt].append(cat)
        out: Dict[str, Any] = {}
        for qt, ms in buckets.items():
            agg = aggregate_metrics(ms, cats[qt])
            out[qt] = agg.get("overall", {})
        return out

    def _dump_hypotheses(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for row in self.hypotheses:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _print_summary_lme(self) -> None:
        print("\n" + "=" * 70)
        print("  Summary")
        print("=" * 70)

        if self.retrieval_times:
            avg_ret = sum(self.retrieval_times) / len(self.retrieval_times)
            avg_ans = sum(self.answer_times) / len(self.answer_times)
            print(f"Avg retrieval time : {avg_ret:.3f}s")
            print(f"Avg answer time    : {avg_ans:.3f}s")

        if not self.all_metrics:
            print("(no metrics recorded)")
            return

        agg = aggregate_metrics(self.all_metrics, self.all_categories)
        overall = agg.get("overall", {})
        print("\nOverall metrics:")
        for key in ("f1", "bleu1", "rougeL_f", "bert_f1", "sbert", "llm_judge"):
            if key in overall:
                s = overall[key]
                print(f"  {key:<14} {s['mean']:.4f} ± {s['std']:.4f}  (n={s['n']})")

        # Per-question-type breakdown (the metric of interest for the paper)
        by_qtype = self._aggregate_by_qtype()
        if by_qtype:
            print("\nPer question_type (LLM-judge preferred, else F1):")
            for qt in sorted(by_qtype):
                stats = by_qtype[qt]
                # Prefer the LLM-judge accuracy if available
                if "llm_judge" in stats and stats["llm_judge"]["n"] > 0:
                    s = stats["llm_judge"]
                    print(f"  {qt:<28s} judge={s['mean']*100:>6.2f}%  (n={s['n']})")
                elif "f1" in stats:
                    s = stats["f1"]
                    print(f"  {qt:<28s} F1   ={s['mean']*100:>6.2f}%  (n={s['n']})")
        print("=" * 70)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="SimpleMem LongMemEval evaluation")
    p.add_argument("--dataset", default="data/longmemeval_oracle.json",
                   help="Path to longmemeval_oracle / _s_cleaned / _m_cleaned .json")
    p.add_argument("--num-samples", type=int, default=None,
                   help="Limit the number of evaluated questions "
                        "(useful for smoke tests).")
    p.add_argument("--result-file", default="results/lme_results.json")
    p.add_argument("--hypothesis-file", default=None,
                   help="If set, dump a jsonl of "
                        "{question_id, hypothesis} compatible with "
                        "LongMemEval's official evaluate_qa.py.")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--llm-judge", action="store_true",
                   help="Run the in-repo LLM-as-judge (see config.JUDGE_*).")
    p.add_argument("--parallel-questions", action="store_true",
                   help="LongMemEval has 1 question per sample; this only "
                        "matters if you later batch multiple Qs together. "
                        "Kept for API symmetry with run_eval.py.")
    p.add_argument("--test-workers", type=int, default=None)
    p.add_argument("--model-override", default=None,
                   help="Override config.LLM_MODEL for this run (e.g. a "
                        "distilled LoCoMo checkpoint served via vLLM).")
    p.add_argument("--base-url-override", default=None,
                   help="Override config.OPENAI_BASE_URL for this run.")
    p.add_argument("--embedding-override", default=None,
                   help="Override config.EMBEDDING_MODEL for this run "
                        "(local path or HF id of a contrastive-finetuned "
                        "SentenceTransformer checkpoint from "
                        "train/train_contrastive.py).")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for the abstention A/B randomization in "
                        "_cat5_answer (set for reproducibility).")
    args = p.parse_args()

    random.seed(args.seed)

    system = SimpleMemSystem(
        clear_db=True,
        model=args.model_override,
        base_url=args.base_url_override,
        embedding_model_name=args.embedding_override,
    )

    evaluator = LongMemEvalEvaluator(
        system=system,
        dataset_path=args.dataset,
        use_llm_judge=args.llm_judge,
        test_workers=args.test_workers,
        hypothesis_file=args.hypothesis_file,
    )
    evaluator.run(
        num_samples=args.num_samples,
        result_file=args.result_file,
        save_results=not args.no_save,
        parallel_questions=args.parallel_questions,
    )


if __name__ == "__main__":
    main()
