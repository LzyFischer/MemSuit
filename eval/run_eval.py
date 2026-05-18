"""
Intra-session evaluation on LoCoMo.

Each sample is evaluated independently:
  1. Clear the vector store (start fresh = intra-session)
  2. Feed all session dialogues into SimpleMem
  3. Answer each QA question and score against the ground truth

Usage:
  python eval/run_eval.py                              # all samples
  python eval/run_eval.py --num-samples 5             # quick test
  python eval/run_eval.py --llm-judge                 # enable LLM judge
  python eval/run_eval.py --parallel-questions        # parallel QA within a sample
  python eval/run_eval.py --result-file results.json  # custom output path
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import pdb

# allow imports from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.dataset import LoCoMoSample, load_locomo
from eval.metrics import aggregate_metrics, compute_metrics
from main import SimpleMemSystem
from models.memory_entry import Dialogue
from utils.llm_client import LLMClient


# ------------------------------------------------------------------
# Adversarial answer generation (category 5)
# ------------------------------------------------------------------

def _cat5_answer(
    question: str,
    contexts: List[Any],
    adversarial_answer: str,
    system: SimpleMemSystem,
) -> str:
    """Binary choice: 'Not mentioned' vs the adversarial answer."""
    options = ["Not mentioned in the conversation", adversarial_answer]
    if random.random() < 0.5:
        options = options[::-1]

    ctx_str = system.answer_generator._format_contexts(contexts)
    prompt = f"""Based on the context, choose the correct answer.

Context:
{ctx_str}

Question: {question}

Option A: {options[0]}
Option B: {options[1]}

If the specific answer is not supported by the context, choose "Not mentioned in the conversation".

Return JSON:
```json
{{"reasoning": "brief", "answer": "chosen option text"}}
```
Return ONLY JSON."""

    messages = [
        {"role": "system", "content": "You are a Q&A assistant. Output valid JSON only."},
        {"role": "user", "content": prompt},
    ]
    for _ in range(3):
        try:
            import config
            response_format = (
                {"type": "json_object"} if getattr(config, "USE_JSON_FORMAT", False) else None
            )
            resp = system.llm_client.chat_completion(
                messages, temperature=0.5, response_format=response_format
            )
            return system.llm_client.extract_json(resp).get("answer", options[0])
        except Exception:
            pass
    return "Not mentioned in the conversation"


# ------------------------------------------------------------------
# Core evaluation class
# ------------------------------------------------------------------

class LoCoMoEvaluator:
    def __init__(
        self,
        system: SimpleMemSystem,
        dataset_path: str,
        use_llm_judge: bool = False,
        test_workers: Optional[int] = None,
    ):
        self.system = system
        self.dataset_path = dataset_path
        self.use_llm_judge = use_llm_judge
        self.test_workers = test_workers

        self.judge_client: Optional[LLMClient] = None
        if use_llm_judge:
            self.judge_client = self._make_judge_client()

        # Running stats
        self.retrieval_times: List[float] = []
        self.answer_times: List[float] = []
        self.all_metrics: List[Dict[str, Any]] = []
        self.all_categories: List[int] = []

    @staticmethod
    def _make_judge_client() -> LLMClient:
        import config
        api_key = getattr(config, "JUDGE_API_KEY", None) or config.OPENAI_API_KEY
        base_url = getattr(config, "JUDGE_BASE_URL", None) or getattr(config, "OPENAI_BASE_URL", None)
        model = getattr(config, "JUDGE_MODEL", None) or config.LLM_MODEL
        thinking = getattr(config, "JUDGE_ENABLE_THINKING", False)
        streaming = getattr(config, "JUDGE_USE_STREAMING", False)
        print(f"Judge model: {model}")
        return LLMClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            enable_thinking=thinking,
            use_streaming=streaming,
        )

    # ------------------------------------------------------------------

    def _sample_to_dialogues(self, sample: LoCoMoSample) -> List[Dialogue]:
        dialogues = []
        did = 1
        for sid in sorted(sample.conversation.sessions):
            session = sample.conversation.sessions[sid]
            for turn in session.turns:
                dialogues.append(
                    Dialogue(
                        dialogue_id=did,
                        speaker=turn.speaker,
                        content=turn.text,
                        timestamp=session.date_time,
                    )
                )
                did += 1
        return dialogues

    def _process_question(self, sample: LoCoMoSample, qa_idx: int) -> Dict[str, Any]:
        qa = sample.qa[qa_idx]
        question = qa.question
        category = qa.category or 0
        ref = "Not mentioned in the conversation" if category == 5 else qa.final_answer

        print(f"\n  [Q{qa_idx + 1}] cat={category}: {question}")

        t0 = time.time()
        contexts = self.system.hybrid_retriever.retrieve(
            question,
            enable_reflection=(False if category == 5 else None),
        )
        t_retrieval = time.time() - t0

        t1 = time.time()
        if category == 5:
            answer = _cat5_answer(
                question, contexts,
                qa.adversarial_answer or "Unknown",
                self.system,
            )
        else:
            answer = self.system.answer_generator.generate_answer(question, contexts)
        t_answer = time.time() - t1

        metrics = compute_metrics(
            pred=answer,
            ref=ref or "",
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

        print(f"    retrieved={len(contexts)}  ret={t_retrieval:.2f}s  ans={t_answer:.2f}s")
        print(f"    answer: {answer}")
        print(f"    ref:    {ref}")
        if metrics:
            print(
                f"    F1={metrics.get('f1', 0):.3f}  "
                f"BLEU-1={metrics.get('bleu1', 0):.3f}  "
                f"ROUGE-L={metrics.get('rougeL_f', 0):.3f}  "
                f"BERT={metrics.get('bert_f1', 0):.3f}"
                + (f"  LLM-judge={metrics.get('llm_judge', 0):.3f}" if self.use_llm_judge else "")
            )

        return {
            "question": question,
            "answer": answer,
            "reference": ref,
            "category": category,
            "retrieval_time": t_retrieval,
            "answer_time": t_answer,
            "num_retrieved": len(contexts),
            "metrics": metrics,
        }

    def _evaluate_sample(
        self, sample: LoCoMoSample, sample_idx: int, parallel_questions: bool
    ) -> List[Dict[str, Any]]:
        print(f"\n{'=' * 70}")
        print(f"Sample {sample_idx}  ({len(sample.qa)} questions)")
        print(f"{'=' * 70}")

        dialogues = self._sample_to_dialogues(sample)
        print(f"Building memory from {len(dialogues)} turns...")
        t0 = time.time()
        self.system.add_dialogues(dialogues)
        self.system.finalize()
        print(f"Memory built in {time.time() - t0:.1f}s")

        if parallel_questions and len(sample.qa) > 1:
            return self._parallel_questions(sample)
        return [self._process_question(sample, i) for i in range(len(sample.qa))]

    def _parallel_questions(self, sample: LoCoMoSample) -> List[Dict[str, Any]]:
        import concurrent.futures
        import config as cfg
        workers = min(
            self.test_workers or getattr(cfg, "MAX_RETRIEVAL_WORKERS", 8),
            len(sample.qa),
            20,
        )
        print(f"  [Parallel] {len(sample.qa)} questions, {workers} workers")
        results_map: Dict[int, Dict[str, Any]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(self._process_question, sample, i): i
                for i in range(len(sample.qa))
            }
            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                try:
                    results_map[i] = future.result()
                except Exception as e:
                    qa = sample.qa[i]
                    print(f"  [Parallel] Q{i + 1} failed: {e}")
                    results_map[i] = {
                        "question": qa.question,
                        "answer": "error",
                        "reference": qa.final_answer,
                        "category": qa.category or 0,
                        "retrieval_time": 0, "answer_time": 0,
                        "num_retrieved": 0, "metrics": {},
                    }
        return [results_map[i] for i in sorted(results_map)]

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(
        self,
        num_samples: Optional[int] = None,
        result_file: str = "eval_results.json",
        save_results: bool = True,
        parallel_questions: bool = False,
    ) -> List[Dict[str, Any]]:
        print("\n" + "=" * 70)
        print("  SimpleMem — LoCoMo Intra-Session Evaluation")
        print("=" * 70)

        samples = load_locomo(self.dataset_path, limit=num_samples)
        all_results: List[Dict[str, Any]] = []

        for idx, sample in enumerate(samples):
            # clear db for each sample → intra-session evaluation
            self.system.vector_store.clear()
            results = self._evaluate_sample(sample, idx, parallel_questions)
            all_results.extend(results)

        self._print_summary()

        if save_results:
            aggregated = aggregate_metrics(self.all_metrics, self.all_categories)
            output = {
                "summary": {
                    "num_samples": len(samples),
                    "num_questions": len(all_results),
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
                "detailed_results": all_results,
            }
            with open(result_file, "w") as f:
                json.dump(output, f, indent=2)
            print(f"\nResults saved to {result_file}")

        return all_results

    def _print_summary(self) -> None:
        print("\n" + "=" * 70)
        print("  Summary")
        print("=" * 70)
        if self.retrieval_times:
            print(f"Avg retrieval time : {sum(self.retrieval_times)/len(self.retrieval_times):.3f}s")
            print(f"Avg answer time    : {sum(self.answer_times)/len(self.answer_times):.3f}s")

        if self.all_metrics:
            agg = aggregate_metrics(self.all_metrics, self.all_categories)
            overall = agg.get("overall", {})
            print("\nOverall metrics:")
            for key in ("f1", "bleu1", "rougeL_f", "bert_f1", "sbert", "llm_judge"):
                if key in overall:
                    s = overall[key]
                    print(f"  {key:<14} {s['mean']:.4f} ± {s['std']:.4f}  (n={s['n']})")

            print("\nPer-category F1:")
            for k in sorted(agg):
                if k.startswith("cat_") and "f1" in agg[k]:
                    cat = k.split("_")[1]
                    s = agg[k]["f1"]
                    print(f"  cat {cat}: {s['mean']:.4f}  (n={s['n']})")
        print("=" * 70)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SimpleMem LoCoMo evaluation")
    parser.add_argument("--dataset", default="data/locomo10.json")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--result-file", default="eval_results.json")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--llm-judge", action="store_true")
    parser.add_argument("--parallel-questions", action="store_true")
    parser.add_argument("--test-workers", type=int, default=None)
    args = parser.parse_args()

    system = SimpleMemSystem(clear_db=True)
    evaluator = LoCoMoEvaluator(
        system=system,
        dataset_path=args.dataset,
        use_llm_judge=args.llm_judge,
        test_workers=args.test_workers,
    )
    evaluator.run(
        num_samples=args.num_samples,
        result_file=args.result_file,
        save_results=not args.no_save,
        parallel_questions=args.parallel_questions,
    )


if __name__ == "__main__":
    main()
