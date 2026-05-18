"""
Run intra-session evaluation restricted to the held-out test split produced
by build_dataset.py. This guarantees we report numbers on the same 1307
questions before vs after self-distillation training.

It works by loading the test split, grouping its (sample_id, qa_idx) pairs
by sample, then patching each LoCoMoSample to keep only those QA pairs
before feeding it to the existing LoCoMoEvaluator.

Usage:
  # Baseline (no fine-tuning, current LLM_MODEL in config.py)
  python train/eval_on_split.py --result-file train/results/baseline.json

  # After fine-tuning: point config.LLM_MODEL at your distilled model first,
  # OR pass --model-override
  python train/eval_on_split.py \
      --model-override qwen25-3b-distill \
      --result-file train/results/distilled.json
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

# allow imports from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.dataset import LoCoMoSample, load_locomo
from eval.run_eval import LoCoMoEvaluator
from main import SimpleMemSystem


def load_split_keys(path: str) -> Dict[str, Set[int]]:
    """Return {sample_id: {qa_idx, ...}} from a split JSONL file."""
    keys: Dict[str, Set[int]] = defaultdict(set)
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            keys[str(row["sample_id"])].add(int(row["qa_idx"]))
    return keys


def filter_samples(
    samples: List[LoCoMoSample], keys: Dict[str, Set[int]]
) -> List[LoCoMoSample]:
    """Keep only samples that appear in the split, restricted to those qa_idx."""
    out: List[LoCoMoSample] = []
    for s in samples:
        if s.sample_id not in keys:
            continue
        wanted = sorted(keys[s.sample_id])
        s.qa = [s.qa[i] for i in wanted if i < len(s.qa)]
        if s.qa:
            out.append(s)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/locomo10_full.json")
    p.add_argument("--split-file", default="train/data/test.jsonl")
    p.add_argument("--result-file", default="train/results/eval.json")
    p.add_argument("--llm-judge", action="store_true")
    p.add_argument("--parallel-questions", action="store_true")
    p.add_argument("--test-workers", type=int, default=None)
    p.add_argument("--model-override", default=None,
                   help="Override config.LLM_MODEL for this run.")
    p.add_argument("--base-url-override", default=None,
                   help="Override config.OPENAI_BASE_URL for this run.")
    args = p.parse_args()

    Path(args.result_file).parent.mkdir(parents=True, exist_ok=True)

    keys = load_split_keys(args.split_file)
    print(f"Split keys: {sum(len(v) for v in keys.values())} questions across {len(keys)} samples")

    # Build system (optionally overriding model)
    system = SimpleMemSystem(
        clear_db=True,
        model=args.model_override,
        base_url=args.base_url_override,
    )

    # Hot-patch the evaluator to load filtered samples
    evaluator = LoCoMoEvaluator(
        system=system,
        dataset_path=args.dataset,
        use_llm_judge=args.llm_judge,
        test_workers=args.test_workers,
    )

    # Mirror evaluator.run() but with sample filtering
    print("\nLoading and filtering samples...")
    samples = load_locomo(args.dataset)
    samples = filter_samples(samples, keys)
    print(
        f"Evaluating {len(samples)} samples, "
        f"{sum(len(s.qa) for s in samples)} questions"
    )

    all_results = []
    for idx, sample in enumerate(samples):
        evaluator.system.vector_store.clear()
        results = evaluator._evaluate_sample(sample, idx, args.parallel_questions)
        all_results.extend(results)

    evaluator._print_summary()

    from eval.metrics import aggregate_metrics
    aggregated = aggregate_metrics(evaluator.all_metrics, evaluator.all_categories)
    output = {
        "summary": {
            "split_file": args.split_file,
            "num_samples": len(samples),
            "num_questions": len(all_results),
            "model": args.model_override or "<config.LLM_MODEL>",
            "avg_retrieval_time": (
                sum(evaluator.retrieval_times) / len(evaluator.retrieval_times)
                if evaluator.retrieval_times else 0
            ),
            "avg_answer_time": (
                sum(evaluator.answer_times) / len(evaluator.answer_times)
                if evaluator.answer_times else 0
            ),
        },
        "aggregated_metrics": aggregated,
        "detailed_results": all_results,
    }
    with open(args.result_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.result_file}")


if __name__ == "__main__":
    main()
