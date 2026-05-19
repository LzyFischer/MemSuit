"""
Granularity Ablation — sweep WINDOW_SIZE while forcing the memory builder
to emit exactly ONE memory entry per window.
=========================================================================

Pipeline (identical to the main LoCoMo eval):

    add_dialogues  →  MemoryBuilder (sliding window, single-entry prompt)
                  →  VectorStore (semantic index)
                  →  HybridRetriever (intent-aware retrieval)
                  →  AnswerGenerator
                  →  metrics (compute_metrics / aggregate_metrics)

Only TWO things change between groups:
  - WINDOW_SIZE             T ∈ {1, 5, 10, 20}
  - MemoryBuilder.single_entry_mode = True  (swaps the extraction prompt)

Everything else — retrieval, answering, scoring — is unchanged.

Usage
-----
    # smoke test (1 LoCoMo sample, all four T's)
    python eval/run_granularity_ablation.py --num-samples 1

    # full run on the full benchmark
    python eval/run_granularity_ablation.py

    # custom sweep
    python eval/run_granularity_ablation.py --turn-counts 1 5 10 20

    # re-print summary from existing JSON files
    python eval/run_granularity_ablation.py --table-only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# allow imports from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from eval.metrics import aggregate_metrics
from eval.run_eval import LoCoMoEvaluator
from main import SimpleMemSystem


DEFAULT_TURN_COUNTS: Tuple[int, ...] = (1, 5, 10, 20)


# ─────────────────────────────────────────────────────────────────────────────
# Single-T runner
# ─────────────────────────────────────────────────────────────────────────────

def run_one_turn_count(
    turn_count:         int,
    dataset_path:       str,
    num_samples:        int | None,
    out_dir:            Path,
    use_llm_judge:      bool,
    parallel_questions: bool,
    test_workers:       int | None,
    db_path:            str | None,
) -> Dict[str, Any]:
    """
    Stand up a fresh SimpleMemSystem with WINDOW_SIZE=turn_count and
    single_entry_mode=True, run the standard LoCoMo evaluation, dump
    eval_T{n}.json into out_dir.
    """
    print("\n" + "#" * 70)
    print(f"#  Granularity ablation — T = {turn_count}  "
          f"(single_entry_mode=True)")
    print("#" * 70)

    # Per-T table inside lancedb_data so concurrent groups don't collide if
    # the user later wants to parallelize externally. Same db_path is fine
    # because LoCoMoEvaluator clears the table per sample anyway.
    system = SimpleMemSystem(
        clear_db=True,
        window_size=turn_count,
        # For T=1 we can't have overlap; force it to 0.  For T>=2 fall back
        # to whatever config.py specifies (default 2).
        overlap_size=0 if turn_count <= 1 else None,
        single_entry_mode=True,
        db_path=db_path,
        table_name=f"granularity_ablation_T{turn_count}",
    )

    out_file = out_dir / f"eval_T{turn_count}.json"
    evaluator = LoCoMoEvaluator(
        system=system,
        dataset_path=dataset_path,
        use_llm_judge=use_llm_judge,
        test_workers=test_workers,
    )
    evaluator.run(
        num_samples=num_samples,
        result_file=str(out_file),
        save_results=True,
        parallel_questions=parallel_questions,
    )

    # The evaluator already writes a JSON, but we also want a compact
    # summary that the plot script can read directly.
    aggregated = aggregate_metrics(evaluator.all_metrics, evaluator.all_categories)
    summary = {
        "turn_count":     turn_count,
        "num_questions":  len(evaluator.all_metrics),
        "model":          system.llm_client.model,
        "aggregated":     aggregated,
        "avg_retrieval_time": (
            sum(evaluator.retrieval_times) / len(evaluator.retrieval_times)
            if evaluator.retrieval_times else 0.0
        ),
        "avg_answer_time": (
            sum(evaluator.answer_times) / len(evaluator.answer_times)
            if evaluator.answer_times else 0.0
        ),
    }
    (out_dir / f"summary_T{turn_count}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Combine per-T summaries into the bar-plot-ready file
# ─────────────────────────────────────────────────────────────────────────────

def build_combined_summary(
    out_dir: Path,
    turn_counts: Tuple[int, ...],
    metric_keys: Tuple[str, ...] = ("f1", "bleu1", "rougeL_f", "bert_f1",
                                    "sbert", "llm_judge"),
) -> Dict[str, Any]:
    """
    Walk summary_T{n}.json files and reshape them into:
        {
          "T_1":  { "overall": { f1: {mean,std,n}, ... },
                    "cat_1":   { f1: {...}, ... }, ... },
          "T_5":  { ... },
          ...
        }
    Same shape as plot_prelim1.py / the prelim study summaries so the bar
    plotter is dead simple.
    """
    combined: Dict[str, Any] = {}
    for t in turn_counts:
        sp = out_dir / f"summary_T{t}.json"
        if not sp.exists():
            print(f"  [warn] missing {sp.name} — skipping T={t}")
            continue
        agg = json.loads(sp.read_text()).get("aggregated", {})
        block: Dict[str, Dict[str, Any]] = {}
        for region in ("overall",
                       *(k for k in agg if k.startswith("cat_"))):
            r = agg.get(region, {})
            block[region] = {}
            for mk in metric_keys:
                if mk in r:
                    block[region][mk] = {
                        "mean":   r[mk]["mean"],
                        "std":    r[mk]["std"],
                        "median": r[mk].get("median", 0.0),
                        "n":      r[mk]["n"],
                    }
        combined[f"T_{t}"] = block
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Table printer
# ─────────────────────────────────────────────────────────────────────────────

def print_table(combined: Dict[str, Any], turn_counts: Tuple[int, ...]) -> None:
    print("\n" + "=" * 70)
    print("Granularity Ablation — F1 (token-level) on LoCoMo")
    print("=" * 70)
    print(f"  {'T':>4}  {'n':>5}  {'mean F1':>8}  {'std':>6}")
    print("  " + "-" * 32)
    for t in turn_counts:
        block = combined.get(f"T_{t}")
        if not block or "f1" not in block.get("overall", {}):
            print(f"  {t:>4}  {'-':>5}  {'-':>8}  {'-':>6}")
            continue
        ov = block["overall"]["f1"]
        print(f"  {t:>4}  {ov['n']:>5d}  {ov['mean']:>8.3f}  {ov['std']:>6.3f}")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset",     default="data/locomo10.json")
    p.add_argument("--out-dir",     default="results/granularity_ablation")
    p.add_argument("--turn-counts", type=int, nargs="+",
                   default=list(DEFAULT_TURN_COUNTS),
                   help="Window sizes (turns) to ablate over")
    p.add_argument("--num-samples", type=int, default=None,
                   help="Limit LoCoMo samples (default = full dataset)")
    p.add_argument("--llm-judge",   action="store_true",
                   help="Also score with LLM-as-judge")
    p.add_argument("--parallel-questions", action="store_true",
                   help="Parallelize QA within each sample")
    p.add_argument("--test-workers", type=int, default=None)
    p.add_argument("--db-path",      default=None,
                   help="LanceDB path (defaults to config.LANCEDB_PATH)")
    p.add_argument("--table-only",   action="store_true",
                   help="Skip eval, rebuild combined summary + print table")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    turn_counts: Tuple[int, ...] = tuple(args.turn_counts)

    if not args.table_only:
        for t in turn_counts:
            done = out_dir / f"summary_T{t}.json"
            if done.exists():
                print(f"[T={t}] {done.name} already exists — skipping.  "
                      f"Delete the file to re-run.")
                continue
            try:
                run_one_turn_count(
                    turn_count         = t,
                    dataset_path       = args.dataset,
                    num_samples        = args.num_samples,
                    out_dir            = out_dir,
                    use_llm_judge      = args.llm_judge,
                    parallel_questions = args.parallel_questions,
                    test_workers       = args.test_workers,
                    db_path            = args.db_path,
                )
            except Exception as e:
                print(f"[T={t}] FAILED: {e}")
                import traceback; traceback.print_exc()

    combined = build_combined_summary(out_dir, turn_counts)
    combined_path = out_dir / "summary_combined.json"
    combined_path.write_text(json.dumps(
        {
            "config": {
                "dataset":     args.dataset,
                "turn_counts": list(turn_counts),
                "num_samples": args.num_samples,
            },
            "summary": combined,
        },
        indent=2, ensure_ascii=False,
    ))
    print(f"\nCombined summary written to {combined_path.resolve()}")
    print_table(combined, turn_counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
