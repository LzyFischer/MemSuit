"""
Offline token-efficiency counter for the BUILD-MEMORY stage on LoCoMo.

This script replays exactly what `MemoryBuilder.add_dialogues(...)` would do
for every LoCoMo sample, but instead of calling the LLM it just counts the
tokens of every prompt that *would* be sent. No vLLM / no OpenAI call.

What is counted
---------------
For each sample (one LoCoMo conversation):
  1. Convert the sample to a Dialogue list (same as eval/run_eval.py).
  2. Slide a window of size WINDOW_SIZE over the dialogues with stride
     WINDOW_SIZE - OVERLAP_SIZE (identical to MemoryBuilder).
  3. For every window — including the trailing "remaining" window flushed
     by `process_remaining()` — build the *exact* extraction prompt that
     MemoryBuilder._extraction_prompt would produce, wrap it in the same
     system + user chat message MemoryBuilder._generate_entries uses, then
     apply the model's chat template and count the resulting tokens.

The output is the mean (and median/sum) of "build-memory input tokens
per conversation".

Notes
-----
- This counts INPUT tokens only. LLM completion tokens cannot be computed
  offline because they depend on what the LLM actually generates.
- The script imports MemoryBuilder verbatim and only calls its
  _extraction_prompt method, so no source files are modified.
- The "[Previous entries ...]" context block is included in MemoryBuilder
  only when previous_entries is non-empty. Offline we don't have real
  previous entries; by default we skip that block (which matches the very
  first window of every sample exactly and is a *small* underestimate for
  later windows: ~30-80 tokens per window depending on entry length).
  Pass --simulate-prev-context to instead pad each window after the first
  with a fixed placeholder of typical length, matching the structure
  MemoryBuilder uses at runtime more closely.

Usage
-----
    python count_build_memory_tokens.py \
        --data data/locomo10.json \
        --tokenizer Qwen/Qwen2.5-3B-Instruct \
        --window-size 20 --overlap-size 2 \
        --out-file build_memory_tokens.json
"""
import argparse
import json
import statistics
import sys
import types
from pathlib import Path
from typing import List

# Make repo root importable
sys.path.insert(0, str(Path(__file__).parent.resolve()))

# ----------------------------------------------------------------------
# Stub out heavy dependencies of the repo (lancedb, openai, sentence-
# transformers) so we can import MemoryBuilder without actually installing
# them. We never call any code from these modules — they only need to be
# importable so the chain database/__init__.py -> vector_store.py and
# utils/__init__.py -> llm_client.py doesn't crash.
# ----------------------------------------------------------------------
def _install_stub(name: str, attrs: dict | None = None):
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    for k, v in (attrs or {}).items():
        setattr(mod, k, v)
    sys.modules[name] = mod


class _AnyClass:
    """A class that accepts anything in init and ignores it."""
    def __init__(self, *a, **kw): pass
    def __getattr__(self, _): return _AnyClass()
    def __call__(self, *a, **kw): return _AnyClass()


# Top-level packages first
_install_stub("lancedb", {"connect": lambda *a, **kw: _AnyClass()})
_install_stub("lancedb.pydantic", {"LanceModel": object,
                                   "Vector": lambda *a, **kw: list})

_install_stub("openai", {"OpenAI": _AnyClass})

_install_stub("sentence_transformers", {"SentenceTransformer": _AnyClass})

_install_stub("pyarrow", {"schema": lambda *a, **kw: None,
                         "field": lambda *a, **kw: None,
                         "list_": lambda *a, **kw: None,
                         "string": lambda *a, **kw: None,
                         "float32": lambda *a, **kw: None,
                         "int32": lambda *a, **kw: None,
                         "int64": lambda *a, **kw: None,
                         "Table": _AnyClass})

# numpy may be missing in extreme environments — only stub if absent
try:
    import numpy  # noqa: F401
except ImportError:
    _install_stub("numpy", {"ndarray": object, "array": lambda *a, **kw: a})

from eval.dataset import load_locomo, LoCoMoSample
from models.memory_entry import Dialogue, MemoryEntry


# ----------------------------------------------------------------------
# Build a MemoryBuilder instance that we can call ._extraction_prompt on
# without ever touching an LLM or a vector store. We just need an object
# with the right attributes (single_entry_mode flag).
# ----------------------------------------------------------------------
def make_builder(window_size: int, overlap_size: int, single_entry_mode: bool):
    # Lazy import so config errors only surface here.
    from core.memory_builder import MemoryBuilder

    class _NullLLM:
        # MemoryBuilder.__init__ doesn't actually CALL the client; it only
        # stores it. So any object works.
        pass

    class _NullStore:
        # Same: stored, not called during _extraction_prompt.
        def add_entries(self, _entries):
            raise RuntimeError("Should not be called in token-counting mode")

    return MemoryBuilder(
        llm_client=_NullLLM(),
        vector_store=_NullStore(),
        window_size=window_size,
        overlap_size=overlap_size,
        enable_parallel_processing=False,
        single_entry_mode=single_entry_mode,
    )


# ----------------------------------------------------------------------
# Same conversion as eval/run_eval.py::_sample_to_dialogues — kept inline
# to avoid pulling in SimpleMemSystem (which would need an LLM).
# ----------------------------------------------------------------------
def sample_to_dialogues(sample: LoCoMoSample) -> List[Dialogue]:
    dialogues: List[Dialogue] = []
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


# ----------------------------------------------------------------------
# Window planner: same logic as MemoryBuilder.add_dialogues +
# process_remaining (sequential path).
# ----------------------------------------------------------------------
def plan_windows(dialogues: List[Dialogue], window_size: int, step_size: int):
    """Return list of window-dialogue-lists in the order MemoryBuilder
    would process them sequentially.
    """
    windows: List[List[Dialogue]] = []
    pos = 0
    while pos + window_size <= len(dialogues):
        windows.append(dialogues[pos: pos + window_size])
        pos += step_size
    # process_remaining(): the trailing partial window, if any.
    if pos < len(dialogues):
        windows.append(dialogues[pos:])
    return windows


# ----------------------------------------------------------------------
# Build the prompt and count tokens for one window.
# ----------------------------------------------------------------------
SYSTEM_MSG = (
    "You are a professional information extraction assistant. "
    "Extract structured, unambiguous facts from conversations. "
    "Output valid JSON only."
)

# A placeholder MemoryEntry list used when --simulate-prev-context is on.
# MemoryBuilder takes the most recent up to 3 entries and prepends each as
# "- <text>\n". Length here is a typical lossless_restatement (~25 words).
_PLACEHOLDER_PREV = [
    MemoryEntry(lossless_restatement=(
        "On 2023-01-01 at 12:00, Speaker A told Speaker B that they "
        "had finished reviewing the quarterly project documentation."
    ))
    for _ in range(3)
]


def count_window_tokens(builder, window, tokenizer, prev_entries):
    # Reproduce MemoryBuilder._generate_entries up to (but not including)
    # the LLM call.
    builder.previous_entries = prev_entries
    dialogue_text = "\n".join(str(d) for d in window)
    context = ""
    if builder.previous_entries:
        context = "\n[Previous entries — avoid duplication]\n"
        for e in builder.previous_entries[:3]:
            context += f"- {e.lossless_restatement}\n"
    user_prompt = builder._extraction_prompt(dialogue_text, context)

    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": user_prompt},
    ]
    # apply_chat_template gives the exact token sequence the LLM sees,
    # including any control / special tokens (Qwen2 ChatML headers etc.).
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    return len(ids)


_QWEN_CHATML_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


class _ChatMLTokenizerShim:
    """Wraps a bare tokenizers.Tokenizer so it has the bits we need:
       - apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
       - encode(text, add_special_tokens=False) -> list of ids

    This is only used when HuggingFace transformers is unavailable; the
    chat template is the Qwen2 ChatML template (matches Qwen2.5-Instruct
    models). For other model families pass --tokenizer with a path that
    transformers can load.
    """

    def __init__(self, tok):
        self._tok = tok

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        sys_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        usr_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        text = _QWEN_CHATML_TEMPLATE.format(system=sys_msg, user=usr_msg)
        return text  # raw string; caller will encode

    def encode(self, text, add_special_tokens=False):
        enc = self._tok.encode(text, add_special_tokens=add_special_tokens)
        return enc.ids


def _load_tokenizer(name_or_path: str):
    """Try transformers.AutoTokenizer first, fall back to the lightweight
    tokenizers package with a Qwen ChatML shim.
    """
    # Debug-only: whitespace-split "tokenizer" for offline smoke tests.
    if name_or_path == "__whitespace__":
        print("[tok] using WHITESPACE tokenizer (debug only, NOT a real token count)")

        class _WS:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                sys_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
                usr_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
                return _QWEN_CHATML_TEMPLATE.format(system=sys_msg, user=usr_msg)
            def encode(self, text, add_special_tokens=False):
                return text.split()
        return _WS()

    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
    except Exception as e:
        print(f"[tok] transformers unavailable ({type(e).__name__}: {e}); "
              f"falling back to tokenizers + Qwen ChatML shim.")
        from tokenizers import Tokenizer
        try:
            tok = Tokenizer.from_pretrained(name_or_path)
        except Exception as e2:
            raise RuntimeError(
                f"Could not load tokenizer {name_or_path!r} via transformers "
                f"or tokenizers. Original error: {e2}"
            ) from e2
        return _ChatMLTokenizerShim(tok)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/locomo10.json",
                   help="Path to locomo10.json")
    p.add_argument("--tokenizer", default="Qwen/Qwen2.5-3B-Instruct",
                   help="HuggingFace tokenizer name or path")
    p.add_argument("--window-size", type=int, default=None,
                   help="Override config.WINDOW_SIZE")
    p.add_argument("--overlap-size", type=int, default=None,
                   help="Override config.OVERLAP_SIZE")
    p.add_argument("--single-entry-mode", action="store_true",
                   help="Use the granularity-ablation single-entry prompt")
    p.add_argument("--simulate-prev-context", action="store_true",
                   help="Pad windows after the first with a fixed-length "
                        "placeholder for the '[Previous entries]' block. "
                        "Default: empty (small underestimate).")
    p.add_argument("--num-samples", type=int, default=None,
                   help="Only process first N conversations (debug)")
    p.add_argument("--out-file", default="build_memory_tokens.json")
    args = p.parse_args()

    # Resolve window / overlap from config defaults if not overridden
    import config as cfg
    window_size = args.window_size if args.window_size is not None else cfg.WINDOW_SIZE
    overlap_size = (
        args.overlap_size if args.overlap_size is not None
        else getattr(cfg, "OVERLAP_SIZE", 0)
    )
    overlap_size = max(0, min(overlap_size, window_size - 1))
    step_size = max(1, window_size - overlap_size)

    print(f"[cfg] window_size={window_size}  overlap_size={overlap_size}  step_size={step_size}")
    print(f"[cfg] single_entry_mode={args.single_entry_mode}  "
          f"simulate_prev_context={args.simulate_prev_context}")

    # Load tokenizer. Try HuggingFace transformers first (the "official"
    # path), then fall back to the lighter `tokenizers` package — which is
    # all we actually need here and doesn't pull in torch.
    print(f"[tok] loading tokenizer: {args.tokenizer}")
    tokenizer = _load_tokenizer(args.tokenizer)

    # Builder (no LLM, no DB)
    builder = make_builder(window_size, overlap_size, args.single_entry_mode)

    # Load dataset
    samples = load_locomo(args.data, limit=args.num_samples)

    per_sample = []
    all_window_tokens = []
    for idx, sample in enumerate(samples):
        dialogues = sample_to_dialogues(sample)
        windows = plan_windows(dialogues, window_size, step_size)

        sample_total = 0
        window_token_counts = []
        for w_idx, window in enumerate(windows):
            prev = []
            if args.simulate_prev_context and w_idx > 0:
                prev = _PLACEHOLDER_PREV
            n_tok = count_window_tokens(builder, window, tokenizer, prev)
            window_token_counts.append(n_tok)
            sample_total += n_tok
            all_window_tokens.append(n_tok)

        per_sample.append({
            "sample_id": sample.sample_id,
            "num_dialogues": len(dialogues),
            "num_windows": len(windows),
            "total_input_tokens": sample_total,
            "mean_tokens_per_window": (
                sample_total / len(windows) if windows else 0
            ),
            "window_token_counts": window_token_counts,
        })
        print(f"[{idx+1:>3}/{len(samples)}] sample={sample.sample_id}  "
              f"dialogues={len(dialogues):>4}  windows={len(windows):>3}  "
              f"input_tokens={sample_total:>7}")

    # Aggregate
    totals = [s["total_input_tokens"] for s in per_sample]
    summary = {
        "config": {
            "window_size": window_size,
            "overlap_size": overlap_size,
            "step_size": step_size,
            "single_entry_mode": args.single_entry_mode,
            "simulate_prev_context": args.simulate_prev_context,
            "tokenizer": args.tokenizer,
            "counts": "input prompt tokens only (no completion tokens)",
        },
        "num_conversations": len(per_sample),
        "total_windows": sum(s["num_windows"] for s in per_sample),
        "mean_input_tokens_per_conversation": (
            statistics.mean(totals) if totals else 0
        ),
        "median_input_tokens_per_conversation": (
            statistics.median(totals) if totals else 0
        ),
        "stdev_input_tokens_per_conversation": (
            statistics.stdev(totals) if len(totals) > 1 else 0
        ),
        "min_input_tokens_per_conversation": min(totals) if totals else 0,
        "max_input_tokens_per_conversation": max(totals) if totals else 0,
        "sum_input_tokens_all_conversations": sum(totals),
        "mean_input_tokens_per_window": (
            statistics.mean(all_window_tokens) if all_window_tokens else 0
        ),
    }

    out = {"summary": summary, "per_sample": per_sample}
    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Pretty print
    print("\n" + "=" * 70)
    print(" Build-Memory Token Efficiency (LoCoMo, offline)")
    print("=" * 70)
    print(f"  conversations         : {summary['num_conversations']}")
    print(f"  total windows         : {summary['total_windows']}")
    print(f"  MEAN input tokens / conv : {summary['mean_input_tokens_per_conversation']:>12,.1f}")
    print(f"  median                : {summary['median_input_tokens_per_conversation']:>12,.1f}")
    print(f"  stdev                 : {summary['stdev_input_tokens_per_conversation']:>12,.1f}")
    print(f"  min / max             : {summary['min_input_tokens_per_conversation']:,}  /  "
          f"{summary['max_input_tokens_per_conversation']:,}")
    print(f"  total across all      : {summary['sum_input_tokens_all_conversations']:>12,}")
    print(f"  mean tokens / window  : {summary['mean_input_tokens_per_window']:>12,.1f}")
    print("=" * 70)
    print(f"\nSaved per-sample details to {args.out_file}")


if __name__ == "__main__":
    main()
