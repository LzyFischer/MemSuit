"""
Step 1/2 of the contrastive-learning before/after analysis.

For ONE encoder (base or fine-tuned), encode every (query, positive_text) pair
from a contrastive_pairs.jsonl file, then compute two distributions of cosine
similarities:

  - POSITIVE pairs   :  cos(q_i, p_i)            -- diagonal of the [N, N] matrix
  - IN-BATCH NEG     :  cos(q_i, p_j), j != i,
                        AND qa_key[j] != qa_key[i]
                        -- off-diagonals, EXCLUDING cells where p_j is just
                        another evidence for the SAME question as q_i.
                        This matches the NoQACollisionBatchSampler invariant
                        from train_contrastive.py: same-QA cross-pairs were
                        never used as negatives during training, so they
                        shouldn't be counted as negatives when we measure
                        the effect of training either.

This mirrors the InfoNCE setup used during training (train_contrastive.py:
info_nce_loss), so the violin plots produced from these numbers directly
show what contrastive training is *supposed* to change: pull diagonals up,
push off-diagonals down.

Output: a single .npz file with arrays {pos, neg, label} for plotting.

Usage:
  # Before training (base encoder)
  python train/compute_similarities.py \
      --pairs-file train/data/contrastive_pairs.jsonl \
      --model-path sentence-transformers/all-MiniLM-L6-v2 \
      --label "Llama-base" \
      --out-file train/data/sims_llama_base.npz

  # After training (fine-tuned encoder, just point to the checkpoint dir)
  python train/compute_similarities.py \
      --pairs-file train/data/contrastive_pairs.jsonl \
      --model-path train/checkpoints/embed-contrastive-llama \
      --label "Llama-finetuned" \
      --out-file train/data/sims_llama_ft.npz

Then run plot_similarity_violin.py on the four .npz files.

Memory note:
  We materialize the full [N, N] cosine matrix in memory. For N ~ a few
  thousand pairs this is fine (a 5000x5000 float32 matrix is ~100 MB). If
  your pair count is much larger, switch to streaming (see --max-pairs).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# allow imports from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_pairs(path: str) -> Tuple[List[str], List[str], List[Tuple[str, int]]]:
    """Load (query, positive_text, qa_key) from contrastive_pairs.jsonl.

    qa_key = (sample_id, qa_idx) identifies the underlying QA. Two rows that
    share a qa_key are different evidences for the SAME question — we must
    not treat one as a negative for the other when computing the in-batch
    negative distribution (this mirrors NoQACollisionBatchSampler in
    train_contrastive.py).
    """
    queries, positives, qa_keys = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            queries.append(r["query"])
            positives.append(r["positive_text"])
            qa_keys.append((r["sample_id"], int(r["qa_idx"])))
    return queries, positives, qa_keys


def is_qwen3_path(path: str) -> bool:
    """Mirror utils/embedding.py:_init_qwen3 heuristic. The training script may
    save a fine-tuned Qwen3 checkpoint to a path that does NOT start with
    'qwen3', so we also peek at the saved config if present."""
    if Path(path).name.lower().startswith("qwen3"):
        return True
    cfg = Path(path) / "config.json"
    if cfg.is_file():
        try:
            with open(cfg) as f:
                c = json.load(f)
            mt = (c.get("model_type") or "").lower()
            arch = " ".join(c.get("architectures", [])).lower()
            if "qwen" in mt or "qwen" in arch:
                return True
        except Exception:
            pass
    return False


def load_encoder(model_path: str):
    """Load either a Qwen3 ST or a standard ST. We deliberately do NOT reuse
    utils.embedding.EmbeddingModel because that one maps the short names like
    'qwen3-0.6b' through config.EMBEDDING_MODEL -- here we want exact control
    over which path is loaded so 'before/after' is unambiguous."""
    from sentence_transformers import SentenceTransformer

    if is_qwen3_path(model_path):
        # Match utils/embedding.py:_init_qwen3 — try flash-attn first, fall back.
        try:
            model = SentenceTransformer(
                model_path,
                model_kwargs={"attn_implementation": "flash_attention_2",
                              "device_map": "auto"},
                tokenizer_kwargs={"padding_side": "left"},
                trust_remote_code=True,
            )
            print(f"  Loaded {model_path} with flash_attention_2")
        except Exception as e:
            print(f"  flash_attention_2 unavailable ({e}); falling back")
            model = SentenceTransformer(model_path, trust_remote_code=True)
        supports_query_prompt = "query" in getattr(model, "prompts", {})
    else:
        model = SentenceTransformer(model_path, trust_remote_code=True)
        supports_query_prompt = False

    return model, supports_query_prompt


def encode(model, texts: List[str], is_query: bool, supports_query_prompt: bool,
           batch_size: int) -> np.ndarray:
    """Wrap model.encode to apply Qwen3's query prompt when applicable. Always
    L2-normalize so the dot product is cosine."""
    kwargs = dict(
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    if is_query and supports_query_prompt:
        kwargs["prompt_name"] = "query"
    return model.encode(texts, **kwargs)


def compute_pos_neg_sims(q_emb: np.ndarray, p_emb: np.ndarray,
                         qa_keys: List[Tuple[str, int]]
                         ) -> Tuple[np.ndarray, np.ndarray, int]:
    """Given L2-normalized [N, D] embeddings, return:
       pos : [N]   cosine of q_i with p_i
       neg : [M]   cosine of q_i with p_j  for j != i AND qa_key[j] != qa_key[i]
       n_excluded : how many off-diagonal cells we dropped because they came
                    from a different evidence of the SAME question

    Why exclude same-qa_key off-diagonals?
      A single QA can produce multiple training pairs (one per evidence
      dia_id). At training time NoQACollisionBatchSampler forbids two such
      pairs from sharing a batch — otherwise the OTHER pair's positive
      would be served as a "negative" for our query, but its positive
      likely DOES answer our query since the question is identical
      (false-negative gradient). The same caveat applies when we *measure*
      the negative-pair similarity distribution: those cells are not real
      negatives, they're collisions, and including them inflates the
      apparent neg distribution.
    """
    sim = q_emb @ p_emb.T  # [N, N], cosine since both sides are normalized
    n = sim.shape[0]
    pos = np.diag(sim).astype(np.float32).copy()

    # Build "true negative" mask: off-diagonal AND different qa_key.
    diag_mask = np.eye(n, dtype=bool)

    # For an N up to ~tens of thousands the N*N qa-key comparison via a
    # small id-array is cheap and avoids any Python-level loop.
    key_to_id: Dict[Tuple[str, int], int] = {}
    ids = np.empty(n, dtype=np.int64)
    for i, k in enumerate(qa_keys):
        if k not in key_to_id:
            key_to_id[k] = len(key_to_id)
        ids[i] = key_to_id[k]
    same_qa = ids[:, None] == ids[None, :]   # [N, N], True where same QA

    neg_mask = ~diag_mask & ~same_qa
    n_excluded = int((same_qa & ~diag_mask).sum())

    neg = sim[neg_mask].astype(np.float32)
    return pos, neg, n_excluded


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs-file", required=True,
                   help="Path to contrastive_pairs.jsonl (train or val).")
    p.add_argument("--model-path", required=True,
                   help="HF id (e.g. 'sentence-transformers/all-MiniLM-L6-v2', "
                        "'Qwen/Qwen3-Embedding-0.6B') OR a local checkpoint dir.")
    p.add_argument("--label", required=True,
                   help="Label stored in the .npz for plotting, e.g. "
                        "'Qwen-base' or 'Llama-finetuned'.")
    p.add_argument("--out-file", required=True,
                   help="Output .npz file.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-pairs", type=int, default=0,
                   help="If >0, sample at most this many pairs (deterministic, "
                        "seed=42). Useful when N is huge and the full [N,N] "
                        "matrix would blow up memory.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    queries, positives, qa_keys = load_pairs(args.pairs_file)
    print(f"Loaded {len(queries)} pairs from {args.pairs_file}")
    n_distinct_qa = len(set(qa_keys))
    if n_distinct_qa < len(qa_keys):
        print(f"  {len(qa_keys)} pairs span {n_distinct_qa} distinct QAs "
              f"({len(qa_keys) - n_distinct_qa} pairs share a QA with another "
              f"pair -- those cross-pairs will be excluded from the negative "
              f"distribution to match training-time NoQACollisionBatchSampler)")

    if args.max_pairs and len(queries) > args.max_pairs:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(queries), size=args.max_pairs, replace=False)
        idx.sort()
        queries = [queries[i] for i in idx]
        positives = [positives[i] for i in idx]
        qa_keys = [qa_keys[i] for i in idx]
        print(f"  subsampled to {len(queries)} pairs (seed={args.seed})")

    print(f"Loading encoder: {args.model_path}")
    model, supports_query_prompt = load_encoder(args.model_path)
    if supports_query_prompt:
        print("  (encoder supports a 'query' prompt -- using it for queries)")

    print("Encoding queries...")
    q_emb = encode(model, queries, is_query=True,
                   supports_query_prompt=supports_query_prompt,
                   batch_size=args.batch_size)
    print("Encoding positives...")
    p_emb = encode(model, positives, is_query=False,
                   supports_query_prompt=supports_query_prompt,
                   batch_size=args.batch_size)

    pos, neg, n_excluded = compute_pos_neg_sims(q_emb, p_emb, qa_keys)
    print(f"pos cosines: n={len(pos)}  mean={pos.mean():.4f}  "
          f"median={np.median(pos):.4f}  std={pos.std():.4f}")
    print(f"neg cosines: n={len(neg)}  mean={neg.mean():.4f}  "
          f"median={np.median(neg):.4f}  std={neg.std():.4f}  "
          f"(excluded {n_excluded} same-QA cross-pairs)")

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        pos=pos,
        neg=neg,
        label=np.array(args.label),
        model_path=np.array(args.model_path),
        pairs_file=np.array(args.pairs_file),
        n_pairs=np.array(len(queries)),
        n_excluded_same_qa=np.array(n_excluded),
    )
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()