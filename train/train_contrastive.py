"""
Phase 2 — Step 2: contrastive fine-tuning of the embedding model with
in-batch InfoNCE.

Inputs:
  train/data/contrastive_pairs.jsonl   (from build_contrastive_pairs.py)

For each pair we have:
  - query     (anchor)
  - positive_text  (the chosen teacher memory entry — positive sample)

In-batch contrastive setup
--------------------------
Within a mini-batch of size B:
  - q_i is the anchor for the i-th example
  - p_i is its positive
  - {p_j : j != i} are the negatives

Each p_j (j != i) is a real memory entry produced by the teacher for a
DIFFERENT (question, evidence) pair, so it is a "hard" negative in the sense
that it is fluent, well-formed, and could plausibly be retrieved by some
query — just not by q_i. This is exactly the failure mode we want the encoder
to learn to discriminate against.

Loss
----
Symmetric InfoNCE over the BxB similarity matrix S where
S[i,j] = cos(enc(q_i), enc(p_j)) / temperature

  loss = 0.5 * (CE(S, diagonal_targets) + CE(S^T, diagonal_targets))

The symmetric form trains both directions (query→passage and passage→query).
This is standard for sentence-pair contrastive learning (cf. SimCSE,
sentence-transformers MultipleNegativesRankingLoss) and is more sample-
efficient than one-sided NCE on small batches.

Sampling guard
--------------
We make sure no two examples in the same batch share the same QA pair (same
question). Different evidences of the same question would otherwise be paired
across i,j and treated as negatives even though the question is the same.

Model
-----
Defaults to config.EMBEDDING_MODEL. Uses sentence-transformers' SentenceTransformer
.fit-style loop, but we write the loop manually to keep batch construction in
our control.

Usage:
  python train/train_contrastive.py \
      --pairs-file   train/data/contrastive_pairs.jsonl \
      --val-pairs    train/data/contrastive_pairs_val.jsonl \
      --base-model   sentence-transformers/all-MiniLM-L6-v2 \
      --output-dir   train/checkpoints/embed-contrastive \
      --epochs 3 --batch-size 32 --lr 2e-5 --temperature 0.05
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
import pdb

# allow imports from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import config


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

@dataclass
class Pair:
    sample_id: str
    qa_idx: int
    query: str
    positive: str

    @property
    def qa_key(self) -> Tuple[str, int]:
        return (self.sample_id, self.qa_idx)


def load_pairs(path: str) -> List[Pair]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out.append(
                Pair(
                    sample_id=r["sample_id"],
                    qa_idx=r["qa_idx"],
                    query=r["query"],
                    positive=r["positive_text"],
                )
            )
    return out


# ----------------------------------------------------------------------
# Batch sampler that avoids same-QA collisions in a batch
# ----------------------------------------------------------------------
#
# Why this matters: a single LoCoMo question can produce multiple training
# pairs (one per evidence dia_id). If two such pairs end up in the same batch,
# the OTHER one's positive will be served as a "negative" for our query --
# but its positive likely DOES answer our query, since the question is the
# same. That gives a false-negative gradient. We avoid it by enforcing that
# no two examples in a batch share (sample_id, qa_idx).
#
# Implementation: greedy shuffling. Each epoch, shuffle pairs, then form
# batches by scanning and only adding a pair if its qa_key is not in the
# current batch. Pairs that would violate the constraint get held over.
# This is O(N) per epoch and produces batches that are exactly batch_size
# whenever there are enough distinct QA keys remaining.
class NoQACollisionBatchSampler:
    """
    Yields batches of indices such that no two indices in the same batch
    share a qa_key. Algorithm:

      pool = shuffle(all indices)
      while pool is non-empty:
          batch, leftover = [], []
          keys_in_batch = set()
          for idx in pool:
              if pair[idx].qa_key in keys_in_batch:
                  leftover.append(idx)
              else:
                  batch.append(idx); keys_in_batch.add(key)
                  if len(batch) == batch_size: break
          # any pool items NOT yet seen go back to the pool tail
          pool = pool[after_break:] + leftover  ← rolled into next batch
          yield batch (drop trailing batches of size 1, which can't form
                       any negatives anyway)

    This guarantees forward progress (every iteration removes at least one
    item from the pool) and produces batches of exactly batch_size whenever
    enough distinct qa_keys remain. The very last batch may be smaller; we
    drop size-1 batches because InfoNCE on B=1 is undefined (no negatives).
    """

    def __init__(self, pairs: List[Pair], batch_size: int, seed: int = 42,
                 drop_singleton_batches: bool = True):
        self.pairs = pairs
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.drop_singleton = drop_singleton_batches

    def __iter__(self) -> Iterator[List[int]]:
        pool = list(range(len(self.pairs)))
        self.rng.shuffle(pool)

        while pool:
            batch: List[int] = []
            keys: set = set()
            leftover: List[int] = []
            consumed_until = 0

            for pos, idx in enumerate(pool):
                if len(batch) == self.batch_size:
                    consumed_until = pos
                    break
                key = self.pairs[idx].qa_key
                if key in keys:
                    leftover.append(idx)
                else:
                    batch.append(idx)
                    keys.add(key)
            else:
                # Loop ran to completion without filling the batch
                consumed_until = len(pool)

            # Items in pool[consumed_until:] were never visited; they go
            # back into the pool along with the leftover items we passed
            # over because of qa_key collisions.
            pool = pool[consumed_until:] + leftover

            if self.drop_singleton and len(batch) < 2:
                # Can't compute InfoNCE on a singleton; the leftover/pool
                # rollover above already preserves the indices for next time,
                # but if the entire remaining pool consists of one qa_key we
                # would loop forever. Detect that and stop.
                if len(pool) == 0 or all(self.pairs[i].qa_key == self.pairs[pool[0]].qa_key
                                          for i in pool):
                    break
                continue
            if batch:
                yield batch

    def __len__(self) -> int:
        # Lower bound; some held-over items reduce final batch counts slightly.
        return math.ceil(len(self.pairs) / self.batch_size)


# ----------------------------------------------------------------------
# Validation: recall@k against an in-split pool
# ----------------------------------------------------------------------

def recall_at_k(model, val_pairs: List[Pair], ks=(1, 5, 10), batch_size: int = 64) -> Dict[str, float]:
    """
    Encode all val queries and all val positives. For each query, rank ALL
    positives by cosine similarity; check whether the gold positive is in
    top-k.
    """
    import torch
    queries = [p.query for p in val_pairs]
    positives = [p.positive for p in val_pairs]

    q_emb = model.encode(queries, batch_size=batch_size, convert_to_tensor=True,
                          show_progress_bar=False, normalize_embeddings=True)
    p_emb = model.encode(positives, batch_size=batch_size, convert_to_tensor=True,
                          show_progress_bar=False, normalize_embeddings=True)
    sims = q_emb @ p_emb.T  # [Nq, Np], both are L2-normalized so this is cosine

    n = len(val_pairs)
    targets = torch.arange(n, device=sims.device)
    out = {}
    for k in ks:
        topk = sims.topk(min(k, n), dim=1).indices  # [Nq, k]
        hit = (topk == targets.unsqueeze(1)).any(dim=1).float().mean().item()
        out[f"recall@{k}"] = hit
    return out


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def info_nce_loss(q_emb, p_emb, temperature: float):
    """
    Symmetric InfoNCE on a batch of L2-normalized embeddings.
    q_emb, p_emb: [B, D]
    """
    import torch.nn.functional as F
    import torch
    sims = (q_emb @ p_emb.T) / temperature  # [B, B]
    targets = torch.arange(sims.size(0), device=sims.device)
    loss_q = F.cross_entropy(sims, targets)
    loss_p = F.cross_entropy(sims.T, targets)
    return 0.5 * (loss_q + loss_p), sims


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs-file", default="train/data/contrastive_pairs.jsonl")
    p.add_argument("--val-pairs", default=None,
                   help="Optional val split for in-batch retrieval recall@k.")
    p.add_argument("--base-model", default=config.EMBEDDING_MODEL)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=0.05,
                   help="InfoNCE temperature. Smaller -> sharper. 0.05 is the "
                        "SimCSE default; 0.02-0.10 is the usual range.")
    p.add_argument("--max-length", type=int, default=256,
                   help="Truncation length for queries and entries. Both are "
                        "single sentences so 256 is generous.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every-epoch", action="store_true", default=True)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true", default=True)
    args = p.parse_args()

    if args.fp16:
        args.bf16 = False

    # Lazy imports
    import torch
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR
    from sentence_transformers import SentenceTransformer

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- Data ----
    pairs = load_pairs(args.pairs_file)
    val_pairs = load_pairs(args.val_pairs) if args.val_pairs else []
    print(f"Loaded {len(pairs)} train pairs / {len(val_pairs)} val pairs")
    if not pairs:
        raise ValueError(f"No pairs in {args.pairs_file}")

    # Diagnostic: how many distinct QAs?
    qa_counts = defaultdict(int)
    for pr in pairs:
        qa_counts[pr.qa_key] += 1
    print(f"  -> {len(qa_counts)} distinct QAs; "
          f"max pairs per QA: {max(qa_counts.values())}")

    # ---- Model ----
    print(f"Loading base model: {args.base_model}")
    model = SentenceTransformer(args.base_model, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Set max_seq_length on the underlying transformer
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = args.max_length

    # AMP dtype
    amp_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)

    # ---- Optimizer / scheduler ----
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    sampler = NoQACollisionBatchSampler(pairs, args.batch_size, seed=args.seed)
    steps_per_epoch = max(1, math.ceil(len(pairs) / args.batch_size))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * prog)))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # ---- Initial val ----
    if val_pairs:
        model.eval()
        with torch.no_grad():
            metrics = recall_at_k(model, val_pairs)
        print(f"[epoch 0 / pre-train] {metrics}")

    # ---- Training loop ----
    global_step = 0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    log_f = open(log_path, "w")

    for epoch in range(1, args.epochs + 1):
        model.train()
        # We re-instantiate sampler per epoch so the RNG advances and order changes
        sampler = NoQACollisionBatchSampler(
            pairs, args.batch_size, seed=args.seed + epoch
        )
        running_loss = 0.0
        n_batches = 0

        for batch_idxs in sampler:
            batch = [pairs[i] for i in batch_idxs]
            queries = [b.query for b in batch]
            positives = [b.positive for b in batch]

            # SentenceTransformer.encode with convert_to_tensor + a forward
            # pass within an autograd-enabled context. The simpler path:
            # tokenize manually and call model.forward / model[0]({...}).
            # SentenceTransformer's __call__ is the cleanest:
            q_features = model.tokenize(queries)
            p_features = model.tokenize(positives)
            q_features = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                        for k, v in q_features.items()}
            p_features = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                        for k, v in p_features.items()}

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                dtype=amp_dtype, enabled=device == "cuda"):
                q_out = model(q_features)
                p_out = model(p_features)
                # SentenceTransformer modules return dict with 'sentence_embedding'
                q_emb = q_out["sentence_embedding"]
                p_emb = p_out["sentence_embedding"]
                # L2-normalize for cosine
                q_emb = torch.nn.functional.normalize(q_emb, dim=-1)
                p_emb = torch.nn.functional.normalize(p_emb, dim=-1)
                loss, sims = info_nce_loss(q_emb, p_emb, args.temperature)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            running_loss += loss.item()
            n_batches += 1

            # In-batch accuracy (diagonal == top-1)
            with torch.no_grad():
                acc = (sims.argmax(dim=1) == torch.arange(sims.size(0), device=sims.device)).float().mean().item()

            log_f.write(json.dumps({
                "epoch": epoch, "step": global_step,
                "loss": loss.item(), "in_batch_acc": acc,
                "lr": scheduler.get_last_lr()[0], "batch_size": len(batch),
            }) + "\n")
            log_f.flush()

            if global_step % 10 == 0:
                print(
                    f"  epoch {epoch} step {global_step}/{total_steps} "
                    f"loss={loss.item():.4f} in_batch_acc={acc:.3f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

        avg_loss = running_loss / max(1, n_batches)
        print(f"[epoch {epoch}] avg_loss={avg_loss:.4f}")

        if args.eval_every_epoch and val_pairs:
            model.eval()
            with torch.no_grad():
                metrics = recall_at_k(model, val_pairs)
            print(f"[epoch {epoch}] {metrics}")
            log_f.write(json.dumps({"epoch": epoch, "val": metrics}) + "\n")
            log_f.flush()

    log_f.close()

    # ---- Save ----
    model.save(str(out_dir))
    print(f"Saved fine-tuned encoder to {out_dir}")
    with open(out_dir / "train_meta.json", "w") as f:
        json.dump({
            "base_model": args.base_model,
            "n_train_pairs": len(pairs),
            "n_val_pairs": len(val_pairs),
            "n_distinct_qa": len(qa_counts),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "temperature": args.temperature,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "max_length": args.max_length,
        }, f, indent=2)

    print(
        "\nTo use this encoder at inference, point config.EMBEDDING_MODEL at "
        f"'{out_dir}' (the SentenceTransformer loader accepts local paths).\n"
        "The model file format is identical to upstream sentence-transformers, "
        "so utils/embedding.py picks it up via _init_standard with no changes."
    )


if __name__ == "__main__":
    main()
