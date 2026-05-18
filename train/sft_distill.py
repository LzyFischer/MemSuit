"""
Self-distillation SFT for the SimpleMem memory builder.

Trains a student model (default: Qwen/Qwen2.5-3B-Instruct -- the same family
as config.LLM_MODEL) to imitate the teacher's answer-aware memory summaries
when given the standard (no-hint) prompt.

Loss: token-level NLL over the assistant response only (prompt is masked out
via TRL's data collator). LoRA adapters keep this runnable on a single
24-32GB GPU.

Usage:
  python train/sft_distill.py \
      --train-file train/data/train.jsonl \
      --val-file   train/data/val.jsonl \
      --base-model Qwen/Qwen2.5-3B-Instruct \
      --output-dir train/checkpoints/qwen25-3b-distill \
      --epochs 3 --lr 2e-4 --batch-size 2 --grad-accum 8

To run with FP16 on smaller GPUs add `--fp16`. For 4-bit QLoRA add
`--load-in-4bit` (requires bitsandbytes).
"""
import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# allow imports from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

# We import torch / transformers lazily so the script can be inspected
# (e.g. `--help`) without a CUDA install present.

SYSTEM_MSG = (
    "You are a professional information extraction assistant. "
    "Extract structured, unambiguous facts from conversations. "
    "Output valid JSON only."
)


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_chat_example(tokenizer, student_prompt: str, teacher_output: str) -> Dict:
    """
    Build one training example using the model's chat template.
    The label is the teacher_output (assistant turn). Everything before is
    the prompt, which we mask out of the loss.
    """
    # Build the full chat: [system, user, assistant]
    full_messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": student_prompt},
        {"role": "assistant", "content": teacher_output},
    ]
    # Build the prompt-only chat (everything up to but not including assistant)
    prompt_messages = full_messages[:2]

    # apply_chat_template returns either a string or token IDs; we use IDs
    # with `add_generation_prompt=True` for the prompt half so it ends right
    # at the assistant tag.
    full_ids = tokenizer.apply_chat_template(
        full_messages, tokenize=True, add_generation_prompt=False
    )
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages, tokenize=True, add_generation_prompt=True
    )

    # Sanity: prompt_ids must be a strict prefix of full_ids
    if full_ids[: len(prompt_ids)] != prompt_ids:
        # Some chat templates append a BOS or extra tokens differently.
        # Fall back to scanning for the longest matching prefix.
        i = 0
        while (
            i < len(prompt_ids) and i < len(full_ids) and prompt_ids[i] == full_ids[i]
        ):
            i += 1
        prompt_len = i
    else:
        prompt_len = len(prompt_ids)

    labels = list(full_ids)
    for i in range(prompt_len):
        labels[i] = -100  # mask prompt tokens

    return {
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": [1] * len(full_ids),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-file", required=True)
    p.add_argument("--val-file", default=None)
    p.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    # LoRA
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    # Precision
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--load-in-4bit", action="store_true",
                   help="QLoRA via bitsandbytes (needs bitsandbytes installed)")
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    args = p.parse_args()

    if args.fp16:
        args.bf16 = False

    # Lazy imports
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
        set_seed,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    set_seed(args.seed)

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Data ----
    train_rows = load_jsonl(args.train_file)
    val_rows = load_jsonl(args.val_file) if args.val_file else []
    print(f"Loaded {len(train_rows)} train / {len(val_rows)} val rows")

    def to_features(rows):
        out = []
        for r in rows:
            ex = build_chat_example(tokenizer, r["student_prompt"], r["teacher_output"])
            if len(ex["input_ids"]) > args.max_length:
                # truncate from the LEFT of the user message, keeping the
                # assistant target intact. Easiest: just skip overflowing rows.
                continue
            out.append(ex)
        return out

    train_features = to_features(train_rows)
    val_features = to_features(val_rows) if val_rows else []
    print(f"After length filtering: {len(train_features)} train / {len(val_features)} val")

    train_ds = Dataset.from_list(train_features)
    val_ds = Dataset.from_list(val_features) if val_features else None

    # ---- Model ----
    model_kwargs = dict(torch_dtype=torch.bfloat16 if args.bf16 else torch.float16)
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ---- Collator ----
    # DataCollatorForSeq2Seq handles label padding with -100 correctly.
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    # ---- Trainer ----
    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps" if val_ds is not None else "no",
        eval_steps=args.eval_steps if val_ds is not None else None,
        save_total_limit=3,
        bf16=args.bf16 and not args.fp16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        seed=args.seed,
        load_best_model_at_end=val_ds is not None,
        metric_for_best_model="eval_loss" if val_ds is not None else None,
        greater_is_better=False if val_ds is not None else None,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Saved LoRA adapter + tokenizer to {args.output_dir}")
    print("To use this checkpoint at inference time, either:")
    print("  1) Merge the adapter into the base model:")
    print("       from peft import PeftModel")
    print(f"       base = AutoModelForCausalLM.from_pretrained('{args.base_model}')")
    print(f"       model = PeftModel.from_pretrained(base, '{args.output_dir}')")
    print("       merged = model.merge_and_unload()")
    print(f"       merged.save_pretrained('{args.output_dir}-merged')")
    print("  2) Then serve via vLLM/Ollama and point config.LLM_MODEL at it.")


if __name__ == "__main__":
    main()
