"""QLoRA fine-tune with transformers + peft + trl + bitsandbytes.

Usage:
    python train/train_qlora.py \
        --base-model /abs/path/to/Llama-3.2-3B-Instruct \
        --train-file dataset/mixed_sft_train.jsonl \
        --eval-file dataset/mixed_sft_eval.jsonl \
        --output-dir train/outputs/llama-aviation-lora
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", type=str, required=True)
    ap.add_argument("--train-file", type=Path, default=Path("dataset/mixed_sft_train.jsonl"))
    ap.add_argument("--eval-file", type=Path, default=Path("dataset/mixed_sft_eval.jsonl"))
    ap.add_argument("--output-dir", type=Path, default=Path("train/outputs/llama-aviation-lora"))
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--per-device-batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    def to_hf(path: Path) -> Dataset:
        rows = load_jsonl(path)
        texts: list[str] = []
        for r in rows:
            msgs = [
                {"role": "system", "content": r["system"]},
                {"role": "user", "content": r["instruction"]},
                {"role": "assistant", "content": r["response"]},
            ]
            texts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))
        return Dataset.from_dict({"text": texts})

    train_ds = to_hf(args.train_file)
    eval_ds = to_hf(args.eval_file)
    print(f"[data] train={len(train_ds)} eval={len(eval_ds)}")

    sft_cfg = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        per_device_eval_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        optim="paged_adamw_8bit",
        seed=args.seed,
        max_seq_length=args.max_seq_len,
        dataset_text_field="text",
        packing=False,
        gradient_checkpointing=True,
        report_to="none",
    )

    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_cfg,
        data_collator=collator,
    )

    print(f"[train-fallback] epochs={args.epochs} effective_batch="
          f"{args.per_device_batch * args.grad_accum}")
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"[done] adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
