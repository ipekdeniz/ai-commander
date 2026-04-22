"""Merge a QLoRA adapter into the base Gemma 3 4B model and save as fp16 HF directory.

Usage:
    python export/merge_lora.py \
        --base-model /abs/path/to/gemma-3-4b-it \
        --lora-dir train/outputs/gemma-aviation-lora \
        --out-dir export/gemma-aviation-4b-merged

The output directory is in HuggingFace format (config.json, *.safetensors, tokenizer files),
ready for llama.cpp's convert_hf_to_gguf.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", type=str, required=True)
    ap.add_argument("--lora-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"[merge] loading base: {args.base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"},
    )

    print(f"[merge] loading adapter: {args.lora_dir}")
    model = PeftModel.from_pretrained(base, str(args.lora_dir))

    print("[merge] merging LoRA into base weights...")
    merged = model.merge_and_unload()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[merge] saving merged model: {args.out_dir}")
    merged.save_pretrained(str(args.out_dir), safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.save_pretrained(str(args.out_dir))
    print(f"[done] merged model → {args.out_dir}")


if __name__ == "__main__":
    main()
