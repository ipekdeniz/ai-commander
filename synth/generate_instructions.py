"""Generate synthetic aviation-emergency Q&A pairs from seeds.yaml using a local teacher model.

Uses HuggingFace transformers with manual batched generation — no vLLM needed.
Teacher model: Qwen3.5-27B-GPTQ-Int4.

Key features:
- Brace-balanced JSON extraction (robust to markdown fences, prose, nested JSON)
- Incremental append to output file after each batch (crash-safe)
- Auto-resume: skips seeds already in output file
- Retry on empty extraction with lower temperature
- Batched generation via manual model.generate

Usage (Windows, offline):
    python synth/generate_instructions.py \\
        --teacher-model C:/models/Qwen3.5-27B-GPTQ-Int4 \\
        --out dataset/aviation_emergency_sft.jsonl \\
        --pairs-per-seed 5 \\
        --freeform-pairs 200 \\
        --batch-size 4

Re-running with the same --out resumes where it left off.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from tqdm import tqdm

from prompts import (
    FREEFORM_PROMPT_TEMPLATE,
    GENERATION_PROMPT_TEMPLATE,
    TEACHER_SYSTEM_PROMPT,
)


@dataclass(frozen=True)
class SeedJob:
    topic: str
    angle: str

    @property
    def key(self) -> str:
        return f"seed::{self.topic}::{self.angle}"

    def render(self, n: int = 5) -> str:
        return GENERATION_PROMPT_TEMPLATE.format(topic=self.topic, angle=self.angle).replace(
            "Generate 5 diverse", f"Generate {n} diverse"
        )


def load_seeds(path: Path) -> list[SeedJob]:
    data = yaml.safe_load(path.read_text())
    jobs: list[SeedJob] = []
    for entry in data["seeds"]:
        topic = entry["topic"]
        for angle in entry.get("angles", ["general"]):
            jobs.append(SeedJob(topic=topic, angle=angle))
    return jobs


# --- JSON extraction ---------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)\n```", re.DOTALL)


def _find_balanced_object(text: str, start: int) -> tuple[int, int] | None:
    """Return (begin, end_exclusive) of a balanced JSON object, handling string escapes."""
    begin = text.find("{", start)
    if begin < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(begin, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return begin, i + 1
    return None


def extract_pairs(raw: str) -> list[dict]:
    """Parse teacher output and return validated {instruction, response} pairs."""
    text = raw.strip()

    # Try markdown code fence first
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()

    # Try direct parse
    obj: dict | None = None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "pairs" in parsed:
            obj = parsed
    except json.JSONDecodeError:
        pass

    # Fallback: walk text, try each balanced {...} object
    if obj is None:
        cursor = 0
        while cursor < len(text):
            span = _find_balanced_object(text, cursor)
            if span is None:
                break
            begin, end = span
            try:
                parsed = json.loads(text[begin:end])
                if isinstance(parsed, dict) and "pairs" in parsed:
                    obj = parsed
                    break
            except json.JSONDecodeError:
                pass
            cursor = begin + 1

    if not isinstance(obj, dict):
        return []

    clean: list[dict] = []
    for p in obj.get("pairs", []):
        if not isinstance(p, dict):
            continue
        instr = (p.get("instruction") or "").strip()
        resp = (p.get("response") or "").strip()
        if not instr or not resp:
            continue
        if len(resp) < 80 or len(resp) > 3000:
            continue
        clean.append({"instruction": instr, "response": resp})
    return clean


def deduplicate(pairs: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for p in pairs:
        key = p["instruction"].lower().strip()[:120]
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


# --- Generation --------------------------------------------------------------

def generate_batch(
    model,
    tokenizer,
    batch_msgs: list[list[dict]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    import torch

    prompts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in batch_msgs
    ]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    decoded: list[str] = []
    input_len = inputs["input_ids"].shape[1]
    for seq in outputs:
        new_tokens = seq[input_len:]
        decoded.append(tokenizer.decode(new_tokens, skip_special_tokens=True))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return decoded


def generate_with_retry(
    model,
    tokenizer,
    user_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[dict]:
    """Single-prompt generate with one retry on empty extraction (lower temperature)."""
    for attempt in range(2):
        t = temperature if attempt == 0 else max(0.3, temperature - 0.3)
        raw_list = generate_batch(
            model,
            tokenizer,
            [[
                {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]],
            max_new_tokens=max_new_tokens,
            temperature=t,
            top_p=top_p,
        )
        pairs = extract_pairs(raw_list[0])
        if pairs:
            return pairs
    return []


# --- Resume helpers ----------------------------------------------------------

def load_existing_keys(out: Path) -> set[str]:
    """Read already-written output file to compute processed seed keys."""
    if not out.exists():
        return set()
    keys: set[str] = set()
    for line in out.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            meta = row.get("meta", {})
            topic = meta.get("topic", "")
            angle = meta.get("angle", "")
            src = meta.get("source", "seed")
            if src == "seed":
                keys.add(f"seed::{topic}::{angle}")
            elif src == "freeform":
                keys.add(f"freeform::{meta.get('batch_idx', 0)}")
        except json.JSONDecodeError:
            continue
    return keys


def append_pairs(out: Path, pairs: list[dict]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


# --- Main --------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=Path, default=Path("synth/seeds.yaml"))
    ap.add_argument("--teacher-model", type=str, required=True,
                    help="Local path to Qwen3.5-27B-GPTQ-Int4 directory")
    ap.add_argument("--out", type=Path, default=Path("dataset/aviation_emergency_sft.jsonl"))
    ap.add_argument("--pairs-per-seed", type=int, default=5)
    ap.add_argument("--freeform-pairs", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--deduplicate-final", action="store_true",
                    help="Run dedup pass at end (outputs *_dedup.jsonl)")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[teacher] loading {args.teacher_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    # Left-padding required for batched causal LM generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("[teacher] ready.")

    seed_jobs = load_seeds(args.seeds)
    freeform_batches = max(1, args.freeform_pairs // 10)

    done_keys = load_existing_keys(args.out)
    if done_keys:
        print(f"[resume] found {len(done_keys)} already-processed keys in {args.out}")

    pending_seeds = [j for j in seed_jobs if j.key not in done_keys]
    pending_freeform = [
        i for i in range(freeform_batches)
        if f"freeform::{i}" not in done_keys
    ]

    print(f"[gen] seed jobs: {len(pending_seeds)}/{len(seed_jobs)} pending, "
          f"freeform: {len(pending_freeform)}/{freeform_batches} pending")

    # Process seeds in batches
    for i in tqdm(range(0, len(pending_seeds), args.batch_size), desc="seed batches"):
        batch = pending_seeds[i:i + args.batch_size]
        batch_msgs = [
            [
                {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
                {"role": "user", "content": job.render(args.pairs_per_seed)},
            ]
            for job in batch
        ]
        raw_list = generate_batch(
            model, tokenizer, batch_msgs,
            args.max_new_tokens, args.temperature, args.top_p,
        )

        batch_out: list[dict] = []
        for job, raw in zip(batch, raw_list):
            pairs = extract_pairs(raw)
            if not pairs:
                # Retry once, single prompt, lower temperature
                pairs = generate_with_retry(
                    model, tokenizer, job.render(args.pairs_per_seed),
                    args.max_new_tokens, args.temperature, args.top_p,
                )
                if not pairs:
                    print(f"[warn] empty extraction after retry: {job.topic} / {job.angle}")
            for p in pairs:
                p["meta"] = {"topic": job.topic, "angle": job.angle, "source": "seed"}
                batch_out.append(p)

        append_pairs(args.out, batch_out)

    # Process freeform batches
    freeform_prompt = FREEFORM_PROMPT_TEMPLATE.format(n=10)
    for idx in tqdm(pending_freeform, desc="freeform"):
        pairs = generate_with_retry(
            model, tokenizer, freeform_prompt,
            args.max_new_tokens, args.temperature, args.top_p,
        )
        if not pairs:
            print(f"[warn] empty freeform extraction (batch {idx})")
        for p in pairs:
            p["meta"] = {"topic": "freeform", "angle": "general",
                         "source": "freeform", "batch_idx": idx}
        append_pairs(args.out, pairs)

    # Final summary
    total_lines = 0
    if args.out.exists():
        total_lines = sum(1 for line in args.out.read_text(encoding="utf-8").splitlines()
                          if line.strip())
    print(f"[done] {total_lines} pairs in {args.out}")

    if args.deduplicate_final and total_lines > 0:
        all_pairs = [json.loads(line) for line in args.out.read_text(encoding="utf-8").splitlines()
                     if line.strip()]
        before = len(all_pairs)
        deduped = deduplicate(all_pairs)
        dedup_path = args.out.with_name(args.out.stem + "_dedup.jsonl")
        with dedup_path.open("w", encoding="utf-8") as f:
            for p in deduped:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"[dedup] {before} → {len(deduped)} pairs → {dedup_path}")


if __name__ == "__main__":
    main()
