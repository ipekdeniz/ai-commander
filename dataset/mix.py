"""Mix aviation synthetic SFT with a general SFT subset, attach system prompt, split train/eval.

Inputs:
    dataset/aviation_emergency_sft.jsonl   (from synth/generate_instructions.py)
    dataset/dolly_subset.jsonl             (see prepare_general() below)
    dataset/oasst_subset.jsonl             (optional, English assistant turns only)

Outputs:
    dataset/mixed_sft_train.jsonl
    dataset/mixed_sft_eval.jsonl

Each output row carries the raw fields (instruction, response, system) so training
code can apply the Llama-3 chat template via tokenizer.apply_chat_template().
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SYSTEM_PROMPT = (
    "You are an aviation safety assistant specialized in emergency procedures. "
    "Provide accurate, conservative, procedure-oriented guidance based on FAA "
    "standards (FAR/AIM, Pilot's Handbook of Aeronautical Knowledge, Airplane "
    "Flying Handbook). Always note when relevant: this is educational material; "
    "actual flight decisions require the aircraft's POH, ATC authority, and a "
    "certified flight instructor."
)

GENERAL_SYSTEM_PROMPT = (
    "You are a helpful, honest, and concise assistant. Provide clear and "
    "accurate answers to the user's questions."
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize_aviation(row: dict) -> dict:
    return {
        "system": SYSTEM_PROMPT,
        "instruction": row["instruction"].strip(),
        "response": row["response"].strip(),
        "source": "aviation_synthetic",
    }


def normalize_general(row: dict) -> dict | None:
    """Expect fields instruction/response/context; context may be empty."""
    instr = (row.get("instruction") or row.get("prompt") or "").strip()
    resp = (row.get("response") or row.get("output") or row.get("completion") or "").strip()
    ctx = (row.get("context") or "").strip()
    if not instr or not resp:
        return None
    if ctx:
        instr = f"{instr}\n\nContext:\n{ctx}"
    if len(resp) < 20 or len(resp) > 3000:
        return None
    return {
        "system": GENERAL_SYSTEM_PROMPT,
        "instruction": instr,
        "response": resp,
        "source": "general",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aviation", type=Path, default=Path("dataset/aviation_emergency_sft.jsonl"))
    ap.add_argument("--general", type=Path, nargs="*", default=[
        Path("dataset/dolly_subset.jsonl"),
        Path("dataset/oasst_subset.jsonl"),
    ])
    ap.add_argument("--aviation-ratio", type=float, default=0.70,
                    help="Target fraction of aviation examples in the final mix")
    ap.add_argument("--train-out", type=Path, default=Path("dataset/mixed_sft_train.jsonl"))
    ap.add_argument("--eval-out", type=Path, default=Path("dataset/mixed_sft_eval.jsonl"))
    ap.add_argument("--eval-fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not 0.0 < args.aviation_ratio < 1.0:
        raise SystemExit("--aviation-ratio must be strictly between 0 and 1")

    rng = random.Random(args.seed)

    aviation_rows = [normalize_aviation(r) for r in read_jsonl(args.aviation)]
    general_rows_all: list[dict] = []
    for p in args.general:
        for r in read_jsonl(p):
            n = normalize_general(r)
            if n is not None:
                general_rows_all.append(n)

    if not aviation_rows:
        raise SystemExit(f"No aviation rows found at {args.aviation}")

    target_general = int(len(aviation_rows) * (1 - args.aviation_ratio) / args.aviation_ratio)
    if len(general_rows_all) > target_general:
        rng.shuffle(general_rows_all)
        general_rows = general_rows_all[:target_general]
    else:
        general_rows = general_rows_all

    # Stratified split: hold out eval_fraction from EACH source, then combine.
    def _split(rows: list[dict], frac: float) -> tuple[list[dict], list[dict]]:
        shuffled = rows[:]
        rng.shuffle(shuffled)
        n_eval = max(1, int(len(shuffled) * frac)) if shuffled else 0
        return shuffled[n_eval:], shuffled[:n_eval]

    aviation_train, aviation_eval = _split(aviation_rows, args.eval_fraction)
    general_train, general_eval = _split(general_rows, args.eval_fraction)

    train_rows = aviation_train + general_train
    eval_rows = aviation_eval + general_eval
    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)

    write_jsonl(args.train_out, train_rows)
    write_jsonl(args.eval_out, eval_rows)

    total = len(train_rows) + len(eval_rows)
    ratio = len(aviation_rows) / total if total else 0.0
    print(f"[mix] aviation={len(aviation_rows)} general={len(general_rows)} "
          f"total={total} (aviation ratio {ratio:.2f})")
    print(f"[mix] train: aviation={len(aviation_train)} + general={len(general_train)} "
          f"= {len(train_rows)}")
    print(f"[mix] eval:  aviation={len(aviation_eval)} + general={len(general_eval)} "
          f"= {len(eval_rows)}")
    print(f"[mix] wrote {args.train_out} and {args.eval_out}")


if __name__ == "__main__":
    main()
