"""Convert raw HuggingFace dolly-15k and oasst1 dumps into simple jsonl rows that mix.py accepts.

Run once on the Windows rig after loading the HF dataset from local cache:

    python dataset/prepare_general.py \
        --dolly-dir /abs/path/to/databricks-dolly-15k \
        --oasst-dir /abs/path/to/oasst1 \
        --out-dolly dataset/dolly_subset.jsonl \
        --out-oasst dataset/oasst_subset.jsonl \
        --max-per-source 3000

Both `--dolly-dir` and `--oasst-dir` expect local snapshots downloaded with
`huggingface-cli download <repo> --local-dir <dir>` on the internet machine.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def dump_dolly(dolly_dir: Path, out: Path, limit: int) -> int:
    ds = load_dataset(str(dolly_dir), split="train")
    count = 0
    with out.open("w", encoding="utf-8") as f:
        for row in ds:
            instr = (row.get("instruction") or "").strip()
            ctx = (row.get("context") or "").strip()
            resp = (row.get("response") or "").strip()
            if not instr or not resp:
                continue
            f.write(json.dumps({
                "instruction": instr,
                "context": ctx,
                "response": resp,
            }, ensure_ascii=False) + "\n")
            count += 1
            if count >= limit:
                break
    return count


def dump_oasst(oasst_dir: Path, out: Path, limit: int) -> int:
    """Flatten oasst1 into single-turn (prompter -> assistant) pairs, English only."""
    ds = load_dataset(str(oasst_dir), split="train")
    by_id: dict[str, dict] = {m["message_id"]: m for m in ds}
    count = 0
    with out.open("w", encoding="utf-8") as f:
        for msg in ds:
            if msg.get("role") != "assistant":
                continue
            if msg.get("lang") != "en":
                continue
            # Accept top-2 assistant replies per thread (rank 0, 1, or unset)
            if msg.get("rank") not in (0, 1, None):
                continue
            parent_id = msg.get("parent_id")
            parent = by_id.get(parent_id) if parent_id else None
            if not parent or parent.get("role") != "prompter":
                continue
            instr = (parent.get("text") or "").strip()
            resp = (msg.get("text") or "").strip()
            if not instr or not resp:
                continue
            f.write(json.dumps({
                "instruction": instr,
                "context": "",
                "response": resp,
            }, ensure_ascii=False) + "\n")
            count += 1
            if count >= limit:
                break
    return count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dolly-dir", type=Path)
    ap.add_argument("--oasst-dir", type=Path)
    ap.add_argument("--out-dolly", type=Path, default=Path("dataset/dolly_subset.jsonl"))
    ap.add_argument("--out-oasst", type=Path, default=Path("dataset/oasst_subset.jsonl"))
    ap.add_argument("--max-per-source", type=int, default=3000)
    args = ap.parse_args()

    if args.dolly_dir:
        args.out_dolly.parent.mkdir(parents=True, exist_ok=True)
        n = dump_dolly(args.dolly_dir, args.out_dolly, args.max_per_source)
        print(f"[dolly] wrote {n} rows → {args.out_dolly}")

    if args.oasst_dir:
        args.out_oasst.parent.mkdir(parents=True, exist_ok=True)
        n = dump_oasst(args.oasst_dir, args.out_oasst, args.max_per_source)
        print(f"[oasst] wrote {n} rows → {args.out_oasst}")


if __name__ == "__main__":
    main()
