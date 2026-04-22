"""Cross-platform end-to-end export: merge LoRA → GGUF f16 → quantize.

Works on Windows (cmd/PowerShell), macOS, and Linux — no bash required.

Usage:
    python export/run_export.py \\
        --base-model C:/models/Llama-3.2-3B-Instruct \\
        --lora-dir train/outputs/llama-aviation-lora \\
        --llama-cpp-dir C:/tools/llama.cpp \\
        --out-name llama-aviation-3b

Equivalent to merge_and_quantize.sh. Prefer this on Windows.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(f"[run] {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"[error] command failed with exit code {result.returncode}")


def find_quantize_binary(llama_cpp_dir: Path) -> Path:
    """Locate llama-quantize binary across common build layouts."""
    candidates = [
        llama_cpp_dir / "build" / "bin" / "Release" / "llama-quantize.exe",
        llama_cpp_dir / "build" / "bin" / "llama-quantize.exe",
        llama_cpp_dir / "build" / "bin" / "Release" / "llama-quantize",
        llama_cpp_dir / "build" / "bin" / "llama-quantize",
        llama_cpp_dir / "llama-quantize.exe",
        llama_cpp_dir / "llama-quantize",
    ]
    for c in candidates:
        if c.exists():
            return c
    sys.exit(
        f"[error] could not find llama-quantize binary under {llama_cpp_dir}. "
        f"Build llama.cpp first (cmake --build build --config Release)."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", type=str, required=True,
                    help="Path to fp16 HF base model directory")
    ap.add_argument("--lora-dir", type=Path, default=Path("train/outputs/llama-aviation-lora"))
    ap.add_argument("--llama-cpp-dir", type=Path, required=True,
                    help="Path to local llama.cpp clone (built)")
    ap.add_argument("--out-name", type=str, default="llama-aviation-3b")
    ap.add_argument("--quant", type=str, default="Q4_K_M",
                    choices=["Q2_K", "Q3_K_M", "Q4_K_S", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"])
    ap.add_argument("--out-dir", type=Path, default=Path("export"))
    ap.add_argument("--skip-merge", action="store_true", help="Reuse existing merged dir")
    ap.add_argument("--skip-convert", action="store_true", help="Reuse existing f16 GGUF")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged_dir = args.out_dir / f"{args.out_name}-merged"
    gguf_f16 = args.out_dir / f"{args.out_name}.f16.gguf"
    gguf_quant = args.out_dir / f"{args.out_name}.{args.quant}.gguf"

    # Step 1: merge LoRA
    if args.skip_merge and merged_dir.exists():
        print(f"[1/3] skipping merge (found {merged_dir})")
    else:
        print(f"[1/3] merging LoRA → {merged_dir}")
        merge_script = Path(__file__).parent / "merge_lora.py"
        run([
            sys.executable, str(merge_script),
            "--base-model", args.base_model,
            "--lora-dir", str(args.lora_dir),
            "--out-dir", str(merged_dir),
        ])

    # Step 2: HF → GGUF f16
    convert_script = args.llama_cpp_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        sys.exit(f"[error] {convert_script} not found")

    if args.skip_convert and gguf_f16.exists():
        print(f"[2/3] skipping convert (found {gguf_f16})")
    else:
        print(f"[2/3] converting HF → GGUF f16 → {gguf_f16}")
        run([
            sys.executable, str(convert_script),
            str(merged_dir),
            "--outfile", str(gguf_f16),
            "--outtype", "f16",
        ])

    # Step 3: quantize
    quantize_bin = find_quantize_binary(args.llama_cpp_dir)
    print(f"[3/3] quantizing → {gguf_quant}")
    run([str(quantize_bin), str(gguf_f16), str(gguf_quant), args.quant])

    size_mb = gguf_quant.stat().st_size / (1024 * 1024)
    print(f"\n[done] iPad-ready GGUF: {gguf_quant}  ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
