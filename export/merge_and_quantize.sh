#!/usr/bin/env bash
# End-to-end export pipeline: merge LoRA, convert HF → GGUF f16, quantize → Q4_K_M.
# Run from the heliAI/ project root after training finishes.
#
# Required env vars (adjust paths):
#   BASE_MODEL         /abs/path/to/gemma-3-4b-it (fp16 HF)
#   LORA_DIR           train/outputs/gemma-aviation-lora
#   LLAMA_CPP_DIR      /abs/path/to/llama.cpp (with built binaries)
#
# Optional:
#   OUT_NAME           default: gemma-aviation-4b
#   QUANT              default: Q4_K_M

set -euo pipefail

: "${BASE_MODEL:?set BASE_MODEL to the fp16 HF base model directory}"
: "${LORA_DIR:=train/outputs/gemma-aviation-lora}"
: "${LLAMA_CPP_DIR:?set LLAMA_CPP_DIR to your local llama.cpp clone}"
: "${OUT_NAME:=gemma-aviation-4b}"
: "${QUANT:=Q4_K_M}"

MERGED_DIR="export/${OUT_NAME}-merged"
GGUF_F16="export/${OUT_NAME}.f16.gguf"
GGUF_QUANT="export/${OUT_NAME}.${QUANT}.gguf"

echo "[1/3] merging LoRA into base..."
python export/merge_lora.py \
    --base-model "$BASE_MODEL" \
    --lora-dir "$LORA_DIR" \
    --out-dir "$MERGED_DIR"

echo "[2/3] converting HF → GGUF f16..."
python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
    "$MERGED_DIR" \
    --outfile "$GGUF_F16" \
    --outtype f16

echo "[3/3] quantizing → ${QUANT}..."
QUANT_BIN="$LLAMA_CPP_DIR/build/bin/llama-quantize"
if [[ ! -x "$QUANT_BIN" ]]; then
    QUANT_BIN="$LLAMA_CPP_DIR/llama-quantize"
fi
"$QUANT_BIN" "$GGUF_F16" "$GGUF_QUANT" "$QUANT"

echo
echo "[done] final iPad-ready file: $GGUF_QUANT"
ls -lh "$GGUF_QUANT"
