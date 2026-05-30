#!/usr/bin/env bash
# Download LTX-2.3 22B distilled weights + the Gemma text encoder.
# Run on the pod after bootstrap. ~67 GB total; with HF_HUB_ENABLE_HF_TRANSFER=1
# this pulls at ~30-40 GB/min on H100-class nodes.
#
# NOTE: repo IDs and filenames track upstream releases and may change over time.
# If a download 404s, check the model card on Hugging Face for current names.
set -euo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p /workspace/models/LTX-2.3 /workspace/models/gemma

echo ">> main distilled checkpoint + spatial upscaler"
# Use the positional file-arg syntax, NOT --include: with some hf CLI versions
# --include returns instantly with no files when downloads are backgrounded.
hf download Lightricks/LTX-2.3 \
  ltx-2.3-22b-distilled-1.1.safetensors \
  ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --local-dir /workspace/models/LTX-2.3

echo ">> gemma text encoder (full repo)"
hf download Lightricks/gemma-3-12b-it-qat-q4_0-unquantized \
  --local-dir /workspace/models/gemma

echo ">> done. models under /workspace/models"
