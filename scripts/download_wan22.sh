#!/usr/bin/env bash
# Download Wan 2.2 TI2V 5B (Apache-2.0, ~32 GB) into the Hugging Face cache so
# that WanPipeline.from_pretrained(MODEL_ID) loads it fast and offline.
set -euo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=1

hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers
echo ">> cached Wan-AI/Wan2.2-TI2V-5B-Diffusers"
