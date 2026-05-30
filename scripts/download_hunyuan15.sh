#!/usr/bin/env bash
# Download HunyuanVideo 1.5 (480p T2V distilled, ~52 GB) into the Hugging Face cache.
#
# IMPORTANT: use the community diffusers-layout repo, NOT tencent/HunyuanVideo-1.5.
# The official Tencent repo ships 8 transformer variants (~15 GB each, 100+ GB
# total) and a custom non-diffusers pipeline class, so a naive download balloons.
set -euo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=1

hf download hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v_distilled
echo ">> cached HunyuanVideo-1.5-Diffusers-480p_t2v_distilled"
