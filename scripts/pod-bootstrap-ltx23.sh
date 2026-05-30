#!/usr/bin/env bash
# Bootstrap a fresh RunPod pod for LTX-2.3 22B distilled inference.
# Run this AFTER the pod is up and you can SSH in, e.g.:
#   cat scripts/pod-bootstrap-ltx23.sh | ssh -p $PORT root@$IP 'bash -s'
#
# Assumes the pod image runpod/pytorch:1.0.3-cu1300-torch290-ubuntu2404
# (or another image with torch 2.9 / CUDA 13). See runbooks/ltx-2.3.md.
set -euo pipefail

echo ">> [1/4] system packages (ffmpeg, aria2, git)"
apt-get update -qq
apt-get install -y -qq ffmpeg aria2 git

echo ">> [2/4] python deps (uv build backend + hf + diffusers)"
# uv + uv-build are needed FIRST: ltx packages use uv_build as their build
# backend, so a source install fails without them.
pip install --break-system-packages -q \
  uv uv-build hf_transfer "huggingface_hub[cli]" diffusers safetensors

echo ">> [3/4] LTX-2 from source"
# PyPI only ships ltx-pipelines wheels for Python 3.11; on 3.12 you must build
# from source or pip silently resolves an ancient version.
cd /opt
rm -rf ltx2
git clone --depth 1 https://github.com/Lightricks/LTX-2.git ltx2
pip install --break-system-packages /opt/ltx2/packages/ltx-core /opt/ltx2/packages/ltx-pipelines

echo ">> [4/4] pin transformers (5.x breaks ltx-core's vision tower path)"
pip install --break-system-packages "transformers>=4.55,<5.0"

mkdir -p /workspace/outputs
echo ">> bootstrap complete."
echo ">> NEXT: verify the GPU, then run download_models.sh"
echo "   python3 -c 'import torch; print(\"cuda:\", torch.cuda.is_available())'"
