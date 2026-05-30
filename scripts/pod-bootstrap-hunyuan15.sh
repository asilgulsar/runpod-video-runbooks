#!/usr/bin/env bash
# Bootstrap a RunPod pod for HunyuanVideo 1.5 (480p T2V distilled).
# Like the Wan path: diffusers-native, no source build. Needs diffusers >=0.37.1
# for the HunyuanVideo15Pipeline class.
#   cat scripts/pod-bootstrap-hunyuan15.sh | ssh -p $PORT root@$IP 'bash -s'
set -euo pipefail

echo ">> [1/2] system packages (ffmpeg, aria2)"
apt-get update -qq
apt-get install -y -qq ffmpeg aria2

echo ">> [2/2] python deps"
pip install --break-system-packages -q \
  "diffusers>=0.37.1" transformers safetensors accelerate hf_transfer imageio imageio-ffmpeg

mkdir -p /workspace/outputs
echo ">> bootstrap complete."
echo ">> NEXT: verify the GPU, then run download_hunyuan15.sh"
echo "   python3 -c 'import torch; print(\"cuda:\", torch.cuda.is_available())'"
