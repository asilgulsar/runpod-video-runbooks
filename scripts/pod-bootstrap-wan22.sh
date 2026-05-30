#!/usr/bin/env bash
# Bootstrap a RunPod pod for Wan 2.2 TI2V 5B.
# Much simpler than the LTX path: diffusers supports Wan natively, so there is
# no source build and no transformers downgrade. ~30-45s on a fresh pod.
#   cat scripts/pod-bootstrap-wan22.sh | ssh -p $PORT root@$IP 'bash -s'
set -euo pipefail

echo ">> [1/2] system packages (ffmpeg, aria2)"
apt-get update -qq
apt-get install -y -qq ffmpeg aria2

echo ">> [2/2] python deps"
# accelerate is required for low_cpu_mem_usage to work; imageio + imageio-ffmpeg
# are what diffusers' export_to_video uses (without them it falls back to cv2).
pip install --break-system-packages -q \
  diffusers transformers safetensors accelerate hf_transfer imageio imageio-ffmpeg

mkdir -p /workspace/outputs
echo ">> bootstrap complete."
echo ">> NEXT: verify the GPU, then run download_wan22.sh"
echo "   python3 -c 'import torch; print(\"cuda:\", torch.cuda.is_available())'"
