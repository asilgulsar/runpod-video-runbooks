#!/usr/bin/env python3
"""Thin wrapper around `python3 -m ltx_pipelines.distilled` for LTX-2.3 22B distilled.

Validates the two constraints that otherwise throw partway through a run:
  - height and width must be multiples of 64
  - num_frames must satisfy (8 * K) + 1   (e.g. 121 = 8*15 + 1)

The distilled CLI in 1.1.2 runs a fixed 8 steps internally, so there is no
--num-inference-steps flag to pass.

Usage:
    python3 inference.py --prompt "a desert at dawn, drone shot" \\
        --output /workspace/outputs/clip.mp4
"""
import argparse
import subprocess
import sys
from pathlib import Path

MODELS = Path("/workspace/models")
DISTILLED = MODELS / "LTX-2.3" / "ltx-2.3-22b-distilled-1.1.safetensors"
UPSAMPLER = MODELS / "LTX-2.3" / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
GEMMA_ROOT = MODELS / "gemma"


def main() -> int:
    p = argparse.ArgumentParser(description="Render an LTX-2.3 distilled clip.")
    p.add_argument("--prompt", required=True)
    p.add_argument("--output", default="/workspace/outputs/clip.mp4")
    p.add_argument("--height", type=int, default=512, help="multiple of 64")
    p.add_argument("--width", type=int, default=832, help="multiple of 64")
    p.add_argument("--num-frames", type=int, default=121, help="(8*K)+1")
    p.add_argument("--frame-rate", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.height % 64 or args.width % 64:
        sys.exit(f"height/width must be multiples of 64 (got {args.width}x{args.height})")
    if (args.num_frames - 1) % 8:
        sys.exit(f"num-frames must be (8*K)+1, e.g. 121 (got {args.num_frames})")
    for path in (DISTILLED, UPSAMPLER, GEMMA_ROOT):
        if not path.exists():
            sys.exit(f"missing model path: {path} -- run download_models.sh first")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "ltx_pipelines.distilled",
        "--distilled-checkpoint-path", str(DISTILLED),
        "--gemma-root", str(GEMMA_ROOT),
        "--spatial-upsampler-path", str(UPSAMPLER),
        "--prompt", args.prompt,
        "--output-path", args.output,
        "--height", str(args.height),
        "--width", str(args.width),
        "--num-frames", str(args.num_frames),
        "--frame-rate", str(args.frame_rate),
        "--seed", str(args.seed),
    ]
    print(f">> rendering {args.width}x{args.height}, {args.num_frames}f -> {args.output}", flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
