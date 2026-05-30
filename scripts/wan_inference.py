#!/usr/bin/env python3
"""Wan 2.2 TI2V 5B text-to-video via diffusers.

The one non-obvious gotcha: `low_cpu_mem_usage=True` must be passed explicitly to
BOTH AutoencoderKLWan.from_pretrained AND WanPipeline.from_pretrained. With
accelerate installed, diffusers defaults it to False, but Wan's VAE has
keep_in_fp32_modules=True, which requires it True -- otherwise the load raises:

    ValueError: `low_cpu_mem_usage` cannot be False when `keep_in_fp32_modules` is True.

Usage:
    python3 wan_inference.py --prompt "a perfume bottle on wet stone, macro" \\
        --output /workspace/outputs/clip.mp4
"""
import argparse
import json
import time
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"


def main() -> None:
    p = argparse.ArgumentParser(description="Render a Wan 2.2 TI2V 5B clip.")
    p.add_argument("--prompt", required=True)
    p.add_argument("--output", default="/workspace/outputs/clip.mp4")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num-frames", type=int, default=81, help="(4*K)+1")
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--guidance", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--metrics", help="append a JSON line with timing to this file")
    args = p.parse_args()

    if (args.num_frames - 1) % 4:
        raise SystemExit(f"num-frames should be (4*K)+1, e.g. 81 (got {args.num_frames})")

    t0 = time.time()
    # low_cpu_mem_usage=True is mandatory on BOTH loads (see module docstring).
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    pipe = WanPipeline.from_pretrained(
        MODEL_ID, vae=vae, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    pipe.to("cuda")
    t_load = time.time() - t0

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    t1 = time.time()
    frames = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        generator=generator,
    ).frames[0]
    t_infer = time.time() - t1

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, args.output, fps=args.fps)
    print(f">> {args.output}  (load {t_load:.1f}s, infer {t_infer:.1f}s)")

    if args.metrics:
        with open(args.metrics, "a") as f:
            f.write(json.dumps({
                "output": args.output,
                "prompt": args.prompt,
                "height": args.height,
                "width": args.width,
                "num_frames": args.num_frames,
                "steps": args.steps,
                "guidance": args.guidance,
                "seed": args.seed,
                "load_s": round(t_load, 1),
                "infer_s": round(t_infer, 1),
            }) + "\n")


if __name__ == "__main__":
    main()
