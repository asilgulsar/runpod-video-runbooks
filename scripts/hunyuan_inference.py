#!/usr/bin/env python3
"""HunyuanVideo 1.5 (480p T2V distilled) text-to-video via diffusers.

Four gotchas are handled here so you don't hit them at runtime:
  1. Uses the diffusers-layout community repo, NOT tencent/HunyuanVideo-1.5
     (that one is 100+ GB across 8 variants and uses a custom pipeline class).
  2. CPU offload is enabled by default -- at 832x480 >= 61 frames the VAE decode
     OOMs without it (tries to allocate ~51 GB on top of the model).
  3. The pipeline does NOT accept a guidance_scale kwarg -- guidance lives on the
     separate pipe.guider component. Use --guidance to set pipe.guider.guidance_scale.
  4. The distilled variant is tuned for 8 steps; more steps just waste compute.

Usage:
    python3 hunyuan_inference.py --prompt "a perfume bottle on wet stone, macro" \\
        --output /workspace/outputs/clip.mp4
"""
import argparse
import json
import time
from pathlib import Path

import torch
from diffusers import HunyuanVideo15Pipeline
from diffusers.utils import export_to_video

MODEL_ID = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v_distilled"


def main() -> None:
    p = argparse.ArgumentParser(description="Render a HunyuanVideo 1.5 distilled clip.")
    p.add_argument("--prompt", required=True)
    p.add_argument("--output", default="/workspace/outputs/clip.mp4")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num-frames", type=int, default=61,
                   help="61 fits at 832x480; for more frames, drop resolution to ~640x384")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--steps", type=int, default=8, help="distilled is tuned for 8")
    p.add_argument("--guidance", type=float, default=None,
                   help="optional; sets pipe.guider.guidance_scale (default 6.0)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-offload", action="store_true",
                   help="disable CPU offload (only with VRAM headroom; OOMs at 832x480 61f)")
    p.add_argument("--metrics", help="append a JSON line with timing to this file")
    args = p.parse_args()

    t0 = time.time()
    pipe = HunyuanVideo15Pipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    # gotcha 2: offload keeps VAE decode from OOMing at 832x480.
    if args.no_offload:
        pipe.to("cuda")
    else:
        pipe.enable_model_cpu_offload()
    # extra VRAM savings on the VAE
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    # gotcha 3: guidance is a separate component, not a __call__ kwarg.
    if args.guidance is not None:
        pipe.guider.guidance_scale = args.guidance
    t_load = time.time() - t0

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    t1 = time.time()
    frames = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
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
                "seed": args.seed,
                "load_s": round(t_load, 1),
                "infer_s": round(t_infer, 1),
            }) + "\n")


if __name__ == "__main__":
    main()
