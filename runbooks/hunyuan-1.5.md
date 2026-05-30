# HunyuanVideo 1.5 (480p T2V distilled) on RunPod

A validated recipe for rendering video with **HunyuanVideo 1.5** (Tencent, 8.3B, 480p text-to-video distilled) on RunPod via `diffusers`. It's diffusers-native like Wan, but has a few sharp edges: the obvious Hugging Face repo is the wrong one, it needs CPU offload to fit, and its guidance API is different.

- **Model:** [`hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v_distilled`](https://huggingface.co/hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v_distilled) (~52 GB)
- **Hardware:** H100 NVL strongly preferred (higher VRAM ceiling); A100 80 GB works with tighter margins
- **Throughput:** ~77s per 2.54s clip (832x480, 61 frames, 8 steps, with offload) on an H100 NVL
- **License:** Tencent Hunyuan Community — check the [license](https://huggingface.co/hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v_distilled) before deploying (notably it carries regional restrictions)

> Shares pod spawn, the guardrail, the SSH poll, the driver/CUDA fix, and teardown with the [LTX-2.3 runbook](ltx-2.3.md). This page covers the HunyuanVideo-specific deltas.

---

## The 4 gotchas

### 1. The obvious repo is the wrong one

A naive `hf download tencent/HunyuanVideo-1.5` silently starts pulling **100+ GB** — that repo ships 8 transformer variants (480p/720p/1080p x t2v/i2v x distilled/regular, ~15 GB each) and uses a **custom, non-diffusers pipeline class**. Use the community diffusers-layout conversion instead:

```
hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v_distilled   # ~52 GB, this runbook
```

Other community variants exist (`480p_t2v` non-distilled, `720p_t2v[_distilled]`, `i2v` variants) — swap the `MODEL_ID` in [`hunyuan_inference.py`](../scripts/hunyuan_inference.py) to use them.

### 2. CPU offload is required at 832x480 >= 61 frames

Without `pipe.enable_model_cpu_offload()`, VAE decode tries to allocate **~51 GB on top of the ~42 GB** already used by the model and OOMs even on a 94 GB H100 NVL:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 51.05 GiB.
```

With offload, peak VRAM stays ~36 GB. The trade-off is ~50% slower inference, but without it the render simply doesn't finish. The inference script enables offload (plus VAE tiling + slicing) by default.

### 3. `__call__()` does not accept `guidance_scale`

Unlike `WanPipeline`, `HunyuanVideo15Pipeline` puts guidance in a separate `pipe.guider` component. Passing `guidance_scale=...` raises `TypeError: unexpected keyword argument 'guidance_scale'`. To change it, set `pipe.guider.guidance_scale` after construction (the script exposes this as `--guidance`). The default of 6.0 is fine for the distilled variant.

### 4. The distilled variant runs 8 steps

Distilled is tuned for fast inference. 30+ steps waste compute with no quality gain. Use 8 (default) or 12 for a slight bump.

---

## Step by step

Steps 1-3 (spawn the pod, set the guardrail, wait for SSH) are identical to the [LTX-2.3 runbook](ltx-2.3.md#step-by-step). H100 NVL is preferred here for the VRAM headroom.

### 4. Bootstrap

```bash
cat scripts/pod-bootstrap-hunyuan15.sh | ssh -i ~/.ssh/id_ed25519 \
  -o StrictHostKeyChecking=no -p "$PORT" root@"$IP" 'bash -s'
```

`apt` (ffmpeg, aria2) + `pip` (diffusers>=0.37.1, accelerate, imageio, ...). ~30-45s, no source build. The `>=0.37.1` pin matters — that's where `HunyuanVideo15Pipeline` landed.

### 5. Verify the GPU, then download weights

```bash
ssh ... 'python3 -c "import torch; print(torch.cuda.is_available())"'   # must be True
cat scripts/download_hunyuan15.sh | ssh ... 'bash -s'                   # ~52 GB into HF cache
```

If CUDA is `False`, apply the [cu128 torch reinstall](ltx-2.3.md#5-driver--cuda-fix-only-if-cuda-is-false).

### 6. Render

```bash
cat scripts/hunyuan_inference.py | ssh ... 'cat > /workspace/hunyuan_inference.py'

ssh ... 'python3 /workspace/hunyuan_inference.py \
  --prompt "a perfume bottle on wet black stone, slow macro push-in" \
  --output /workspace/outputs/clip.mp4'
```

**Tested configurations:**

| Resolution | Frames | Result |
|---|---|---|
| 640x384 | 25 (1.04s) | ~20s end to end (smoke test) |
| **832x480** | **61 (2.54s)** | **~77s, default** |
| 832x480 | 121 (5.04s) | OOMs even with offload — drop to 640x384 or use a 720p variant |

### 7. Pull outputs and tear down

Identical to the [LTX teardown](ltx-2.3.md#8-pull-outputs-and-tear-down).

---

## Performance (validated 2026-04)

| Spec | Value |
|---|---|
| GPU | H100 NVL 94 GB (~$2.59/hr) |
| Per 2.54s clip (832x480, 61f, 8 steps) | **~77s** (7s load + 62s inference + ~8s offload overhead) |
| VRAM peak | **~36 GB** (with offload) |
| Compute cost per clip | **~$0.06** |
| 4-clip session (cold pod) | ~32 min, ~$0.83 |

---

## How the three models compare (same H100 NVL)

| Metric | LTX 2.3 22B distilled | Wan 2.2 5B | HunyuanVideo 1.5 distilled |
|---|---|---|---|
| Time per clip | ~29s (5s clip) | ~53s (5s clip) | ~77s (2.5s clip; 5s OOMs at 832x480) |
| VRAM peak | 50-55 GB | ~34 GB | ~36 GB (offload) |
| Setup | source build | no build | no build |
| Diffusers-native | no (ltx-pipelines from source) | yes | yes (0.37.1+) |
| License | check LTX-2 terms | **Apache-2.0** | Tencent Community (regional limits) |
| Reach for it when... | speed/cost is king | permissive license, consistent quality | comparing the open frontier, or you need the 720p variant |

**Bottom line:** HunyuanVideo 1.5 is the most expensive per clip of the three and OOMs at 5-second clips on 832x480. Pick it for specific quality/license reasons or the 720p variant — not as a throughput default.

---

## Known issues (HunyuanVideo-specific)

| Symptom | Cause | Fix |
|---|---|---|
| download balloons to 100+ GB | pulled `tencent/HunyuanVideo-1.5` (8 variants) | use the `hunyuanvideo-community/...-Diffusers-480p_t2v_distilled` repo |
| `_class_name: HunyuanVideo_1_5_Pipeline` not found | the Tencent repo uses a custom non-diffusers pipeline | use the community conversion; class is `HunyuanVideo15Pipeline` (diffusers >=0.37.1) |
| `unexpected keyword argument 'guidance_scale'` | guidance lives on `pipe.guider`, not `__call__` | don't pass it; set `pipe.guider.guidance_scale` (the `--guidance` flag) |
| `OutOfMemoryError: Tried to allocate 51.05 GiB` | VAE decode at 832x480 >= 61f | enable CPU offload (default) and/or lower frames/resolution |
| inference 2-3x slower than expected | CPU offload shuffles components per step | expected trade-off; only drop offload with real VRAM headroom |

Shared driver/CUDA gotchas live in the [LTX runbook's known-issues table](ltx-2.3.md#known-issues).

---

## Credits

HunyuanVideo 1.5 is built by [Tencent](https://huggingface.co/tencent); the diffusers-layout conversion used here is maintained by the [hunyuanvideo-community](https://huggingface.co/hunyuanvideo-community) org. Check the model's license before commercial use.
