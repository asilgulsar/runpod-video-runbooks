# Wan 2.2 (TI2V 5B) on RunPod

A validated recipe for rendering video with **Wan 2.2 TI2V 5B** on RunPod. Wan is **Apache-2.0** and supported natively by `diffusers`, which makes it much simpler to stand up than LTX — no source build, no `transformers` downgrade, no separate text encoder.

- **Model:** [`Wan-AI/Wan2.2-TI2V-5B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) — Apache-2.0
- **Hardware:** single H100 or A100 80 GB (peak VRAM ~34 GB)
- **Throughput:** ~53s per 5-second clip (832x480, 81 frames, 40 steps) on an H100 NVL
- **Cost:** roughly **$0.04 per 5-second clip** of compute

> Shares most of its operational steps with the [LTX-2.3 runbook](ltx-2.3.md) — pod spawn, the auto-terminate guardrail, the SSH poll, the driver/CUDA fix, and teardown are identical. This page covers the Wan-specific deltas.

---

## The one gotcha that isn't in the LTX runbook

**`low_cpu_mem_usage=True` must be passed explicitly to both model loads.** With `accelerate` installed (which you want — it's required for this flag to function), `diffusers` 0.37 defaults `low_cpu_mem_usage` to `False`. But Wan's VAE sets `keep_in_fp32_modules=True`, and the two can't coexist as `False`/`True`. The load fails with:

```
ValueError: `low_cpu_mem_usage` cannot be False when `keep_in_fp32_modules` is True.
```

Pass it explicitly on **both** `AutoencoderKLWan.from_pretrained(...)` and `WanPipeline.from_pretrained(...)`. The included [`wan_inference.py`](../scripts/wan_inference.py) already does this.

---

## Step by step

Steps 1-3 (spawn the pod, set the guardrail, wait for SSH) are identical to the [LTX-2.3 runbook](ltx-2.3.md#step-by-step) — same pod image (`runpod/pytorch:1.0.3-cu1300-torch290-ubuntu2404`) and same GPU fallback cascade.

### 4. Bootstrap

```bash
cat scripts/pod-bootstrap-wan22.sh | ssh -i ~/.ssh/id_ed25519 \
  -o StrictHostKeyChecking=no -p "$PORT" root@"$IP" 'bash -s'
```

Just `apt` (ffmpeg, aria2) and `pip` (diffusers, transformers, safetensors, accelerate, hf_transfer, imageio, imageio-ffmpeg). ~30-45s — no git clone, no build step.

### 5. Verify the GPU, then download weights

```bash
ssh ... 'python3 -c "import torch; print(torch.cuda.is_available())"'   # must be True
cat scripts/download_wan22.sh | ssh ... 'bash -s'                       # ~32 GB into HF cache
```

If CUDA is `False`, apply the same cu128 torch reinstall as the [LTX driver fix](ltx-2.3.md#5-driver--cuda-fix-only-if-cuda-is-false).

### 6. Render

```bash
cat scripts/wan_inference.py | ssh ... 'cat > /workspace/wan_inference.py'

ssh ... 'python3 /workspace/wan_inference.py \
  --prompt "a perfume bottle on wet black stone, slow macro push-in" \
  --output /workspace/outputs/clip.mp4'
```

**Locked constraints** (validated):

| Flag | Rule | Default |
|---|---|---|
| `--height` / `--width` | no strict x64 like LTX, but x16 is safe | `480` x `832` |
| `--num-frames` | `(4 * K) + 1` | `81` = 4x20 + 1 |
| `--fps` | output frame rate | `16` (81f -> 5.06s) |
| `--steps` | lower = faster, mild quality loss | `40` |
| `--guidance` | classifier-free guidance scale | `5.0` |

### 7. Pull outputs and tear down

Identical to the [LTX teardown](ltx-2.3.md#8-pull-outputs-and-tear-down): `scp` the clips back, `podTerminate`, and kill the guardrail timer.

---

## Performance (validated 2026-04)

| Spec | Value |
|---|---|
| GPU | H100 NVL 94 GB (~$2.59/hr) |
| Per 5.06s clip (832x480, 81f, 40 steps) | **~53s** (6s load + 47s inference) |
| VRAM peak | **~34 GB** |
| Compute cost per clip | **~$0.04** |
| 4-clip session (cold pod) | ~14 min, ~$0.52 |

## Wan 2.2 5B vs LTX 2.3 22B (same H100 NVL)

| Metric | Wan 2.2 5B | LTX 2.3 22B distilled |
|---|---|---|
| Inference per 5s clip | ~47s | ~22s |
| End to end (incl. load) | ~53s | ~29s |
| VRAM peak | ~34 GB | ~50-55 GB |
| Cost per clip | ~$0.04 | ~$0.02 |
| Setup | 30-45s, no source build | ~40s + source build |
| License | **Apache-2.0** | check the LTX-2 license terms |

**Rule of thumb:** reach for **Wan 2.2** when a permissive license matters or you want consistent quality across many shots; reach for **LTX 2.3** when raw speed and cost-per-clip are the priority.

---

## Known issues (Wan-specific)

| Symptom | Cause | Fix |
|---|---|---|
| `low_cpu_mem_usage cannot be False when keep_in_fp32_modules is True` | flag not passed explicitly | pass `low_cpu_mem_usage=True` to **both** loads (the script does this) |
| `export_to_video requires the OpenCV library` | imageio missing, falls back to cv2 which is also missing | `pip install imageio imageio-ffmpeg` (the bootstrap does this) |
| inference >2 min per clip | probably running on CPU | apply the cu128 torch reinstall (driver mismatch) |

All shared driver/CUDA gotchas live in the [LTX runbook's known-issues table](ltx-2.3.md#known-issues).

---

## Credits

Wan 2.2 is built and released by [Wan-AI](https://huggingface.co/Wan-AI) under Apache-2.0. This runbook documents how to run it on RunPod.
