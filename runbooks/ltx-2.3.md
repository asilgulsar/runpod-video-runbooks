# LTX-2.3 (22B distilled) on RunPod

A validated, end-to-end recipe for rendering video with **LTX-2.3 22B distilled** on a fresh RunPod pod. Each step here was burned in by trial and error; the version pins matter.

- **Model:** [Lightricks LTX-2](https://github.com/Lightricks/LTX-2) — 22B distilled checkpoint, fixed 8 steps
- **Hardware:** single H100 (94/80 GB) or A100 80 GB
- **Throughput:** ~26-29s per 5-second clip (832x512, 121 frames) on an H100 NVL
- **Cost:** roughly **$0.02 per 5-second clip** of compute

> Software stacks for open video models drift fast. This was validated in 2026-04 against the versions pinned below. If something breaks, start by checking the [known issues](#known-issues) table — the error string you're seeing is probably there.

---

## The 4 gotchas that waste the most time

1. **Pick the right pod image: `runpod/pytorch:1.0.3-cu1300-torch290-ubuntu2404`.** Older `torch 2.4 / cu12.4` images ship a stale NCCL that breaks when the LTX install pulls a newer torch. The torch 2.9 / CUDA 13 image works.
2. **Driver-vs-CUDA mismatch is the #1 runtime failure.** Community pods sometimes have an older driver (570.x, max CUDA 12.8) even when the image targets CUDA 13. If `torch.cuda.is_available()` is `False` after install, reinstall torch for cu128 (see [step 5](#5-driver--cuda-fix-only-if-cuda-is-false)).
3. **`transformers` must be `>=4.55,<5.0`.** transformers 5.x flattens `SiglipVisionModel`, so the nested `model.vision_tower.vision_model` path LTX expects disappears. Pin it and forget it.
4. **Install `ltx-pipelines` from source.** PyPI only ships the current wheel for Python 3.11; on 3.12 (Ubuntu 24.04 default) pip silently resolves an ancient version. Build from the GitHub source — and install `uv` + `uv-build` first, because the package uses `uv_build` as its build backend.

---

## Step by step

The three scripts referenced below live in [`../scripts/`](../scripts).

### 1. Spawn the pod

Spawn an on-demand pod via the RunPod GraphQL API. Export your key first: `export RUNPOD_API_KEY=...`

```bash
PUBKEY=$(cat ~/.ssh/id_ed25519.pub)
curl -s -X POST https://api.runpod.io/graphql \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"mutation {
    podFindAndDeployOnDemand(input: {
      cloudType: COMMUNITY, gpuCount: 1,
      volumeInGb: 200, containerDiskInGb: 60,
      minVcpuCount: 8, minMemoryInGb: 80,
      gpuTypeId: \\\"NVIDIA H100 NVL\\\",
      name: \\\"ltx23\\\",
      imageName: \\\"runpod/pytorch:1.0.3-cu1300-torch290-ubuntu2404\\\",
      ports: \\\"22/tcp\\\",
      volumeMountPath: \\\"/workspace\\\",
      env: [{key: \\\"PUBLIC_KEY\\\", value: \\\"$PUBKEY\\\"}]
    }) { id costPerHr machine { gpuDisplayName } }
  }\"}"
```

**GPU fallback cascade** — on a `SUPPLY_CONSTRAINT` error, try the next one:

| Order | GPU | Pool | VRAM | ~$/hr |
|---|---|---|---|---|
| 1 | NVIDIA H100 NVL | community | 94 GB | 2.59 |
| 2 | NVIDIA H100 PCIe | community | 80 GB | 1.99 |
| 3 | NVIDIA A100 80GB PCIe | community | 80 GB | 1.19 |
| 4 | NVIDIA H100 80GB HBM3 | community | 80 GB | 2.69 |
| 5 | NVIDIA A100 80GB PCIe | secure | 80 GB | 1.39 |

H100 PCIe community is frequently out of stock — don't rely on it. A100 80 GB is the reliable fallback (~1.5-2x slower per clip).

### 2. Set an auto-terminate guardrail

The `runpod/pytorch` container has no systemd, so an in-pod `shutdown` does nothing. Use an external timer so a hang doesn't bill you for hours:

```bash
POD_ID=...   # from step 1
nohup bash -c "sleep 5400 && curl -s -X POST https://api.runpod.io/graphql \
  -H 'Authorization: Bearer $RUNPOD_API_KEY' -H 'Content-Type: application/json' \
  -d '{\"query\":\"mutation { podTerminate(input: {podId: \\\"$POD_ID\\\"}) }\"}'" \
  > /dev/null 2>&1 &
echo $! > /tmp/terminate_pid.txt   # kill this PID after a clean teardown
```

5400s = 90 min. Use 7200s for longer sessions. Always `kill` the saved PID after you tear down manually, or it will fire on your next pod.

### 3. Wait for SSH, then bootstrap

Boot + image pull is ~3 min. SSH is on a dynamically mapped public port — poll the pod's `runtime.ports` until port 22 appears, then:

```bash
cat scripts/pod-bootstrap-ltx23.sh | ssh -i ~/.ssh/id_ed25519 \
  -o StrictHostKeyChecking=no -p "$PORT" root@"$IP" 'bash -s'
```

The script installs ffmpeg/aria2/git, the uv build backend + diffusers, builds `ltx-core` and `ltx-pipelines` from source, and pins transformers. ~40s on an H100.

### 4. Verify the GPU first

Before downloading 67 GB of weights, confirm CUDA actually works:

```bash
ssh ... 'python3 -c "import torch; print(torch.cuda.is_available())"'
```

### 5. Driver / CUDA fix (only if CUDA is `False`)

The pod has an older driver than the image targets. Reinstall torch for cu128 — **without** `--no-deps` (that strips the CUDA runtime libs and you'll get `libcudart.so.12: cannot open shared object file`):

```bash
ssh ... 'pip install --break-system-packages --force-reinstall \
  torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128'
```

Re-check `torch.cuda.is_available()` — it should now be `True`.

### 6. Download weights

```bash
cat scripts/download_models.sh | ssh ... 'bash -s'
```

Pulls the 43 GB distilled checkpoint, the 950 MB spatial upscaler, and the ~24 GB Gemma text encoder (~67 GB, ~2 min on H100 nodes).

### 7. Render

```bash
# copy the wrapper up once
cat scripts/inference.py | ssh ... 'cat > /workspace/inference.py'

ssh ... 'python3 /workspace/inference.py \
  --prompt "a lone figure walking across red desert dunes at dawn, drone shot" \
  --output /workspace/outputs/clip.mp4'
```

**Locked constraints** (the wrapper validates these so you don't fail mid-run):

| Flag | Rule | Example |
|---|---|---|
| `--height` / `--width` | multiples of 64 | `512` x `832` (832x480 fails) |
| `--num-frames` | `(8 * K) + 1` | `121` = 8x15 + 1 |
| steps | fixed at 8, no flag | (distilled is internal) |
| `--frame-rate` | default 24 | higher works; gen time scales with frame count |

### 8. Pull outputs and tear down

```bash
scp -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no -P "$PORT" \
  root@"$IP":/workspace/outputs/'*.mp4' ./outputs/

curl -s -X POST https://api.runpod.io/graphql \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$POD_ID\\\"}) }\"}"

kill "$(cat /tmp/terminate_pid.txt)" 2>/dev/null || true   # cancel the guardrail timer
```

---

## Performance (validated 2026-04)

| Spec | Value |
|---|---|
| GPU | H100 NVL 94 GB (community, ~$2.59/hr) |
| Per 5.04s clip (832x512, 121f, 8 steps) | **26-29s** end to end |
| VRAM during inference | ~50-55 GB |
| Compute cost per clip | **~$0.02** |
| Cold start (bootstrap + 67 GB download) | ~5 min one-time |
| 8-clip session (cold pod) | ~16 min, ~$0.70 |

The ~5-min cold-start tax amortizes fast across a multi-clip session.

---

## Known issues

| Symptom | Cause | Fix |
|---|---|---|
| `'SiglipVisionModel' object has no attribute 'vision_model'` | transformers 5.x flattened the structure | `pip install "transformers>=4.55,<5.0"` |
| `scaled_dot_product_attention() got an unexpected keyword argument 'enable_gqa'` | torch <2.5 missing the kwarg diffusers passes | upgrade torch to >=2.6 |
| `infer_schema(func): Parameter q has unsupported type 'torch.Tensor'` | torch 2.4 custom_op can't introspect the new union syntax | upgrade torch (root fix) |
| `libcudart.so.12: cannot open shared object file` | a `pip install --no-deps` skipped the CUDA runtime libs | reinstall torch **without** `--no-deps` |
| `ncclCommWindowDeregister` undefined symbol | a too-new torch wheel needs newer NCCL than the pod has | stay on torch 2.9; don't let the install force-upgrade it |
| `CUDA initialization: NVIDIA driver too old (12080)` | pod driver 570.x, image built for CUDA 13 | reinstall torch for cu128 ([step 5](#5-driver--cuda-fix-only-if-cuda-is-false)) |
| `unrecognized arguments: --num-inference-steps` | distilled CLI dropped the flag | remove it; distilled is a fixed 8 steps |
| `ValueError: Resolution (832x480) is not divisible by 64` | two-stage pipeline requires x64 dims | use 832x512 or 768x432 |
| `hf download --include` returns instantly with no files | hf CLI bug when downloads are backgrounded | use the positional file-arg syntax instead |
| `No matching distribution found for ltx-pipelines==<ver>` | PyPI wheel is py3.11-only | install from GitHub source (the bootstrap script does this) |

---

## Note on text overlays (local post-processing)

Homebrew `ffmpeg` on macOS is often built **without** `--enable-libfreetype`, so the `drawtext` filter is unavailable. The portable workaround: render each text card as a transparent PNG (e.g. with PIL), then composite with ffmpeg's `overlay` filter (which is always present) and stitch clips with `xfade` crossfades.

---

## Credits

LTX-2 is built and released by [Lightricks](https://github.com/Lightricks/LTX-2). This runbook only documents how to run it on RunPod — check the model's own license for usage terms.
