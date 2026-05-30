# Benchmarks: LTX-2.3 vs Wan 2.2 vs HunyuanVideo 1.5

Throughput and cost numbers for the three runbook models, measured on the **same GPU** (H100 NVL 94 GB, ~$2.59/hr on RunPod) with each model's **default shipping recipe**. The raw numbers come straight from the per-model runbooks; this page adds length- and resolution-normalized metrics so the three are actually comparable, plus an honest account of what these numbers do and don't tell you.

> **Read the caveats before quoting any of this.** These are single-run figures with each model on a *different* default recipe (different clip length, resolution, and step count). They compare **default recipes**, not models at an equal step budget. See [Caveats](#caveats--how-to-read-this).

## Raw numbers (as published in the runbooks)

| Model | Params | Pipeline | Clip | Res | Frames | Steps | Total time | VRAM peak | $/clip | License |
|---|---|---|---|---|---|---|---|---|---|---|
| [LTX-2.3](../runbooks/ltx-2.3.md) | 22B distilled | ltx-pipelines (source) | 5.04s | 832x512 | 121 | 8 | **26-29s** | 50-55 GB | ~$0.02 | LTX-2 terms |
| [Wan 2.2](../runbooks/wan-2.2.md) | 5B | WanPipeline (diffusers) | 5.06s | 832x480 | 81 | 40 | **~53s** | ~34 GB | ~$0.04 | Apache-2.0 |
| [HunyuanVideo 1.5](../runbooks/hunyuan-1.5.md) | 8.3B distilled | HunyuanVideo15Pipeline (diffusers) | 2.54s | 832x480 | 61 | 8 | **~77s** | ~36 GB | ~$0.06 | Tencent Community |

HunyuanVideo's clip is **half the length** because a 5s clip OOMs at 832x480 even on a 94 GB card, and it runs with **CPU offload** (required to fit), which costs roughly 50% of its speed. Keep that in mind for every Hunyuan row below.

## Normalized metrics (the comparable view)

Raw "seconds per clip" is apples-to-oranges: a 5s LTX clip and a 2.5s Hunyuan clip aren't the same unit of work. These metrics divide out clip length, frame count, and resolution. Regenerate them anytime with `python3 benchmarks/plot.py`.

| Model | Steps | Compute-s per video-s | Frames per compute-s | **Pixel throughput (MP/s)** | $/video-s | VRAM |
|---|---|---|---|---|---|---|
| LTX-2.3 | 8 | 5.75 | 4.17 | **1.777** | $0.0040 | 50-55 GB |
| Wan 2.2 | 40 | 10.47 | 1.53 | **0.610** | $0.0079 | ~34 GB |
| HunyuanVideo 1.5 | 8 | 30.31 | 0.79 | **0.316** | $0.0236 | ~36 GB |

**Pixel throughput** = `(width x height x frames) / total_seconds` — the most apples-to-apples speed metric, since it accounts for both resolution and frame count. Relative to the fastest:

| Model | Pixel throughput | Cost per second of video |
|---|---|---|
| LTX-2.3 | **1.00x** (fastest) | **1.00x** (cheapest) |
| Wan 2.2 | 0.34x | 2.0x |
| HunyuanVideo 1.5 | 0.18x | 5.9x |

### What the normalized numbers say

- **LTX-2.3 is the throughput/cost leader by a wide margin** — ~2.9x Wan's pixel throughput and ~5.6x Hunyuan's, at the lowest cost per second of video. If you only care about volume of footage per dollar, it's not close.
- **Wan 2.2 pays for 40 steps.** It's the only un-distilled model here, running 5x the step count of the other two. Its per-step work is cheap (5B params), but 40 steps is a lot of forward passes. This is the single most obvious thing to optimize — see below.
- **HunyuanVideo 1.5 is the most expensive per second of video and the most VRAM-constrained** (offload-forced, half-length clips). You reach for it for specific quality/license reasons or its 720p variant — not as a throughput default. The runbook says the same.
- **VRAM and speed don't correlate the way you'd guess.** LTX is the *fastest* but uses the *most* VRAM (50-55 GB); the 5B Wan is slower but fits in ~34 GB. Parameter count drives VRAM; step count and per-step cost drive wall-clock.

## Caveats — how to read this

These numbers are useful for **order-of-magnitude planning**, not for a leaderboard. The honest limitations:

1. **Single run each.** `runs: 1` in the data. No averaging, no error bars, no warm-vs-cold-cache separation beyond the documented load time. Treat the figures as ±10-15%.
2. **Different default recipes, not an equal-footing fight.** Wan runs 40 steps (un-distilled); LTX and Hunyuan run 8 (step-distilled). This compares *what each model ships with*, which is what you'd actually run on day one — but it is **not** "which architecture is fastest at N steps." The equal-step comparison is exactly what the [optimization experiments](#what-to-optimize-next) below will measure.
3. **HunyuanVideo runs handicapped.** CPU offload (required to fit) costs ~50% of its speed, and its clip is half-length because 5s OOMs. Its numbers reflect "made it fit on this card," not the model's ceiling on bigger hardware.
4. **No quality scoring.** This is throughput and cost only. Visual quality is subjective, prompt-dependent, and not captured here. A faster model that needs more retries isn't actually cheaper. Quality benchmarking is future work.
5. **Point-in-time pricing.** $2.59/hr is a community H100 NVL rate from 2026-04. RunPod pricing and availability move; recompute cost with your own rate (it's a single field in `results.jsonl`).
6. **One GPU SKU.** All three on H100 NVL. Relative ordering may shift on A100 80 GB (less VRAM headroom hurts LTX and Hunyuan first) or on consumer cards.

## What to optimize next

The normalized table points straight at the highest-value experiments, which feed the [research note](../RESEARCH.md) (work in progress):

- **Cut Wan's step count.** 40 -> 8 steps via a Lightning / step-distillation LoRA could close most of the gap to LTX if quality holds. Biggest single lever in the table.
- **Get HunyuanVideo to 5s without offload.** VAE-decode OOM is the wall. Group offloading, temporal VAE tiling, or a quantized VAE are the candidates — if any lets it run un-offloaded, its throughput roughly doubles.
- **Quantize the VRAM hogs.** LTX sits at 50-55 GB; fp8 / int8 weight quant could buy headroom for higher resolution or longer clips on the same card.
- **Caching (TeaCache / FBCache)** advertises 1.5-2x on video DiTs with small quality cost — applies to all three in principle.

Each of these is a cheap, bounded RunPod experiment. The point of publishing the baseline is to measure those deltas against it honestly.

## Reproduce it yourself

1. Pick a model and follow its [runbook](../README.md) top to bottom on an H100 NVL.
2. The runbooks print load time, inference time, and VRAM peak; record them.
3. Drop a row into `benchmarks/results.jsonl` (schema below) and run `python3 benchmarks/plot.py` to regenerate the normalized table. Add `--charts` for PNG bar charts (needs `matplotlib`).

If your numbers differ materially, open an issue or PR — more data points (especially other GPUs, or averaged runs) are exactly what this folder wants.

## Data schema (`results.jsonl`)

One JSON object per line. Fields:

| Field | Meaning |
|---|---|
| `model`, `variant`, `params_b` | model name, variant, parameter count in billions |
| `distilled`, `diffusers_native` | step-distilled? runs in stock diffusers? |
| `pipeline`, `license` | pipeline class / install path, model license |
| `gpu`, `gpu_hourly_usd` | GPU SKU and the hourly rate used for cost |
| `resolution`, `frames`, `clip_seconds`, `steps` | the recipe that was run |
| `total_seconds` (+`_range`) | end-to-end wall time per clip (measured) |
| `load_seconds`, `inference_seconds`, `offload_overhead_seconds` | breakdown where the runbook reports it |
| `vram_peak_gb` (+`_range`), `cpu_offload` | peak VRAM (measured), whether offload was on |
| `cost_per_clip_usd` | derived: `total_seconds * gpu_hourly_usd / 3600` |
| `validated`, `runs`, `notes` | when validated, run count, free-text caveats |

Normalized metrics (`compute_s_per_video_s`, `frames_per_compute_s`, `megapixels_per_s`, `usd_per_video_s`) are **derived** by `plot.py`, not stored — so the data file stays raw and auditable.
