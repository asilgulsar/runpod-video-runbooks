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
| LTX-2.3 | 8 | 5.75 | 4.17 | 1.777 | $0.0040 | 50-55 GB |
| Wan 40-step *(baseline)* | 40 | 10.47 | 1.53 | 0.610 | $0.0079 | ~34 GB |
| Hunyuan 61f *(baseline)* | 8 | 30.31 | 0.79 | 0.316 | $0.0236 | ~36 GB |
| **Wan Turbo 4-step** | 4 | 1.58 | 10.12 | **4.044** | $0.0014 | ~34 GB |
| **Wan Turbo 8-step** | 8 | 1.90 | 8.44 | **3.370** | $0.0018 | ~34 GB |
| Hunyuan 5s 832x480 | 8 | 29.68 | 0.81 | 0.323 | $0.0258 | 68 GB |
| Hunyuan 5s 640x384 | 8 | 16.09 | 1.49 | 0.367 | $0.0139 | 68 GB |

The bottom four rows are **Tier A measured optimizations** (this PR — step-distillation and VAE-tiling); the top three are default-recipe baselines. Regenerate with `python3 benchmarks/plot.py`.

**Pixel throughput** = `(width x height x frames) / total_seconds` — the most apples-to-apples speed metric, since it accounts for both resolution and frame count. Relative to the fastest:

| Model | Pixel throughput | Cost per second of video |
|---|---|---|
| Wan Turbo 4-step | **1.00x** (fastest) | **1.00x** (cheapest) |
| Wan Turbo 8-step | 0.83x | 1.3x |
| LTX-2.3 | 0.44x | 2.9x |
| Wan 40-step *(baseline)* | 0.15x | 5.6x |
| Hunyuan 5s 640x384 | 0.09x | 9.9x |
| Hunyuan 5s 832x480 | 0.08x | 18.4x |
| Hunyuan 61f *(baseline)* | 0.08x | 16.9x |

> **Basis note.** The Tier A rows use **warm-render** `total_seconds` (steady-state per-clip render, the right metric for a served model); their one-time cold load is in `load_seconds` separately. The baseline rows include a small (~6-7s) load in `total_seconds`. So Wan Turbo's lead is a *render-throughput* lead — see [Measured results](#measured-results-tier-a) for the same-basis, same-pod deltas. Tier A ran on a $3.19/hr SECURE H100 NVL; baselines on $2.59/hr community (throughput is rate-independent; $/clip is not).

### What the normalized numbers say

- **On default recipes LTX-2.3 led; once Wan is step-distilled, Wan takes the render-throughput lead.** The baseline table had LTX ~2.9x ahead of Wan. After Tier A, **Wan Turbo 4-step hits 4.04 MP/s — ~2.3x LTX** — because dropping 40 steps to 4 (and CFG to 1) cuts per-clip render from 47s to 8s. The gap was step count, not architecture.
- **Wan 2.2 paid for 40 steps — and that bill is now optional.** The single biggest lever in the whole study, confirmed: the [measured A1 delta](#measured-results-tier-a) is **5.8x faster render at the same 34 GB VRAM**.
- **HunyuanVideo's wall was clip length, not throughput — and it's now breakable.** Tier A renders a full **5s clip at 832x480** (was OOM); throughput stays low (an 8.3B model doing 121 frames with offload), but the *capability* doubled. You reach for it on quality/license — now without the half-length compromise.
- **VRAM and speed still don't correlate the way you'd guess.** LTX is fast but heavy (50-55 GB); 5B Wan Turbo is *both* fastest and lightest (~34 GB). Hunyuan's 5s mode peaks at ~68 GB alloc / 78 GB reserved — tiling keeps it under the 94 GB cliff.

## Measured results (Tier A)

Tier A turned the two highest-value predictions from [What to optimize next](#what-to-optimize-next) into measured numbers. Budget approved: ~$6 / ~2.5 GPU-hr; actual spend ≈ **$2.65** on ~0.8 GPU-hr. One SECURE H100 NVL ($3.19/hr). Two questions, both answered yes:

- **A1** — does step-distillation close Wan's gap to LTX? *Yes — it overtakes LTX.*
- **A2** — can VAE tiling break HunyuanVideo's 5-second OOM ceiling? *Yes — full 5s now fits.*

All Tier A figures are **warm render** (steady-state per-clip, the metric that matters for a served model); one-time cold load is in `results.jsonl` `load_seconds`, not here. `runs: 1` each — treat as ±10-15%, same as the baselines.

### A1 — Wan step + CFG distillation (the dominant lever)

First, the naive control: just truncate the base model's 40-step schedule and keep guidance at 5.0. You get speed, but the base weights were never trained for low step counts, so motion smears and detail collapses below ~20 steps. Speed is real; quality is not.

| Steps | Guidance | Warm render | vs 40-step |
|---|---|---|---|
| 40 (default) | 5.0 | 46.6s | 1.00x |
| 20 | 5.0 | 26.2s | 1.78x |
| 8 | 5.0 | 14.0s | 3.33x |
| 6 | 5.0 | 12.0s | 3.88x |

The fix is **trained** few-step weights. Swapping in the distilled Turbo transformer (`yetter-ai/Wan2.2-TI2V-5B-Turbo-Diffusers`) and dropping guidance to 1.0 holds quality at 4-8 steps — same pod, same 832×480×81f, same 34 GB:

| Recipe | Steps | Guidance | Warm render | vs base | VRAM |
|---|---|---|---|---|---|
| Base, un-distilled | 40 | 5.0 | 46.6s | 1.00x | 34 GB |
| **Turbo 8-step** *(recommended)* | 8 | 1.0 | 9.6s | **4.9x** | 34 GB |
| **Turbo 4-step** *(fastest)* | 4 | 1.0 | 8.0s | **5.8x** | 34 GB |

Headline: **5.8x faster render at identical resolution, frame count, and VRAM.** Two practical notes:

- **8-step is the pick, not 4-step.** 4-step saves only 1.6s over 8-step, because fixed overhead (VAE decode + text encode ≈ 6.4s) dominates at low step counts. 8-step retains motion better for near-zero speed cost.
- **Sourcing correction (cost us a run).** The widely-cited **lightx2v** step-distill LoRAs are **14B-only** (hidden dim 5120) and do **not** fit TI2V-5B (hidden dim 3072) — they throw a state_dict shape mismatch on load. The working 5B path is the full **transformer swap** above, not a LoRA. The cold load is a one-time ~32s (inflated because the base transformer is loaded then discarded during the swap); warm render is the 8.0/9.6s above.

### A2 — HunyuanVideo: breaking the 5-second OOM ceiling

The baseline runbook caps Hunyuan at a **2.54s** clip because a full 5s at 832×480 OOMs the VAE decode even on a 94 GB card. Temporal VAE tiling (`tile_sample_min_num_frames=16`) + fp8 layerwise casting + CPU offload breaks that wall:

| Config | Res | Frames | Clip | Warm render | Peak VRAM (alloc / reserved) |
|---|---|---|---|---|---|
| Baseline (offload, no tiling) | 832×480 | 61 | 2.54s | 65.3s | 35.6 / 55.1 GB |
| **Tiled, full 5s** | 832×480 | 121 | 5.04s | 149.6s | 68.0 / **78.0** GB |
| Tiled 5s, tile=8 | 832×480 | 121 | 5.04s | 152.9s | 68.0 / 78.0 GB |
| Tiled 5s, low-res | 640×384 | 121 | 5.04s | 81.1s | 67.7 / 78.0 GB |

The capability delta: **clip length doubled (2.54s → 5.04s) at the headline 832×480**, landing at 78 GB reserved with ~16 GB of headroom on the 94 GB card. One counter-intuitive finding worth keeping:

- **Tiled-decode peak VRAM is set by the temporal tile size + offload buffers, not spatial resolution.** 832×480 and 640×384 peak at the *same* ~68/78 GB. Dropping resolution buys **compute time** (149.6s → 81.1s), not memory headroom — so if you're VRAM-bound, shrink the tile, not the frame.
- `tile_sample_min_num_frames=8` vs `16` made no VRAM difference and ran slightly slower (152.9s vs 149.6s), so 16 is the pick.

Throughput stays low (an 8.3B model rendering 121 frames *with* offload is inherently slow) — you still reach for Hunyuan on quality/license, not speed. What changed is that you no longer pay for it in clip length.

## Caveats — how to read this

These numbers are useful for **order-of-magnitude planning**, not for a leaderboard. The honest limitations:

1. **Single run each.** `runs: 1` in the data. No averaging, no error bars, no warm-vs-cold-cache separation beyond the documented load time. Treat the figures as ±10-15%.
2. **Different default recipes, not an equal-footing fight.** Wan runs 40 steps (un-distilled); LTX and Hunyuan run 8 (step-distilled). This compares *what each model ships with*, which is what you'd actually run on day one — but it is **not** "which architecture is fastest at N steps." [Tier A](#measured-results-tier-a) starts closing this gap by re-running Wan at 4-8 distilled steps; the baseline table itself is still default-recipe-vs-default-recipe.
3. **HunyuanVideo runs handicapped.** CPU offload (required to fit) costs ~50% of its speed, and its clip is half-length because 5s OOMs. Its numbers reflect "made it fit on this card," not the model's ceiling on bigger hardware.
4. **No quality scoring.** This is throughput and cost only. Visual quality is subjective, prompt-dependent, and not captured here. A faster model that needs more retries isn't actually cheaper. Quality benchmarking is future work.
5. **Point-in-time pricing, and two different rates.** Baselines used a $2.59/hr community H100 NVL (2026-04); the Tier A rows used a $3.19/hr **SECURE** H100 NVL (2026-05, a different pod). Throughput (MP/s) and the relative speedups are rate-independent, but $/clip is not — so the two groups' dollar columns aren't directly comparable. RunPod pricing and availability move; recompute cost with your own rate (it's a single field per row in `results.jsonl`).
6. **One GPU SKU.** All three on H100 NVL. Relative ordering may shift on A100 80 GB (less VRAM headroom hurts LTX and Hunyuan first) or on consumer cards.

## What to optimize next

The normalized table pointed at four levers, feeding the [research note](../RESEARCH.md). Tier A measured the top two ([above](#measured-results-tier-a)); two remain open.

**Measured in this PR:**

- **Cut Wan's step count — done, biggest lever.** 40 → 4-8 steps via the `yetter-ai/Wan2.2-TI2V-5B-Turbo-Diffusers` transformer swap (a full swap, *not* a LoRA — the popular lightx2v distill LoRAs are [14B-only](#a1--wan-step--cfg-distillation-the-dominant-lever)). Measured **5.8x faster render at the same 34 GB VRAM**.
- **Get HunyuanVideo to 5s — done, with offload.** Temporal VAE tiling + fp8 casting [break the 5s OOM ceiling](#a2--hunyuanvideo-breaking-the-5-second-oom-ceiling) at 832×480; clip length doubled (2.54s → 5.04s). Dropping offload for the ~2x throughput bump is still open — tiling fixed the *length* wall, not the offload tax.

**Still open:**

- **Quantize the VRAM hogs.** LTX sits at 50-55 GB; fp8 / int8 weight quant could buy headroom for higher resolution or longer clips on the same card. fp8 layerwise casting is already proven on Hunyuan here — porting it to LTX is the obvious next step.
- **Caching (TeaCache / FBCache)** advertises 1.5-2x on video DiTs with small quality cost — applies to all three in principle, untested here.

Each remaining item is a cheap, bounded RunPod experiment. The point of publishing the baseline was to measure these deltas against it honestly — Tier A did that for the first two.

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
