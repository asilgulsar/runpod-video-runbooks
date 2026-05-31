# Making open video DiTs faster, cheaper, and fit bigger

A working research note on optimizing the three models in this repo — **LTX-2.3** (22B distilled), **Wan 2.2** (TI2V 5B), **HunyuanVideo 1.5** (8.3B distilled) — for inference on an H100 NVL (94 GB) or A100 80 GB. It pairs a literature scan (quantization, memory, speed/quality, as of mid-2026) with the validated baseline in [benchmarks/](benchmarks/BENCHMARKS.md), and turns it into a concrete, costed experiment plan.

**How to read this.** Numbers without a "(measured here)" tag come from papers, vendor blogs, or third-party reports — and a lot of the strongest published figures are for *other* DiTs (CogVideoX, Flux), so treat them as directional until confirmed on a pod. That gap is the point: the [planned experiments](#planned-experiments-costed) exist to generate the Wan/Hunyuan/LTX datapoints that don't yet exist publicly. Every claim links a source at the bottom.

---

## The thesis in one paragraph

The single biggest lever in the whole study is **cutting Wan 2.2 from its 40 un-distilled steps to 4-8** with distilled few-step weights — that also drops classifier-free guidance (another ~2x). **Measured here: 46.6s → 8.0s warm render, ~5.8x at identical VRAM** ([Tier A](#tier-a--measured-done)). Nothing else here is close. The other two models are *already* step- and CFG-distilled (8 steps each), so for them the wins are smaller and stack-able: **FP8 weight casting** (~50% transformer VRAM, near-free), **SageAttention-FP8** (one line on the H100's SM90), **inference caching** (TeaCache ~2x), **torch.compile** (~1.5x), and — for HunyuanVideo specifically — **VAE tiling** to break its 5-second OOM ceiling. Distill first (it shrinks the base everything else multiplies against), then cache/attention/compile on top.

---

## Start here: the per-model playbook

The right moves differ a lot by model, mostly because **LTX-2.3 is not diffusers-native** (it runs through Lightricks' `ltx-pipelines` package), so the entire diffusers optimization toolbox — `enable_layerwise_casting`, group offload, `set_attention_backend`, `enable_cache` — simply doesn't apply to it.

### LTX-2.3 (22B distilled, `ltx-pipelines` from source)
Already 8-step and CFG-free, so there's **no distillation or caching headroom**. The lever is VRAM (it's the heaviest at 50-55 GB):
- **`--quantization fp8-cast`** in its own pipeline drops the 22B from ~50 GB toward ~18-32 GB. This is the main move. It buys headroom for higher resolution or longer clips on the same card.
- **Trap:** `--quantization fp8-cast` and `--offload {cpu,disk}` are **mutually exclusive** — you pick one. Test which gives you the envelope you need.
- Its compile/attention path is bespoke (a custom RMSNorm+compile kernel reportedly ~1.43x); none of the diffusers attention/cache APIs reach it.
- **Bottom line:** small optimization surface, but it's already the throughput/cost leader. Don't over-invest here.

### Wan 2.2 (TI2V 5B, diffusers `WanPipeline`, 40 steps)
This is where the money is.
1. **Step + CFG distillation (do this first). ✅ Measured here — and a sourcing correction.** The popular `Wan2.2-Lightning` / `lightx2v` step-distill LoRAs are **14B-only** (hidden dim 5120, built around the A14B model's high/low-noise *expert* split) — they **do not fit TI2V-5B** (single dense transformer, hidden dim 3072) and throw a state_dict shape mismatch on load. For the 5B the working path is a **full transformer swap** to `yetter-ai/Wan2.2-TI2V-5B-Turbo-Diffusers`, run at **8 steps, guidance=1.0** (no expert split — the 5B is one transformer). **Measured: 40-step 46.6s → 8-step 9.6s (4.9x) / 4-step 8.0s (5.8x)** at identical 832×480×81f and 34 GB ([Tier A](#tier-a--measured-done)). **Use 8 steps, not 4** — 4-step saves only ~1.6s (fixed VAE/text-encode overhead dominates at low step counts) and loses motion.
2. **Then stack on the smaller base:** `set_attention_backend("sage")` for SageAttention-FP8 (H100/SM90 only — the A100 falls back to INT8), `torch.compile` (~1.5x, lock your resolution to avoid recompiles), and FP8 layerwise casting for VRAM headroom. On Hopper, **torchao fp8-dynamic + compile** is the one path that buys VRAM *and* speed together.
3. **Free orthogonal win:** quantize the T5 text encoder to FP8 — zero impact on video quality.
4. **Gotchas:** diffusers' native FirstBlockCache **errors on Wan 2.2** (issue #12012) — use standalone **TeaCache** instead; GGUF-from-diffusers is **broken** for Wan 2.2 (meta-tensor, #12009); there's a reported tiled-VAE-decode bug (#125).

### HunyuanVideo 1.5 (8.3B distilled, diffusers `HunyuanVideo15Pipeline`)
Already 8-step distilled and uses `pipe.guider`, so its problem is **VRAM/OOM, not speed** — specifically the VAE decode (~51 GB spike on top of ~42 GB).
1. **Break the 5s OOM ceiling (832×480, 121f): ✅ Measured here — it holds.** Temporal VAE tiling (`tile_sample_min_num_frames=16`) + FP8 layerwise casting + CPU offload renders the **full 832×480×121f (5.04s)** clip without OOM — **measured peak 68 GB alloc / 78 GB reserved, ~16 GB headroom** on the 94 GB card ([Tier A](#tier-a--measured-done)). The pre-experiment "640×384 is the dependable answer" hedge was **too conservative** — full res fits; 640×384 is the *faster* 5s option (81s vs 150s), not a memory necessity. **Measured surprise:** tiled-decode peak VRAM is set by the **temporal tile size + offload buffers, not spatial resolution** (both resolutions peak at the same ~68/78 GB) — so when VRAM-bound, shrink the tile, not the frame. Sequential offload alone won't fix it — it's a peak-allocation problem, not steady-state.
2. **Speed:** **TeaCache** (~2x, official Hunyuan patch) and **SageAttention-FP8** (one line). **Prefer Sage over FA3-fp8** — FA3-fp8 went visibly blurry on Hunyuan in third-party tests.
3. **Faster previews:** `taehv1_5` tiny-VAE decode (4-6x faster decode, 12-18x less VRAM) for draft passes; swap the full VAE back for finals.
4. **Free orthogonal win:** T5 FP8. **Don't quantize the VAE** — fix decode OOM with tiling, not quantization (quantizing the decoder is where color shifts / blockiness show up).

---

## Technique reference

Condensed from the full briefs. Legend: ✅ usable today in our stack · ⚠️ works with caveats / not our exact pipeline · ❌ not applicable / not ready.

### Quantization

| Technique | LTX-2.3 | Wan 2.2 | Hunyuan 1.5 | What you get |
|---|---|---|---|---|
| FP8 layerwise cast (`enable_layerwise_casting`, e4m3fn) | ⚠️ via native `fp8-cast` | ✅ | ✅ | ~25-50% transformer VRAM, ~free, negligible quality loss. **Storage-only — no FLOP speedup.** |
| torchao FP8-dynamic + `torch.compile` (Hopper) | ❌ not diffusers-native | ✅ likely (untested) | ✅ likely (untested) | The one that buys **VRAM *and* speed**. ~27% faster than bf16 on CogVideoX-5b/H100. |
| torchao INT8-dynamic (Ampere/A100) | ❌ | ✅ likely | ✅ likely | ~40% VRAM cut; needs compile or it's slow. |
| bitsandbytes 4-bit NF4 | ⚠️ when diffusers pipeline ships | ✅ | ✅ (documented) | ~60% VRAM cut, **same speed as bf16**, small quality hit. |
| T5 / text-encoder FP8 (independent) | ✅ | ✅ | ✅ | Frees several GB, **zero** video-quality impact. |
| VAE quantization | ⚠️ prefer tiling | ⚠️ prefer tiling | ❌ use tiling+offload | Risky; color/blockiness. Not worth it. |

Key traps: **8-bit is ~2x slower than 4-bit** for less benefit (use 4-bit NF4), and **weight-only int4/int8 *without* `torch.compile` is ~9x slower** — only use weight-only with compile, or use dynamic-activation configs. FP8 *compute* needs Hopper (H100 ✅, A100 ❌).

### Memory

| Technique | LTX-2.3 | Wan 2.2 | Hunyuan 1.5 | What you get |
|---|---|---|---|---|
| Group offload + CUDA-stream prefetch (`enable_group_offload(use_stream=True)`) | ❌ not diffusers-native | ✅ | ✅ | Lower VRAM than model-offload and *faster than sequential* (transfer overlaps compute). The right offload tier for big models. |
| Model CPU offload | ⚠️ via pkg `--offload` (disables fp8) | ✅ | ✅ (current default) | Modest VRAM cut, modest speed tax. |
| VAE spatial tiling / slicing | ✅ `--spatial-tile` | ⚠️ decode bug #125 | ✅ key OOM fix | Cuts decode peak; watch seams. |
| VAE **temporal** tiling (`tile_sample_min_num_frames`) | ⚠️ check pkg | ⚠️ same caveat | ✅ the OOM fix | Splits along the frame axis — exactly what blows up at high frame counts. |
| Sequential / leaf offload | ❌ (pkg offload) | ✅ slow fallback | ✅ slow fallback | Lowest VRAM, big speed penalty. Last resort to fit at all. |

### Speed & quality

| Technique | LTX-2.3 | Wan 2.2 | Hunyuan 1.5 | What you get |
|---|---|---|---|---|
| Step + CFG distillation | ❌ already distilled | ✅ **the win** — measured **5.8x** (Turbo swap, 8-step) | ❌ already distilled | ~5-10x fewer forward passes + CFG→1. **Dominant lever**, [confirmed](#tier-a--measured-done). |
| TeaCache (standalone) | ⚠️ little to cache at 8 steps | ✅ ~2x (official patch) | ✅ ~2x (official patch) | Training-free, "no noticeable degradation," tune threshold 0.2-0.3. |
| diffusers `enable_cache` (FBCache/FasterCache) | ⚠️ not wired | ⚠️ **errors on Wan 2.2 (#12012)** | ✅ has CacheConfig | ~1.5-2x; set `is_guidance_distilled=True` for distilled runs. |
| SageAttention-FP8 (`set_attention_backend("sage")`) | ⚠️ source pipeline | ✅ (SM90/H100) | ✅ (SM90/H100) | ~1.1-1.3x end-to-end, near-zero quality hit. **Prefer over FA3-fp8.** |
| torch.compile (regional / max-autotune) | ⚠️ bespoke | ✅ ~1.5x | ✅ ~1.5x | Free quality; lock resolution to avoid recompiles. |
| RIFLEx (RoPE freq fix, length 2-4x) | ⚠️ check RoPE | ✅ (in diffusers) | ✅ (in diffusers) | Longer clips, training-free, no motion collapse. Length, not speed. |
| taehv tiny-VAE decode | ❌ no LTX weights | ✅ `taew2_2` | ✅ `taehv1_5` | 4-6x faster decode, 12-18x less VRAM. Draft/preview; slight detail loss. |

---

## Don't chase these (dead-ends and traps)

The most valuable output of the scan — what *not* to spend a pod-hour on:

- **SVDQuant / Nunchaku** — strictly **image-only** (Flux/Qwen-Image/SANA/PixArt). No video model, no diffusers integration, none on the public roadmap.
- **GGUF *via diffusers*** — currently **broken** for Wan 2.2 (meta-tensor, #12009) and LTX-2 (#12981). GGUF works, but **ComfyUI-only** for our stack today. (Note: `unsloth/LTX-2.3-GGUF` covers our exact LTX model if we ever add a ComfyUI path.)
- **Sliding Tile Attention / Sparse VideoGen / Radial / FPSAttention / DraftAttention** — high-ceiling sparse-attention research, but **custom kernels, research/ComfyUI only, not diffusers-usable.** Don't budget on them.
- **Block-swapping (kijai WanVideoWrapper)** — **ComfyUI-only**; diffusers' group offload (`block_level`) is the functional equivalent.
- **FramePack** — it's a **model swap** (FramePack-trained checkpoints), not a flag you bolt onto our weights. Good for a *separate* long-clip track; diffusers ships the Hunyuan variant.
- **Quantizing the VAE** — both the quant and memory scans independently say no; fix OOM with **tiling**.
- **Pure 4-step Wan distillation** — weak motion. Use **8 steps (4+4)**.
- **diffusers FirstBlockCache on Wan 2.2** — errors ("No context", #12012); use standalone **TeaCache**.
- **FA3-fp8 on Hunyuan** — went blurry in tests; use **SageAttention**.
- **ViDiT-Q / DVD-Quant** — good to cite as technique background (W4A8 near-lossless, 2-2.5x memory), but research codebases with no diffusers integration and none of our three models. Read, don't run.

---

## Planned experiments (costed)

De-duplicated across the three scans and ranked. Costs assume a community **H100 NVL at ~$2.59/hr** (the benchmark's rate); on owned hardware the dollar figures are effectively just power. Each experiment measures a **delta against the published [baseline](benchmarks/BENCHMARKS.md)**.

### Tier A — MEASURED (done)
**Spent ≈ $2.65 / ~0.8 GPU-hr on a SECURE H100 NVL ($3.19/hr). Both landed — full numbers in [BENCHMARKS.md](benchmarks/BENCHMARKS.md#measured-results-tier-a).**

| # | Experiment | Result (measured) |
|---|---|---|
| A1 | **Wan step+CFG distillation:** 40-step base vs Turbo transformer swap at 8 / 4 steps (guidance 1.0), 832×480×81f | ✅ **5.8x faster render** (46.6s → 8.0s) at identical 34 GB VRAM; 8-step (9.6s, 4.9x) recommended. Correction: lightx2v LoRAs are 14B-only — 5B needs the `yetter-ai/…Turbo` transformer swap, not a LoRA. |
| A2 | **Hunyuan VAE-tiling + fp8** to render 5s @ 832×480 without OOM | ✅ **Full 832×480×121f (5.04s) fits** — 78 GB reserved, ~16 GB headroom; clip length doubled vs the baseline's 2.54s. Peak VRAM is tile-size-bound, not resolution-bound. |

### Tier B — strong, stackable, low-risk (~2.4 GPU-hr, ~$6)
| # | Experiment | Why | GPU-hr |
|---|---|---|---|
| B1 | **Wan FP8 layerwise-cast** vs bf16 (VRAM + s/clip) | Cheap headroom; baseline for the stack | ~0.3 |
| B2 | **Wan SageAttention-FP8 + torch.compile** on the A1 winner (isolate each) | The post-distill speed stack on H100 | ~1.0 |
| B3 | **TeaCache on Hunyuan** (threshold 0.2/0.3, 5 prompts) | ~2x, training-free, low risk | ~0.75 |
| B4 | **LTX-2.3 native `fp8-cast`** vs bf16 (VRAM + speed at equal quality) | The main LTX lever | ~0.3 |

### Tier C — forward-looking / nice-to-have (~3.0 GPU-hr, ~$8)
| # | Experiment | Why | GPU-hr |
|---|---|---|---|
| C1 | **Wan torchao fp8-dynamic (H100) + int8-dynamic (A100)** with compile | Generates the Wan VRAM+speed datapoint that doesn't exist publicly | ~1.0 |
| C2 | **taehv tiny-VAE** decode swap (Wan `taew2_2`, Hunyuan `taehv1_5`) | Faster draft passes | ~0.5 |
| C3 | **RIFLEx** length 2x on Wan/Hunyuan | Free longer clips | ~0.75 |
| C4 | **LTX fp8-cast vs `--offload`** mutual-exclusivity tax | Quantifies the trap | ~0.5 |
| C5 | **Hunyuan T5 text-encoder FP8** isolated | Free orthogonal VRAM win | ~0.3 |

**Run-all ≈ 8 GPU-hr ≈ $21.** Tier A first — it shrinks Wan's base clip time and makes B2/C1 cheaper. Results land back in `benchmarks/results.jsonl` so the normalized table updates automatically.

---

## How this connects to the benchmark

The [baseline](benchmarks/BENCHMARKS.md) is deliberately the *default shipping recipe* for each model — which is why Wan's 40 steps look slow next to the two distilled models. That's not a flaw in Wan; it's the headline opportunity. Every experiment above is a measured delta against that baseline, added as a new row so the comparison stays honest and reproducible.

---

## Sources

**Diffusers core:** [memory optimization](https://huggingface.co/docs/diffusers/main/en/optimization/memory) · [attention backends](https://huggingface.co/docs/diffusers/main/optimization/attention_backends) · [caching API](https://huggingface.co/docs/diffusers/main/en/api/cache) · [torchao](https://huggingface.co/docs/diffusers/en/quantization/torchao) · [quantization backends blog](https://huggingface.co/blog/diffusers-quantization) · group offload PRs [#10503](https://github.com/huggingface/diffusers/pull/10503)/[#10516](https://github.com/huggingface/diffusers/pull/10516)/[#11106](https://github.com/huggingface/diffusers/pull/11106)

**Quantization:** [diffusers-torchao (CogVideoX video benchmarks)](https://github.com/sayakpaul/diffusers-torchao) · [bitsandbytes in diffusers](https://github.com/huggingface/diffusers/blob/main/docs/source/en/quantization/bitsandbytes.md) · [Nunchaku/SVDQuant (image-only)](https://github.com/nunchaku-tech/nunchaku) · [ViDiT-Q](https://arxiv.org/abs/2406.02540) · [DVD-Quant](https://arxiv.org/pdf/2505.18663) · [unsloth/LTX-2.3-GGUF](https://huggingface.co/unsloth/LTX-2.3-GGUF) · diffusers GGUF bugs [#12009](https://github.com/huggingface/diffusers/issues/12009)/[#12981](https://github.com/huggingface/diffusers/issues/12981) · [LTX quantization formats](https://ltx.io/model/model-blog/quantization-formats-explained)

**Memory:** [FramePack (arXiv 2504.12626)](https://arxiv.org/html/2504.12626v3) · [diffusers FramePack pipeline](https://huggingface.co/docs/diffusers/en/api/pipelines/framepack) · [video VAEs in diffusers (blog)](https://huggingface.co/blog/Bekhouche/compressing-time-a-comparative-study-of-video-vaes) · [AutoencoderKLHunyuanVideo](https://huggingface.co/docs/diffusers/api/models/autoencoder_kl_hunyuan_video) · [kijai block-swapping](https://deepwiki.com/kijai/ComfyUI-WanVideoWrapper/6.2-block-swapping-and-device-management)

**Speed & quality:** [TeaCache](https://github.com/ali-vilab/TeaCache) ([arXiv 2411.19108](https://arxiv.org/abs/2411.19108)) · [Wan2.2-Lightning](https://huggingface.co/lightx2v/Wan2.2-Lightning) / [LightX2V step-distill](https://lightx2v-en.readthedocs.io/en/latest/method_tutorials/step_distill.html) · [SageAttention](https://github.com/thu-ml/SageAttention) ([2++, arXiv 2505.21136](https://arxiv.org/html/2505.21136v1)) · [Sliding Tile Attention](https://arxiv.org/abs/2502.04507) · [torch.compile + diffusers](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/) · [RIFLEx](https://arxiv.org/abs/2502.15894) · [taehv tiny-VAE](https://github.com/madebyollin/taehv) · diffusers FBCache-on-Wan bug [#12012](https://github.com/huggingface/diffusers/issues/12012)

---

## Status & contributing

This is a **living note**, not a finished study. The baseline is single-run; the literature numbers are mostly for other DiTs until the experiments above replace them with measured ones — **Tier A is now done** (Wan step-distillation + Hunyuan 5s tiling), Tiers B/C remain open. The most valuable contributions are **measured deltas on these exact three models** — if you run any of the experiments (or have your own numbers), open a PR adding a row to `benchmarks/results.jsonl` and a note here.
