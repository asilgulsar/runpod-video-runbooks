# runpod-video-runbooks

Battle-tested runbooks for running **open video-generation models on [RunPod](https://www.runpod.io/)** — with the exact version pins, GPU choices, and error fixes that actually make them work.

Every recipe here was burned in by trial and error on real pods. The goal is simple: copy a runbook, follow it top to bottom, and get a rendered clip without losing an hour to dependency hell.

## Why this exists

Open video models (LTX, Wan, HunyuanVideo) move fast and their dependency stacks are fragile. The same three things break for everyone:

- **Driver vs. CUDA mismatches** that silently disable the GPU
- **`transformers` / `torch` version drift** that throws cryptic attribute errors
- **Packages that only ship wheels for one Python version**, forcing a source install

These runbooks pin the working stack and document the fix for every error string you're likely to hit.

## Runbooks

| Model | License | Status | Runbook |
|---|---|---|---|
| LTX-2.3 (22B distilled) | Open weights | Validated | [runbooks/ltx-2.3.md](runbooks/ltx-2.3.md) |
| Wan 2.2 (TI2V 5B) | Apache-2.0 | Validated | [runbooks/wan-2.2.md](runbooks/wan-2.2.md) |
| HunyuanVideo 1.5 (480p T2V) | Tencent Community | Validated | [runbooks/hunyuan-1.5.md](runbooks/hunyuan-1.5.md) |

## Benchmarks

All three models timed on the **same GPU** (H100 NVL 94 GB), then normalized by clip length and resolution so they're actually comparable — raw "seconds per clip" isn't, because each model ships a different default recipe:

| Model | Pixel throughput (MP/s) | Cost per second of video | VRAM peak |
|---|---|---|---|
| LTX-2.3 | **1.78** (1.00x) | **$0.0040** | 50-55 GB |
| Wan 2.2 | 0.61 (0.34x) | $0.0079 | ~34 GB |
| HunyuanVideo 1.5 | 0.32 (0.18x) | $0.0236 | ~36 GB |

Full methodology, raw numbers, and the honest caveats (single-run; Wan runs 40 un-distilled steps vs 8 for the others) live in **[benchmarks/BENCHMARKS.md](benchmarks/BENCHMARKS.md)**. Regenerate the normalized table from the raw data with `python3 benchmarks/plot.py`.

## Research

**[RESEARCH.md](RESEARCH.md)** is a working note on making these three models faster, cheaper, and fit bigger — a mid-2026 scan of quantization, memory, and speed/quality techniques mapped to each model, with a per-model playbook, an honest "don't chase these" list, and a costed experiment plan. The headline: Wan 2.2's 40 un-distilled steps are the biggest single optimization target; the other two are already step-distilled, so their wins come from FP8 casting, attention backends, and inference caching.

## Quick start

1. Get a [RunPod](https://www.runpod.io/) account and an API key. Export it:
   ```bash
   export RUNPOD_API_KEY=your_key_here
   ```
2. Pick a runbook from the table above.
3. Follow it top to bottom. Each one covers: spawning the pod, the bootstrap script, downloading weights, running inference, and tearing down so you stop paying.

## What's in each runbook

- **The critical gotchas up front** — the handful of things that waste the most time
- **Step-by-step commands** — pod spawn, bootstrap, model download, inference, teardown
- **A known-issues table** — symptom → cause → fix for the errors you'll actually see
- **Real cost benchmarks** — time and dollars per clip on real hardware

## Repo layout

```
runbooks/    # one markdown runbook per model
scripts/     # bootstrap + helper scripts referenced by the runbooks
benchmarks/  # cross-model throughput + cost numbers, with a script to regenerate them
```

## Contributing

Found a new gotcha, a better GPU fallback, or got a new model working? Open an issue or a PR. The most valuable contributions are **exact error strings and their fixes** — that's what people search for.

## Disclaimer

These runbooks reference third-party models and infrastructure. Check each model's own license before commercial use. RunPod pricing and GPU availability change over time; the cost figures here are point-in-time references, not guarantees.

## License

[MIT](LICENSE)
