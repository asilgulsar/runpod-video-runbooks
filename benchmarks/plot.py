#!/usr/bin/env python3
"""Compute normalized throughput/cost metrics from results.jsonl and (optionally) chart them.

Raw per-clip numbers aren't comparable across models because each model's default
recipe uses a different clip length, resolution, frame count, and step budget. This
script derives length- and resolution-normalized metrics so the comparison is fair:

  - compute_s_per_video_s : seconds of GPU time per second of output video (lower = faster)
  - frames_per_compute_s  : frames generated per second of GPU time (higher = faster)
  - megapixels_per_s       : pixel throughput = (W*H*frames) / total_s (higher = faster)
  - usd_per_video_s        : dollars per second of output video (lower = cheaper)

Usage:
    python3 plot.py                 # print the normalized table (no deps)
    python3 plot.py --charts        # also write PNG bar charts (needs matplotlib)

The pure-Python table needs only the standard library, so it runs anywhere.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "results.jsonl")


def load_rows(path=DATA):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def derive(row):
    """Add normalized metrics to a row (non-destructive copy)."""
    w, h = (int(x) for x in row["resolution"].lower().split("x"))
    frames = row["frames"]
    clip_s = row["clip_seconds"]
    total_s = row["total_seconds"]
    cost = row["cost_per_clip_usd"]
    mp_per_frame = (w * h) / 1_000_000.0
    d = dict(row)
    d["label"] = row.get("label", row["model"])
    d["megapixels_per_frame"] = round(mp_per_frame, 3)
    d["compute_s_per_video_s"] = round(total_s / clip_s, 2)
    d["frames_per_compute_s"] = round(frames / total_s, 2)
    d["megapixels_per_s"] = round((mp_per_frame * frames) / total_s, 3)
    d["usd_per_video_s"] = round(cost / clip_s, 4)
    return d


def print_table(rows):
    cols = [
        ("label", "Model", 20),
        ("steps", "Steps", 6),
        ("compute_s_per_video_s", "s_compute/s_video", 18),
        ("frames_per_compute_s", "frames/s", 9),
        ("megapixels_per_s", "MP/s", 7),
        ("usd_per_video_s", "$/s_video", 10),
        ("vram_peak_gb", "VRAM_GB", 8),
    ]
    header = "  ".join(label.ljust(width) for _, label, width in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        line = "  ".join(str(r.get(key, "")).ljust(width) for key, _, width in cols)
        print(line)
    print()
    # Relative-to-fastest summary on the headline metric (pixel throughput).
    fastest = max(rows, key=lambda r: r["megapixels_per_s"])
    base = fastest["megapixels_per_s"]
    print(f"Pixel throughput, relative to fastest ({fastest['label']} = 1.00x):")
    for r in sorted(rows, key=lambda r: -r["megapixels_per_s"]):
        print(f"  {r['label']:<20} {r['megapixels_per_s'] / base:>5.2f}x")
    print()


def write_charts(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping charts. "
              "Install with: pip install matplotlib", file=sys.stderr)
        return
    labels = [r["label"] for r in rows]
    specs = [
        ("megapixels_per_s", "Pixel throughput (MP/s, higher is better)", "throughput_mp_s.png"),
        ("usd_per_video_s", "Cost per second of video (USD, lower is better)", "cost_per_video_s.png"),
    ]
    for key, title, fname in specs:
        vals = [r[key] for r in rows]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labels, vals)
        ax.set_title(title)
        ax.set_ylabel(key)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:g}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        out = os.path.join(HERE, fname)
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--charts", action="store_true", help="also write PNG bar charts (needs matplotlib)")
    args = ap.parse_args()
    rows = [derive(r) for r in load_rows()]
    print_table(rows)
    if args.charts:
        write_charts(rows)


if __name__ == "__main__":
    main()
