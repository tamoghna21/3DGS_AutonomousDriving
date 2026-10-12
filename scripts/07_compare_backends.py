# SPDX-License-Identifier: MIT
"""Step 7: the same training run with two rasterizers. Reads the logs of
    python scripts/03_train.py                                      (pure PyTorch)
    python scripts/03_train.py --backend gsplat --out-dir outputs/train_gsplat
(identical settings, seed and data; only the rasterizer differs) and writes
  figures/fig_backend_time_to_quality.png   held-out PSNR against training minutes
  outputs/bench/backends.md                 time, speed and final metrics

    python scripts/07_compare_backends.py
"""
import argparse
import json
import os

import matplotlib
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ("#2a78d6", "#eb6834")


def curve(log):
    """(minutes of training, held-out PSNR) at every evaluation, from the
    per-1000-step speeds (evaluation and snapshots are not timed)."""
    t, pts, prev = 0.0, [], 0
    for e in log:
        t += (e["step"] - prev) / e["steps_per_sec"]
        prev = e["step"]
        if "test_psnr" in e:
            pts.append((t / 60, e["test_psnr"], e["step"]))
    return pts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--torch-dir", default=os.path.join(ROOT, "outputs", "train"))
    p.add_argument("--gsplat-dir", default=os.path.join(ROOT, "outputs", "train_gsplat"))
    p.add_argument("--fig-dir", default=os.path.join(ROOT, "figures"))
    p.add_argument("--out", default=os.path.join(ROOT, "outputs", "bench", "backends.md"))
    args = p.parse_args()
    runs = {}
    for name, d in (("pure PyTorch", args.torch_dir), ("gsplat (CUDA kernels)", args.gsplat_dir)):
        with open(os.path.join(d, "train_log.json")) as f:
            runs[name] = json.load(f)

    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=160, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ends = []
    for (name, tl), col in zip(runs.items(), SERIES):
        pts = curve(tl["log"])
        ax.plot([x for x, _, _ in pts], [y for _, y, _ in pts], "-o", color=col, lw=2, ms=7, mec=SURFACE, mew=2,
                label=name)
        ends.append(pts[-1][:2])
    # End labels: the earlier-finishing run to the right of its last point, the other below-left of its own.
    (xa, ya), (xb, yb) = sorted(ends)
    ax.annotate(f"{ya:.2f} dB after {xa:.0f} min", (xa, ya), textcoords="offset points", xytext=(9, -3),
                ha="left", va="center", color=INK, fontsize=8)
    ax.annotate(f"{yb:.2f} dB after {xb:.0f} min", (xb, yb), textcoords="offset points", xytext=(-4, -15),
                ha="right", color=INK, fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, color=GRID, lw=1.0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_xlabel("training time (minutes, RTX A1000 4 GB)", color=INK2)
    ax.set_ylabel("held-out PSNR (dB)", color=INK2)
    ax.set_title("Same training, two rasterizers (evaluations every 5,000 steps)", color=INK, fontsize=10,
                 loc="left")
    leg = ax.legend(frameon=False, fontsize=8, loc="lower right")
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(args.fig_dir, "fig_backend_time_to_quality.png"), facecolor=SURFACE)

    lines = ["| Rasterizer | Training time (30k steps) | Steps/s (median) | Gaussians | Peak VRAM | "
             "Held-out PSNR | SSIM | LPIPS |", "|---|---|---|---|---|---|---|---|"]
    for name, tl in runs.items():
        m = tl["meta"]
        sps = sorted(e["steps_per_sec"] for e in tl["log"])[len(tl["log"]) // 2]
        f = m["final_test"]
        lines.append(f"| {name} | {m['train_seconds'] / 60:.0f} min | {sps:.1f} | {m['n_gaussians']:,} | "
                     f"{m['peak_vram_allocated_mb'] / 1024:.2f} GB | {f['psnr']:.2f} dB | {f['ssim']:.3f} | "
                     f"{f['lpips']:.3f} |")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("COMPARE_DONE")


if __name__ == "__main__":
    main()
