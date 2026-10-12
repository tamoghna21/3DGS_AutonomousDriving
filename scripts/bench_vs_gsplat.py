# SPDX-License-Identifier: MIT
"""The trained street scene, rendered by the pure-PyTorch rasterizer and by
gsplat's CUDA kernels: how different are the images, and how much slower is
plain PyTorch? Times rendering (forward) and a training step's render +
backward on every training view.

gsplat is used only for this comparison and is not a dependency of the
project. Run it in a separate environment that has gsplat installed:

    python scripts/bench_vs_gsplat.py

Writes outputs/bench/bench_vs_gsplat.{json,md}, figures/fig_renderer_speed.png and
figures/fig_renderer_match.png.
"""
import argparse
import json
import math
import os
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from gsplat import rasterization
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from splat import gaussians as gs, privacy, render as R, scene as sc  # noqa: E402
from splat.figs import label, to_u8  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ("#2a78d6", "#eb6834")


def ours(act, w2c, K, W, H, sh):
    m, q, s, o, c = act
    return R.render(m, q, s, o, c, w2c, K, W, H, sh)["rgb"]


def theirs(act, w2c, K, W, H, sh):
    m, q, s, o, c = act
    return rasterization(m, q, s, o, c, w2c[None], K[None], W, H, sh_degree=sh, packed=False)[0][0]


def timed(fn, backward):
    torch.cuda.synchronize()
    t0 = time.time()
    img = fn()
    if backward:
        img.mean().backward()
    torch.cuda.synchronize()
    return img.detach(), time.time() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=os.path.join(ROOT, "data", "pandaset_028"))
    p.add_argument("--ckpt", default=os.path.join(ROOT, "outputs", "train", "ckpt.pt"))
    p.add_argument("--out-dir", default=os.path.join(ROOT, "outputs", "bench"))
    p.add_argument("--fig-dir", default=os.path.join(ROOT, "figures"))
    p.add_argument("--show-frame", type=int, default=30, help="front-camera training frame for the side-by-side")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda")
    params, _, meta = gs.load(args.ckpt, device)
    for v in params.values():
        v.requires_grad_(True)
    sh = meta["sh_degree"]
    scene = sc.load_scene(args.data_dir)
    W, H = scene["width"], scene["height"]
    views = [r for r in scene["images"] if r["split"] == "train"]

    res = {"ours": {"fwd": [], "fwdbwd": []}, "gsplat": {"fwd": [], "fwdbwd": []}, "psnr_between": []}
    shown = None
    for k, r in enumerate(views):
        w2c = torch.tensor(r["w2c"], dtype=torch.float32, device=device)
        K = torch.tensor(r["K"], dtype=torch.float32, device=device)
        imgs = {}
        for name, fn in (("ours", ours), ("gsplat", theirs)):
            with torch.no_grad():
                img, t = timed(lambda: fn(gs.activated(params), w2c, K, W, H, sh), False)
            imgs[name] = img
            for v in params.values():
                v.grad = None
            _, tb = timed(lambda: fn(gs.activated(params), w2c, K, W, H, sh), True)
            if k >= 3:                                         # skip warm-up views
                res[name]["fwd"].append(t)
                res[name]["fwdbwd"].append(tb)
        mse = ((imgs["ours"].clamp(0, 1) - imgs["gsplat"].clamp(0, 1)) ** 2).mean().item()
        res["psnr_between"].append(10 * math.log10(1 / max(mse, 1e-12)))
        if r["camera"] == "front_camera" and r["frame"] == args.show_frame:
            shown = (r, imgs["ours"].clamp(0, 1), imgs["gsplat"].clamp(0, 1), res["psnr_between"][-1])

    summary = {name: {k: float(np.median(v)) * 1e3 for k, v in res[name].items()} for name in ("ours", "gsplat")}
    summary["psnr_between_median_db"] = float(np.median(res["psnr_between"]))
    summary["psnr_between_min_db"] = float(np.min(res["psnr_between"]))
    summary["n_gaussians"] = meta["n_gaussians"]
    summary["resolution"] = [W, H]
    summary["gpu"] = torch.cuda.get_device_name(0)
    with open(os.path.join(args.out_dir, "bench_vs_gsplat.json"), "w") as f:
        json.dump({"summary": summary, "per_view": res}, f, indent=1)
    o, g = summary["ours"], summary["gsplat"]
    table = ("| Renderer | Render (forward) | Render + backward | Same image? |\n|---|---|---|---|\n"
             f"| pure PyTorch (this repo) | {o['fwd']:.0f} ms | {o['fwdbwd']:.0f} ms | PSNR vs gsplat: median "
             f"{summary['psnr_between_median_db']:.1f} dB, worst {summary['psnr_between_min_db']:.1f} dB |\n"
             f"| gsplat 1.5.3 (CUDA) | {g['fwd']:.1f} ms | {g['fwdbwd']:.1f} ms | reference |\n"
             f"\nMedian over {len(views) - 3} training views, {meta['n_gaussians']:,} Gaussians, {W}x{H}, "
             f"{summary['gpu']}. Pure PyTorch is {o['fwd'] / g['fwd']:.0f}x slower to render and "
             f"{o['fwdbwd'] / g['fwdbwd']:.0f}x slower per training step.\n")
    with open(os.path.join(args.out_dir, "bench_vs_gsplat.md"), "w") as f:
        f.write(table)
    print(table)

    # Figure: horizontal bars, value at the tip, one chart per measure (shared scale, no second axis).
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.4), dpi=160, facecolor=SURFACE, sharex=True)
    for ax, key, title in ((axes[0], "fwd", "Render one 480x270 frame"),
                           (axes[1], "fwdbwd", "Render + backward (one training step)")):
        ax.set_facecolor(SURFACE)
        vals = [summary["gsplat"][key], summary["ours"][key]]
        ax.barh([0, 1], vals, height=0.42, color=[SERIES[1], SERIES[0]])
        for y, v in zip([0, 1], vals):
            ax.text(v + max(vals) * 0.015, y, f"{v:.0f} ms" if v >= 10 else f"{v:.1f} ms", va="center",
                    color=INK, fontsize=9)
        ax.set_yticks([0, 1], ["gsplat (CUDA kernels)", "pure PyTorch"], color=INK2, fontsize=9)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="x", colors=INK2, labelsize=8)
        ax.tick_params(axis="y", length=0)
        ax.grid(True, axis="x", color=GRID, lw=1.0)
        ax.set_axisbelow(True)
        ax.set_xlabel("milliseconds (median)", color=INK2, fontsize=8)
    axes[0].set_xlim(0, max(summary["ours"]["fwdbwd"], summary["ours"]["fwd"]) * 1.18)
    fig.tight_layout()
    fig.savefig(os.path.join(args.fig_dir, "fig_renderer_speed.png"), facecolor=SURFACE)

    # Side by side on one view (raw Gaussians, no sky): pure PyTorch | gsplat | |difference| x 50.
    if shown is not None:
        r, a, b, db = shown
        w2c, K = np.asarray(r["w2c"]), np.asarray(r["K"])
        boxes = privacy.boxes_at(scene["tracks"], r["timestamp"], static_only=True)
        diff = ((a - b).abs().mean(-1) * 50).clamp(0, 1).cpu().numpy()
        diff = (matplotlib.colormaps["magma"](diff)[..., :3] * 255).astype(np.uint8)
        row = np.concatenate([label(privacy.blur(to_u8(a), boxes, w2c, K), "pure PyTorch (Gaussians only, no sky)", 14),
                              label(privacy.blur(to_u8(b), boxes, w2c, K), "gsplat CUDA kernels (same Gaussians)", 14),
                              label(diff, f"|difference| x 50  (PSNR between: {db:.1f} dB)", 14)], axis=1)
        Image.fromarray(row).save(os.path.join(args.fig_dir, "fig_renderer_match.png"))
    print("BENCH_DONE")


if __name__ == "__main__":
    main()
