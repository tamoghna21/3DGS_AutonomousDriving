# SPDX-License-Identifier: MIT
"""Step 4: replay the drive. Held-out metrics (PSNR, SSIM, LPIPS; also PSNR
over static pixels only, since moving objects are masked out of training),
rendering speed, and the figures:

  figures/fig_heldout_cameras.png    a held-out moment, all three cameras: photo | splat | error
  figures/fig_heldout_front.png      three held-out moments, front camera
  figures/fig_replay.gif             the drive (front camera): photo | splat | rendered depth
  outputs/replay/replay.mp4          the same at full frame rate
  outputs/replay/metrics.{json,md}

Published images are privacy-blurred (license plates, people); metrics use the
unblurred images.

    python scripts/04_replay.py
"""
import argparse
import json
import os
import sys
import time

import matplotlib
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from splat import figs, metrics, privacy, scene as sc  # noqa: E402
from splat.figs import heatmap, label, to_u8  # noqa: E402
from splat.model import load_model, render_view  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAM_NAME = {"front_left_camera": "front-left", "front_camera": "front", "front_right_camera": "front-right"}


def depth_color(depth: torch.Tensor, alpha: torch.Tensor, max_depth: float = 80.0) -> np.ndarray:
    """Expected depth / alpha on a turbo colormap of inverse depth (warm = near, blue = far), sky black."""
    d = (depth / alpha.clamp_min(1e-3)).clamp_min(0.5)
    inv = (2.0 / d).clamp(0, 1) ** 0.5                     # 2 m -> 1, far -> 0
    rgb = matplotlib.colormaps["turbo"](inv.cpu().numpy())[..., :3]
    rgb = rgb * alpha.clamp(0, 1).cpu().numpy()[..., None]
    return (rgb * 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=os.path.join(ROOT, "data", "pandaset_028"))
    p.add_argument("--ckpt", default=os.path.join(ROOT, "outputs", "train", "ckpt.pt"))
    p.add_argument("--out-dir", default=os.path.join(ROOT, "outputs", "replay"))
    p.add_argument("--fig-dir", default=os.path.join(ROOT, "figures"))
    p.add_argument("--cameras-frame", type=int, default=44, help="held-out frame shown for all three cameras")
    p.add_argument("--front-frames", type=int, nargs="+", default=[12, 36, 68])
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    device = torch.device("cuda")

    params, sky, color, meta = load_model(args.ckpt, device)
    sh = meta["sh_degree"]
    scene = sc.load_scene(args.data_dir)
    W, H = scene["width"], scene["height"]

    # 1) Held-out metrics, overall and per camera.
    test = sc.load_views(scene, "test")
    per_view, preds = [], {}
    with torch.no_grad():
        for i, rec in enumerate(test["records"]):
            w2c, K = test["w2c"][i].to(device), test["K"][i].to(device)
            cam = int(test["cam_index"][i])
            gt = test["images"][i].to(device).float() / 255.0
            mask = test["masks"][i].to(device)
            pred = render_view(params, sky, color, w2c, K, W, H, sh, cam=cam)["rgb"].clamp(0, 1)
            per_view.append({"camera": rec["camera"], "frame": rec["frame"], **metrics.all_metrics(pred, gt, ~mask)})
            preds[(rec["camera"], rec["frame"])] = (pred, gt, rec)
    keys = ("psnr", "ssim", "lpips", "psnr_static")
    mean = {k: float(np.mean([m[k] for m in per_view])) for k in keys}
    by_cam = {c: {k: float(np.mean([m[k] for m in per_view if m["camera"] == c])) for k in keys}
              for c in scene["cameras"]}

    # 2) Held-out figures: photo | splat | error. All three cameras at one
    #    moment, and the front camera at three moments.
    def heldout_row(c, f):
        pred, gt, rec = preds[(c, f)]
        boxes = privacy.boxes_at(scene["tracks"], rec["timestamp"])
        w2c, K = np.asarray(rec["w2c"]), np.asarray(rec["K"])
        m = next(v for v in per_view if v["camera"] == c and v["frame"] == f)
        return np.concatenate([
            label(privacy.blur(to_u8(gt), boxes, w2c, K), f"photo: {CAM_NAME[c]}, held-out frame {f}", 14),
            label(privacy.blur(to_u8(pred), boxes, w2c, K), f"splat: PSNR {m['psnr']:.1f} dB", 14),
            label(heatmap(pred, gt), "|error| (black 0, bright >= 0.1)", 14)], axis=1)
    rows = [heldout_row(c, args.cameras_frame) for c in ("front_left_camera", "front_camera", "front_right_camera")]
    Image.fromarray(np.concatenate(rows, 0)).save(os.path.join(args.fig_dir, "fig_heldout_cameras.png"))
    rows = [heldout_row("front_camera", f) for f in args.front_frames]
    Image.fromarray(np.concatenate(rows, 0)).save(os.path.join(args.fig_dir, "fig_heldout_front.png"))

    # 3) The drive, front camera, every frame: photo | splat | depth.
    recs = sorted([r for r in scene["images"] if r["camera"] == "front_camera"], key=lambda r: r["frame"])
    mp4 = figs.mp4_writer(os.path.join(args.out_dir, "replay.mp4"), fps=10)
    gif = []
    with torch.no_grad():
        for rec in recs:
            w2c = torch.tensor(rec["w2c"], dtype=torch.float32, device=device)
            K = torch.tensor(rec["K"], dtype=torch.float32, device=device)
            out = render_view(params, sky, color, w2c, K, W, H, sh, cam=0, with_depth=True)
            gt = np.asarray(Image.open(os.path.join(args.data_dir, rec["file"])).convert("RGB"))
            boxes = privacy.boxes_at(scene["tracks"], rec["timestamp"])
            w2c_np, K_np = np.asarray(rec["w2c"]), np.asarray(rec["K"])
            tag = "held out" if rec["split"] == "test" else "training view"
            frame = np.concatenate([
                label(privacy.blur(gt, boxes, w2c_np, K_np), f"camera, frame {rec['frame']} ({tag})", 14),
                label(privacy.blur(to_u8(out["rgb"].clamp(0, 1)), boxes, w2c_np, K_np), "Gaussian splat", 14),
                label(depth_color(out["depth"], out["alpha"]), "splat depth", 14)], axis=1)
            frame = frame[: frame.shape[0] // 2 * 2]           # even height for H.264
            mp4.append_data(frame)
            gif.append(frame)
    mp4.close()
    figs.save_gif(os.path.join(args.fig_dir, "fig_replay.gif"), gif[::2], 1200, 0.2)

    # 4) Rendering speed at the training resolution (render only).
    w2cs = [torch.tensor(r["w2c"], dtype=torch.float32, device=device) for r in recs]
    K = torch.tensor(recs[0]["K"], dtype=torch.float32, device=device)
    with torch.no_grad():
        for w in w2cs[:5]:
            render_view(params, sky, color, w, K, W, H, sh, cam=0)
        torch.cuda.synchronize()
        t0 = time.time()
        for w in w2cs:
            render_view(params, sky, color, w, K, W, H, sh, cam=0)
        torch.cuda.synchronize()
    fps = len(w2cs) / (time.time() - t0)

    result = {"heldout_mean": mean, "heldout_by_camera": by_cam, "heldout_per_view": per_view,
              "render_fps": fps, "resolution": [W, H], "n_gaussians": meta["n_gaussians"],
              "train_minutes": meta["train_seconds"] / 60, "peak_vram_allocated_mb": meta["peak_vram_allocated_mb"],
              "peak_vram_reserved_mb": meta["peak_vram_reserved_mb"], "gpu": meta["gpu"], "steps": meta["steps"]}
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=1)
    lines = ["| Held-out views | PSNR | PSNR, static pixels | SSIM | LPIPS |", "|---|---|---|---|---|"]
    lines.append(f"| all 30 | {mean['psnr']:.2f} dB | {mean['psnr_static']:.2f} dB | {mean['ssim']:.3f} | "
                 f"{mean['lpips']:.3f} |")
    for c, m in by_cam.items():
        lines.append(f"| {CAM_NAME[c]} (10) | {m['psnr']:.2f} dB | {m['psnr_static']:.2f} dB | {m['ssim']:.3f} | "
                     f"{m['lpips']:.3f} |")
    lines += ["", "| Gaussians | Training time | Peak VRAM (training) | Render speed |", "|---|---|---|---|",
              f"| {meta['n_gaussians']:,} | {meta['train_seconds'] / 60:.0f} min | "
              f"{meta['peak_vram_allocated_mb'] / 1024:.2f} GB allocated ({meta['peak_vram_reserved_mb'] / 1024:.2f} "
              f"GB reserved) | {fps:.1f} FPS at {W}x{H} |"]
    table = "\n".join(lines) + "\n"
    with open(os.path.join(args.out_dir, "metrics.md"), "w") as f:
        f.write(table)
    print(table)
    print("REPLAY_DONE")


if __name__ == "__main__":
    main()
