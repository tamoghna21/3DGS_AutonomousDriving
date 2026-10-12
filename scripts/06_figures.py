# SPDX-License-Identifier: MIT
"""Step 6: figures about the model itself.

  figures/fig_training_progress.png   one held-out view during training, LiDAR init to final
  figures/fig_training_progress.gif   the same, every snapshot
  figures/fig_training_curves.png     held-out PSNR and Gaussian count over training
  figures/fig_anatomy.png             what the splat is made of: render and depth, then a 3x crop:
                                      Gaussian centers, the Gaussians as opaque ellipses
  figures/fig_hero.png                header image: a held-out photo wiping into the splat

    python scripts/06_figures.py
"""
import argparse
import glob
import json
import os
import re
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from splat import figs, gaussians as gs, privacy, scene as sc  # noqa: E402
from splat.figs import label, to_u8  # noqa: E402
from splat.model import load_model, render_view  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module  # noqa: E402

depth_color = import_module("04_replay").depth_color

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ("#2a78d6", "#eb6834")


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, color=GRID, lw=1.0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=os.path.join(ROOT, "data", "pandaset_028"))
    p.add_argument("--train-dir", default=os.path.join(ROOT, "outputs", "train"))
    p.add_argument("--fig-dir", default=os.path.join(ROOT, "figures"))
    p.add_argument("--view", default="front_camera/44")
    p.add_argument("--progress-steps", type=int, nargs="+", default=[0, 200, 1000, 5000, 30000])
    p.add_argument("--anatomy-crop", type=int, nargs=4, default=[300, 150, 160, 90],
                   help="x0 y0 width height of the zoomed crop, in training-resolution pixels")
    p.add_argument("--hero-view", default="front_camera/52")
    p.add_argument("--hero-scale", type=int, default=3)
    args = p.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)
    device = torch.device("cuda")
    scene = sc.load_scene(args.data_dir)
    W, H = scene["width"], scene["height"]
    cam_name, frame = args.view.split("/")
    rec = next(r for r in scene["images"] if r["camera"] == cam_name and r["frame"] == int(frame))
    w2c, K = np.asarray(rec["w2c"]), np.asarray(rec["K"])
    boxes = privacy.boxes_at(scene["tracks"], rec["timestamp"])
    static_boxes = privacy.boxes_at(scene["tracks"], rec["timestamp"], static_only=True)
    photo = np.asarray(Image.open(os.path.join(args.data_dir, rec["file"])).convert("RGB"))
    photo = privacy.blur(photo, boxes, w2c, K)

    # 1) Training progress on the held-out view.
    snaps = {}
    for f in glob.glob(os.path.join(args.train_dir, "progress", "step_*.png")):
        step, n = map(int, re.findall(r"step_(\d+)_(\d+)", f)[0])
        snaps[step] = (f, n)
    steps = sorted(snaps)

    def snap(step):
        f, n = snaps[step]
        img = privacy.blur(np.asarray(Image.open(f).convert("RGB")), static_boxes, w2c, K)
        what = "LiDAR initialization" if step == 0 else f"step {step:,}"
        return label(img, f"{what}, {n:,} Gaussians", 14)
    chosen = [s for s in args.progress_steps if s in snaps]
    tiles = [snap(s) for s in chosen] + [label(photo, f"photo (held-out frame {frame})", 14)]
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[i:i + 3], 1) for i in range(0, len(tiles), 3)]
    Image.fromarray(np.concatenate(rows, 0)).save(os.path.join(args.fig_dir, "fig_training_progress.png"))
    gif = [np.concatenate([snap(s), label(photo, "photo (held out)", 14)], 1) for s in steps]
    figs.save_gif(os.path.join(args.fig_dir, "fig_training_progress.gif"), gif + [gif[-1]] * 6, 960, 0.35)

    # 2) Training curves: two charts, one y-axis each.
    with open(os.path.join(args.train_dir, "train_log.json")) as fh:
        tl = json.load(fh)
    ev = [e for e in tl["log"] if "test_psnr" in e]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), dpi=160, facecolor=SURFACE)
    ax = axes[0]
    style(ax)
    x = [e["step"] for e in ev]
    ax.plot(x, [e["test_psnr"] for e in ev], "-o", color=SERIES[0], lw=2, ms=6, mec=SURFACE, mew=2,
            label="all pixels")
    ax.plot(x, [e["test_psnr_static"] for e in ev], "-o", color=SERIES[1], lw=2, ms=6, mec=SURFACE, mew=2,
            label="static pixels (moving objects masked)")
    ax.set_xlabel("training step")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Held-out PSNR, 30 views", color=INK, fontsize=11, loc="left")
    leg = ax.legend(frameon=False, fontsize=8, loc="lower right")
    for t in leg.get_texts():
        t.set_color(INK2)
    ax = axes[1]
    style(ax)
    ref = [(0, tl["refine"][0]["n"] - tl["refine"][0]["cloned"] - tl["refine"][0]["split"] + tl["refine"][0]["pruned"])]
    ref += [(r["step"], r["n"]) for r in tl["refine"]]
    ref += [(e["step"], e["n_gaussians"]) for e in tl["log"] if e["step"] > ref[-1][0]]
    ax.plot([r[0] for r in ref], [r[1] / 1e6 for r in ref], "-", color=SERIES[0], lw=2)
    ax.set_xlabel("training step")
    ax.set_ylabel("Gaussians (millions)")
    ax.set_title("Gaussian count (densification until step "
                 f"{tl['meta']['args']['refine_stop']:,})", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(args.fig_dir, "fig_training_curves.png"), facecolor=SURFACE)
    plt.close(fig)

    # 3) Anatomy of the splat on the held-out view: the full frame, then a crop
    #    rendered at 3x (a splat renders at any resolution) to show single Gaussians.
    params, sky, color, meta = load_model(os.path.join(args.train_dir, "ckpt.pt"), device)
    cam = scene["cameras"].index(cam_name)
    x0, y0, cw, ch = args.anatomy_crop
    z = 480 // cw
    Kc = K.copy()
    Kc[:2] *= z
    Kc[0, 2] -= x0 * z
    Kc[1, 2] -= y0 * z
    Wc, Hc = cw * z, ch * z

    def run(p, KK, w, h, sky_, color_, deg, depth=False):
        return render_view(p, sky_, color_, torch.tensor(w2c, dtype=torch.float32, device=device),
                           torch.tensor(KK, dtype=torch.float32, device=device), w, h, deg, cam=cam if color_ else None,
                           with_depth=depth)

    with torch.no_grad():
        full = run(params, K, W, H, sky, color, meta["sh_degree"], depth=True)
        full_img = privacy.blur(to_u8(full["rgb"].clamp(0, 1)), static_boxes, w2c, K)
        full_img = np.array(Image.fromarray(full_img))
        full_img[y0:y0 + ch, [x0, x0 + cw - 1]] = 255           # mark the crop
        full_img[[y0, y0 + ch - 1], x0:x0 + cw] = 255
        crop = run(params, Kc, Wc, Hc, sky, color, meta["sh_degree"])
        crop_img = privacy.blur(to_u8(crop["rgb"].clamp(0, 1)), static_boxes, w2c, Kc)
        # Centers: each Gaussian in the crop as a 2x2 dot in its base (degree-0) color, far drawn first.
        proj = crop["proj"]
        idx = proj["idx"]
        rgb0 = (params["sh0"][idx, 0] * gs.SH_C0 + 0.5).clamp(0, 1)
        order = torch.argsort(proj["depths"], descending=True)
        uv = proj["means2d"][order].floor().long()
        dots = torch.zeros(Hc + 1, Wc + 1, 3, device=device)
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < Wc) & (uv[:, 1] >= 0) & (uv[:, 1] < Hc)
        for dy in (0, 1):
            for dx in (0, 1):
                dots[uv[ok, 1] + dy, uv[ok, 0] + dx] = rgb0[order][ok]
        centers = to_u8(dots[:Hc, :Wc])                      # single dots: nothing readable to blur
        # Ellipses: Gaussians with opacity > 0.2 made opaque, in random colors; full size, then shrunk.
        g = torch.Generator(device=device).manual_seed(0)
        n = params["means"].shape[0]
        rand = torch.rand(n, 3, device=device, generator=g) * 0.8 + 0.2

        def ellipses(shrink):
            ell = {"means": params["means"], "quats": params["quats"], "scales": params["scales"] + np.log(shrink),
                   "opacities": torch.where(torch.sigmoid(params["opacities"]) > 0.2, 6.0, -10.0),
                   "sh0": gs.rgb_to_sh0(rand)[:, None, :], "shN": torch.zeros_like(params["shN"])}
            return to_u8(run(ell, Kc, Wc, Hc, None, None, 0)["rgb"].clamp(0, 1))
        ell_full, ell_small = ellipses(1.0), ellipses(0.4)
    n_vis = int(idx.numel())
    tiles = [label(full_img, f"splat, held-out frame {frame} ({meta['n_gaussians']:,} Gaussians)", 13),
             label(depth_color(full["depth"], full["alpha"]), "rendered depth (warm = near, blue = far)", 13),
             label(crop_img, f"the white box, rendered at {z}x", 13),
             label(centers, f"Gaussian centers ({n_vis:,} touch the box)", 13),
             label(ell_full, "opacity > 0.2, made opaque, random colors", 13),
             label(ell_small, "the same, shrunk to 40%: single Gaussians", 13)]
    Image.fromarray(np.concatenate([np.concatenate(tiles[:3], 1), np.concatenate(tiles[3:], 1)], 0)).save(
        os.path.join(args.fig_dir, "fig_anatomy.png"))

    # 4) Header image: the photo wipes diagonally into the splat at the same
    #    (held-out) pose. The photo is the 480x270 training-resolution frame,
    #    upscaled; the splat is rendered directly at hero-scale x that resolution.
    hc, hf = args.hero_view.split("/")
    hr = next(r for r in scene["images"] if r["camera"] == hc and r["frame"] == int(hf))
    s = args.hero_scale
    Wh, Hh = W * s, H * s
    Kh = np.asarray(hr["K"]).copy()
    Kh[:2] *= s                                            # pixel centers at +0.5: scaling is exact
    w2c_h = np.asarray(hr["w2c"])
    photo_h = np.asarray(Image.open(os.path.join(args.data_dir, hr["file"])).convert("RGB").resize((Wh, Hh),
                                                                                                    Image.LANCZOS))
    photo_h = privacy.blur(photo_h, privacy.boxes_at(scene["tracks"], hr["timestamp"]), w2c_h, Kh, strength=6.0 * s)
    with torch.no_grad():
        out = render_view(params, sky, color, torch.tensor(w2c_h, dtype=torch.float32, device=device),
                          torch.tensor(Kh, dtype=torch.float32, device=device), Wh, Hh, meta["sh_degree"],
                          cam=scene["cameras"].index(hc))
    splat_h = privacy.blur(to_u8(out["rgb"].clamp(0, 1)),
                           privacy.boxes_at(scene["tracks"], hr["timestamp"], static_only=True), w2c_h, Kh,
                           strength=6.0 * s)
    yy, xx = np.mgrid[0:Hh, 0:Wh]
    edge = Wh * 0.42 + (Hh - yy) * 0.35                    # a slanted seam
    hero = np.where((xx < edge)[..., None], photo_h, splat_h)
    hero[np.abs(xx - edge) < 2] = 255
    hero = label(label(hero, "camera photo", 28), "Gaussian splat (pure PyTorch)", 28, right=True)
    Image.fromarray(hero).save(os.path.join(args.fig_dir, "fig_hero.png"))
    print("FIGURES_DONE")


if __name__ == "__main__":
    main()
