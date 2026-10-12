# SPDX-License-Identifier: MIT
"""Step 3: train 3D Gaussians on the PandaSet drive, rendering with the
pure-PyTorch rasterizer (splat/render.py).

Each step renders one random training image and compares it to the photo:
    loss = 0.8 L1 + 0.2 (1 - SSIM)          over pixels that are not masked
         + depth_lambda |D - z| / z          at that frame's LiDAR returns
where D is the rendered expected depth. Moving objects and people are masked
out of the photometric loss (they break the static-scene assumption).
Adaptive density control (clone / split / prune / opacity reset) grows the
model from the LiDAR initialization, up to --max-gaussians.

    python scripts/03_train.py
"""
import argparse
import json
import math
import os
import sys
import time

# The chunked rasterizer allocates tensors of a different size every chunk;
# expandable segments stop the CUDA caching allocator from fragmenting on that.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from splat import gaussians as gs, metrics, scene as sc  # noqa: E402
from splat.model import render_view  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description="Train Gaussian splats on a prepared PandaSet scene.")
    p.add_argument("--data-dir", default=os.path.join(ROOT, "data", "pandaset_028"))
    p.add_argument("--out-dir", default=os.path.join(ROOT, "outputs", "train"))
    p.add_argument("--steps", type=int, default=30_000)
    p.add_argument("--init-points", type=int, default=400_000, help="LiDAR map points used to initialize")
    p.add_argument("--max-gaussians", type=int, default=1_000_000)
    p.add_argument("--sh-degree", type=int, default=3)
    p.add_argument("--position-lr-scale", type=float, default=5.0,
                   help="meters; means lr = 1.6e-4 x this, decaying to 1%% by the end")
    p.add_argument("--ssim-lambda", type=float, default=0.2)
    p.add_argument("--depth-lambda", type=float, default=0.1, help="weight of the LiDAR depth loss (0 = off)")
    p.add_argument("--no-color-correction", action="store_true")
    p.add_argument("--grad-thresh", type=float, default=2e-4, help="densify above this NDC position gradient")
    p.add_argument("--refine-stop", type=int, default=15_000)
    p.add_argument("--needle-reg", type=float, default=0.0,
                   help="weight of the anti-needle loss: mean over Gaussians of relu(log(longest / middle axis) - "
                        "log(--needle-ratio)). Needles fit the recorded views and smear when the camera moves")
    p.add_argument("--needle-ratio", type=float, default=5.0)
    p.add_argument("--opacity-reg", type=float, default=0.0, help="weight of mean opacity (SplatAD uses 0.005)")
    p.add_argument("--save-at", type=int, nargs="*", default=[], help="also save checkpoints after these steps")
    p.add_argument("--eval-every", type=int, default=5_000)
    p.add_argument("--progress-view", default="front_camera/44", help="held-out view saved during training")
    p.add_argument("--backend", choices=("torch", "gsplat"), default="torch",
                   help="torch: the pure-PyTorch rasterizer. gsplat: gsplat's CUDA kernels, only to compare "
                        "training speed (needs gsplat installed)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(views, i, device):
    w2c = views["w2c"][i].to(device)
    return (w2c, views["K"][i].to(device), int(views["cam_index"][i]),
            views["images"][i].to(device, non_blocking=True).float() / 255.0,
            views["masks"][i].to(device, non_blocking=True))


@torch.no_grad()
def evaluate(params, sky, color, views, sh, device, full: bool = False):
    W, H = views["width"], views["height"]
    out = []
    for i in range(len(views["records"])):
        w2c, K, cam, gt, mask = to_device(views, i, device)
        pred = render_view(params, sky, color, w2c, K, W, H, sh, cam=cam)["rgb"].clamp(0, 1)
        out.append(metrics.all_metrics(pred, gt, ~mask) if full else
                   {"psnr": metrics.psnr(pred, gt), "psnr_static": metrics.psnr(pred, gt, ~mask)})
    return {k: float(np.mean([o[k] for o in out])) for k in out[0]}


def progress_steps(total):
    s = {0, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 2500}
    s |= set(range(3000, total + 1, 1000))
    return s


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda")
    os.makedirs(os.path.join(args.out_dir, "progress"), exist_ok=True)
    print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) on {torch.cuda.get_device_name(0)}")

    scene = sc.load_scene(args.data_dir)
    train = sc.load_views(scene, "train", with_depth=args.depth_lambda > 0)
    test = sc.load_views(scene, "test")
    W, H = train["width"], train["height"]
    print(f"{len(train['records'])} train / {len(test['records'])} test views at {W}x{H}, cameras {scene['cameras']}")

    xyz, rgb = sc.load_points(scene)
    params = gs.init_from_points(xyz, rgb, args.init_points, args.sh_degree, args.seed, device=device)
    print(f"initialized {len(params['means']):,} Gaussians from {len(xyz):,} LiDAR map points")

    # Sky: start from the median color of the top rows of the front-camera photos.
    front = train["cam_index"] == 0
    sky_color = train["images"][front][:, : H // 10].reshape(-1, 3).float().median(dim=0).values / 255.0
    sky_size = (64, 256)
    sky = gs.Sky(sky_color, *sky_size).to(device)
    color = None if args.no_color_correction else gs.ColorCorrection(len(scene["cameras"])).to(device)
    color_or_identity = color if color is not None else gs.ColorCorrection(len(scene["cameras"]))

    means_lr = 1.6e-4 * args.position_lr_scale
    lrs = {"means": means_lr, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2,
           "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    optimizers = {k: torch.optim.Adam([params[k]], lr=lr, eps=1e-15) for k, lr in lrs.items()}
    extra_opt = torch.optim.Adam([{"params": sky.parameters(), "lr": 1e-2}] +
                                 ([{"params": color.parameters(), "lr": 1e-3}] if color is not None else []))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / args.steps))
    density = gs.DensityControl(len(params["means"]), device, grad_thresh=args.grad_thresh,
                                refine_stop=args.refine_stop, max_gaussians=args.max_gaussians)

    prog_cam, prog_frame = args.progress_view.split("/")
    prog_i = next(i for i, r in enumerate(test["records"]) if r["camera"] == prog_cam and r["frame"] == int(prog_frame))
    snap_at = progress_steps(args.steps)

    rng = np.random.default_rng(args.seed)
    torch.cuda.reset_peak_memory_stats()
    log, refine_log, running = [], [], []
    train_seconds, tic = 0.0, time.time()
    block_tic, block_start = time.time(), 0
    for step in tqdm(range(args.steps + 1), desc="training", mininterval=10):
        if step in snap_at:                                    # progress snapshot (not timed)
            train_seconds += time.time() - tic
            with torch.no_grad():
                w2c, K, cam, _, _ = to_device(test, prog_i, device)
                img = render_view(params, sky, color, w2c, K, W, H, min(step // 1000, args.sh_degree),
                                  cam=cam)["rgb"].clamp(0, 1)
            Image.fromarray((img.cpu().numpy() * 255 + 0.5).astype(np.uint8)).save(
                os.path.join(args.out_dir, "progress", f"step_{step:05d}_{len(params['means'])}.png"))
            tic = time.time()
        if step == args.steps:
            break

        i = int(rng.integers(len(train["records"])))
        w2c, K, cam, gt, mask = to_device(train, i, device)
        sh = min(step // 1000, args.sh_degree)                # one more SH band every 1000 steps
        out = render_view(params, sky, color, w2c, K, W, H, sh, cam=cam, with_depth=args.depth_lambda > 0,
                          backend=args.backend)
        out["proj"]["means2d"].retain_grad()
        pred = out["rgb"]

        valid = (~mask).float()
        n_valid = valid.sum().clamp_min(1.0)
        l1 = ((pred - gt).abs().mean(-1) * valid).sum() / n_valid
        ssim_loss = 1.0 - (metrics.ssim_map(pred, gt) * valid).sum() / n_valid
        loss = (1 - args.ssim_lambda) * l1 + args.ssim_lambda * ssim_loss
        depth_err = torch.zeros((), device=device)
        if args.depth_lambda > 0:
            uvz = train["lidar"][i].to(device, non_blocking=True)
            u = uvz[:, 0].long().clamp(0, W - 1)
            v = uvz[:, 1].long().clamp(0, H - 1)
            d = out["depth"][v, u]
            depth_err = ((d - uvz[:, 2]).abs() / uvz[:, 2]).mean()
            loss = loss + args.depth_lambda * depth_err
        if args.needle_reg > 0:
            # Sorted log-scales: longest minus middle axis = log of how needle-like a Gaussian is.
            # Flat discs (two long axes, one short) are allowed; that's what roads and walls need.
            ls = params["scales"].sort(dim=1, descending=True).values
            loss = loss + args.needle_reg * torch.relu(ls[:, 0] - ls[:, 1] - math.log(args.needle_ratio)).mean()
        if args.opacity_reg > 0:
            loss = loss + args.opacity_reg * torch.sigmoid(params["opacities"]).mean()
        loss.backward()

        density.accumulate(out["proj"], W, H)
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        extra_opt.step()
        extra_opt.zero_grad(set_to_none=True)
        scheduler.step()
        report = density.step(step + 1, params, optimizers)
        if report is not None:
            refine_log.append(report)
        if step % 50 == 0:
            running.append((loss.item(), l1.item(), depth_err.item()))

        if step + 1 in args.save_at:
            gs.save(os.path.join(args.out_dir, f"ckpt_{step + 1:05d}.pt"), params, {"sky": sky, "color": color_or_identity},
                    {"steps": step + 1, "sh_degree": args.sh_degree, "sky_size": list(sky_size),
                     "cameras": scene["cameras"], "n_gaussians": len(params["means"]), "args": vars(args)})
        if (step + 1) % 1000 == 0:
            r = np.array(running)
            entry = {"step": step + 1, "loss": float(r[:, 0].mean()), "l1": float(r[:, 1].mean()),
                     "depth_rel_err": float(r[:, 2].mean()), "train_psnr": metrics.psnr(pred.detach().clamp(0, 1), gt),
                     "n_gaussians": len(params["means"]), "sh_degree": sh,
                     "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20,
                     "steps_per_sec": (step + 1 - block_start) / (time.time() - block_tic)}
            running = []
            if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
                train_seconds += time.time() - tic
                entry.update({f"test_{k}": v for k, v in evaluate(params, sky, color, test, sh, device).items()})
                tic = time.time()
            log.append(entry)
            block_tic, block_start = time.time(), step + 1
            tqdm.write(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in entry.items()}))
    train_seconds += time.time() - tic

    final = evaluate(params, sky, color, test, args.sh_degree, device, full=True)
    meta = {
        "steps": args.steps, "sh_degree": args.sh_degree, "n_gaussians": len(params["means"]),
        "train_seconds": train_seconds, "sky_size": list(sky_size), "cameras": scene["cameras"],
        "peak_vram_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
        "peak_vram_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "width": W, "height": H, "args": vars(args), "final_test": final,
    }
    gs.save(os.path.join(args.out_dir, "ckpt.pt"), params, {"sky": sky, "color": color_or_identity}, meta)
    with open(os.path.join(args.out_dir, "train_log.json"), "w") as f:
        json.dump({"meta": meta, "log": log, "refine": refine_log}, f, indent=1)
    print("TRAIN_DONE", json.dumps({k: meta[k] for k in ("n_gaussians", "train_seconds", "peak_vram_allocated_mb",
                                                          "final_test")}))


if __name__ == "__main__":
    main()
