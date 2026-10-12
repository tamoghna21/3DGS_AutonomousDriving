# SPDX-License-Identifier: MIT
"""Step 8: how realistic is the splat off the recorded path?

A driving simulator moves the car where the recording didn't: into the next
lane, for example. There are no photos from there, so, following SplatAD's
protocol, realism is measured as a distribution distance: the Frechet distance
(FD) between DINOv2 features of renders along a laterally shifted path and of
the real camera images (all frames of the same cameras). Lower is better; 0 m
is the recorded path. Also reported: on-path held-out PSNR, SSIM, LPIPS.

Shifts are to the left (into the oncoming lane); the car drives in the right
lane, so a shift to the right would put the camera inside the parked cars.

    python scripts/08_eval_offpath.py --ckpts outputs/train/ckpt.pt --labels baseline
    python scripts/08_eval_offpath.py --ckpts a.pt b.pt --labels A B --out outputs/study/ab.json

FD values depend on the number of images and are only comparable within this
script's settings, not to published tables.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import linalg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from splat import cameras as cm, metrics, scene as sc  # noqa: E402
from splat.figs import to_u8  # noqa: E402
from splat.model import load_model, render_view  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONT3 = ("front_camera", "front_left_camera", "front_right_camera")
_DINO = None


def dino_features(images: list, device, crop_bottom: list) -> np.ndarray:
    """CLS-token features of DINOv2 ViT-L/14 at 224x224 (ImageNet normalization).
    crop_bottom[i] rows are cut off image i first (the back camera's hood)."""
    global _DINO
    if _DINO is None:
        _DINO = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", verbose=False).to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    feats = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(images), 16):
            batch = []
            for img, cb in zip(images[i:i + 16], crop_bottom[i:i + 16]):
                t = torch.from_numpy(img[: img.shape[0] - cb] if cb else img).to(device).permute(2, 0, 1).float()
                batch.append(F.interpolate(t[None] / 255.0, size=(224, 224), mode="bicubic", align_corners=False))
            x = (torch.cat(batch).clamp(0, 1) - mean) / std
            feats.append(_DINO(x).float().cpu())
    return torch.cat(feats).numpy().astype(np.float64)


def frechet_distance(a: np.ndarray, b: np.ndarray) -> float:
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean = linalg.sqrtm(s1 @ s2)
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def evaluate(ckpt, scene, cameras, shifts, device, sample_dir=None):
    params, sky, color, meta = load_model(ckpt, device)
    sh = meta["sh_degree"]
    W, H = scene["width"], scene["height"]
    hood = {"back_camera": 67}
    out = {"ckpt": ckpt, "n_gaussians": int(params["means"].shape[0]), "step": meta.get("step", meta["steps"])}

    # On-path: held-out frames of the evaluated cameras.
    per = []
    with torch.no_grad():
        for rec in scene["images"]:
            if rec["split"] != "test" or rec["camera"] not in cameras:
                continue
            gt = torch.from_numpy(np.asarray(Image.open(os.path.join(scene["data_dir"], rec["file"])))).to(device)
            gt = gt.float() / 255.0
            mask = torch.from_numpy(np.asarray(Image.open(os.path.join(scene["data_dir"], rec["mask"]))) > 127)
            pred = render_view(params, sky, color, torch.tensor(rec["w2c"], dtype=torch.float32, device=device),
                               torch.tensor(rec["K"], dtype=torch.float32, device=device), W, H, sh,
                               cam=meta["cameras"].index(rec["camera"]))["rgb"].clamp(0, 1)
            per.append(metrics.all_metrics(pred, gt, ~mask.to(device)))
    for k in ("psnr", "psnr_static", "ssim", "lpips"):
        out[f"onpath_{k}"] = float(np.mean([m[k] for m in per]))

    # Off-path: every frame of every evaluated camera, the whole rig shifted left
    # along the car's sideways axis.
    recs = sorted([r for r in scene["images"] if r["camera"] in cameras], key=lambda r: (r["camera"], r["frame"]))
    front = {r["frame"]: np.asarray(r["c2w"]) for r in scene["images"] if r["camera"] == "front_camera"}
    for s in shifts:
        imgs, crops = [], []
        with torch.no_grad():
            for rec in recs:
                c2w = cm.shift_rig(np.asarray(rec["c2w"]), front[rec["frame"]], right=-s)
                rgb = render_view(params, sky, color,
                                  torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=device),
                                  torch.tensor(rec["K"], dtype=torch.float32, device=device), W, H, sh,
                                  cam=meta["cameras"].index(rec["camera"]))["rgb"]
                imgs.append(to_u8(rgb.clamp(0, 1)))
                crops.append(hood.get(rec["camera"], 0))
        if sample_dir:                                     # a few renders for inspection (not published)
            os.makedirs(sample_dir, exist_ok=True)
            pick = [i for i, r in enumerate(recs) if r["frame"] in (20, 60)]
            Image.fromarray(np.concatenate([np.concatenate([imgs[i] for i in pick[j:j + 3]], 1)
                                            for j in range(0, len(pick), 3)], 0)).save(
                os.path.join(sample_dir, f"shift_{s:g}m.jpg"), quality=85)
        out[f"feats_{s:g}"] = dino_features(imgs, device, crops)
    return out, recs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--data-dir", default=os.path.join(ROOT, "data", "pandaset_028_6cam"),
                   help="any prepared scene that contains the evaluated cameras (images are the same in all)")
    p.add_argument("--cameras", nargs="+", default=list(FRONT3))
    p.add_argument("--shifts", type=float, nargs="+", default=[0.0, 2.0, 3.0], help="meters to the left")
    p.add_argument("--out", default=os.path.join(ROOT, "outputs", "study", "offpath.json"))
    p.add_argument("--samples", action="store_true", help="save a few shifted renders per checkpoint")
    args = p.parse_args()
    device = torch.device("cuda")
    scene = sc.load_scene(args.data_dir)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    results, real_feats = [], None
    for ckpt, label in zip(args.ckpts, args.labels):
        sample_dir = os.path.join(os.path.dirname(args.out), "samples", label) if args.samples else None
        r, recs = evaluate(ckpt, scene, args.cameras, args.shifts, device, sample_dir)
        if real_feats is None:                             # real images: every frame of the same cameras
            real = [np.asarray(Image.open(os.path.join(scene["data_dir"], x["file"])).convert("RGB")) for x in recs]
            real_feats = dino_features(real, device, [67 if x["camera"] == "back_camera" else 0 for x in recs])
        row = {"label": label, **{k: v for k, v in r.items() if not k.startswith("feats_")}}
        for s in args.shifts:
            row[f"fd_{s:g}m"] = frechet_distance(real_feats, r[f"feats_{s:g}"])
        results.append(row)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)

    with open(args.out, "w") as f:
        json.dump({"cameras": args.cameras, "shifts_m_left": args.shifts, "n_images_per_set": len(recs),
                   "feature": "DINOv2 ViT-L/14 CLS, 224x224", "results": results}, f, indent=1)
    head = "| Model | Gaussians | On-path PSNR | SSIM | LPIPS | " + " | ".join(
        f"FD {s:g} m" for s in args.shifts) + " |"
    lines = [head, "|" + "---|" * (5 + len(args.shifts))]
    for r in results:
        lines.append(f"| {r['label']} | {r['n_gaussians']:,} | {r['onpath_psnr']:.2f} dB | {r['onpath_ssim']:.3f} | "
                     f"{r['onpath_lpips']:.3f} | " + " | ".join(f"{r[f'fd_{s:g}m']:.1f}" for s in args.shifts) + " |")
    note = (f"\nCameras: {', '.join(args.cameras)}. FD: Frechet distance between DINOv2 features of {len(recs)} "
            "renders (all frames, shifted left) and the real images; lower is better.\n")
    with open(os.path.splitext(args.out)[0] + ".md", "w") as f:
        f.write("\n".join(lines) + "\n" + note)
    print("\n".join(lines))
    print("OFFPATH_DONE")


if __name__ == "__main__":
    main()
