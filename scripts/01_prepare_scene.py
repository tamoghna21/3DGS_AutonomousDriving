# SPDX-License-Identifier: MIT
"""Step 1: turn one PandaSet sequence into a training scene.

    python scripts/01_prepare_scene.py --seq 028
    python scripts/01_prepare_scene.py --seq 028 --out-dir data/pandaset_028_6cam \
        --cameras front_camera front_left_camera front_right_camera left_camera right_camera back_camera

Writes data/pandaset_<seq>/:
  images/<camera>/NN.png   frames downscaled 4x (1920x1080 -> 480x270)
  masks/<camera>/NN.png    255 where a moving object, a person or the ego car is (left out of the loss)
  scene.json               per image: camera, frame, split, intrinsics, camera-to-world;
                           object tracks (boxes per frame) for masks, figures and blurring
  points.npz               static LiDAR map: all sweeps merged, moving objects removed,
                           10 cm voxels, colored from the training images
  lidar_depth.npz          per training image, that frame's static LiDAR points as (u, v, depth)

The world frame is PandaSet's, shifted so the front camera's mean position is
the origin (Z stays up).
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from splat import pandaset as ps  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description="Prepare a PandaSet sequence for Gaussian splatting.")
    p.add_argument("--pandaset-dir", default=os.path.join(ROOT, "pandaset_archive"))
    p.add_argument("--seq", default="028")
    p.add_argument("--out-dir", default=None, help="default: data/pandaset_<seq>")
    p.add_argument("--cameras", nargs="+", default=list(ps.CAMERAS))
    p.add_argument("--downscale", type=int, default=4)
    p.add_argument("--test-every", type=int, default=8, help="hold out frames f with f %% N == N/2")
    p.add_argument("--voxel", type=float, default=0.10, help="LiDAR map voxel size, meters")
    p.add_argument("--max-range", type=float, default=120.0, help="drop LiDAR points farther from the sensor")
    p.add_argument("--ego-radius", type=float, default=3.0, help="drop points this close to the LiDAR (the ego car)")
    p.add_argument("--bottom-mask", nargs="*", default=["back_camera:67"],
                   help="camera:rows, mask the bottom rows of a camera's (downscaled) frames, e.g. where the "
                        "back camera sees the ego car's hood")
    return p.parse_args()


def scale_intrinsics(K: np.ndarray, s: int) -> np.ndarray:
    """PandaSet intrinsics put pixel centers at integers (OpenCV). The renderer
    puts them at integer + 0.5 (gsplat), so shift by half a pixel, then scale."""
    Ks = K.copy()
    Ks[0, 0] /= s
    Ks[1, 1] /= s
    Ks[0, 2] = (K[0, 2] + 0.5) / s
    Ks[1, 2] = (K[1, 2] + 0.5) / s
    return Ks


def project(xyz: torch.Tensor, w2c: torch.Tensor, K: torch.Tensor):
    pc = xyz @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    uv = pc[:, :2] / z.clamp_min(1e-6)[:, None] * torch.stack([K[0, 0], K[1, 1]]) + torch.stack([K[0, 2], K[1, 2]])
    return uv, z


def visible_points(uv, z, W, H, rel_tol=0.10, abs_tol=0.3, window=3, near=0.5):
    """Which projected points are the front surface: a sparse z-buffer, eroded
    with a min filter so points seen through gaps between nearer points are
    rejected too. Returns a mask and the pixel index of every point."""
    inside = (z > near) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    pix = uv[:, 1].long().clamp(0, H - 1) * W + uv[:, 0].long().clamp(0, W - 1)
    zbuf = torch.full((H * W,), float("inf"), device=z.device)
    zbuf.scatter_reduce_(0, pix[inside], z[inside], reduce="amin")
    zmin = -F.max_pool2d(-zbuf.view(1, 1, H, W), window, stride=1, padding=window // 2).view(-1)
    return inside & (z <= zmin[pix] * (1 + rel_tol) + abs_tol), pix


def main():
    args = parse_args()
    seq_dir = os.path.join(args.pandaset_dir, args.seq)
    out = args.out_dir or os.path.join(ROOT, "data", f"pandaset_{args.seq}")
    os.makedirs(out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s = args.downscale
    n_frames = len(json.load(open(os.path.join(seq_dir, "lidar", "timestamps.json"))))

    cams = {c: ps.load_camera(seq_dir, c) for c in args.cameras}
    lidar_poses, lidar_times = ps.load_lidar_poses(seq_dir)
    offset = cams[args.cameras[0]]["c2w"][:, :3, 3].mean(axis=0)
    per_frame = [ps.load_cuboids(seq_dir, f) for f in range(n_frames)]
    for boxes in per_frame:
        for b in boxes:
            b["center"] = (np.asarray(b["center"]) - offset).tolist()
    tracks = ps.track_boxes(per_frame, lidar_times)
    dynamic = {u: t for u, t in tracks.items() if t["dynamic"]}
    print(f"sequence {args.seq}: {n_frames} frames, {len(tracks)} objects, {len(dynamic)} masked "
          f"(moving or people)")

    # --- images, masks, cameras -------------------------------------------
    half = args.test_every // 2
    bottom_rows = {c: int(n) for c, n in (b.split(":") for b in args.bottom_mask)}
    images = []
    for cam in args.cameras:
        os.makedirs(os.path.join(out, "images", cam), exist_ok=True)
        os.makedirs(os.path.join(out, "masks", cam), exist_ok=True)
        Ks = scale_intrinsics(cams[cam]["K"], s)
        for f in tqdm(range(n_frames), desc=f"images {cam}", mininterval=5):
            img = Image.open(cams[cam]["image_path"](f)).convert("RGB")
            W, H = img.width // s, img.height // s
            img.resize((W, H), Image.LANCZOS).save(os.path.join(out, "images", cam, f"{f:02d}.png"))
            c2w = cams[cam]["c2w"][f].copy()
            c2w[:3, 3] -= offset
            w2c = np.linalg.inv(c2w)
            t_cam = float(cams[cam]["timestamps"][f])
            mask = Image.new("L", (W, H), 0)
            draw = ImageDraw.Draw(mask)
            for tr in dynamic.values():
                box = ps.box_at_time(tr, t_cam)
                if box is None:
                    continue
                # half a meter of margin: annotation error, timing, and shadows at the wheels
                poly = ps.project_polygon(ps.box_corners(box, grow=(0.3, 0.5, 0.2)), w2c, Ks)
                if poly is not None:
                    draw.polygon([tuple(q - 0.5) for q in poly], fill=255)
            if cam in bottom_rows:                       # the ego car itself (the back camera's hood)
                draw.rectangle([0, H - bottom_rows[cam], W, H], fill=255)
            mask.save(os.path.join(out, "masks", cam, f"{f:02d}.png"))
            images.append({"camera": cam, "frame": f, "split": "test" if f % args.test_every == half else "train",
                           "file": f"images/{cam}/{f:02d}.png", "mask": f"masks/{cam}/{f:02d}.png",
                           "K": Ks.tolist(), "c2w": c2w.tolist(), "timestamp": t_cam, "width": W, "height": H})
    W, H = images[0]["width"], images[0]["height"]

    # --- LiDAR: static points per sweep, then one merged map ---------------
    sweeps = []
    for f in tqdm(range(n_frames), desc="lidar", mininterval=5):
        sw = ps.load_lidar(seq_dir, f)
        xyz = sw["xyz"] - offset
        sensor = lidar_poses[f][:3, 3] - offset
        dist = np.linalg.norm(xyz - sensor, axis=1)
        keep = (np.linalg.norm(xyz[:, :2] - sensor[:2], axis=1) > args.ego_radius) & (dist < args.max_range)
        for tr in dynamic.values():
            box = ps.box_at_time(tr, float(lidar_times[f]))
            if box is not None:
                keep &= ~ps.points_in_box(xyz, box, grow=0.3)
        sweeps.append(xyz[keep].astype(np.float32))
    all_pts = np.concatenate(sweeps)
    keys = np.floor(all_pts / args.voxel).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    merged = np.zeros((len(counts), 3))
    np.add.at(merged, inv, all_pts)
    merged = (merged / counts[:, None]).astype(np.float32)
    print(f"LiDAR: {len(all_pts):,} static returns from {n_frames} sweeps -> {len(merged):,} voxels of "
          f"{args.voxel * 100:.0f} cm")

    # --- color the map from the training images ----------------------------
    # Each point takes its color from the closest training view that sees it
    # (z-buffer test); points no training camera sees are dropped.
    pts = torch.from_numpy(merged).to(device)
    best_z = torch.full((len(pts),), float("inf"), device=device)
    rgb = torch.zeros(len(pts), 3, dtype=torch.uint8, device=device)
    n_seen = torch.zeros(len(pts), dtype=torch.int32, device=device)
    depth_store = {}
    for im in tqdm([i for i in images if i["split"] == "train"], desc="color map + depth", mininterval=5):
        K = torch.tensor(im["K"], dtype=torch.float32, device=device)
        w2c = torch.tensor(np.linalg.inv(np.asarray(im["c2w"])), dtype=torch.float32, device=device)
        img = torch.from_numpy(np.array(Image.open(os.path.join(out, im["file"])))).to(device).view(-1, 3)
        msk = torch.from_numpy(np.array(Image.open(os.path.join(out, im["mask"])))).to(device).view(-1)
        uv, z = project(pts, w2c, K)
        vis, pix = visible_points(uv, z, W, H)
        vis &= msk[pix] == 0
        n_seen += vis.int()
        closer = vis & (z < best_z)
        best_z[closer] = z[closer]
        rgb[closer] = img[pix[closer]]
        # LiDAR depth for this image: this frame's sweep only (no smearing across time)
        sw = torch.from_numpy(sweeps[im["frame"]]).to(device)
        uv, z = project(sw, w2c, K)
        vis, pix = visible_points(uv, z, W, H)
        vis &= msk[pix] == 0
        depth_store[f"{im['camera']}/{im['frame']:02d}"] = torch.cat([uv[vis], z[vis, None]], 1).cpu().numpy()
    seen = (n_seen > 0).cpu().numpy()
    np.savez_compressed(os.path.join(out, "points.npz"), xyz=merged[seen], rgb=rgb.cpu().numpy()[seen],
                        n_views=n_seen.cpu().numpy()[seen])
    np.savez_compressed(os.path.join(out, "lidar_depth.npz"), **depth_store)
    n_depth = np.mean([len(v) for v in depth_store.values()])
    print(f"kept {seen.sum():,} map points seen by a training camera; {n_depth:,.0f} LiDAR depth samples per "
          f"training image on average")

    # --- scene.json ----------------------------------------------------------
    lidar_c2w = lidar_poses.copy()
    lidar_c2w[:, :3, 3] -= offset
    scene = {
        "sequence": args.seq, "cameras": args.cameras, "downscale": s, "width": W, "height": H,
        "world_offset": offset.tolist(), "test_every": args.test_every, "n_frames": n_frames,
        "lidar_positions": lidar_c2w[:, :3, 3].tolist(), "lidar_timestamps": lidar_times.tolist(),
        "images": images,
        "tracks": {u: {k: v for k, v in t.items()} for u, t in tracks.items()},
    }
    with open(os.path.join(out, "scene.json"), "w") as fh:
        json.dump(scene, fh)
    n_train = sum(i["split"] == "train" for i in images)
    print(f"wrote {out}: {n_train} train / {len(images) - n_train} test images at {W}x{H}")
    print("PREPARE_DONE")


if __name__ == "__main__":
    main()
