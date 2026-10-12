# SPDX-License-Identifier: MIT
"""Step 2: check the prepared scene before training.

  figures/fig_lidar_check.png  LiDAR depth projected into the three cameras,
                               moving objects and people (masked) tinted red
  figures/fig_scene_map.png    top view: colored LiDAR map, camera path,
                               held-out frames, masked object tracks

    python scripts/02_check_scene.py
"""
import argparse
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from splat import pandaset as ps, privacy, scene as sc  # noqa: E402
from splat.figs import label  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAM_ORDER = ("front_left_camera", "front_camera", "front_right_camera")
CAM_NAME = {"front_left_camera": "front-left", "front_camera": "front", "front_right_camera": "front-right"}


def depth_overlay(img, uvz, mask, max_depth=60.0):
    out = img.astype(np.float32)
    out[mask] = out[mask] * 0.5 + np.array([255, 0, 0]) * 0.5
    order = np.argsort(-uvz[:, 2])                         # far first, near points on top
    u, v, z = uvz[order, 0].astype(int), uvz[order, 1].astype(int), uvz[order, 2]
    col = matplotlib.colormaps["turbo"](np.clip(z / max_depth, 0, 1))[:, :3] * 255
    out[v, u] = col
    return out.clip(0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=os.path.join(ROOT, "data", "pandaset_028"))
    p.add_argument("--fig-dir", default=os.path.join(ROOT, "figures"))
    p.add_argument("--frames", type=int, nargs="+", default=[10, 50])
    args = p.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)
    scene = sc.load_scene(args.data_dir)
    depth = np.load(os.path.join(args.data_dir, "lidar_depth.npz"))
    by_key = {(im["camera"], im["frame"]): im for im in scene["images"]}

    # 1) LiDAR over the cameras
    rows = []
    for f in args.frames:
        tiles = []
        for cam in CAM_ORDER:
            im = by_key[(cam, f)]
            img = np.asarray(Image.open(os.path.join(args.data_dir, im["file"])).convert("RGB"))
            mask = np.asarray(Image.open(os.path.join(args.data_dir, im["mask"]))) > 127
            boxes = privacy.boxes_at(scene["tracks"], im["timestamp"])
            img = privacy.blur(img, boxes, np.asarray(im["w2c"]), np.asarray(im["K"]))
            key = f"{cam}/{f:02d}"
            uvz = depth[key] if key in depth else np.zeros((0, 3))
            tiles.append(label(depth_overlay(img, uvz, mask), f"{CAM_NAME[cam]}, frame {f}", size=16))
        rows.append(np.concatenate(tiles, axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(os.path.join(args.fig_dir, "fig_lidar_check.png"))

    # 2) Top view
    xyz, rgb = sc.load_points(scene)
    rng = np.random.default_rng(0)
    sub = rng.choice(len(xyz), min(len(xyz), 400_000), replace=False)
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    ax.set_facecolor("black")
    ax.scatter(xyz[sub, 0], xyz[sub, 1], c=rgb[sub] / 255.0, s=0.15, linewidths=0)
    for cam, color in zip(CAM_ORDER, ("#4cc9f0", "#ffffff", "#f72585")):
        ims = sorted([im for im in scene["images"] if im["camera"] == cam], key=lambda r: r["frame"])
        c = np.array([np.asarray(im["c2w"])[:3, 3] for im in ims])
        if cam == "front_camera":
            ax.plot(c[:, 0], c[:, 1], "-", color=color, lw=1.2, label="camera path")
            test = np.array([im["split"] == "test" for im in ims])
            ax.scatter(c[test, 0], c[test, 1], s=28, facecolors="none", edgecolors="#ffd166", lw=1.2,
                       label="held-out frames", zorder=5)
    for tr in scene["tracks"].values():
        if tr["dynamic"]:
            cen = np.asarray(tr["centers"])
            ax.plot(cen[:, 0], cen[:, 1], "-", color="#ef233c", lw=1.5, alpha=0.9)
    ax.plot([], [], "-", color="#ef233c", lw=1.5, label="masked objects (moving, people)")
    for tr in scene["tracks"].values():
        if not tr["dynamic"] and tr["label"] in ps.VEHICLES:
            b = {"center": np.median(tr["centers"], 0), "size": np.median(tr["sizes"], 0),
                 "yaw": float(np.median(tr["yaws"]))}
            c4 = ps.box_corners(b)[[0, 2, 6, 4, 0], :2]
            ax.plot(c4[:, 0], c4[:, 1], "-", color="#06d6a0", lw=0.6, alpha=0.8)
    ax.plot([], [], "-", color="#06d6a0", lw=0.8, label="parked vehicles (kept)")
    pos = np.array([np.asarray(im["c2w"])[:3, 3] for im in scene["images"]])
    pad = 35
    ax.set_xlim(pos[:, 0].min() - pad, pos[:, 0].max() + pad)
    ax.set_ylim(pos[:, 1].min() - pad, pos[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper left", fontsize=8, facecolor="#222222", labelcolor="white", framealpha=0.8)
    ax.set_title(f"PandaSet sequence {scene['sequence']}: LiDAR map ({len(xyz):,} points, colored from the "
                 f"cameras) and the drive", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(args.fig_dir, "fig_scene_map.png"))
    print("CHECK_DONE")


if __name__ == "__main__":
    main()
