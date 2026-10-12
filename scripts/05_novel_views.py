# SPDX-License-Identifier: MIT
"""Step 5: drive the car where it didn't drive: lane changes.

A driving simulator moves the whole camera rig with the car, so every camera is
shifted along the car's sideways axis (splat/cameras.py: shift_rig) and keeps
its own orientation and color correction.

  figures/fig_novel_views.png    one moment, the three front cameras: photo, splat on
                                 the recorded path, splat 3 m to the left (the next lane)
  figures/fig_lane_change.gif    the drive replayed with a lane change, every camera of
                                 the rig the model was trained on (six or three)
  figures/fig_moving_removed.png held-out frames with moving traffic: photo, the mask,
                                 and the splat (static scene only)
  outputs/novel/lane_change.mp4  full-frame-rate version

With --only-gif --gif-name fig_surround_lane_change.gif and a six-camera model,
only the lane-change animation is made, with all six cameras.

All images are privacy-blurred.

    python scripts/05_novel_views.py --data-dir data/pandaset_028_6cam --ckpt outputs/study/6cam_reg/ckpt.pt
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from splat import cameras as cm, figs, privacy, scene as sc  # noqa: E402
from splat.figs import label, to_u8  # noqa: E402
from splat.model import load_model, render_view  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONT3 = ("front_left_camera", "front_camera", "front_right_camera")
RIG6 = (("front_left_camera", "front_camera", "front_right_camera"), ("left_camera", "back_camera", "right_camera"))
NAME = {"front_left_camera": "front-left", "front_camera": "front", "front_right_camera": "front-right",
        "left_camera": "left", "right_camera": "right", "back_camera": "back"}


class Renderer:
    """Renders the trained model from any camera pose, in a given camera's colors,
    with license plates blurred."""

    def __init__(self, ckpt, scene, device):
        self.params, self.sky, self.color, self.meta = load_model(ckpt, device)
        self.static_boxes = privacy.boxes_at(scene["tracks"], 0.0, static_only=True)
        self.device = device

    @torch.no_grad()
    def __call__(self, c2w, K, W, H, camera="front_camera", blur=True):
        w2c = np.linalg.inv(c2w)
        out = render_view(self.params, self.sky, self.color,
                          torch.tensor(w2c, dtype=torch.float32, device=self.device),
                          torch.tensor(K, dtype=torch.float32, device=self.device), W, H,
                          self.meta["sh_degree"], cam=self.meta["cameras"].index(camera))
        rgb = to_u8(out["rgb"].clamp(0, 1))
        return privacy.blur(rgb, self.static_boxes, w2c, K) if blur else rgb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=os.path.join(ROOT, "data", "pandaset_028"))
    p.add_argument("--ckpt", default=os.path.join(ROOT, "outputs", "train", "ckpt.pt"))
    p.add_argument("--out-dir", default=os.path.join(ROOT, "outputs", "novel"))
    p.add_argument("--fig-dir", default=os.path.join(ROOT, "figures"))
    p.add_argument("--frame", type=int, default=30, help="moment used for the camera grid")
    p.add_argument("--shift", type=float, default=3.0, help="meters to the left (one lane) for the grid")
    p.add_argument("--gif-name", default="fig_lane_change.gif")
    p.add_argument("--gif-width", type=int, default=1080)
    p.add_argument("--gif-step", type=int, default=2, help="keep every Nth frame in the GIF (the MP4 keeps all)")
    p.add_argument("--only-gif", action="store_true", help="skip the camera grid and the moving-objects figure")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    device = torch.device("cuda")
    scene = sc.load_scene(args.data_dir)
    W, H = scene["width"], scene["height"]
    render = Renderer(args.ckpt, scene, device)
    rec = {(r["camera"], r["frame"]): r for r in scene["images"]}
    front = {f: np.asarray(rec[("front_camera", f)]["c2w"]) for f in range(scene["n_frames"])}

    # 1) One moment, three front cameras: photo | splat on the path | splat one lane over.
    rows = [[], [], []] if not args.only_gif else None
    for cam in (FRONT3 if rows is not None else ()):
        r = rec[(cam, args.frame)]
        c2w, K = np.asarray(r["c2w"]), np.asarray(r["K"])
        photo = np.asarray(Image.open(os.path.join(args.data_dir, r["file"])).convert("RGB"))
        photo = privacy.blur(photo, privacy.boxes_at(scene["tracks"], r["timestamp"]), np.asarray(r["w2c"]), K)
        tag = "held out" if r["split"] == "test" else "training view"
        rows[0].append(label(photo, f"photo, {NAME[cam]} ({tag})", 14))
        rows[1].append(label(render(c2w, K, W, H, cam), f"splat, {NAME[cam]}, recorded path", 14))
        shifted = cm.shift_rig(c2w, front[args.frame], right=-args.shift)
        rows[2].append(label(render(shifted, K, W, H, cam), f"splat, {NAME[cam]}, {args.shift:g} m left", 14))
    if rows is not None:
        Image.fromarray(np.concatenate([np.concatenate(r, 1) for r in rows], 0)).save(
            os.path.join(args.fig_dir, "fig_novel_views.png"))

    # 2) Lane change along the drive, every camera of the rig moving together.
    layout = RIG6 if all(c in render.meta["cameras"] for c in RIG6[1]) else (FRONT3,)
    poses = {c: cm.interpolate([rec[(c, f)]["c2w"] for f in range(scene["n_frames"])], 160)
             for row in layout for c in row}
    fronts = cm.interpolate([front[f] for f in range(scene["n_frames"])], 160)
    mp4 = figs.mp4_writer(os.path.join(args.out_dir, os.path.splitext(args.gif_name)[0].replace("fig_", "") + ".mp4"),
                          fps=20)
    gif = []
    for k in range(160):
        t = k / 159
        # drift one lane to the left, hold, come back
        lateral = -args.shift * (cm.smoothstep((t - 0.15) / 0.25) - cm.smoothstep((t - 0.6) / 0.25))
        grid = []
        for row in layout:
            tiles = []
            for c in row:
                K = np.asarray(rec[(c, 0)]["K"])
                tiles.append(label(render(cm.shift_rig(poses[c][k], fronts[k], right=lateral), K, W, H, c),
                                   NAME[c], 13))
            grid.append(np.concatenate(tiles, 1))
        frame = label(np.concatenate(grid, 0), f"lane change: {-lateral:.1f} m to the left", 16, right=True)
        mp4.append_data(frame)
        gif.append(frame)
    mp4.close()
    figs.save_gif(os.path.join(args.fig_dir, args.gif_name), gif[::args.gif_step], args.gif_width, 0.05 * args.gif_step)
    if args.only_gif:
        print("NOVEL_DONE")
        return

    # 3) Moving traffic removed: photo | mask | splat.
    rows = []
    for cam_name, f in (("front_left_camera", 60), ("front_camera", 76)):
        r = rec[(cam_name, f)]
        photo = np.asarray(Image.open(os.path.join(args.data_dir, r["file"])).convert("RGB"))
        mask = np.asarray(Image.open(os.path.join(args.data_dir, r["mask"]))) > 127
        boxes = privacy.boxes_at(scene["tracks"], r["timestamp"])
        w2c, K = np.asarray(r["w2c"]), np.asarray(r["K"])
        photo_b = privacy.blur(photo, boxes, w2c, K)
        shown = photo_b.astype(np.float32)
        shown[mask] = shown[mask] * 0.45 + np.array([239, 35, 60]) * 0.55
        splat = render(np.asarray(r["c2w"]), K, W, H, cam_name, blur=False)
        splat = privacy.blur(splat, boxes, w2c, K)
        split = "held out" if r["split"] == "test" else "training view"
        rows.append(np.concatenate([label(photo_b, f"photo ({NAME[cam_name]}, frame {f}, {split})", 14),
                                    label(shown.astype(np.uint8), "masked: moving objects and people", 14),
                                    label(splat, "splat: the static street", 14)], axis=1))
    Image.fromarray(np.concatenate(rows, 0)).save(os.path.join(args.fig_dir, "fig_moving_removed.png"))
    print("NOVEL_DONE")


if __name__ == "__main__":
    main()
