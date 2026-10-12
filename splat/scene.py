# SPDX-License-Identifier: MIT
"""Load a scene written by scripts/01_prepare_scene.py."""
import json
import os

import numpy as np
import torch
from PIL import Image


def load_scene(data_dir: str) -> dict:
    with open(os.path.join(data_dir, "scene.json")) as f:
        scene = json.load(f)
    scene["data_dir"] = data_dir
    for im in scene["images"]:
        im["w2c"] = np.linalg.inv(np.asarray(im["c2w"])).tolist()
    return scene


def load_views(scene: dict, split: str, with_depth: bool = False) -> dict:
    """All images of one split, kept on the CPU as uint8; the training loop
    moves one to the GPU per step.

    Returns images (N,H,W,3) uint8, masks (N,H,W) bool (True = ignore),
    w2c (N,4,4), K (N,3,3), camera index (N,) into scene['cameras'], the
    image records, and optionally LiDAR depth samples per image."""
    d = scene["data_dir"]
    recs = [im for im in scene["images"] if im["split"] == split]
    images = np.stack([np.asarray(Image.open(os.path.join(d, im["file"])).convert("RGB")) for im in recs])
    masks = np.stack([np.asarray(Image.open(os.path.join(d, im["mask"]))) > 127 for im in recs])
    out = {
        "images": torch.from_numpy(images),
        "masks": torch.from_numpy(masks),
        "w2c": torch.tensor(np.stack([im["w2c"] for im in recs]), dtype=torch.float32),
        "K": torch.tensor(np.stack([im["K"] for im in recs]), dtype=torch.float32),
        "cam_index": torch.tensor([scene["cameras"].index(im["camera"]) for im in recs]),
        "records": recs,
        "width": scene["width"], "height": scene["height"],
    }
    if with_depth:
        z = np.load(os.path.join(d, "lidar_depth.npz"))
        out["lidar"] = [torch.from_numpy(z[f"{im['camera']}/{im['frame']:02d}"]) for im in recs]
    return out


def load_points(scene: dict):
    z = np.load(os.path.join(scene["data_dir"], "points.npz"))
    return z["xyz"], z["rgb"]
