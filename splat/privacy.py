# SPDX-License-Identifier: MIT
"""Blur license plates and people in images meant for publication.

PandaSet frames are not anonymized, and a Gaussian splat reproduces whatever
it was trained on, so both camera frames and renders go through this before
they land in figures/. It uses the 3D cuboid annotations, no detector:
- people: their whole box;
- vehicles: a plate zone (0.6 m wide, 0.2 to 1.0 m above the ground) on
  whichever end faces the camera, if a plate there would be at least 6 pixels
  wide (smaller plates can't be read).
Metrics are always computed on the unblurred images.
"""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import pandaset as ps


def _plate_zone(box: dict, end: int) -> np.ndarray:
    """The 8 corners of a thin slab around one end face (end = +1 front, -1 rear)."""
    w, l, h = box["size"]
    hw = min(0.30, w / 2)
    y0, y1 = sorted((end * (l / 2 - 0.20), end * (l / 2 + 0.15)))
    z0, z1 = -h / 2 + 0.2, min(-h / 2 + 1.0, h / 2)
    local = np.array([[x, y, z] for x in (-hw, hw) for y in (y0, y1) for z in (z0, z1)])
    c, s = math.cos(box["yaw"]), math.sin(box["yaw"])
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return local @ rot.T + np.asarray(box["center"])


PLATE_WIDTH_M = 0.3
MIN_PLATE_PX = 6.0


def blur_mask(boxes, w2c: np.ndarray, K: np.ndarray, width: int, height: int) -> Image.Image:
    """L-mode mask, 255 where the image should be blurred."""
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    cam = np.linalg.inv(w2c)[:3, 3]
    for b in boxes:
        polys = []
        if b["label"] in ps.ALWAYS_MASK:
            polys.append(ps.project_polygon(ps.box_corners(b, grow=0.25), w2c, K))
        elif b["label"] in ps.VEHICLES:
            c, s = math.cos(b["yaw"]), math.sin(b["yaw"])
            axis = np.array([-s, c, 0.0])               # the box's length direction (local +y)
            for end in (1, -1):
                face = np.asarray(b["center"]) + end * axis * b["size"][1] / 2
                depth = (w2c[:3, :3] @ face + w2c[:3, 3])[2]
                if depth <= 0 or K[0, 0] * PLATE_WIDTH_M / depth < MIN_PLATE_PX:
                    continue                            # behind the camera, or too small to read
                if np.dot(end * axis, cam - face) > 0:  # this end faces the camera
                    polys.append(ps.project_polygon(_plate_zone(b, end), w2c, K))
        for poly in polys:
            if poly is not None and len(poly) >= 3:
                draw.polygon([tuple(q - 0.5) for q in poly], fill=255)
    return mask


def blur(img: np.ndarray, boxes, w2c: np.ndarray, K: np.ndarray, strength: float = 6.0) -> np.ndarray:
    """img (H, W, 3) uint8 at the resolution K describes."""
    h, w = img.shape[:2]
    mask = blur_mask(boxes, w2c, K, w, h)
    if not mask.getbbox():
        return img
    pil = Image.fromarray(img)
    blurred = pil.filter(ImageFilter.GaussianBlur(strength))
    soft = mask.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(1.0))
    return np.asarray(Image.composite(blurred, pil, soft))


def boxes_at(tracks: dict, t: float, static_only: bool = False) -> list:
    """Every annotated object's box at time t (static objects at their median pose)."""
    out = []
    for tr in tracks.values():
        if tr["dynamic"]:
            if static_only:
                continue
            b = ps.box_at_time(tr, t)
        else:
            b = {"label": tr["label"], "center": list(np.median(tr["centers"], axis=0)),
                 "size": list(np.median(tr["sizes"], axis=0)), "yaw": float(np.median(np.unwrap(tr["yaws"])))}
        if b is not None:
            out.append(b)
    return out
