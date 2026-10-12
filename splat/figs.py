# SPDX-License-Identifier: MIT
"""Small helpers for figures: labels, error heatmaps, video writers."""
import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def to_u8(t: torch.Tensor) -> np.ndarray:
    return (t.clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)


def label(img: np.ndarray, text: str, size: int = 22, right: bool = False) -> np.ndarray:
    """Black box with white text in the top-left (or top-right) corner."""
    im = Image.fromarray(np.ascontiguousarray(img))
    d = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=size)
    x0, y0, x1, y1 = d.textbbox((8, 6), text, font=font)
    dx = im.width - (x1 + 8) if right else 0
    d.rectangle([dx, 0, dx + x1 + 8, y1 + 6], fill=(0, 0, 0))
    d.text((dx + 8, 6), text, fill=(255, 255, 255), font=font)
    return np.asarray(im)


def heatmap(pred: torch.Tensor, gt: torch.Tensor, vmax: float = 0.1) -> np.ndarray:
    """Per-pixel mean absolute error, magma colormap (black = 0, bright = vmax or more)."""
    err = ((pred - gt).abs().mean(-1) / vmax).clamp(0, 1).cpu().numpy()
    return (matplotlib.colormaps["magma"](err)[..., :3] * 255).astype(np.uint8)


def mp4_writer(path: str, fps: int = 10):
    # H.264 (yuv420p) needs even dimensions; frames here are 270 or 540 px tall
    return imageio.get_writer(path, fps=fps, quality=8, macro_block_size=2)


def save_gif(path: str, frames, width: int, frame_seconds: float) -> None:
    out = []
    for f in frames:
        h = round(f.shape[0] * width / f.shape[1])
        out.append(np.asarray(Image.fromarray(f).resize((width, h), Image.LANCZOS)))
    imageio.mimsave(path, out, duration=frame_seconds, loop=0)
