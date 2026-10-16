# SPDX-License-Identifier: MIT
"""Diagram of the rasterizer's three steps after projection: bin Gaussians
into tiles, sort each tile's list front to back and take the next chunk of
every list, composite the chunk as one tensor.

    python scripts/diagram_compositing.py      -> figures/fig_compositing_diagram.png
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#d9d8d3", "#fcfcfb"
BLUE, BLUE_LIGHT, ORANGE, GRAY = "#2a78d6", "#cde2fb", "#eb6834", "#e4e3df"


def panel_bin(ax):
    tx, ty = 10, 6
    for i in range(tx + 1):
        ax.plot([i, i], [0, ty], color=GRID, lw=1)
    for j in range(ty + 1):
        ax.plot([0, tx], [j, j], color=GRID, lw=1)
    # one Gaussian with its footprint box and the tiles it touches
    cx, cy, w, h, ang = 3.6, 3.3, 3.0, 1.2, 25
    t = np.radians(ang)
    hx = 0.5 * np.sqrt((w * np.cos(t)) ** 2 + (h * np.sin(t)) ** 2)
    hy = 0.5 * np.sqrt((w * np.sin(t)) ** 2 + (h * np.cos(t)) ** 2)
    cols = range(int(np.floor(cx - hx)), int(np.ceil(cx + hx)))
    rows = range(int(np.floor(cy - hy)), int(np.ceil(cy + hy)))
    for i in cols:
        for j in rows:
            ax.add_patch(Rectangle((i, j), 1, 1, color=BLUE_LIGHT, zorder=0))
    ax.add_patch(Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy, fill=False, ec=BLUE, lw=1.2, ls="--", zorder=3))
    ax.add_patch(Ellipse((cx, cy), w, h, angle=ang, color=BLUE, alpha=0.85, zorder=4))
    others = [(7.4, 4.4, 1.6, 0.9, -30, ORANGE), (8.0, 1.5, 0.8, 0.5, 10, "#1baf7a"),
              (6.2, 2.4, 1.0, 0.7, 60, "#eda100"), (1.4, 1.0, 0.6, 0.4, 0, "#e87ba4")]
    for x, y, ew, eh, a, c in others:
        ax.add_patch(Ellipse((x, y), ew, eh, angle=a, color=c, alpha=0.85, zorder=4))
    ax.text(0, ty + 0.45, f"the blue Gaussian's box (dashed) overlaps {len(cols)} x {len(rows)} tiles:\n"
            f"{len(cols) * len(rows)} (Gaussian, tile) entries in the list", ha="left", va="top", fontsize=8,
            color=INK2)
    ax.set_xlim(-0.2, tx + 0.2)
    ax.set_ylim(ty + 1.6, -0.3)
    ax.set_aspect("equal")
    ax.axis("off")


def panel_chunk(ax):
    lengths = [14, 9, 22, 5, 17, 12, 20, 3]
    done = {6}                                     # this tile turned opaque early
    off, B = 6, 5
    for r, n in enumerate(lengths):
        y = len(lengths) - 1 - r
        ended = n <= off
        opaque = r in done
        for k in range(n):
            if k < off:
                col = GRAY
            elif ended or opaque:
                col = "#f0efec"
            elif k < off + B:
                col = BLUE
            else:
                col = BLUE_LIGHT
            ax.add_patch(Rectangle((k, y + 0.12), 0.9, 0.76, color=col, lw=0))
        note = "list ended" if ended else ("opaque: T < 1e-4, dropped" if opaque else "")
        ax.text(-0.6, y + 0.5, f"tile {r}", ha="right", va="center", fontsize=8, color=INK2)
        if note:
            ax.text(max(n, off) + 0.4, y + 0.5, note, ha="left", va="center", fontsize=7.5, color=MUTED)
    ax.add_patch(Rectangle((off - 0.08, -0.1), B + 0.06, len(lengths) + 0.2, fill=False, ec=INK, lw=1.4))
    ax.text(off + B / 2, len(lengths) + 0.35, "this chunk: the next B Gaussians\nof every active tile", ha="center",
            va="bottom", fontsize=8, color=INK)
    ax.annotate("", xy=(23.5, -0.75), xytext=(0, -0.75),
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
    ax.text(0, -1.0, "front to back (sorted by depth)", ha="left", va="top", fontsize=8, color=INK2)
    ax.set_xlim(-3.6, 25.5)
    ax.set_ylim(-2.0, len(lengths) + 1.6)
    ax.axis("off")


def box(ax, x, y, w, h, d, face, edge=INK2):
    """An isometric box: front face w x h at (x, y), depth d up and to the right."""
    dx, dy = d * 0.55, d * 0.4
    ax.add_patch(Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], closed=True, fc=face, ec=edge, lw=1))
    ax.add_patch(Polygon([(x, y + h), (x + w, y + h), (x + w + dx, y + h + dy), (x + dx, y + h + dy)], closed=True,
                         fc=face, ec=edge, lw=1, alpha=0.75))
    ax.add_patch(Polygon([(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)], closed=True,
                         fc=face, ec=edge, lw=1, alpha=0.6))


def panel_tensor(ax):
    box(ax, 0.6, 1.5, 3.2, 3.2, 2.4, BLUE_LIGHT)
    ax.text(2.2, 1.25, "A active tiles", ha="center", va="top", fontsize=8, color=INK2)
    ax.text(0.45, 3.1, "B Gaussians\n(the chunk)", ha="right", va="center", fontsize=8, color=INK2)
    ax.text(4.3, 6.0, "64 px per tile (depth)", ha="center", va="bottom", fontsize=8, color=INK2)
    steps = [
        ("alpha = exp( coeffs @ [1, x, y, x², xy, y²] )", "one matmul: (A, B, 6) @ (6, 64)"),
        ("T = T_in · exp( cumsum log(1 - alpha) )", "transmittance along the chunk"),
        ("color += sum over B of  f · alpha · T", "one batched matmul"),
        ("T_out = T at the end of the chunk", "carried into the next chunk"),
    ]
    y = 6.3
    for main, sub in steps:
        ax.text(6.6, y, main, ha="left", va="center", fontsize=8.5, color=INK, family="DejaVu Sans Mono")
        ax.text(6.6, y - 0.42, sub, ha="left", va="center", fontsize=7.5, color=MUTED)
        y -= 1.25
    arrow = FancyArrowPatch((10.4, 1.7), (4.6, 0.7), connectionstyle="arc3,rad=-0.35", arrowstyle="->",
                            color=INK2, lw=1, mutation_scale=10)
    ax.add_patch(arrow)
    ax.text(8.6, 0.55, "next chunk", ha="center", va="top", fontsize=8, color=INK2)
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 7.6)
    ax.axis("off")


def main():
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=150, facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.0, 1.25, 1.35]})
    for ax in axes:
        ax.set_facecolor(SURFACE)
    panel_bin(axes[0])
    panel_chunk(axes[1])
    panel_tensor(axes[2])
    fig.tight_layout(w_pad=2.5, rect=(0, 0, 1, 0.86))
    titles = ["1. Bin: each footprint box lists\nthe 8x8-pixel tiles it overlaps",
              "2. Sort each tile's list by depth,\nthen walk all lists one chunk at a time",
              "3. Composite the chunk as one tensor\n(with a hand-written backward pass)"]
    for ax, t in zip(axes, titles):
        fig.text(ax.get_position().x0, 0.97, t, ha="left", va="top", fontsize=10.5, color=INK)
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    fig.savefig(os.path.join(ROOT, "figures", "fig_compositing_diagram.png"), facecolor=SURFACE)
    print("DIAGRAM_DONE")


if __name__ == "__main__":
    main()
