# SPDX-License-Identifier: MIT
"""The full scene model: Gaussians + sky + per-camera color, rendered for one
camera. Shared by training, evaluation and the visualization scripts."""
import torch

from . import gaussians as gs, render as R


def _render_gsplat(m, q, s, o, sh, w2c, K, width, height, sh_degree, with_depth):
    """Reference backend: gsplat's CUDA rasterizer, same inputs and outputs as
    render.render (used only to compare training speed; gsplat is not a
    dependency of this project)."""
    from gsplat import rasterization
    img, alpha, info = rasterization(m, q, s, o, sh, w2c[None], K[None], width, height, sh_degree=sh_degree,
                                     packed=False, render_mode="RGB+D" if with_depth else "RGB")
    radii = info["radii"][0]
    vis = (radii > 0).all(-1) if radii.dim() == 2 else radii > 0
    out = {"rgb": img[0, ..., :3], "alpha": alpha[0, ..., 0],
           "proj": {"idx": vis.nonzero(as_tuple=True)[0], "means2d": info["means2d"]}}
    if with_depth:
        out["depth"] = img[0, ..., 3]                      # sum_i z_i a_i T_i, like render.render
    return out


def render_view(params: dict, sky, color, w2c: torch.Tensor, K: torch.Tensor, width: int, height: int,
                sh_degree: int, cam: int | None = None, with_depth: bool = False, backend: str = "torch",
                **kw) -> dict:
    """rgb = Gaussians composited over the sky, then (if `cam` is given) mapped
    into that camera's colors. Returns rgb (H,W,3), alpha (H,W), and depth
    (H,W) if with_depth, plus 'proj' from the projection."""
    m, q, s, o, sh = gs.activated(params)
    if backend == "gsplat":
        out = _render_gsplat(m, q, s, o, sh, w2c, K, width, height, sh_degree, with_depth)
    else:
        out = R.render(m, q, s, o, sh, w2c, K, width, height, sh_degree, with_depth=with_depth, **kw)
    if sky is not None:
        out["sky"] = sky(torch.linalg.inv(w2c), K, width, height)
        out["rgb"] = out["rgb"] + (1.0 - out["alpha"])[..., None] * out["sky"]
    if color is not None and cam is not None:
        out["rgb"] = color(out["rgb"], cam)
    return out


def load_model(path: str, device="cuda"):
    """Checkpoint from scripts/03_train.py -> (params, sky, color, meta)."""
    params, extra, meta = gs.load(path, device)
    sky = gs.Sky([0.5, 0.5, 0.5], *meta["sky_size"]).to(device)
    sky.load_state_dict(extra["sky"])
    color = gs.ColorCorrection(len(meta["cameras"])).to(device)
    color.load_state_dict(extra["color"])
    return params, sky, color, meta
