# SPDX-License-Identifier: MIT
"""Gaussian parameters, initialization from the LiDAR map, the sky, per-camera
color correction, and adaptive density control (the 3DGS paper's clone /
split / prune / opacity reset).

Parameter names follow gsplat's convention:
    means (N,3)  scales (N,3, log)  quats (N,4)  opacities (N, logit)
    sh0 (N,1,3)  shN (N,K-1,3)   spherical-harmonic colors
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from .render import SH_C0, quat_to_rotmat


def rgb_to_sh0(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / SH_C0


def init_from_points(xyz: np.ndarray, rgb: np.ndarray, n_init: int, sh_degree: int, seed: int,
                     init_opacity: float = 0.1, device="cuda") -> dict:
    """A random subset of the colored LiDAR map -> Gaussians. Each starts as a
    sphere whose radius is the RMS distance to its 3 nearest neighbors (as in
    the 3DGS reference code), with the point's camera color as its constant
    (degree-0) SH term."""
    rng = np.random.default_rng(seed)
    if len(xyz) > n_init:
        pick = rng.choice(len(xyz), n_init, replace=False)
        xyz, rgb = xyz[pick], rgb[pick]
    dist, _ = cKDTree(xyz).query(xyz, k=4)
    radius = np.sqrt((dist[:, 1:] ** 2).mean(axis=1)).clip(1e-3, 2.0)
    n = len(xyz)
    params = {
        "means": torch.tensor(xyz, dtype=torch.float32),
        "scales": torch.log(torch.tensor(radius, dtype=torch.float32)).unsqueeze(-1).repeat(1, 3),
        "quats": F.normalize(torch.randn(n, 4, generator=torch.Generator().manual_seed(seed)), dim=-1),
        "opacities": torch.logit(torch.full((n,), init_opacity)),
        "sh0": rgb_to_sh0(torch.tensor(rgb, dtype=torch.float32)[:, None, :] / 255.0),
        "shN": torch.zeros(n, (sh_degree + 1) ** 2 - 1, 3),
    }
    return {k: v.to(device).requires_grad_(True) for k, v in params.items()}


def activated(params: dict) -> tuple:
    return (params["means"], params["quats"], torch.exp(params["scales"]), torch.sigmoid(params["opacities"]),
            torch.cat([params["sh0"], params["shN"]], dim=1))


# ---------------------------------------------------------------------------
# Sky and per-camera color
# ---------------------------------------------------------------------------

class Sky(torch.nn.Module):
    """The sky as a learned panorama (equirectangular, world Z up), looked up
    per pixel by viewing direction and composited behind the Gaussians:
    C = sum_i c_i a_i T_i + T_final * sky(ray direction).
    LiDAR never returns from the sky, so no Gaussians start there."""

    def __init__(self, color, height: int = 64, width: int = 256):
        super().__init__()
        c = torch.as_tensor(color, dtype=torch.float32).clamp(1e-3, 1 - 1e-3)
        self.texture = torch.nn.Parameter(torch.logit(c).view(1, 3, 1, 1).repeat(1, 1, height, width))

    def forward(self, c2w: torch.Tensor, K: torch.Tensor, width: int, height: int) -> torch.Tensor:
        dev = self.texture.device
        ys, xs = torch.meshgrid(torch.arange(height, device=dev) + 0.5, torch.arange(width, device=dev) + 0.5,
                                indexing="ij")
        d = torch.stack([(xs - K[0, 2]) / K[0, 0], (ys - K[1, 2]) / K[1, 1], torch.ones_like(xs)], -1)
        d = F.normalize(d @ c2w[:3, :3].T, dim=-1)                       # world-space ray directions
        lon = torch.atan2(d[..., 1], d[..., 0]) / math.pi                 # -1..1
        lat = -torch.asin(d[..., 2].clamp(-1, 1)) / (math.pi / 2)        # -1 (up) .. 1 (down)
        grid = torch.stack([lon, lat], -1)[None]
        rgb = F.grid_sample(torch.sigmoid(self.texture), grid, mode="bilinear", padding_mode="border",
                            align_corners=False)
        return rgb[0].permute(1, 2, 0)                                    # (H, W, 3)


class ColorCorrection(torch.nn.Module):
    """A 3x3 color matrix and offset per camera, for the cameras' different
    exposure and white balance. The first camera is the reference (identity,
    not learned), so the scene's colors stay in its color space."""

    def __init__(self, n_cameras: int):
        super().__init__()
        self.delta = torch.nn.Parameter(torch.zeros(max(n_cameras - 1, 0), 3, 4))

    def forward(self, rgb: torch.Tensor, cam: int) -> torch.Tensor:
        if cam == 0:
            return rgb
        A = torch.eye(3, 4, device=rgb.device) + self.delta[cam - 1]
        return rgb @ A[:, :3].T + A[:, 3]


# ---------------------------------------------------------------------------
# Adaptive density control (3DGS paper, Sec. 5; gsplat's DefaultStrategy)
# ---------------------------------------------------------------------------

def _edit_params(params: dict, optimizers: dict, fn):
    """Apply fn(name, tensor) -> tensor to every parameter and to its Adam
    moments (new rows get zero moments), keeping the optimizers in sync."""
    for name in list(params.keys()):
        old = params[name]
        new = fn(name, old.detach()).requires_grad_(True)
        opt = optimizers[name]
        state = opt.state.pop(old, None)
        opt.param_groups[0]["params"] = [new]
        if state is not None:
            for key in ("exp_avg", "exp_avg_sq"):
                v = state[key]
                state[key] = fn(name + ":moment", v) if not torch.is_tensor(v) or v.dim() > 0 else v
            opt.state[new] = state
        params[name] = new


class DensityControl:
    """Clone small Gaussians and split large ones where the image-space
    position gradient is high (under-reconstructed regions), prune transparent
    or oversized ones, and periodically reset opacities so floaters must earn
    their opacity back. A hard cap keeps the count within the GPU budget."""

    def __init__(self, n: int, device, grad_thresh=2e-4, grow_scale_m=0.2, prune_opacity=0.005,
                 prune_scale_m=5.0, refine_start=500, refine_stop=15_000, refine_every=100,
                 reset_every=3000, max_gaussians=1_000_000):
        self.grad_thresh = grad_thresh
        self.grow_scale = grow_scale_m
        self.prune_opacity = prune_opacity
        self.prune_scale = prune_scale_m
        self.refine_start, self.refine_stop = refine_start, refine_stop
        self.refine_every, self.reset_every = refine_every, reset_every
        self.max_gaussians = max_gaussians
        self.device = device
        self._reset_stats(n)

    def _reset_stats(self, n):
        self.grad_sum = torch.zeros(n, device=self.device)
        self.count = torch.zeros(n, device=self.device)

    @torch.no_grad()
    def accumulate(self, proj: dict, width: int, height: int):
        """After backward: add each visible Gaussian's image-space position
        gradient, in normalized device coordinates (pixels x W/2, H/2) so the
        threshold matches the paper's 0.0002."""
        g = proj["means2d"].grad
        if g is None:
            return
        idx = proj["idx"]
        if g.dim() == 3:                       # gsplat backend: (1, N, 2) for all Gaussians
            g = g[0].index_select(0, idx)
        g = g * torch.tensor([width / 2.0, height / 2.0], device=g.device)
        self.grad_sum.index_add_(0, idx, g.norm(dim=-1))
        self.count.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))

    @torch.no_grad()
    def step(self, step: int, params: dict, optimizers: dict) -> dict | None:
        """Call once per training step after the optimizer step."""
        report = None
        if self.refine_start <= step < self.refine_stop and step % self.refine_every == 0:
            report = self._refine(step, params, optimizers)
        if step < self.refine_stop and step > 0 and step % self.reset_every == 0:
            cap = torch.logit(torch.tensor(2 * self.prune_opacity, device=self.device))
            _edit_params(params, optimizers,
                         lambda name, t: t.clamp(max=cap) if name == "opacities"
                         else (torch.zeros_like(t) if name == "opacities:moment" else t))
        return report

    def _refine(self, step, params, optimizers):
        n = params["means"].shape[0]
        avg = self.grad_sum / self.count.clamp_min(1)
        high = (avg > self.grad_thresh) & (self.count > 0)
        max_scale = torch.exp(params["scales"]).max(dim=-1).values
        clone = high & (max_scale <= self.grow_scale)
        split = high & (max_scale > self.grow_scale)
        # Budget: each clone adds one Gaussian, each split one net (two replace one).
        room = max(self.max_gaussians - n, 0)
        n_grow = int(clone.sum() + split.sum())
        if n_grow > room:
            keep_best = torch.zeros_like(high)
            cand = (clone | split).nonzero(as_tuple=True)[0]
            keep_best[cand[torch.topk(avg[cand], room).indices]] = True
            clone &= keep_best
            split &= keep_best
        n_clone, n_split = int(clone.sum()), int(split.sum())

        # Split: two samples from the Gaussian itself, scaled down by 1.6.
        sel = split.nonzero(as_tuple=True)[0]
        rot = quat_to_rotmat(params["quats"][sel])
        sc = torch.exp(params["scales"][sel])
        samples = torch.randn(2, len(sel), 3, device=self.device) * sc
        new_means = params["means"][sel] + torch.einsum("nij,bnj->bni", rot, samples)

        def grow(name, t):
            base = name.split(":")[0]
            moment = name.endswith(":moment")
            c = t[clone]
            s = t[split].repeat(2, *[1] * (t.dim() - 1))
            if moment:
                c, s = torch.zeros_like(c), torch.zeros_like(s)
            elif base == "means":
                s = new_means.reshape(-1, 3)
            elif base == "scales":
                s = s - math.log(1.6)
            return torch.cat([t[~split], c, s])

        _edit_params(params, optimizers, grow)

        # Prune: nearly transparent, or (after the first opacity reset) huge.
        opac = torch.sigmoid(params["opacities"])
        prune = opac < self.prune_opacity
        if step > self.reset_every:
            prune |= torch.exp(params["scales"]).max(dim=-1).values > self.prune_scale
        keep = ~prune
        _edit_params(params, optimizers, lambda name, t: t[keep])
        self._reset_stats(params["means"].shape[0])
        return {"step": step, "n": int(params["means"].shape[0]), "cloned": n_clone, "split": n_split,
                "pruned": int(prune.sum())}


def save(path: str, params: dict, extra: dict, meta: dict) -> None:
    torch.save({"params": {k: v.detach().cpu() for k, v in params.items()},
                "extra": {k: v.state_dict() for k, v in extra.items()}, "meta": meta}, path)


def load(path: str, device="cuda"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    params = {k: v.to(device) for k, v in ckpt["params"].items()}
    return params, ckpt["extra"], ckpt["meta"]
