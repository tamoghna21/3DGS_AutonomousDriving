# SPDX-License-Identifier: MIT
"""A differentiable 3D Gaussian splatting renderer in plain PyTorch.

No custom CUDA kernels: every step is an ordinary PyTorch tensor operation. The algorithm is the 3DGS paper's (Kerbl et al.,
SIGGRAPH 2023), with gsplat's numerical conventions so the two can be compared
pixel by pixel (tests/test_vs_gsplat.py):

  1. project    world -> camera -> pixels, and the 3D covariance -> a 2D one
                (EWA splatting: J W Sigma W^T J^T)
  2. bin        list every (Gaussian, 8x8-pixel tile) pair its footprint touches
  3. sort       by (tile, depth), so each tile sees its Gaussians front to back
  4. composite  C = sum_i c_i a_i T_i  with  T_i = prod_{j<i} (1 - a_j)

The CUDA rasterizer gives each tile a thread block and each pixel a thread,
which walks the tile's list until the pixel is opaque. Here every tile is
processed at once: step 4 takes the next chunk of each tile's sorted list,
evaluates those Gaussians at all the tile's pixels as one (tiles, chunk, pixels)
tensor, and carries the transmittance T into the next chunk. A tile leaves the
loop when its list ends or all its pixels are opaque (T < 1e-4), the tensor
version of the kernel's early exit.

Each chunk's compositing has a hand-written backward pass (CompositeChunk)
that keeps only the chunk's inputs and recomputes alpha and T, so memory stays
flat however long the lists get. composite_chunk_autograd is the same forward
pass left to autograd, as a reference.
"""
import math

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ALPHA_MIN = 1.0 / 255.0     # a Gaussian contributes to a pixel only if alpha >= 1/255
ALPHA_MAX = 0.999           # alpha cap (the 3DGS paper uses 0.99, gsplat 0.999)
T_MIN = 1e-4                # a pixel is opaque once its transmittance drops below this

# Real spherical-harmonic constants (degree 0..3), as in the 3DGS reference code.
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
         -1.0925484305920792, 0.5462742152960396)
SH_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
         -0.4570457994644658, 1.445305721320277, -0.5900435899266435)


# ---------------------------------------------------------------------------
# Gaussian parameters -> geometry and color
# ---------------------------------------------------------------------------

def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """(N,4) quaternions (w,x,y,z), normalized here, -> (N,3,3) rotation matrices."""
    w, x, y, z = F.normalize(q, dim=-1).unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).view(-1, 3, 3)


def eval_sh(degree: int, sh: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """View-dependent color from spherical-harmonic coefficients.
    sh: (N, K, 3) with K >= (degree+1)^2, dirs: (N, 3) unit vectors from the
    camera to each Gaussian. Returns (N, 3) RGB (before the +0.5 offset)."""
    s = sh.unbind(dim=1)                    # one op (and one backward) instead of 16 slices
    out = SH_C0 * s[0]
    if degree == 0:
        return out
    x, y, z = (d[:, None] for d in dirs.unbind(-1))
    out = out - SH_C1 * y * s[1] + SH_C1 * z * s[2] - SH_C1 * x * s[3]
    if degree == 1:
        return out
    xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
    out = (out + SH_C2[0] * xy * s[4] + SH_C2[1] * yz * s[5] + SH_C2[2] * (2 * zz - xx - yy) * s[6]
           + SH_C2[3] * xz * s[7] + SH_C2[4] * (xx - yy) * s[8])
    if degree == 2:
        return out
    return (out + SH_C3[0] * y * (3 * xx - yy) * s[9] + SH_C3[1] * xy * z * s[10]
            + SH_C3[2] * y * (4 * zz - xx - yy) * s[11] + SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * s[12]
            + SH_C3[4] * x * (4 * zz - xx - yy) * s[13] + SH_C3[5] * z * (xx - yy) * s[14]
            + SH_C3[6] * x * (xx - 3 * yy) * s[15])


# ---------------------------------------------------------------------------
# 1. Projection
# ---------------------------------------------------------------------------

def project(means, quats, scales, opacities, w2c, K, width, height,
            near: float = 0.01, far: float = 1e10, eps2d: float = 0.3):
    """Project N Gaussians into one pinhole camera (OpenCV convention: +Z
    forward, +Y down; pixel centers at integer + 0.5, as in gsplat).

    means (N,3), quats (N,4), scales (N,3) [already exp'd], opacities (N,)
    [already sigmoid'd], w2c (4,4), K (3,3).

    Returns a dict for the Gaussians that can touch the image:
      idx      (V,)   their indices into the N inputs
      means2d  (V,2)  pixel position of the projected center
      conics   (V,3)  (a, b, c) of the inverse 2D covariance [[a, b], [b, c]]
      depths   (V,)   camera-space z
      radii    (V,2)  half-size of the footprint's bounding box in pixels (int)
    """
    R, t = w2c[:3, :3], w2c[:3, 3]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    p = means @ R.T + t                                    # camera space
    x, y, z = p.unbind(-1)
    rz = 1.0 / z.clamp_min(near)

    # Projection Jacobian (the 3DGS paper's J, from EWA splatting): the local
    # affine approximation of the perspective divide at each Gaussian's center.
    # Like gsplat, x/z and y/z are clamped to 1.3x the field of view so Gaussians
    # far outside the image don't get huge, unstable footprints.
    lim_x_pos = (width - cx) / fx + 0.3 * (0.5 * width / fx)
    lim_x_neg = cx / fx + 0.3 * (0.5 * width / fx)
    lim_y_pos = (height - cy) / fy + 0.3 * (0.5 * height / fy)
    lim_y_neg = cy / fy + 0.3 * (0.5 * height / fy)
    tx = (x * rz).clamp(-lim_x_neg, lim_x_pos)
    ty = (y * rz).clamp(-lim_y_neg, lim_y_pos)
    #     J = [[fx/z, 0, -fx tx/z],
    #          [0, fy/z, -fy ty/z]]

    # 3D covariance Sigma = R_q S S^T R_q^T. With M = J W R_q S (W = camera
    # rotation), the 2D covariance J W Sigma W^T J^T is simply M M^T. These are
    # N tiny matrix products, written out elementwise: batched matmuls of 2x3
    # matrices waste most of a GPU matrix-multiply tile (and are slow).
    RS = quat_to_rotmat(quats) * scales[:, None, :]                    # columns of R_q scaled by s
    WRS = (R[None, :, :, None] * RS[:, None, :, :]).sum(dim=2)        # W R_q S, (N,3,3)
    w0, w1, w2 = WRS.unbind(dim=1)                                     # its rows
    m0 = (fx * rz)[:, None] * (w0 - tx[:, None] * w2)                  # rows of M = J W R_q S
    m1 = (fy * rz)[:, None] * (w1 - ty[:, None] * w2)
    # Low-pass filter: add 0.3 px^2 so every Gaussian covers at least about a pixel.
    a = (m0 * m0).sum(-1) + eps2d
    b = (m0 * m1).sum(-1)
    c = (m1 * m1).sum(-1) + eps2d
    det = a * c - b * b

    with torch.no_grad():
        # Footprint: the tight bounding box of the ellipse where alpha >= 1/255
        # (opacity-aware, capped at 3.33 sigma; gsplat's rule). The binning in
        # step 2 uses it; it is not differentiated.
        extend = torch.sqrt(2.0 * torch.log((opacities / ALPHA_MIN).clamp_min(1.0))).clamp(max=3.33)
        rx = torch.ceil(extend * torch.sqrt(a.clamp_min(0)))
        ry = torch.ceil(extend * torch.sqrt(c.clamp_min(0)))
        u = fx * x * rz + cx
        v = fy * y * rz + cy
        keep = ((z > near) & (z < far) & (det > 0) & (opacities >= ALPHA_MIN) & ((rx > 0) | (ry > 0))
                & (u + rx > 0) & (u - rx < width) & (v + ry > 0) & (v - ry < height))
        idx = keep.nonzero(as_tuple=True)[0]

    # index_select rather than t[idx]: its backward is a fast scatter-add.
    rz, x, y, z, a, b, c, det = (t.index_select(0, idx) for t in (rz, x, y, z, a, b, c, det))
    means2d = torch.stack([fx * x * rz + cx, fy * y * rz + cy], dim=-1)
    conics = torch.stack([c / det, -b / det, a / det], dim=-1)
    return {"idx": idx, "means2d": means2d, "conics": conics, "depths": z,
            "radii": torch.stack([rx[idx], ry[idx]], dim=-1).int()}


# ---------------------------------------------------------------------------
# 2-3. Binning and sorting
# ---------------------------------------------------------------------------

def bin_and_sort(means2d, radii, depths, width, height, tile):
    """Every (Gaussian, tile) pair whose footprint box overlaps the tile,
    grouped by tile and sorted front to back within each tile.

    Returns gauss_ids (M,) [indices into the V projected Gaussians], and per tile
    its start offset and count in that list. Not differentiable (indices only)."""
    tiles_x, tiles_y = math.ceil(width / tile), math.ceil(height / tile)
    with torch.no_grad():
        # Pixel j covers [j, j+1); its center is j + 0.5. The box [u - r, u + r]
        # touches pixels floor(u - r) .. ceil(u + r) - 1.
        u, v = means2d[:, 0], means2d[:, 1]
        rx, ry = radii[:, 0].float(), radii[:, 1].float()
        tx0 = (torch.floor(u - rx) / tile).floor().clamp(0, tiles_x - 1).long()
        tx1 = ((torch.ceil(u + rx) - 1) / tile).floor().clamp(0, tiles_x - 1).long()
        ty0 = (torch.floor(v - ry) / tile).floor().clamp(0, tiles_y - 1).long()
        ty1 = ((torch.ceil(v + ry) - 1) / tile).floor().clamp(0, tiles_y - 1).long()
        nx, ny = tx1 - tx0 + 1, ty1 - ty0 + 1
        counts = nx * ny                                       # tiles per Gaussian

        # Depth order first; a stable sort by tile afterwards keeps it inside every tile.
        order = torch.argsort(depths)
        counts_o = counts[order]
        total = int(counts_o.sum().item())
        gauss = torch.repeat_interleave(order, counts_o)        # (M,) one entry per pair
        first = torch.cumsum(counts_o, 0) - counts_o            # where each Gaussian's run starts
        k = torch.arange(total, device=means2d.device) - torch.repeat_interleave(first, counts_o)
        w = nx[gauss]
        tile_ids = (ty0[gauss] + k // w) * tiles_x + (tx0[gauss] + k % w)
        tile_ids, perm = torch.sort(tile_ids, stable=True)
        gauss = gauss[perm]

        n_tiles = tiles_x * tiles_y
        tile_count = torch.bincount(tile_ids, minlength=n_tiles)
        tile_start = torch.cumsum(tile_count, 0) - tile_count
    return gauss, tile_start, tile_count


_TILE_GEOMETRY = {}


def tile_geometry(width, height, tile, device):
    """Per tile, the center in pixels (n_tiles, 2); and the quadratic basis
    [1, x, y, x^2, xy, y^2] of the tile's pixel centers in tile-local
    coordinates (tile*tile, 6), the same for every tile."""
    key = (width, height, tile, str(device))
    if key not in _TILE_GEOMETRY:
        tiles_x, tiles_y = math.ceil(width / tile), math.ceil(height / tile)
        ty, tx = torch.meshgrid(torch.arange(tiles_y, device=device), torch.arange(tiles_x, device=device),
                                indexing="ij")
        centers = (torch.stack([tx, ty], -1).view(-1, 2).float() + 0.5) * tile
        ys, xs = torch.meshgrid(torch.arange(tile, device=device), torch.arange(tile, device=device), indexing="ij")
        x = xs.reshape(-1).float() + 0.5 - tile / 2
        y = ys.reshape(-1).float() + 0.5 - tile / 2
        basis = torch.stack([torch.ones_like(x), x, y, x * x, x * y, y * y], dim=-1)
        _TILE_GEOMETRY[key] = (centers, basis)
    return _TILE_GEOMETRY[key]


# ---------------------------------------------------------------------------
# 4. Front-to-back compositing
# ---------------------------------------------------------------------------
#
# Inside a tile, write a pixel as (x, y) relative to the tile center and a
# Gaussian's center as (mx, my). The exponent of its falloff, plus log opacity,
#     log o - 1/2 [a (x-mx)^2 + 2b (x-mx)(y-my) + c (y-my)^2]
# is a quadratic in (x, y):  k0 + k1 x + k2 y + k3 x^2 + k4 xy + k5 y^2, so
#     alpha = exp(coeffs @ [1, x, y, x^2, xy, y^2]^T)
# for every Gaussian of a chunk and every pixel of its tile is ONE matrix
# multiply, (A, B, 6) @ (6, P), instead of a dozen elementwise passes.
# (Tile-local coordinates keep the terms small enough for float32. Don't enable
# TF32 matmuls: their 10-bit mantissa is too coarse for this.)

def chunk_coefficients(feat, centers):
    """feat (A, B, >=6): [u, v, conic a, b, c, log opacity, ...] of each chunk
    slot; centers (A, 2) of the tiles. Returns the 6 quadratic coefficients."""
    u, v, a, b, c, log_o = feat[..., :6].unbind(-1)
    mx = u - centers[:, None, 0]
    my = v - centers[:, None, 1]
    return torch.stack([log_o - 0.5 * (a * mx * mx + 2 * b * mx * my + c * my * my),
                        a * mx + b * my, b * mx + c * my, -0.5 * a, -b, -0.5 * c], dim=-1)


def _alpha(coeffs, basis):
    """alpha (A, B, P) with gsplat's rules: capped at 0.999, zero below 1/255.
    Also returns where it is differentiable (neither capped nor zeroed)."""
    raw = torch.exp(torch.matmul(coeffs, basis.T))
    live = (raw >= ALPHA_MIN) & (raw < ALPHA_MAX)
    alpha = raw.clamp_(max=ALPHA_MAX).masked_fill_(raw < ALPHA_MIN, 0.0)
    return alpha, live


def composite_chunk_autograd(coeffs, feats, log_T, basis):
    """Reference version: plain tensor ops, autograd differentiates them.
    coeffs (A,B,6), feats (A,B,C) [what gets composited], log_T (A,P) [log
    transmittance in front of the chunk]. Returns the chunk's contribution
    sum_i f_i a_i T_i (A,P,C) and the log transmittance behind it (A,P)."""
    raw = torch.exp(torch.matmul(coeffs, basis.T))
    alpha = torch.where(raw >= ALPHA_MIN, raw.clamp(max=ALPHA_MAX), torch.zeros_like(raw))
    # T_i = T_in * prod_{j<i} (1 - a_j), as a cumulative sum of logs
    log_1m = torch.log1p(-alpha)
    cum = torch.cumsum(log_1m, dim=1)
    weight = alpha * torch.exp(log_T[:, None, :] + cum - log_1m)        # a_i T_i
    return torch.bmm(weight.transpose(1, 2), feats), log_T + cum[:, -1]


class CompositeChunk(torch.autograd.Function):
    """The same computation with a hand-written backward pass: the math of the
    CUDA kernel's backward, as tensor ops. Only the inputs are saved; alpha and
    T are recomputed in backward (cheap: one matmul and a cumulative sum).

    With w_i = a_i T_i and contrib = sum_i f_i w_i, for an upstream gradient g
    on contrib and g_T on log T_out:
        dL/df_i     = sum_p w_ip g_p
        dL/da_i     = T_i (f_i . g) - (S_i . g + g_T) / (1 - a_i),
                      S_i = sum_{k>i} f_k w_k  (what lies behind Gaussian i)
        dL/dlogT_in = g_T + contrib . g
    and alpha = exp(coeffs @ basis^T) gives dL/dcoeffs = (dL/da * a) @ basis."""

    @staticmethod
    def forward(ctx, coeffs, feats, log_T, basis):
        alpha, _ = _alpha(coeffs, basis)
        log_1m = torch.log1p(-alpha)
        cum = torch.cumsum(log_1m, dim=1)
        log_T_out = log_T + cum[:, -1]
        w = cum.sub_(log_1m).add_(log_T[:, None, :]).exp_().mul_(alpha)  # a_i T_i, in place
        contrib = torch.bmm(w.transpose(1, 2), feats)
        ctx.save_for_backward(coeffs, feats, log_T, basis)
        return contrib, log_T_out

    @staticmethod
    def backward(ctx, g, g_T):
        coeffs, feats, log_T, basis = ctx.saved_tensors
        alpha, live = _alpha(coeffs, basis)
        log_1m = torch.log1p(-alpha)
        T = torch.cumsum(log_1m, dim=1).sub_(log_1m).add_(log_T[:, None, :]).exp_()   # T_i
        w = alpha * T
        h = torch.bmm(feats, g.transpose(1, 2))           # f_i . g at every pixel, (A, B, P)
        wh = w * h
        total = wh.sum(dim=1)                              # contrib . g, (A, P)
        behind = total[:, None, :] - torch.cumsum(wh, dim=1)                # S_i . g = sum_{k>i} w_k (f_k . g)
        g_alpha = T * h - (behind + g_T[:, None, :]) / (1.0 - alpha)
        g_coeffs = torch.matmul(g_alpha * alpha * live, basis)              # through alpha = exp(power)
        g_feats = torch.bmm(w, g)
        return g_coeffs, g_feats, g_T + total, None


def rasterize(means2d, conics, opacities, features, depths, radii, width, height,
              tile: int = 8, chunk_elems: int = 1 << 22, backward: str = "manual"):
    """Alpha-composite projected Gaussians into an image.

    features (V, C) is what gets composited (RGB, optionally depth as a 4th
    channel). Returns (H, W, C) sum_i f_i a_i T_i and the final transmittance
    (H, W); the caller adds the background as T * background.

    chunk_elems bounds the size of the (tiles, chunk, pixels) tensors: lower it
    to save memory, raise it for fewer, larger kernel launches. backward is
    "manual" (CompositeChunk) or "autograd" (the reference version under
    activation checkpointing; same result, slower)."""
    device = means2d.device
    C = features.shape[1]
    tiles_x, tiles_y = math.ceil(width / tile), math.ceil(height / tile)
    n_tiles, P = tiles_x * tiles_y, tile * tile
    centers, basis = tile_geometry(width, height, tile, device)

    out = torch.zeros(n_tiles, P, C, device=device)
    log_T = torch.zeros(n_tiles, P, device=device)
    if means2d.shape[0] > 0:
        gauss, tile_start, tile_count = bin_and_sort(means2d, radii, depths, width, height, tile)
        # One gather per chunk instead of one per attribute: pack what compositing needs.
        packed = torch.cat([means2d, conics, torch.log(opacities)[:, None], features], dim=1)
        pad = torch.zeros_like(packed[:1])
        pad[0, 5] = -1e4                                   # log opacity of the padding slot: alpha = 0
        packed = torch.cat([packed, pad], dim=0)
        pad_row = packed.shape[0] - 1
        M = gauss.shape[0]

        active = (tile_count > 0).nonzero(as_tuple=True)[0]
        offset = 0
        grad = torch.is_grad_enabled() and packed.requires_grad
        while active.numel() > 0:
            remaining = int((tile_count[active] - offset).max().item())
            B = max(16, min(remaining, chunk_elems // (active.numel() * P)))
            pos = offset + torch.arange(B, device=device)
            slot = tile_start[active, None] + pos[None, :]                    # (A, B)
            valid = pos[None, :] < tile_count[active, None]
            ids = torch.where(valid, gauss[slot.clamp(max=M - 1)], torch.full_like(slot, pad_row))
            feat = packed.index_select(0, ids.view(-1)).view(*ids.shape, -1)  # (A, B, 6 + C)
            coeffs = chunk_coefficients(feat, centers[active])
            log_T_in = log_T.index_select(0, active)
            if backward == "manual":
                contrib, log_T_new = CompositeChunk.apply(coeffs, feat[..., 6:], log_T_in, basis)
            elif grad:
                contrib, log_T_new = checkpoint(composite_chunk_autograd, coeffs, feat[..., 6:], log_T_in,
                                                basis, use_reentrant=False)
            else:
                contrib, log_T_new = composite_chunk_autograd(coeffs, feat[..., 6:], log_T_in, basis)
            out = out.index_add(0, active, contrib)
            log_T = log_T.index_copy(0, active, log_T_new)
            offset += B
            with torch.no_grad():
                more = (tile_count[active] > offset) & (log_T_new.max(dim=1).values > math.log(T_MIN))
            active = active[more]

    def to_image(t):
        t = t.view(tiles_y, tiles_x, tile, tile, -1).permute(0, 2, 1, 3, 4)
        return t.reshape(tiles_y * tile, tiles_x * tile, -1)[:height, :width]

    return to_image(out), to_image(torch.exp(log_T))[..., 0]


# ---------------------------------------------------------------------------
# Everything together
# ---------------------------------------------------------------------------

def render(means, quats, scales, opacities, sh, w2c, K, width, height, sh_degree: int,
           background=None, with_depth: bool = False, tile: int = 8, chunk_elems: int = 1 << 22,
           near: float = 0.01, far: float = 1e10, backward: str = "manual"):
    """Render one camera. Activated parameters in (scales = exp(log_scales),
    opacities = sigmoid(logits)); sh (N, K, 3).

    background: None (black), a (3,) color, or an (H, W, 3) image (e.g. a sky
    texture looked up per pixel). Returns a dict with
      rgb (H,W,3), alpha (H,W) = 1 - T, depth (H,W) expected depth (if with_depth),
      and 'proj' (the projection outputs, for densification statistics)."""
    proj = project(means, quats, scales, opacities, w2c, K, width, height, near=near, far=far)
    idx = proj["idx"]
    # Color from SH, seen along the ray from the camera center to each Gaussian.
    cam_center = -w2c[:3, :3].T @ w2c[:3, 3]
    dirs = F.normalize(means.index_select(0, idx) - cam_center, dim=-1)
    rgb = (eval_sh(sh_degree, sh.index_select(0, idx), dirs) + 0.5).clamp_min(0.0)
    feats = torch.cat([rgb, proj["depths"][:, None]], dim=1) if with_depth else rgb
    img, T = rasterize(proj["means2d"], proj["conics"], opacities.index_select(0, idx), feats, proj["depths"],
                       proj["radii"], width, height, tile=tile, chunk_elems=chunk_elems,
                       backward=backward)
    out = {"rgb": img[..., :3], "alpha": 1.0 - T, "proj": proj}
    if background is not None:
        bg = background if background.dim() == 3 else background.view(1, 1, 3)
        out["rgb"] = out["rgb"] + T[..., None] * bg
    if with_depth:
        out["depth"] = img[..., 3]           # sum_i z_i a_i T_i; divide by alpha for the surface depth
    return out
