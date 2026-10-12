# SPDX-License-Identifier: MIT
"""Image metrics on (H,W,3) float tensors in [0,1]: PSNR, SSIM, LPIPS (AlexNet),
optionally restricted to a mask; and the SSIM map used in the training loss."""
import math

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim as _ssim

_lpips = None


def psnr(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None) -> float:
    err = (pred - target) ** 2
    mse = (err.mean() if valid is None else err[valid].mean()).item()
    return 99.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)


def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    a = pred.permute(2, 0, 1)[None]
    b = target.permute(2, 0, 1)[None]
    return _ssim(a, b, data_range=1.0, size_average=True, win_size=11).item()


def lpips(pred: torch.Tensor, target: torch.Tensor) -> float:
    global _lpips
    if _lpips is None:
        import lpips as lpips_pkg
        _lpips = lpips_pkg.LPIPS(net="alex", verbose=False).to(pred.device).eval()
    with torch.no_grad():
        a = pred.permute(2, 0, 1)[None] * 2 - 1       # LPIPS expects [-1, 1]
        b = target.permute(2, 0, 1)[None] * 2 - 1
        return _lpips(a, b).item()


def all_metrics(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None) -> dict:
    out = {"psnr": psnr(pred, target), "ssim": ssim(pred, target), "lpips": lpips(pred, target)}
    if valid is not None:
        out["psnr_static"] = psnr(pred, target, valid)
    return out


_WINDOW = {}


def ssim_map(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-pixel SSIM (11x11 Gaussian window, sigma 1.5, the standard
    constants) of (H,W,3) images, averaged over channels -> (H,W). Same-size
    output (reflect padding) so it can be masked."""
    dev = pred.device
    if dev not in _WINDOW:
        g = torch.exp(-((torch.arange(11, device=dev) - 5.0) ** 2) / (2 * 1.5 ** 2))
        g = g / g.sum()
        _WINDOW[dev] = (g[:, None] * g[None, :]).expand(3, 1, 11, 11).contiguous()
    w = _WINDOW[dev]
    x = F.pad(pred.permute(2, 0, 1)[None], (5, 5, 5, 5), mode="reflect")
    y = F.pad(target.permute(2, 0, 1)[None], (5, 5, 5, 5), mode="reflect")

    def blur(t):
        return F.conv2d(t, w, groups=3)
    mx, my = blur(x), blur(y)
    sxx = blur(x * x) - mx * mx
    syy = blur(y * y) - my * my
    sxy = blur(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return s[0].mean(0)
