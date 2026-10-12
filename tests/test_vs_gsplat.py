# SPDX-License-Identifier: MIT
"""Check the pure-PyTorch renderer against gsplat's CUDA kernels.

Random Gaussians in front of a camera are rendered by both; the test compares
the images, the alpha maps and the gradients of a random loss with respect to
every parameter. gsplat is used only here, as a reference; it is not a
dependency of the project. Run it in a separate environment that has gsplat
installed:

    python tests/test_vs_gsplat.py
"""
import os
import sys
import time

import torch
from gsplat import rasterization

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from splat import render as R  # noqa: E402
from common import random_scene  # noqa: E402


def ours(p, w2c, K, W, H, deg):
    out = R.render(p["means"], p["quats"], torch.exp(p["log_scales"]), torch.sigmoid(p["logit_opac"]), p["sh"],
                   w2c, K, W, H, deg)
    return out["rgb"], out["alpha"]


def theirs(p, w2c, K, W, H, deg):
    rgb, alpha, _ = rasterization(p["means"], p["quats"], torch.exp(p["log_scales"]), torch.sigmoid(p["logit_opac"]),
                                  p["sh"], w2c[None], K[None], W, H, sh_degree=deg, packed=False)
    return rgb[0], alpha[0, ..., 0]


def main():
    device = torch.device("cuda")
    W, H = 480, 272
    K = torch.tensor([[400.0, 0, 241.3], [0, 400.0, 133.7], [0, 0, 1]], device=device)
    w2c = torch.eye(4, device=device)
    ok = True
    for n, deg, seed in [(2_000, 0, 0), (20_000, 3, 1), (100_000, 3, 2)]:
        p = random_scene(n, seed, device)
        target = torch.rand(H, W, 3, device=device, generator=torch.Generator(device=device).manual_seed(seed))
        results = {}
        for name, fn in (("ours", ours), ("gsplat", theirs)):
            for v in p.values():
                v.grad = None
            torch.cuda.synchronize()
            t0 = time.time()
            rgb, alpha = fn(p, w2c, K, W, H, deg)
            loss = ((rgb - target) ** 2).mean() + alpha.mean()
            loss.backward()
            torch.cuda.synchronize()
            results[name] = (rgb.detach(), alpha.detach(), {k: v.grad.clone() for k, v in p.items()},
                             time.time() - t0)
        (r1, a1, g1, t1), (r2, a2, g2, t2) = results["ours"], results["gsplat"]
        img_err = (r1 - r2).abs()
        print(f"N={n:>7,}  SH degree {deg}:  max |rgb diff| {img_err.max():.2e}  mean {img_err.mean():.2e}  "
              f"max |alpha diff| {(a1 - a2).abs().max():.2e}   fwd+bwd: ours {t1 * 1e3:.0f} ms, "
              f"gsplat {t2 * 1e3:.0f} ms")
        for k in g1:
            num = (g1[k] - g2[k]).norm()
            rel = (num / g2[k].norm().clamp_min(1e-12)).item()
            cos = torch.nn.functional.cosine_similarity(g1[k].flatten(), g2[k].flatten(), dim=0).item()
            print(f"    grad {k:<11} relative error {rel:.2e}   cosine {cos:.6f}")
            ok &= cos > 0.999
        ok &= img_err.mean().item() < 1e-3
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
