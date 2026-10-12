# SPDX-License-Identifier: MIT
"""The hand-written backward pass (CompositeChunk) against autograd on the
reference forward pass: same images, same gradients, and how much faster.

    python tests/test_backward.py
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from splat import render as R  # noqa: E402
from common import random_scene  # noqa: E402


def run(p, w2c, K, W, H, mode, target):
    for v in p.values():
        v.grad = None
    torch.cuda.synchronize()
    t0 = time.time()
    out = R.render(p["means"], p["quats"], torch.exp(p["log_scales"]), torch.sigmoid(p["logit_opac"]), p["sh"],
                   w2c, K, W, H, 3, with_depth=True, backward=mode)
    loss = ((out["rgb"] - target) ** 2).mean() + out["alpha"].mean() + 1e-3 * out["depth"].mean()
    loss.backward()
    torch.cuda.synchronize()
    return out["rgb"].detach(), {k: v.grad.clone() for k, v in p.items()}, time.time() - t0


def main():
    device = torch.device("cuda")
    W, H = 480, 270
    K = torch.tensor([[400.0, 0, 241.3], [0, 400.0, 133.7], [0, 0, 1]], device=device)
    w2c = torch.eye(4, device=device)
    ok = True
    for n, seed in [(5_000, 0), (100_000, 1)]:
        p = random_scene(n, seed, device)
        target = torch.rand(H, W, 3, device=device)
        for _ in range(2):                                   # warm-up, then measure
            r1, g1, t1 = run(p, w2c, K, W, H, "manual", target)
            r2, g2, t2 = run(p, w2c, K, W, H, "autograd", target)
        print(f"N={n:>7,}: max |rgb diff| {(r1 - r2).abs().max():.1e}   fwd+bwd manual {t1 * 1e3:.0f} ms, "
              f"autograd {t2 * 1e3:.0f} ms")
        for k in g1:
            rel = ((g1[k] - g2[k]).norm() / g2[k].norm()).item()
            print(f"    grad {k:<11} relative difference {rel:.1e}")
            ok &= rel < 1e-3
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
