# SPDX-License-Identifier: MIT
"""Shared test helper: random Gaussians in front of an identity camera."""
import torch


def random_scene(n, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    means = torch.empty(n, 3)
    means[:, 0] = (torch.rand(n, generator=g) - 0.5) * 16
    means[:, 1] = (torch.rand(n, generator=g) - 0.5) * 9
    means[:, 2] = torch.rand(n, generator=g) * 25 + 2
    params = {
        "means": means,
        "quats": torch.randn(n, 4, generator=g),
        "log_scales": torch.log(torch.rand(n, 3, generator=g) * 0.25 + 0.01),
        "logit_opac": torch.randn(n, generator=g) * 1.5,
        "sh": torch.randn(n, 16, 3, generator=g) * 0.3,
    }
    return {k: v.to(device).requires_grad_(True) for k, v in params.items()}
