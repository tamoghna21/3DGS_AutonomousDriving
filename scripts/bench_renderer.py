# SPDX-License-Identifier: MIT
"""How fast is the pure-PyTorch renderer, and what do its design choices buy?
Times render + backward on training views for every combination of
  --backwards  manual (hand-written CompositeChunk) | autograd (reference
               forward under activation checkpointing)
  --tiles      tile size in pixels
  --chunk-elems  size bound of the (tiles, chunk, pixels) tensors
with Gaussians from the trained checkpoint (--ckpt) or from the LiDAR map
(--counts N ...).

    python scripts/bench_renderer.py --ckpt outputs/train/ckpt.pt --backwards manual autograd --tiles 8 16
    python scripts/bench_renderer.py --counts 200000 500000 1000000
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from splat import gaussians as gs, render as R, scene as sc  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=os.path.join(ROOT, "data", "pandaset_028"))
    p.add_argument("--ckpt", default=None, help="use the trained Gaussians instead of a LiDAR initialization")
    p.add_argument("--counts", type=int, nargs="+", default=[200_000, 500_000, 1_000_000])
    p.add_argument("--backwards", nargs="+", default=["manual"])
    p.add_argument("--tiles", type=int, nargs="+", default=[8])
    p.add_argument("--chunk-elems", type=int, nargs="+", default=[1 << 22])
    p.add_argument("--views", type=int, default=24)
    p.add_argument("--out", default=None, help="write results here (.json; a .md table next to it)")
    args = p.parse_args()
    device = torch.device("cuda")
    scene = sc.load_scene(args.data_dir)
    views = sc.load_views(scene, "train")
    W, H = views["width"], views["height"]
    picks = [(i * 37) % len(views["records"]) for i in range(args.views)]

    if args.ckpt:
        params, _, meta = gs.load(args.ckpt, device)
        models = [(f"trained ({meta['n_gaussians']:,})", params, meta["sh_degree"])]
    else:
        xyz, rgb = sc.load_points(scene)
        models = [(f"LiDAR init ({n:,})", None, 3) for n in args.counts]

    rows = []
    for k, (name, params, sh) in enumerate(models):
        if params is None:
            params = gs.init_from_points(xyz, rgb, args.counts[k], 3, 0, device=device)
        for v in params.values():
            v.requires_grad_(True)
        for bw in args.backwards:
            for tile in args.tiles:
                for ce in args.chunk_elems:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    times = []
                    for j in [picks[0], picks[1]] + picks:          # two warm-up renders
                        torch.cuda.synchronize()
                        t0 = time.time()
                        m, q, s, o, c = gs.activated(params)
                        out = R.render(m, q, s, o, c, views["w2c"][j].to(device), views["K"][j].to(device), W, H,
                                       sh, with_depth=True, tile=tile, chunk_elems=ce, backward=bw)
                        (out["rgb"].mean() + 1e-3 * out["depth"].mean()).backward()
                        torch.cuda.synchronize()
                        times.append(time.time() - t0)
                        for v in params.values():
                            v.grad = None
                    row = {"gaussians": name, "backward": bw, "tile": tile, "chunk_elems": ce,
                           "median_ms": float(np.median(times[2:])) * 1e3,
                           "peak_mb": torch.cuda.max_memory_allocated() / 2**20}
                    rows.append(row)
                    print(json.dumps(row))
        del params
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"views": args.views, "resolution": [W, H], "gpu": torch.cuda.get_device_name(0),
                       "rows": rows}, f, indent=1)
        lines = ["| Gaussians | Backward | Tile | Render + backward (median) | Peak VRAM |", "|---|---|---|---|---|"]
        lines += [f"| {r['gaussians']} | {r['backward']} | {r['tile']}x{r['tile']} | {r['median_ms']:.0f} ms | "
                  f"{r['peak_mb'] / 1024:.2f} GB |" for r in rows]
        with open(os.path.splitext(args.out)[0] + ".md", "w") as f:
            f.write("\n".join(lines) + f"\n\n{args.views} training views, {W}x{H}, "
                    f"{torch.cuda.get_device_name(0)}.\n")


if __name__ == "__main__":
    main()
