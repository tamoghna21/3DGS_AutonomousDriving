# SPDX-License-Identifier: MIT
"""Step 9: charts for the off-path study (scripts/run_offpath_study.sh).

  figures/fig_offpath_by_model.png     realism score (FD, lower is better) against
                                       lane shift, one line per model
  figures/fig_offpath_by_steps.png     the realism score on the recorded path, 2 m and
                                       3 m left against training steps, one panel per run
  outputs/study/summary.md             the numbers

    python scripts/09_offpath_figures.py \\
        --model "3 cameras" outputs/study/eval_baseline_front3.json "3 cams@30k" \\
        --model "6 cameras" outputs/study/eval_6cam_front3.json "6cam@30k" ...
"""
import argparse
import json
import os
import re

import matplotlib
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, color=GRID, lw=1.0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def legend(ax, loc):
    leg = ax.legend(frameon=False, fontsize=8, loc=loc)
    for t in leg.get_texts():
        t.set_color(INK2)


def steps_of(label):
    m = re.search(r"@([\d.]+)k", label)
    return float(m.group(1)) * 1000 if m else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", nargs=3, action="append", metavar=("NAME", "JSON", "LABEL"), required=True,
                   help="a model for the by-model chart: display name, eval json, row label in it")
    p.add_argument("--steps", nargs=2, action="append", metavar=("NAME", "JSON"), default=[],
                   help="a run whose checkpoints (labels ending @<steps>k) form the by-steps chart")
    p.add_argument("--exclude", nargs="*", default=[], help="checkpoint labels to leave out of the by-steps chart")
    p.add_argument("--fig-dir", default=os.path.join(ROOT, "figures"))
    p.add_argument("--out", default=os.path.join(ROOT, "outputs", "study", "summary.md"))
    args = p.parse_args()

    def load(path):
        with open(path) as f:
            return json.load(f)

    # 1) Realism against lane shift, one line per model.
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=160, facecolor=SURFACE)
    style(ax)
    lines = ["| Model | On-path PSNR | SSIM | LPIPS | FD 0 m | FD 2 m | FD 3 m |", "|---|---|---|---|---|---|---|"]
    for (name, path, lab), col in zip(args.model, SERIES):
        d = load(path)
        row = next(r for r in d["results"] if r["label"] == lab)
        xs = d["shifts_m_left"]
        ys = [row[f"fd_{s:g}m"] for s in xs]
        ax.plot(xs, ys, "-o", color=col, lw=2, ms=7, mec=SURFACE, mew=2, label=name)
        lines.append(f"| {name} | {row['onpath_psnr']:.2f} dB | {row['onpath_ssim']:.3f} | {row['onpath_lpips']:.3f} | "
                     + " | ".join(f"{y:.0f}" for y in ys) + " |")
    ax.set_xticks(xs, [("recorded path" if s == 0 else f"{s:g} m left") for s in xs])
    ax.set_ylabel("realism gap: FD to real photos (lower is better)")
    ax.set_title("Driving one lane over: how far renders drift from real photos", color=INK, fontsize=10,
                 loc="left")
    legend(ax, "upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(args.fig_dir, "fig_offpath_by_model.png"), facecolor=SURFACE)
    plt.close(fig)

    # 2) Training length: the realism gap on the recorded path, 2 m and 3 m left,
    #    against steps; one panel per run, same scale (all FD, starting at 0).
    if args.steps:
        fig, axes = plt.subplots(1, len(args.steps), figsize=(5 * len(args.steps), 3.6), dpi=160,
                                 facecolor=SURFACE, sharey=True, squeeze=False)
        axes = axes[0]
        lines += ["", "| Run | Steps | On-path PSNR | FD 0 m | FD 2 m | FD 3 m |", "|---|---|---|---|---|---|"]
        top = 0
        for ax, (name, path) in zip(axes, args.steps):
            style(ax)
            rows = sorted([r for r in load(path)["results"] if steps_of(r["label"]) and r["label"] not in args.exclude],
                          key=lambda r: steps_of(r["label"]))
            x = [steps_of(r["label"]) / 1000 for r in rows]
            for key, lab, col in (("fd_0m", "recorded path", SERIES[0]), ("fd_2m", "2 m left", SERIES[1]),
                                  ("fd_3m", "3 m left", SERIES[2])):
                y = [r[key] for r in rows]
                top = max(top, max(y))
                ax.plot(x, y, "-o", color=col, lw=2, ms=6, mec=SURFACE, mew=2, label=lab)
            for r in rows:
                lines.append(f"| {name} | {steps_of(r['label']):,.0f} | {r['onpath_psnr']:.2f} dB | "
                             f"{r['fd_0m']:.0f} | {r['fd_2m']:.0f} | {r['fd_3m']:.0f} |")
            ax.set_xlabel("training steps (thousands)")
            ax.set_title(name, color=INK, fontsize=10, loc="left")
        axes[0].set_ylabel("FD to real photos (lower is better)")
        axes[0].set_ylim(0, top * 1.12)
        legend(axes[0], "lower left")
        fig.tight_layout()
        fig.savefig(os.path.join(args.fig_dir, "fig_offpath_by_steps.png"), facecolor=SURFACE)
        plt.close(fig)

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n\nFD: Frechet distance between DINOv2 ViT-L/14 features of renders "
                "(all frames of the three front cameras, rig shifted left) and the real photos; lower is better.\n")
    print("\n".join(lines))
    print("STUDY_FIGURES_DONE")


if __name__ == "__main__":
    main()
