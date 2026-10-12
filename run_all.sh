#!/usr/bin/env bash
# Main pipeline, in order (about 2.5 hours on a 4 GB RTX A1000, almost all of it training).
# The gsplat comparison and the camera and configuration variants at the end are optional and commented out.
set -euo pipefail
cd "$(dirname "$0")"
PY="conda run --no-capture-output -n pandaset-splat python -u"
SEQ="${SEQ:-028}"
DATA="data/pandaset_${SEQ}"

$PY scripts/01_prepare_scene.py --seq "$SEQ"            # PandaSet -> images, masks, LiDAR map   (~2 min)
$PY scripts/02_check_scene.py --data-dir "$DATA"         # LiDAR over the cameras, top view          (~20 s)
$PY scripts/03_train.py --data-dir "$DATA"               # train the splat                           (~2 h)
$PY scripts/04_replay.py --data-dir "$DATA"              # held-out metrics, replay clip             (~1 min)
$PY scripts/05_novel_views.py --data-dir "$DATA"         # lane-change figures and animation         (~2 min)
$PY scripts/06_figures.py --data-dir "$DATA"             # training progress, anatomy                (~20 s)

# Renderer design ablation (pure PyTorch, ~5 min):
$PY scripts/bench_renderer.py --data-dir "$DATA" --ckpt outputs/train/ckpt.pt --backwards manual autograd \
    --tiles 8 16 --out outputs/bench/renderer_ablation.json
$PY tests/test_backward.py                               # hand-written backward vs autograd

# Optional, needs a separate environment with gsplat installed (only used as a reference):
#   GS="conda run --no-capture-output -n <env-with-gsplat> python -u"
#   $GS tests/test_vs_gsplat.py                                                  # images + gradients match
#   $GS scripts/bench_vs_gsplat.py --data-dir "$DATA"                            # speed on the trained scene
#   $GS scripts/03_train.py --data-dir "$DATA" --backend gsplat --out-dir outputs/train_gsplat   # ~25 min
#   $PY scripts/07_compare_backends.py                                           # time to quality

# Camera and configuration variants under lane shifts (pure PyTorch; see the README):
#   $PY scripts/01_prepare_scene.py --seq "$SEQ" --out-dir data/pandaset_${SEQ}_6cam --cameras \
#       front_camera front_left_camera front_right_camera left_camera right_camera back_camera
#   bash scripts/run_offpath_study.sh && bash scripts/run_offpath_study2.sh && bash scripts/run_offpath_study3.sh
#   then scripts/08_eval_offpath.py on outputs/train/ckpt.pt and scripts/09_offpath_figures.py (README lists the arguments)
