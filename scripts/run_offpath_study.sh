#!/usr/bin/env bash
# Off-path study: what makes the splat hold up when the car drives a different
# path (2-3 m lane shifts)? All runs use the pure-PyTorch renderer.
#   A: six cameras                 (vs the three-camera model in outputs/train)
#   B: six cameras + anti-needle and opacity regularizers
# Each run saves checkpoints at 7.5k, 15k, 22.5k and 30k steps; every checkpoint
# is scored on and off the recorded path by scripts/08_eval_offpath.py.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="conda run --no-capture-output -n pandaset-splat python -u"
D6=data/pandaset_028_6cam
OUT=outputs/study
FRONT="front_camera front_left_camera front_right_camera"
ALL6="front_camera front_left_camera front_right_camera left_camera right_camera back_camera"
mkdir -p "$OUT" logs

run() {   # name, extra training args...
    local name=$1; shift
    $PY scripts/03_train.py --data-dir "$D6" --out-dir "$OUT/$name" --save-at 7500 15000 22500 "$@" \
        > "logs/study_$name.log" 2>&1
    local ck="$OUT/$name/ckpt_07500.pt $OUT/$name/ckpt_15000.pt $OUT/$name/ckpt_22500.pt $OUT/$name/ckpt.pt"
    $PY scripts/08_eval_offpath.py --ckpts $ck --labels "$name@7.5k" "$name@15k" "$name@22.5k" "$name@30k" \
        --cameras $FRONT --out "$OUT/eval_${name}_front3.json" --samples > "logs/study_eval_$name.log" 2>&1
    $PY scripts/08_eval_offpath.py --ckpts "$OUT/$name/ckpt.pt" --labels "$name@30k" \
        --cameras $ALL6 --out "$OUT/eval_${name}_all6.json" >> "logs/study_eval_$name.log" 2>&1
}

run 6cam
run 6cam_reg --needle-reg 0.01 --opacity-reg 0.005
echo STUDY_AB_DONE
