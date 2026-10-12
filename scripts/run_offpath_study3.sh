#!/usr/bin/env bash
# Off-path study, part 3: six cameras with enough budget. Part 1 showed six
# cameras lose less realism per meter of lane shift but, on the same 1M budget
# and 30k steps, fit the front cameras worse. Here: 60k steps (each front camera
# gets as many updates as the three-camera model got in 30k), a 1.2M cap (the
# six-camera LiDAR map is 18% larger), densification until step 30k, and the
# anti-needle loss (free: no cost on the recorded path in part 2).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="conda run --no-capture-output -n pandaset-splat python -u"
D6=data/pandaset_028_6cam
OUT=outputs/study
NAME=6cam_60k
ALL6="front_camera front_left_camera front_right_camera left_camera right_camera back_camera"
mkdir -p "$OUT" logs

$PY scripts/03_train.py --data-dir "$D6" --out-dir "$OUT/$NAME" --steps 60000 --max-gaussians 1200000 \
    --refine-stop 30000 --eval-every 10000 --needle-reg 0.01 --save-at 7500 15000 30000 45000 \
    > "logs/study_$NAME.log" 2>&1
CK="$OUT/$NAME/ckpt_07500.pt $OUT/$NAME/ckpt_15000.pt $OUT/$NAME/ckpt_30000.pt $OUT/$NAME/ckpt_45000.pt $OUT/$NAME/ckpt.pt"
$PY scripts/08_eval_offpath.py --ckpts $CK --labels "$NAME@7.5k" "$NAME@15k" "$NAME@30k" "$NAME@45k" "$NAME@60k" \
    --out "$OUT/eval_${NAME}_front3.json" --samples > "logs/study_eval_$NAME.log" 2>&1
$PY scripts/08_eval_offpath.py --ckpts "$OUT/$NAME/ckpt.pt" "$OUT/6cam/ckpt.pt" --labels "$NAME@60k" "6cam@30k" \
    --cameras $ALL6 --out "$OUT/eval_${NAME}_all6.json" >> "logs/study_eval_$NAME.log" 2>&1
echo STUDY_3_DONE
