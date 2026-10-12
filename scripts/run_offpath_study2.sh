#!/usr/bin/env bash
# Off-path study, part 2: one change at a time on the three-camera setup
# (outputs/train is the reference: SH degree 3, no shape regularizer).
#   sh1:    view-dependent color limited to SH degree 1 (degree 3 extrapolates
#           to wrong, saturated colors at viewing angles the cameras never had)
#   needle: anti-needle shape loss only (no opacity regularizer, which pruned
#           far too many Gaussians in part 1)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="conda run --no-capture-output -n pandaset-splat python -u"
D3=data/pandaset_028
OUT=outputs/study
mkdir -p "$OUT" logs

run() {   # name, extra training args...
    local name=$1; shift
    $PY scripts/03_train.py --data-dir "$D3" --out-dir "$OUT/$name" --save-at 7500 15000 22500 "$@" \
        > "logs/study_$name.log" 2>&1
    local ck="$OUT/$name/ckpt_07500.pt $OUT/$name/ckpt_15000.pt $OUT/$name/ckpt_22500.pt $OUT/$name/ckpt.pt"
    $PY scripts/08_eval_offpath.py --ckpts $ck --labels "$name@7.5k" "$name@15k" "$name@22.5k" "$name@30k" \
        --out "$OUT/eval_${name}_front3.json" --samples > "logs/study_eval_$name.log" 2>&1
}

run sh1 --sh-degree 1
run needle --needle-reg 0.01
echo STUDY_2_DONE
