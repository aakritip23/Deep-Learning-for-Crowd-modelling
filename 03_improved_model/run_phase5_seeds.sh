#!/bin/bash
# Phase 5: two extra seeds of the individual Transformer (official protocol) -> measures seed-to-seed noise and the
# gain of a seed ensemble (the social model is 3x slower on CPU, so the cheap model is used as the probe).
PY="${PY:-python}"; D="${D:-../01_original_social_lstm/social-lstm/data}"; E="${E:-30}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
for s in 1 2; do
  $PY -m trajpred.train --data-root $D --out-dir runs/official/indiv_tf_s$s --epochs $E --no-social --seed $s > runs/official/indiv_tf_s$s.log 2>&1 &
done
wait
echo PHASE5_DONE
