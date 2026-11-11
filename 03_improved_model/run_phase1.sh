#!/bin/bash
# Phase 1: all candidates on the official protocol (train -> validation), same data/metrics. CPU-friendly, 1 thread each.
PY="${PY:-python}"; D="${D:-../01_original_social_lstm/social-lstm/data}"; O=runs/official; E="${E:-30}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
$PY -m trajpred.train_lstm --data-root $D --out-dir $O/social_lstm_prof  --epochs $E > $O/social_lstm_prof.log 2>&1 &
$PY -m trajpred.train_lstm --data-root $D --out-dir $O/vanilla_lstm_prof --epochs $E --no-social > $O/vanilla_lstm_prof.log 2>&1 &
$PY -m trajpred.train_lstm --data-root $D --out-dir $O/social_lstm_tuned --epochs $E --optim adam --lr 1e-3 --dropout 0.1 > $O/social_lstm_tuned.log 2>&1 &
$PY -m trajpred.train      --data-root $D --out-dir $O/indiv_tf  --epochs $E --no-social > $O/indiv_tf.log 2>&1 &
$PY -m trajpred.train      --data-root $D --out-dir $O/social_tf --epochs $E > $O/social_tf.log 2>&1 &
wait
echo ALL_DONE
