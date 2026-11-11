#!/bin/bash
# Phase 2: same candidates, but evaluated on 7 whole videos never seen in training (no scene leakage; mimics the
# unseen-video TrajNet test).  Pool = train + validation minus the held-out videos.
PY="${PY:-python}"; D="${D:-../01_original_social_lstm/social-lstm/data}"; O=runs/holdout; E="${E:-30}"
HOLD="biwi_hotel,crowds_zara02,students003,PETS09-S2L1,bookstore_3,hyang_5,gates_3"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
C="--protocol holdout --holdout $HOLD"
$PY -m trajpred.train      --data-root $D --out-dir $O/indiv_tf  --epochs $E --no-social $C > $O/indiv_tf.log 2>&1 &
$PY -m trajpred.train      --data-root $D --out-dir $O/social_tf --epochs $E $C > $O/social_tf.log 2>&1 &
$PY -m trajpred.train_lstm --data-root $D --out-dir $O/social_lstm_prof --epochs $E $C > $O/social_lstm_prof.log 2>&1 &
# the cheap Vanilla LSTM starts once a core is free (when the official individual-Transformer run has finished)
( until [ -f runs/official/indiv_tf/final_metrics.json ]; do sleep 20; done
  $PY -m trajpred.train_lstm --data-root $D --out-dir $O/vanilla_lstm_prof --epochs $E --no-social $C > $O/vanilla_lstm_prof.log 2>&1 ) &
wait
echo ALL_DONE
