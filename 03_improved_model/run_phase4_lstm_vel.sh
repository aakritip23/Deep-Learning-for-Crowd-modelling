#!/bin/bash
# Phase 4: refined (velocity-formulation) Social / Vanilla LSTM - the fair "improve the existing model" candidate.
# Jobs start automatically when earlier runs have finished and freed a core.
PY="${PY:-python}"; D="${D:-../01_original_social_lstm/social-lstm/data}"; E="${E:-20}"
HOLD="biwi_hotel,crowds_zara02,students003,PETS09-S2L1,bookstore_3,hyang_5,gates_3"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
A="--mode vel --optim adam --lr 1e-3 --dropout 0.1 --epochs $E --patience 6"
when () { until [ -f "$1" ]; do sleep 20; done; }
( when runs/official/indiv_tf_src/final_metrics.json
  $PY -m trajpred.train_lstm --data-root $D --out-dir runs/official/social_lstm_vel $A > runs/official/social_lstm_vel.log 2>&1 ) &
( when runs/official/social_lstm_prof/final_metrics.json
  $PY -m trajpred.train_lstm --data-root $D --out-dir runs/official/vanilla_lstm_vel $A --no-social > runs/official/vanilla_lstm_vel.log 2>&1
  $PY -m trajpred.train_lstm --data-root $D --out-dir runs/holdout/vanilla_lstm_vel $A --no-social --protocol holdout --holdout $HOLD > runs/holdout/vanilla_lstm_vel.log 2>&1 ) &
( when runs/holdout/social_lstm_prof/final_metrics.json
  $PY -m trajpred.train_lstm --data-root $D --out-dir runs/holdout/social_lstm_vel $A --protocol holdout --holdout $HOLD > runs/holdout/social_lstm_vel.log 2>&1 ) &
wait
echo PHASE4_DONE
