#!/bin/bash
# Phase 3: does a learned source-dataset embedding help?  Jobs start automatically when CPU cores free up
# (triggered by the end of earlier runs) so the main comparison is not slowed down.
PY="${PY:-python}"; D="${D:-../01_original_social_lstm/social-lstm/data}"; E="${E:-30}"
HOLD="biwi_hotel,crowds_zara02,students003,PETS09-S2L1,bookstore_3,hyang_5,gates_3"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
when () { until [ -f "$1" ]; do sleep 20; done; }     # when <file>  -> block until the file exists
( when runs/official/vanilla_lstm_prof/final_metrics.json
  $PY -m trajpred.train --data-root $D --out-dir runs/official/indiv_tf_src --epochs $E --no-social --source-emb > runs/official/indiv_tf_src.log 2>&1 ) &
( when runs/holdout/indiv_tf/final_metrics.json
  $PY -m trajpred.train --data-root $D --out-dir runs/official/social_tf_src --epochs $E --source-emb > runs/official/social_tf_src.log 2>&1 ) &
( when runs/holdout/vanilla_lstm_prof/final_metrics.json
  $PY -m trajpred.train --data-root $D --out-dir runs/holdout/indiv_tf_src --epochs $E --no-social --source-emb --protocol holdout --holdout $HOLD > runs/holdout/indiv_tf_src.log 2>&1 ) &
wait
echo PHASE3_DONE
