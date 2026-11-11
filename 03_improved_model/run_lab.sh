#!/bin/bash
# ============================================================================================================
# Lab pipeline for the Social Transformer (pedestrian trajectory prediction, TrajNet).
#
#   DATA=/path/to/data  bash run_lab.sh              full pipeline (GPU is used automatically if present)
#   SMOKE=1 DATA=/path/to/data  bash run_lab.sh      tiny dry run of EVERY stage (minutes) - run this first
#
# DATA must contain the TrajNet folders  train/  validation/  test/  (= 01_original_social_lstm/social-lstm/data).
# Re-running the same command resumes unfinished runs (<run>/last.pt) and skips finished ones.
#
# Stages
#   1  reporting runs   social model + individual-Transformer ablation x seeds, on two protocols:
#                         official : train -> validation          (as in the baseline repo; same videos in both)
#                         holdout  : 7 whole videos never seen in training (no scene leakage; mimics the test)
#   2  tables           mean +- std over seeds, per source dataset, with the constant-velocity reference
#   3  ensemble         seed-ensemble (+ flip TTA) of the social models vs single models
#   4  final fit        train on train+validation (50% more data) with fixed epochs, FINAL_SEEDS seeds
#   5  submission       ensemble of the final-fit models -> TrajNet test files with every '?' filled in, + checks
# ============================================================================================================
set -euo pipefail

PY="${PY:-python}"
DATA="${DATA:?set DATA=<folder containing train/ validation/ test/>}"
DATA="$(cd "$DATA" && pwd)"          # absolute path (resolved from where you call the script), before we cd below
cd "$(dirname "$0")"
for d in train validation test; do [ -d "$DATA/$d" ] || { echo "ERROR: $DATA/$d not found - DATA must contain train/ validation/ test/"; exit 1; }; done
OUT="${OUT:-runs/lab}"
EPOCHS="${EPOCHS:-30}"                 # same schedule as the CPU comparison (cosine decay over EPOCHS)
BATCH="${BATCH:-128}"
WORKERS="${WORKERS:-4}"
SEEDS="${SEEDS:-0 1 2}"                # stage 1 seeds
FINAL_SEEDS="${FINAL_SEEDS:-0 1 2 3 4}" # stage 4 seeds (the submission ensemble)
ARCH="${ARCH:-}"                      # extra flags shared by all Transformer runs, e.g. ARCH="--d-model 192 --nlayers 4 --ff 384"
                                       # (--source-emb gave no gain for the social model in the CPU comparison; see RESULTS.md)
HOLD="biwi_hotel,crowds_zara02,students003,PETS09-S2L1,bookstore_3,hyang_5,gates_3"

TRAIN_EXTRA=""; EVAL_EXTRA=""
if [ "${SMOKE:-0}" = "1" ]; then
  echo ">>> SMOKE MODE: tiny subset, 1 epoch - checks that every stage runs, not the quality <<<"
  OUT="${OUT}_smoke"; rm -rf "$OUT"
  EPOCHS=1; SEEDS="0 1"; FINAL_SEEDS="0 1"; WORKERS=0
  TRAIN_EXTRA="--max-files-per-source 2 --max-train-samples 600 --batch-size 64"
  EVAL_EXTRA="--max-files-per-source 2"
  BATCH=64
fi
mkdir -p "$OUT/official" "$OUT/holdout" "$OUT/final"

train () {   # train <run-dir> <args...>
  local dir="$1"; shift
  if [ -f "$dir/final_metrics.json" ]; then echo "[skip] $dir already finished"; return; fi
  echo "[train] $dir"
  $PY -m trajpred.train --data-root "$DATA" --out-dir "$dir" --epochs "$EPOCHS" --batch-size "$BATCH" \
      --num-workers "$WORKERS" $ARCH $TRAIN_EXTRA "$@" 2>&1 | tee -a "$dir.log"
}

# ---- optional stage 0: quick hyper-parameter sweep (SWEEP=1) -----------------------------------------------
# One seed, official protocol, social model.  Look at <OUT>/sweep_table.md, put the best flags into ARCH, then
# re-run WITHOUT SWEEP.  (The reporting runs below use ARCH for every Transformer run.)
if [ "${SWEEP:-0}" = "1" ]; then
  mkdir -p "$OUT/sweep"
  while IFS='|' read -r name flags; do
    [ -z "$name" ] && continue
    train "$OUT/sweep/$name" --seed 0 $flags
  done <<'EOF'
base|
wide192x4|--d-model 192 --nlayers 4 --ff 384
wide256x4|--d-model 256 --nlayers 4 --ff 512
lr1e-3|--lr 1e-3
lr2.5e-4|--lr 2.5e-4
dropout0.2|--dropout 0.2
fde1.0|--fde-weight 1.0
maxnbrs32|--max-nbrs 32
EOF
  $PY summarize_runs.py "$OUT/sweep" | tee "$OUT/sweep_table.md"
  echo "Sweep done. Pick the best row, set e.g. ARCH=\"--d-model 192 --nlayers 4 --ff 384\", re-run without SWEEP."
  exit 0
fi

# ---- stage 1: reporting runs ------------------------------------------------------------------------------
for s in $SEEDS; do
  train "$OUT/official/social_tf_s$s" --seed "$s"
  train "$OUT/official/indiv_tf_s$s"  --seed "$s" --no-social
  train "$OUT/holdout/social_tf_s$s"  --seed "$s" --protocol holdout --holdout "$HOLD"
  train "$OUT/holdout/indiv_tf_s$s"   --seed "$s" --no-social --protocol holdout --holdout "$HOLD"
done

# ---- stage 2: tables --------------------------------------------------------------------------------------
$PY summarize_runs.py "$OUT/official" "$OUT/holdout" --group-seeds | tee "$OUT/results_tables.md"

# ---- stage 3: ensemble / TTA ------------------------------------------------------------------------------
for proto in official holdout; do
  ck=$(ls "$OUT/$proto"/social_tf_s*/best.pt)
  extra=""; [ "$proto" = "holdout" ] && extra="--protocol holdout --holdout $HOLD"
  first=$(echo $ck | cut -d' ' -f1)
  {
    echo "== social_tf $proto : single model (seed 0)";   $PY -m trajpred.predict --data-root "$DATA" --ckpt "$first" $extra $EVAL_EXTRA | tail -1
    echo "== social_tf $proto : single model + flip TTA"; $PY -m trajpred.predict --data-root "$DATA" --ckpt "$first" --tta $extra $EVAL_EXTRA | tail -1
    echo "== social_tf $proto : seed ensemble + flip TTA"; $PY -m trajpred.predict --data-root "$DATA" --ckpt $ck --tta $extra $EVAL_EXTRA | tail -1
  } | tee "$OUT/ensemble_$proto.txt"
done

# ---- stage 4: final fit on train+validation ---------------------------------------------------------------
# NOTE: with --protocol full the "validation" numbers printed during training are IN-SAMPLE (monitoring only);
# the last-epoch weights are kept (--select last), so no information from validation is used to pick checkpoints.
for s in $FINAL_SEEDS; do
  train "$OUT/final/social_full_s$s" --seed "$s" --protocol full --select last
done

# ---- stage 5: submission ----------------------------------------------------------------------------------
$PY -m trajpred.predict --data-root "$DATA" --ckpt $(ls "$OUT"/final/social_full_s*/best.pt) --tta \
    --write-submission "$OUT/submission" $EVAL_EXTRA
$PY check_submission.py --data-root "$DATA" --submission "$OUT/submission" ${EVAL_EXTRA:+--max-files-per-source 2}
echo
echo "DONE.  Tables: $OUT/results_tables.md | ensemble: $OUT/ensemble_*.txt | submission: $OUT/submission/"
