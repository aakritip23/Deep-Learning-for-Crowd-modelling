# Social Transformer for pedestrian trajectory prediction (TrajNet)

Goal: beat the Social LSTM reference (`../01_original_social_lstm`; Avg 1.3865 / FDE 2.098 / ADE 0.675 on the TrajNet test server).
Results: [`RESULTS.md`](RESULTS.md) (30-epoch CPU study: model comparison, significance tests, limitations) and the final Colab-GPU run in
[`../04_results/`](../04_results/); research background in [`../05_notes/research_notes.md`](../05_notes/research_notes.md).

Task: observe 8 steps (3.2 s), predict the next 12 (4.8 s) for every pedestrian.  Metrics: ADE (= MAD, mean over the 12 steps),
FDE (= FAD, last step) and **Avg = (ADE + FDE) / 2**, the TrajNet ranking score.  One deterministic prediction per pedestrian.

## Layout
| File | Purpose |
|---|---|
| `trajpred/data.py` | parsing, scene grouping, canonical frame (origin = last observed point, x-axis = heading), neighbours, augmentation, split protocols |
| `trajpred/models.py` | **Social Transformer**: Transformer encoder + neighbour cross-attention + residual-over-constant-velocity decoder |
| `trajpred/train.py` | training / validation loop, checkpoint + auto-resume, CSV log, early stopping |
| `trajpred/predict.py` | seed ensemble + flip TTA, scoring, TrajNet submission writer |
| `trajpred/reference_lstm.py`, `train_lstm.py` | Social / Vanilla LSTM re-implemented in the same harness (abs = original formulation, vel = refined) |
| `trajpred/baselines.py`, `metrics.py` | constant-velocity references, ADE/FDE/Avg per source |
| `trajpred/viz.py` | report figures (comparison, curves, per-source, example trajectories) |
| `colab_run.py` + `colab_social_transformer.ipynb` | **one-command pipeline** (also the Colab notebook): train -> tables -> figures -> paired tests -> TrajNet submission (modes smoke / quick / full) |
| `run_lab.sh` | **the lab pipeline** (see below) |
| `compare_models.py` | paired comparison of two models/ensembles with a bootstrap confidence interval |
| `summarize_runs.py` | comparison tables over many runs (mean +/- std over seeds with `--group-seeds`) |
| `check_submission.py` | validates a TrajNet submission directory |
| `evaluate_baselines.py` | numpy-only scoring of constant-velocity baselines |
| `tests/test_basic.py` | sanity tests |
| `run_phase*.sh` | the exact CPU comparison runs that produced `RESULTS.md` (their logs and metrics are archived in `../04_results/cpu_full_study/`) |

## Protocols
* `official`  - train -> validation (as in the baseline repo). The validation files are rotated, noisy copies of training files (the dataset's augmentation: same frames and pedestrian ids), so this protocol is in-sample and optimistic.
* `holdout`   - 7 whole videos (`biwi_hotel, crowds_zara02, students003, PETS09-S2L1, bookstore_3, hyang_5, gates_3`, ~22 % of all trajectories) are never seen in training; mimics the unseen-video test set.
  Each held-out trajectory is present three times (original + two augmented copies; 7,425 rows = 2,475 unique trajectories), so use `compare_models.py --cluster` for confidence intervals.
* `full`      - final fit on train + validation (validation becomes in-sample; last-epoch weights are kept, nothing is selected on validation).
* The test split has hidden ground truth (`?` in the files): it can only be predicted and submitted, not scored locally.

## Quick start (run from this folder)
```bash
pip install -r ../requirements.txt        # numpy, torch, matplotlib (pandas is only needed for ../02_data)
python tests/test_basic.py --data-root ../01_original_social_lstm/social-lstm/data     # sanity tests (seconds)
python colab_run.py --data-root ../01_original_social_lstm/social-lstm/data --mode smoke   # dry run of every stage on a tiny subset
python colab_run.py --data-root ../01_original_social_lstm/social-lstm/data --mode quick   # 8-epoch run: tables, figures, paired tests, submission
```
Everything is written to `results_colab/` (change with `--out`); `--no-comparison` trains only the final model, `--no-holdout` skips the held-out-videos protocol.
Re-running the same command resumes: finished runs are skipped.

## Colab usage (run the finalised model yourself)
Open `colab_social_transformer.ipynb` in Colab (GPU runtime) and run the cells top to bottom. Step 1 clones this repository (`REPO_URL` is preset)
or, with `SOURCE = "upload"`, takes the repository zip (GitHub: `Code > Download ZIP`); the TrajNet text data are part of the repository.
`MODE = "quick"` (8 epochs, about 9 minutes of training on a Colab GPU for everything) is the fast test, `"full"` reproduces `RESULTS.md`.
The run of this repository is archived in [`../04_results/colab_final_run/`](../04_results/colab_final_run/).

## Lab usage
```bash
DATA=../01_original_social_lstm/social-lstm/data SMOKE=1 bash run_lab.sh   # 1) dry run of every stage on a tiny subset (minutes) - always do this first
DATA=../01_original_social_lstm/social-lstm/data bash run_lab.sh           # 2) full pipeline; GPU is used automatically; re-run the same command to resume
```
`DATA` must contain `train/ validation/ test/`.  Useful overrides:
`SEEDS="0 1 2"`, `FINAL_SEEDS="0 1 2 3 4"`, `EPOCHS=30`, `BATCH=128`, `WORKERS=4`, `OUT=runs/lab`,
`ARCH="--source-emb --d-model 192 --nlayers 4 --ff 384"` (model flags shared by all Transformer runs).
Outputs: `<OUT>/results_tables.md` (mean +/- std over seeds, per source), `<OUT>/ensemble_*.txt`,
`<OUT>/final/*` (final-fit models) and `<OUT>/submission/` (TrajNet test files with every `?` filled; checked automatically).
`run_lab.sh` has so far only been dry-run in smoke mode; the full lab run (multi-seed ensemble, final fit on train + validation) has not been executed.

## Design notes
* The model predicts **residuals over a constant-velocity extrapolation** (zero-initialised => starts at the constant-velocity score).
* Loss = mean Euclidean distance + 0.5 x final-step distance, i.e. it optimises the ADE/FDE ranking metric directly.
* Neighbours = pedestrians sharing the same 20-frame window (nearest 16); most scenes contain a single pedestrian, so the social module can only help on a subset - `--no-social` is the ablation.
* Nothing depends on a GPU: `--device auto`; checkpoints in `<run>/last.pt` (resumes automatically), best weights in `<run>/best.pt`.
