# Pedestrian trajectory prediction: a Social Transformer against the Social LSTM baseline (TrajNet)

Deep-learning project on crowd modelling. **Task (TrajNet):** observe a pedestrian for 8 steps (3.2 s) and predict the next 12 steps (4.8 s) - one deterministic path per pedestrian.
It is scored by **ADE** (mean error over the 12 steps), **FDE** (error at the last step) and the TrajNet ranking score **Avg = (ADE + FDE) / 2**, in metres, lower is better.

The reference is the course's Social LSTM (Alahi et al., CVPR 2016; published TrajNet test-server score **Avg 1.3865** = (FDE 2.098 + ADE 0.675) / 2).
This repository contains a **Social Transformer** (0.65 M parameters), the evaluation code, the final run and the study that led to the model choice.

## Result
Final run on a Colab GPU (`04_results/colab_final_run/`, 8 epochs, seed 0), Avg in metres:

| Model | official split (train -> validation) | 7 held-out videos |
|---|---|---|
| Constant velocity (reference) | 1.206 | 1.169 |
| Social LSTM, course configuration (re-implemented) | 1.849 | - |
| Social LSTM, refined (velocity formulation) | 1.123 | - |
| Individual Transformer (no neighbours) | 1.075 | 1.053 |
| **Social Transformer (final)** | **1.042** | **1.028** |

With the full 30-epoch schedule (CPU study, [`03_improved_model/RESULTS.md`](03_improved_model/RESULTS.md)) the Social Transformer reaches **1.008 / 1.020**, while the re-implemented
course-configuration Social LSTM scores 2.193 / 1.833 - the Social Transformer is **54 % / 44 % lower**. Paired bootstrap tests over trajectories put the Social Transformer ahead of
the individual Transformer and of the refined LSTMs, with 95 % intervals that exclude zero, on the held-out videos too when the unique trajectories are resampled (seed-to-seed standard deviation 0.0014, measured on 3 seeds of the individual Transformer).

**How to read these numbers**
* The TrajNet test labels are hidden, so there is **no test-server score yet**; the submission file is `04_results/colab_final_run/submission.zip`. The published 1.3865 is a test-server number and is
  not directly comparable with the validation numbers above. The comparison under identical local conditions is the re-implemented Social LSTM; its held-out score is 1.57x the constant-velocity score,
  consistent with the published 1.55x of the TrajNet linear baseline.
* The *official split* is in-sample and optimistic: its validation files are rotated, noisy copies of training files (the dataset's own augmentation). The *held-out* column (7 whole videos never seen in training;
  2,475 unique trajectories, each present three times as original plus two augmented copies) is the more trustworthy one.
* Single runs per model; the CPU and the GPU run of the same 8-epoch schedule agree within 0.002 Avg (1.044 vs 1.042 official, 1.027 vs 1.028 held-out).

## Repository layout
| Folder | Content |
|---|---|
| [`01_original_social_lstm/`](01_original_social_lstm/) | course-provided Social LSTM reference (3 documented local edits) and the TrajNet text data (131 files) |
| [`02_data/`](02_data/) | exploratory data analysis: scripts, statistics (CSV) and plots |
| [`03_improved_model/`](03_improved_model/) | **the Social Transformer**: package `trajpred/`, training / evaluation scripts, Colab notebook, lab pipeline, tests, `RESULTS.md` (comparison study) |
| [`04_results/`](04_results/) | final Colab run (tables, figures, paired tests, logs, final weights, TrajNet submission) and the archived logs of the 30-epoch CPU study |
| [`05_notes/`](05_notes/) | `Literature_Review.pdf` (phase-1 review) and `research_notes.md` (research on approaches and the TrajNet leaderboard that led to the model choice) |

## Method in short
* **Canonical frame:** origin = last observed position, x-axis = heading, so the model is invariant to translation and rotation. Training augmentation: mirroring, scaling (0.8-1.2), noise.
* **Encoder:** Transformer (3 layers, d = 128, 4 heads) over the 8 observed (position, velocity) steps.
* **Social part:** cross-attention from the pedestrian to its (up to 16) nearest neighbours in the same 20-frame window, with a learned "nobody there" key - it replaces the occupancy-grid pooling of the Social LSTM.
* **Decoder:** MLP producing residual velocities on top of a constant-velocity extrapolation for the 12 future steps (zero-initialised, so training starts at the constant-velocity score).
* **Training:** loss = mean Euclidean error (ADE) + 0.5 x final-step error (FDE), i.e. the ranking metric itself; AdamW, cosine schedule, batch 128. Flip test-time augmentation (about -0.004 Avg) is used for the submission files, not in the tables above.
* **Why not a multimodal / generative model:** TrajNet scores a single prediction per pedestrian, so best-of-K models would not improve the metric.

Main findings (details and significance tests in `03_improved_model/RESULTS.md`):
1. The Social LSTM formulation of the course (absolute positions, teacher forcing) is fragile: validation error is best after 2-3 epochs and then explodes, and even its best epoch is worse than constant velocity.
2. Predicting velocities instead fixes most of it; in that LSTM family the social-pooling grid gives no gain.
3. A Transformer encoder is better still, and neighbour attention adds a further clear gain (-0.033 Avg official, -0.025 held-out in the final run).

## Reproduce
**Google Colab (GPU, about 10 minutes of training):** open `03_improved_model/colab_social_transformer.ipynb` in Colab, choose a GPU runtime and run the cells (Step 1 clones this repository; `REPO_URL` is preset).
`MODE = "quick"` gives the numbers above, `"full"` the 30-epoch schedule.

**Local machine / lab:**
```bash
pip install -r requirements.txt
cd 03_improved_model
python tests/test_basic.py --data-root ../01_original_social_lstm/social-lstm/data         # sanity tests
python colab_run.py --data-root ../01_original_social_lstm/social-lstm/data --mode smoke   # dry run of every stage on a tiny subset (minutes)
python colab_run.py --data-root ../01_original_social_lstm/social-lstm/data --mode quick   # tables, figures, paired tests, submission (hours on a CPU)
DATA=../01_original_social_lstm/social-lstm/data bash run_lab.sh                           # multi-seed ensembles + final fit; see 03_improved_model/README.md
```
**Re-score the archived final weights** (from `03_improved_model/`; official split, Avg 1.042):
```bash
python -m trajpred.predict --data-root ../01_original_social_lstm/social-lstm/data --ckpt ../04_results/colab_final_run/runs/official/social_tf/best.pt
```

## Limitations
* No score on the hidden TrajNet test split; one held-out split of 7 videos; single seeds in the final-run table.
* The re-implemented baseline is not the course code itself (its social neighbourhood was set to 4 m); the LSTM family was only lightly tuned.
* `run_lab.sh` (3-seed ensembles, final fit on train + validation) has been dry-run in smoke mode only.
* On most scenes a pedestrian is alone (median 1 per scene), so the social module can only help on the crowded subset.

## Credits and data
* Baseline: Social LSTM (Alahi et al., CVPR 2016), code from <https://github.com/quancore/social-lstm> (commit `bdddf36`) as provided by the course.
* Data: TrajNet text files as shipped with the baseline - ETH / Hotel (`biwi`), UCY (`crowds`), PETS09 (`mot`) and the Stanford Drone Dataset (`stanford`).
