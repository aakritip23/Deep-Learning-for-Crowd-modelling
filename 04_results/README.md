# Results

Metrics are the TrajNet ones in metres, lower is better: **ADE** (mean error over the 12 predicted steps), **FDE** (error of the last step) and the ranking score
**Avg = (ADE + FDE) / 2**. *official* = train -> validation (in-sample: the validation files are rotated, noisy copies of training files); *held-out* = 7 whole videos never seen in training (the more trustworthy column; 7,425 rows = 2,475 unique trajectories, each present three times).
The ground truth of the TrajNet test split is hidden, so no test score exists.

| Folder | What it is |
|---|---|
| [`colab_final_run/`](colab_final_run/) | **final run of the project**: `03_improved_model/colab_social_transformer.ipynb` on a Colab GPU, `quick` mode, seed 0 |
| [`cpu_full_study/`](cpu_full_study/) | logs / metrics / configs of the 30-epoch CPU study behind `03_improved_model/RESULTS.md` (model comparison, seeds, ablations) |

## `colab_final_run/` (8 epochs; 8.8 minutes of training in total on the Colab GPU)
| Model | official ADE / FDE / **Avg** | held-out ADE / FDE / **Avg** |
|---|---|---|
| Constant velocity (reference) | 0.843 / 1.569 / **1.206** | 0.797 / 1.540 / **1.169** |
| Social LSTM, course configuration (re-implemented, 4 epochs) | 1.282 / 2.415 / **1.849** | - |
| Social LSTM, refined (velocity formulation) | 0.781 / 1.465 / **1.123** | - |
| Individual Transformer (no neighbours) | 0.753 / 1.397 / **1.075** | 0.715 / 1.391 / **1.053** |
| **Social Transformer (final)** | **0.733 / 1.351 / 1.042** | **0.700 / 1.356 / 1.028** |

Paired bootstrap (95 % CI, B - A, negative = B better): Individual -> Social Transformer -0.0329 [-0.0375, -0.0284] (official), -0.0248 [-0.0314, -0.0189] (held-out, rows as in `paired_tests.txt`).
Because every held-out trajectory is present three times, the proper held-out interval resamples the 2,475 unique trajectories (`compare_models.py --cluster`): **-0.0248 [-0.0352, -0.0150]**, still clearly below zero.

| Path | Content |
|---|---|
| `RESULTS_SUMMARY.md` | the tables above per source dataset (biwi, crowds, mot, stanford) + the paired tests |
| `results/` | `fig_comparison.png`, `fig_curves_official.png` (validation Avg per epoch), `fig_per_source_official.png`, `fig_examples_official.png`, `fig_examples_holdout.png` (predicted vs true trajectories); `results_official.md`, `results_holdout.md`, `results.json`, `paired_tests.txt` |
| `runs/<protocol>/<model>/` | `config.json`, `log.csv` (per-epoch validation), `final_metrics.json`; `best.pt` only for `social_tf` (the final weights, 0.65 M parameters) |
| `submission.zip` | the 20 TrajNet test files with every `?` replaced by the prediction of `runs/official/social_tf/best.pt` (+ flip TTA); accepted by `03_improved_model/check_submission.py` (37,776 predicted rows) |

Model names: `social_tf` = Social Transformer, `indiv_tf` = Individual Transformer, `social_lstm_vel` = refined Social LSTM, `social_lstm_prof` = Social LSTM with the course configuration.
The checkpoints of the other runs are not archived; the notebook recreates them.

Re-score the archived weights (from `03_improved_model/`; reproduces the table above):
```bash
python -m trajpred.predict --data-root ../01_original_social_lstm/social-lstm/data --ckpt ../04_results/colab_final_run/runs/official/social_tf/best.pt
```

## `cpu_full_study/` (30 epochs max, CPU, 2026-11-11)
`official/` and `holdout/` contain one folder per run (`config.json`, `log.csv`, `final_metrics.json`) and the console output `<run>.log`; `comparison_tables.md` lists all runs per source dataset.
Run names: `social_tf`, `indiv_tf` (`_s1`, `_s2` = other seeds), `*_src` = with source-dataset embedding, `social_lstm_prof` / `vanilla_lstm_prof` = course configuration (re-implemented),
`*_lstm_vel` = refined velocity formulation, `social_lstm_tuned` = tuned optimiser / dropout (position formulation). Only the two final Social Transformer weights are kept
(`official/social_tf/best.pt` -> Avg 1.008, `holdout/social_tf/best.pt` -> 1.020); the other checkpoints are not archived. The write-up is `03_improved_model/RESULTS.md`.
