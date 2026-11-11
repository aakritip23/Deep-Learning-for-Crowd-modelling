# Social Transformer - results (mode: quick, epochs: {'tf': 8, 'lstm_prof': 4, 'lstm_vel': 8})


Metrics are TrajNet ADE / FDE / Avg = (ADE+FDE)/2 in metres (lower is better). 'vs const. vel.' = Avg divided by the constant-velocity Avg. 'full-study Avg' = the 30-epoch CPU study in RESULTS.md (a short run is expected to be slightly worse).


**Official split (train -> validation)** - 11403 trajectories, metres, lower is better

| Model | ADE | FDE | **Avg** | vs const. vel. | biwi Avg | crowds Avg | mot Avg | stanford Avg | full-study Avg |
|---|---|---|---|---|---|---|---|---|---|
| Social Transformer (final) | 0.733 | 1.351 | **1.042** | 0.86x | 0.642 | 1.001 | 1.149 | 1.057 | 1.008 |
| Individual Transformer (ablation) | 0.753 | 1.397 | **1.075** | 0.89x | 0.684 | 1.052 | 1.165 | 1.085 | 1.054 |
| Social LSTM, refined (velocity) | 0.781 | 1.465 | **1.123** | 0.93x | 0.733 | 1.091 | 1.225 | 1.136 | 1.105 |
| Constant velocity (reference) | 0.843 | 1.569 | **1.206** | 1.00x | 0.705 | 1.130 | 1.233 | 1.232 | 1.206 |
| Social LSTM, professor's settings (re-impl.) | 1.282 | 2.415 | **1.849** | 1.53x | 1.389 | 1.715 | 2.141 | 1.886 | 2.193 |

**Held-out videos (7 unseen videos)** - 7425 trajectories, metres, lower is better

| Model | ADE | FDE | **Avg** | vs const. vel. | biwi Avg | crowds Avg | mot Avg | stanford Avg | full-study Avg |
|---|---|---|---|---|---|---|---|---|---|
| Social Transformer (final) | 0.700 | 1.356 | **1.028** | 0.88x | 0.620 | 1.009 | 1.113 | 1.090 | 1.020 |
| Individual Transformer (ablation) | 0.715 | 1.391 | **1.053** | 0.90x | 0.626 | 1.040 | 1.121 | 1.113 | 1.044 |
| Constant velocity (reference) | 0.797 | 1.540 | **1.169** | 1.00x | 0.655 | 1.127 | 1.217 | 1.268 | 1.169 |


```
Paired bootstrap over trajectories: diff = B - A in Avg (negative = B better), 95% CI
official Individual TF -> Social TF           ALL        n= 11403  A 1.075  B 1.042  diff(B-A) -0.0329 [-0.0375, -0.0284]  B better on 54%  -> B better
official Refined Social LSTM -> Social TF     ALL        n= 11403  A 1.123  B 1.042  diff(B-A) -0.0815 [-0.0912, -0.0715]  B better on 59%  -> B better
holdout  Individual TF -> Social TF           ALL        n=  7425  A 1.053  B 1.028  diff(B-A) -0.0248 [-0.0314, -0.0189]  B better on 52%  -> B better
```


Figures: fig_comparison.png, fig_curves_official.png, fig_per_source_official.png, fig_examples_official.png, fig_examples_holdout.png (folder results/).
