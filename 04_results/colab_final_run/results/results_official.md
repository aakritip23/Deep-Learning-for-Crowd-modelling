**Official split (train -> validation)** - 11403 trajectories, metres, lower is better

| Model | ADE | FDE | **Avg** | vs const. vel. | biwi Avg | crowds Avg | mot Avg | stanford Avg | full-study Avg |
|---|---|---|---|---|---|---|---|---|---|
| Social Transformer (final) | 0.733 | 1.351 | **1.042** | 0.86x | 0.642 | 1.001 | 1.149 | 1.057 | 1.008 |
| Individual Transformer (ablation) | 0.753 | 1.397 | **1.075** | 0.89x | 0.684 | 1.052 | 1.165 | 1.085 | 1.054 |
| Social LSTM, refined (velocity) | 0.781 | 1.465 | **1.123** | 0.93x | 0.733 | 1.091 | 1.225 | 1.136 | 1.105 |
| Constant velocity (reference) | 0.843 | 1.569 | **1.206** | 1.00x | 0.705 | 1.130 | 1.233 | 1.232 | 1.206 |
| Social LSTM, professor's settings (re-impl.) | 1.282 | 2.415 | **1.849** | 1.53x | 1.389 | 1.715 | 2.141 | 1.886 | 2.193 |
