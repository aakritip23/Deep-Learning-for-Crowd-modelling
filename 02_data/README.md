# Exploratory data analysis (TrajNet text files)

Statistics and plots of the TrajNet data shipped with the baseline (`../01_original_social_lstm/social-lstm/data`; the path is found automatically).

| File | Purpose |
|---|---|
| `eda_all_datasets.py` | statistics for all 131 files - observations, frames, pedestrians, trajectory distance / duration, speed, direction consistency, nearest-neighbour distance, crowd size and local density -> `dataset_comparison.csv` (one row per file) |
| `visualize_eda.py` | bar charts and scatter plots from `dataset_comparison.csv` -> `plots/*.png` |
| `plot_myown.py` | the same statistics step by step for a single file (`biwi_eth`) -> `trajectory_statistics.csv`, `pedestrian_speed.csv`, `speed_summary.csv`, `pedestrian_direction.csv`, `direction_summary.csv`, `pedestrian_interactions.csv`, `interaction_summary.csv`, `crowd_density_per_frame.csv`, `spatial_crowd_density.csv`; opens a plot window |
| `*.csv`, `plots/` | the stored outputs of the three scripts |

```bash
pip install -r ../requirements.txt      # numpy, pandas, matplotlib
python eda_all_datasets.py              # -> dataset_comparison.csv
python visualize_eda.py                 # -> plots/
python plot_myown.py                    # single-file analysis (biwi_eth)
```
