
"""
Visualize EDA outputs from eda_all_datasets.py.

Run from the 02_data folder:
    python visualize_eda.py

Input:
    dataset_comparison.csv

Output:
    plots/*.png

This script visualizes existing results; it does not rerun EDA calculations.
"""

from pathlib import Path
import warnings

import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

DATA_FOLDER = Path(__file__).resolve().parent
INPUT_FILE = DATA_FOLDER / "dataset_comparison.csv"
PLOT_FOLDER = DATA_FOLDER / "plots"
PLOT_FOLDER.mkdir(exist_ok=True)


def save_bar_chart(df, value_column, title, y_label, output_name, top_n=20):
    """Save a bar chart of a metric for individual datasets."""
    if value_column not in df.columns:
        print(f"Skipping '{title}': column '{value_column}' was not found.")
        return

    plot_df = df[["Dataset", "Scene", "Split", value_column]].copy()
    plot_df[value_column] = pd.to_numeric(plot_df[value_column], errors="coerce")
    plot_df = plot_df.dropna(subset=[value_column])

    if plot_df.empty:
        print(f"Skipping '{title}': no valid values.")
        return

    # Keep the chart readable when there are many datasets.
    plot_df = plot_df.sort_values(value_column, ascending=False).head(top_n)
    plot_df["Label"] = plot_df["Dataset"] + " (" + plot_df["Scene"] + ")"
    plot_df = plot_df.sort_values(value_column, ascending=True)

    fig, ax = plt.subplots(figsize=(11, max(5, 0.35 * len(plot_df))))
    ax.barh(plot_df["Label"], plot_df[value_column])
    ax.set_title(title)
    ax.set_xlabel(y_label)
    ax.set_ylabel("Dataset (scene)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    output_path = PLOT_FOLDER / output_name
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def save_scatter(df, x_column, y_column, title, x_label, y_label, output_name):
    """Save a scatter plot to inspect relationships between two metrics."""
    missing = [col for col in (x_column, y_column) if col not in df.columns]
    if missing:
        print(f"Skipping '{title}': missing column(s): {', '.join(missing)}.")
        return

    plot_df = df[[x_column, y_column, "Scene"]].copy()
    plot_df[x_column] = pd.to_numeric(plot_df[x_column], errors="coerce")
    plot_df[y_column] = pd.to_numeric(plot_df[y_column], errors="coerce")
    plot_df = plot_df.dropna(subset=[x_column, y_column])

    if plot_df.empty:
        print(f"Skipping '{title}': no valid values.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    for scene, group in plot_df.groupby("Scene"):
        ax.scatter(group[x_column], group[y_column], label=str(scene), alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    if plot_df["Scene"].nunique() > 1:
        ax.legend(title="Scene")
    fig.tight_layout()

    output_path = PLOT_FOLDER / output_name
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_FILE}\n"
            "Run eda_all_datasets.py first to generate dataset_comparison.csv."
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = {"Dataset", "Scene", "Split"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            "dataset_comparison.csv is missing required columns: "
            + ", ".join(sorted(missing))
        )

    print(f"Loaded: {INPUT_FILE}")
    print(f"Datasets found: {len(df)}")
    print(f"Plots will be saved to: {PLOT_FOLDER}\n")

    # Dataset-level comparisons. Charts show the highest values for readability.
    save_bar_chart(
        df, "Average Speed",
        "Average pedestrian speed by dataset",
        "Average speed (coordinate units per frame, unless calibrated)",
        "01_average_speed_by_dataset.png",
    )

    save_bar_chart(
        df, "Average Trajectory Distance",
        "Average trajectory distance by dataset",
        "Average trajectory distance (coordinate units)",
        "02_trajectory_distance_by_dataset.png",
    )

    save_bar_chart(
        df, "Average Pedestrians/Frame",
        "Average number of pedestrians per frame",
        "Average pedestrians per frame",
        "03_average_crowd_size_by_dataset.png",
    )

    save_bar_chart(
        df, "Average Local Density",
        "Average local density by dataset",
        "Local density (coordinate-space estimate; radius used in EDA = 2)",
        "04_local_density_by_dataset.png",
    )

    save_scatter(
        df, "Average Speed", "Average Nearest Distance",
        "Average speed vs. nearest-pedestrian distance",
        "Average speed (coordinate units per frame)",
        "Average nearest-pedestrian distance (coordinate units)",
        "05_speed_vs_nearest_distance.png",
    )

    save_scatter(
        df, "Average Speed", "Average Local Density",
        "Average speed vs. local crowd density",
        "Average speed (coordinate units per frame)",
        "Average local density (coordinate-space estimate)",
        "06_speed_vs_local_density.png",
    )

    print("\nVisualization complete.")
    print("Open the 'plots' folder to view the PNG files.")
    print("Interpret comparisons cautiously: coordinate scales and frame rates may differ by dataset.")


if __name__ == "__main__":
    main()
