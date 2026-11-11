import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Single-file analysis of biwi_eth (baseline data folder). Paths are resolved relative to this
# script, and the CSV files below are written next to it, wherever the script is started from.
HERE = Path(__file__).resolve().parent
os.chdir(HERE)
file = str(HERE.parent / "01_original_social_lstm" / "social-lstm" / "data" / "test" / "biwi" / "biwi_eth.txt")

# Read data
data = np.genfromtxt(
    file,
    dtype=float,
    missing_values="?",
    filling_values=np.nan
)

print("Matrix shape:", data.shape)

# Dataset format:
# column 0 = frame
# column 1 = pedestrian ID
# column 2 = X position
# column 3 = Y position

frames = data[:, 0]
ped_ids = data[:, 1]
x = data[:, 2]
y = data[:, 3]

print("Data shape:", data.shape)
print("Unique pedestrian IDs:", np.unique(ped_ids[~np.isnan(ped_ids)]))

# Get unique pedestrian IDs
pedestrians = np.unique(ped_ids[~np.isnan(ped_ids)])

print("Number of pedestrians:", len(pedestrians))
print("Pedestrian IDs:", pedestrians)

print("Frame range:", np.nanmin(frames), "to", np.nanmax(frames))
print("X range:", np.nanmin(x), "to", np.nanmax(x))
print("Y range:", np.nanmin(y), "to", np.nanmax(y))

# -------------------------------
# Basic Dataset Statistics
# -------------------------------

print("\n===== DATASET STATISTICS =====")

print("Total observations:", len(frames))

print("Total unique frames:", len(np.unique(frames[~np.isnan(frames)])))

print("Total pedestrians:", len(pedestrians))

# print("\nObservations per pedestrian:")

# for ped_id in pedestrians:
#     mask = ped_ids == ped_id
#     count = np.sum(mask & ~np.isnan(x) & ~np.isnan(y))
#     print(f"Pedestrian {int(ped_id)}: {count} observations")
# -------------------------------
# Plot all pedestrian trajectories
# -------------------------------

plt.figure(figsize=(10, 8))

for ped_id in pedestrians:

    mask = ped_ids == ped_id

    # Ignore missing coordinates
    valid = mask & ~np.isnan(x) & ~np.isnan(y)

    plt.plot(
        x[valid],
        y[valid],
        marker=".",
        markersize=2,
        linewidth=1,
        label=f"Ped {int(ped_id)}"
    )

plt.xlabel("X")
plt.ylabel("Y")
plt.title("BIWI ETH - Pedestrian Trajectories")
plt.grid(True)
plt.legend()



# ============================================================
# TRAJECTORY LENGTH ANALYSIS
# ============================================================

import pandas as pd

results = []

for ped_id in pedestrians:

    # Select this pedestrian
    mask = ped_ids == ped_id

    # Keep only observations with valid X and Y
    valid = mask & ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(frames)

    ped_frames = frames[valid]
    ped_x = x[valid]
    ped_y = y[valid]

    # Skip if fewer than 2 valid observations
    if len(ped_x) < 2:
        continue

    # Number of valid observations
    num_frames = len(ped_frames)

    # Duration of trajectory
    duration = ped_frames.max() - ped_frames.min()

    # Distance between consecutive points
    dx = np.diff(ped_x)
    dy = np.diff(ped_y)

    distances = np.sqrt(dx**2 + dy**2)

    # Total distance travelled
    total_distance = np.sum(distances)

    # Average distance between consecutive observations
    avg_distance = total_distance / (num_frames - 1)

    # Store results
    results.append({
        "Pedestrian": int(ped_id),
        "Frames": num_frames,
        "Duration": duration,
        "Total Distance": total_distance,
        "Avg Distance/Frame": avg_distance
    })


# Convert results to a table
trajectory_df = pd.DataFrame(results)

# Sort by pedestrian ID
trajectory_df = trajectory_df.sort_values("Pedestrian")

# Display the table
print(trajectory_df)

# Save results
trajectory_df.to_csv(
    "trajectory_statistics.csv",
    index=False
)

print("\nTrajectory statistics saved to trajectory_statistics.csv")

# ============================================================
# SPEED / VELOCITY ANALYSIS
# ============================================================

speed_results = []

for ped_id in pedestrians:

    # Select this pedestrian
    mask = ped_ids == ped_id

    # Keep only valid observations
    valid = mask & ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(frames)

    ped_frames = frames[valid]
    ped_x = x[valid]
    ped_y = y[valid]

    # Sort observations by frame
    order = np.argsort(ped_frames)

    ped_frames = ped_frames[order]
    ped_x = ped_x[order]
    ped_y = ped_y[order]

    # Need at least 2 observations
    if len(ped_x) < 2:
        continue

    # --------------------------------------------------------
    # Difference between consecutive observations
    # --------------------------------------------------------

    dt = np.diff(ped_frames)

    dx = np.diff(ped_x)
    dy = np.diff(ped_y)

    # Avoid division by zero
    valid_dt = dt > 0

    dt = dt[valid_dt]
    dx = dx[valid_dt]
    dy = dy[valid_dt]

    # --------------------------------------------------------
    # Velocity components
    # --------------------------------------------------------

    vx = dx / dt
    vy = dy / dt

    # --------------------------------------------------------
    # Speed
    # --------------------------------------------------------

    speed = np.sqrt(vx**2 + vy**2)

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    for i in range(len(speed)):

        speed_results.append({
            "Pedestrian": int(ped_id),
            "Frame": ped_frames[1:][valid_dt][i],
            "Vx": vx[i],
            "Vy": vy[i],
            "Speed": speed[i]
        })


# Convert to DataFrame
speed_df = pd.DataFrame(speed_results)

# Save individual velocity observations
speed_df.to_csv(
    "pedestrian_speed.csv",
    index=False
)

print("\n===== SPEED ANALYSIS =====")

print(speed_df.head())

print("\nSpeed data saved to pedestrian_speed.csv")

# ============================================================
# SPEED SUMMARY PER PEDESTRIAN
# ============================================================

speed_summary = (
    speed_df
    .groupby("Pedestrian")["Speed"]
    .agg(
        Average_Speed="mean",
        Maximum_Speed="max",
        Minimum_Speed="min"
    )
    .reset_index()
)

print("\n===== SPEED SUMMARY =====")
print(speed_summary)

# Save summary
speed_summary.to_csv(
    "speed_summary.csv",
    index=False
)

print("\nSpeed summary saved to speed_summary.csv")

# ============================================================
# DIRECTION ANALYSIS
# ============================================================

direction_results = []

for ped_id in pedestrians:

    # Select this pedestrian
    mask = ped_ids == ped_id

    # Keep only valid observations
    valid = mask & ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(frames)

    ped_frames = frames[valid]
    ped_x = x[valid]
    ped_y = y[valid]

    # Sort observations by frame
    order = np.argsort(ped_frames)

    ped_frames = ped_frames[order]
    ped_x = ped_x[order]
    ped_y = ped_y[order]

    # Need at least 2 observations
    if len(ped_x) < 2:
        continue

    # Change in position
    dx = np.diff(ped_x)
    dy = np.diff(ped_y)

    # Direction angle in radians
    angles_rad = np.arctan2(dy, dx)

    # Convert radians to degrees
    angles_deg = np.degrees(angles_rad)

    # Store direction for each movement
    for i in range(len(angles_deg)):

        direction_results.append({
            "Pedestrian": int(ped_id),
            "Frame": ped_frames[i + 1],
            "Dx": dx[i],
            "Dy": dy[i],
            "Direction (degrees)": angles_deg[i]
        })


# Convert to DataFrame
direction_df = pd.DataFrame(direction_results)

# Save direction data
direction_df.to_csv(
    "pedestrian_direction.csv",
    index=False
)

print("\n===== DIRECTION ANALYSIS =====")
print(direction_df.head())

print("\nDirection data saved to pedestrian_direction.csv")

# ============================================================
# DIRECTION SUMMARY PER PEDESTRIAN
# ============================================================

direction_summary = (
    direction_df
    .groupby("Pedestrian")["Direction (degrees)"]
    .agg(
        Mean_Direction="mean",
        Direction_Std="std"
    )
    .reset_index()
)

print("\n===== DIRECTION SUMMARY =====")
print(direction_summary)

# Save summary
direction_summary.to_csv(
    "direction_summary.csv",
    index=False
)

print("\nDirection summary saved to direction_summary.csv")

# ============================================================
# PEDESTRIAN INTERACTION ANALYSIS
# ============================================================

interaction_results = []

# Distance threshold for considering another pedestrian "nearby"
NEARBY_DISTANCE = 2.0

# Get all frames that actually contain data
unique_frames = np.unique(frames[~np.isnan(frames)])

for frame in unique_frames:

    # Select valid pedestrians at this frame
    frame_mask = (
        (frames == frame)
        & ~np.isnan(ped_ids)
        & ~np.isnan(x)
        & ~np.isnan(y)
    )

    frame_ped_ids = ped_ids[frame_mask]
    frame_x = x[frame_mask]
    frame_y = y[frame_mask]

    # Need at least 2 pedestrians to calculate interactions
    if len(frame_ped_ids) < 2:
        continue

    # --------------------------------------------------------
    # Calculate interactions for each pedestrian
    # --------------------------------------------------------

    for i in range(len(frame_ped_ids)):

        ped_id = frame_ped_ids[i]

        # Position of current pedestrian
        current_x = frame_x[i]
        current_y = frame_y[i]

        # Calculate distance from this pedestrian
        # to every other pedestrian
        dx = frame_x - current_x
        dy = frame_y - current_y

        distances = np.sqrt(dx**2 + dy**2)

        # Remove distance from pedestrian to themselves
        distances = distances[distances > 0]

        if len(distances) == 0:
            continue

        # Nearest pedestrian
        nearest_distance = np.min(distances)

        # Average distance to all other pedestrians
        average_distance = np.mean(distances)

        # Number of pedestrians within threshold
        nearby_count = np.sum(distances <= NEARBY_DISTANCE)

        interaction_results.append({
            "Frame": frame,
            "Pedestrian": int(ped_id),
            "Nearest Distance": nearest_distance,
            "Average Neighbor Distance": average_distance,
            "Nearby Pedestrians": nearby_count
        })


# Convert to DataFrame
interaction_df = pd.DataFrame(interaction_results)

# Sort
interaction_df = interaction_df.sort_values(
    ["Frame", "Pedestrian"]
)

# Save detailed interaction data
interaction_df.to_csv(
    "pedestrian_interactions.csv",
    index=False
)

print("\n===== PEDESTRIAN INTERACTION ANALYSIS =====")

print(interaction_df.head(20))

print("\nInteraction data saved to pedestrian_interactions.csv")

# ============================================================
# INTERACTION SUMMARY PER PEDESTRIAN
# ============================================================

interaction_summary = (
    interaction_df
    .groupby("Pedestrian")
    .agg(
        Minimum_Nearest_Distance=(
            "Nearest Distance",
            "min"
        ),
        Average_Nearest_Distance=(
            "Nearest Distance",
            "mean"
        ),
        Average_Neighbor_Distance=(
            "Average Neighbor Distance",
            "mean"
        ),
        Maximum_Nearby_Pedestrians=(
            "Nearby Pedestrians",
            "max"
        ),
        Average_Nearby_Pedestrians=(
            "Nearby Pedestrians",
            "mean"
        )
    )
    .reset_index()
)

print("\n===== INTERACTION SUMMARY =====")
print(interaction_summary)

# Save summary
interaction_summary.to_csv(
    "interaction_summary.csv",
    index=False
)

print("\nInteraction summary saved to interaction_summary.csv")

# ============================================================
# CROWD DENSITY ANALYSIS
# ============================================================

# ------------------------------------------------------------
# 1. NUMBER OF PEDESTRIANS PER FRAME
# ------------------------------------------------------------

density_results = []

unique_frames = np.unique(frames[~np.isnan(frames)])

for frame in unique_frames:

    # Valid pedestrians at this frame
    frame_mask = (
        (frames == frame)
        & ~np.isnan(ped_ids)
        & ~np.isnan(x)
        & ~np.isnan(y)
    )

    number_of_pedestrians = np.sum(frame_mask)

    density_results.append({
        "Frame": frame,
        "Number of Pedestrians": number_of_pedestrians
    })


density_df = pd.DataFrame(density_results)

# Sort by frame
density_df = density_df.sort_values("Frame")

# Save
density_df.to_csv(
    "crowd_density_per_frame.csv",
    index=False
)

print("\n===== CROWD DENSITY =====")
print(density_df.head(20))

print("\nCrowd density data saved to crowd_density_per_frame.csv")

# ------------------------------------------------------------
# 2. CROWD DENSITY SUMMARY
# ------------------------------------------------------------

average_crowd = density_df["Number of Pedestrians"].mean()
maximum_crowd = density_df["Number of Pedestrians"].max()
minimum_crowd = density_df["Number of Pedestrians"].min()

print("\n===== CROWD DENSITY SUMMARY =====")

print(f"Average pedestrians per frame: {average_crowd:.2f}")
print(f"Maximum pedestrians in a frame: {maximum_crowd}")
print(f"Minimum pedestrians in a frame: {minimum_crowd}")

# ------------------------------------------------------------
# 3. SPATIAL CROWD DENSITY
# ------------------------------------------------------------

spatial_density_results = []

for frame in unique_frames:

    frame_mask = (
        (frames == frame)
        & ~np.isnan(ped_ids)
        & ~np.isnan(x)
        & ~np.isnan(y)
    )

    frame_x = x[frame_mask]
    frame_y = y[frame_mask]

    number_of_pedestrians = len(frame_x)

    # Need at least 2 pedestrians to define an area
    if number_of_pedestrians < 2:
        continue

    # Bounding box
    x_range = frame_x.max() - frame_x.min()
    y_range = frame_y.max() - frame_y.min()

    area = x_range * y_range

    # Avoid division by zero
    if area <= 0:
        continue

    density = number_of_pedestrians / area

    spatial_density_results.append({
        "Frame": frame,
        "Number of Pedestrians": number_of_pedestrians,
        "Area": area,
        "Spatial Density": density
    })


spatial_density_df = pd.DataFrame(spatial_density_results)

spatial_density_df = spatial_density_df.sort_values("Frame")

# Save
spatial_density_df.to_csv(
    "spatial_crowd_density.csv",
    index=False
)

print("\n===== SPATIAL CROWD DENSITY =====")
print(spatial_density_df.head(20))

print("\nSpatial density data saved to spatial_crowd_density.csv")

# ------------------------------------------------------------
# 4. HIGH / LOW DENSITY FRAMES
# ------------------------------------------------------------

if len(spatial_density_df) > 0:

    highest_density = spatial_density_df.loc[
        spatial_density_df["Spatial Density"].idxmax()
    ]

    lowest_density = spatial_density_df.loc[
        spatial_density_df["Spatial Density"].idxmin()
    ]

    print("\n===== DENSITY EXTREMES =====")

    print("\nHighest density:")
    print(highest_density)

    print("\nLowest density:")
    print(lowest_density)

plt.show()