from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ======================
# ======================================

# TrajNet text files shipped with the baseline (train/ validation/ test/); resolved relative to this script
DATA_ROOT = (
    Path(__file__).resolve().parent.parent
    / "01_original_social_lstm" / "social-lstm" / "data"
)

NEARBY_DISTANCE = 2.0
DENSITY_RADIUS = 2.0


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data(file_path):

    data = np.genfromtxt(
        file_path,
        dtype=float,
        missing_values="?",
        filling_values=np.nan
    )

    frames = data[:, 0]
    ped_ids = data[:, 1]
    x = data[:, 2]
    y = data[:, 3]

    return frames, ped_ids, x, y


# ============================================================
# 2. BASIC DATASET STATISTICS
# ============================================================

def basic_stats(frames, ped_ids, x, y):

    valid_ped_ids = ped_ids[~np.isnan(ped_ids)]

    pedestrians = np.unique(valid_ped_ids)

    valid_frames = frames[~np.isnan(frames)]

    return {
        "Total Observations": len(frames),
        "Unique Frames": len(np.unique(valid_frames)),
        "Pedestrians": len(pedestrians),
        "X Min": np.nanmin(x),
        "X Max": np.nanmax(x),
        "Y Min": np.nanmin(y),
        "Y Max": np.nanmax(y)
    }


# ============================================================
# 3. TRAJECTORY STATISTICS
# ============================================================

def trajectory_stats(frames, ped_ids, x, y):

    pedestrians = np.unique(
        ped_ids[~np.isnan(ped_ids)]
    )

    results = []

    for ped_id in pedestrians:

        mask = ped_ids == ped_id

        valid = (
            mask
            & ~np.isnan(x)
            & ~np.isnan(y)
            & ~np.isnan(frames)
        )

        ped_frames = frames[valid]
        ped_x = x[valid]
        ped_y = y[valid]

        if len(ped_x) < 2:
            continue

        # Sort by frame
        order = np.argsort(ped_frames)

        ped_frames = ped_frames[order]
        ped_x = ped_x[order]
        ped_y = ped_y[order]

        # Number of observations
        num_observations = len(ped_x)

        # Duration
        duration = (
            ped_frames[-1] - ped_frames[0]
        )

        # Distance travelled
        dx = np.diff(ped_x)
        dy = np.diff(ped_y)

        distances = np.sqrt(
            dx**2 + dy**2
        )

        total_distance = np.sum(distances)

        avg_distance = (
            total_distance /
            (num_observations - 1)
        )

        results.append({
            "Pedestrian": int(ped_id),
            "Observations": num_observations,
            "Duration": duration,
            "Total Distance": total_distance,
            "Avg Distance/Step": avg_distance
        })

    return pd.DataFrame(results)


# ============================================================
# 4. SPEED / VELOCITY STATISTICS
# ============================================================

def speed_stats(frames, ped_ids, x, y):

    pedestrians = np.unique(
        ped_ids[~np.isnan(ped_ids)]
    )

    results = []

    for ped_id in pedestrians:

        mask = ped_ids == ped_id

        valid = (
            mask
            & ~np.isnan(x)
            & ~np.isnan(y)
            & ~np.isnan(frames)
        )

        ped_frames = frames[valid]
        ped_x = x[valid]
        ped_y = y[valid]

        if len(ped_x) < 2:
            continue

        # Sort by frame
        order = np.argsort(ped_frames)

        ped_frames = ped_frames[order]
        ped_x = ped_x[order]
        ped_y = ped_y[order]

        # Differences
        dt = np.diff(ped_frames)
        dx = np.diff(ped_x)
        dy = np.diff(ped_y)

        # Remove zero/invalid time differences
        valid_dt = dt > 0

        dt = dt[valid_dt]
        dx = dx[valid_dt]
        dy = dy[valid_dt]

        if len(dt) == 0:
            continue

        vx = dx / dt
        vy = dy / dt

        speed = np.sqrt(
            vx**2 + vy**2
        )

        results.append({
            "Pedestrian": int(ped_id),
            "Average Speed": np.mean(speed),
            "Maximum Speed": np.max(speed),
            "Minimum Speed": np.min(speed)
        })

    return pd.DataFrame(results)


# ============================================================
# 5. DIRECTION STATISTICS
# ============================================================

def direction_stats(frames, ped_ids, x, y):

    pedestrians = np.unique(
        ped_ids[~np.isnan(ped_ids)]
    )

    results = []

    for ped_id in pedestrians:

        mask = ped_ids == ped_id

        valid = (
            mask
            & ~np.isnan(x)
            & ~np.isnan(y)
            & ~np.isnan(frames)
        )

        ped_frames = frames[valid]
        ped_x = x[valid]
        ped_y = y[valid]

        if len(ped_x) < 2:
            continue

        # Sort by frame
        order = np.argsort(ped_frames)

        ped_x = ped_x[order]
        ped_y = ped_y[order]

        # Movement between consecutive observations
        dx = np.diff(ped_x)
        dy = np.diff(ped_y)

        # Direction in radians
        angles_rad = np.arctan2(dy, dx)

        # ----------------------------------------------------
        # Circular mean
        # ----------------------------------------------------

        mean_sin = np.mean(np.sin(angles_rad))
        mean_cos = np.mean(np.cos(angles_rad))

        mean_direction_rad = np.arctan2(
            mean_sin,
            mean_cos
        )

        mean_direction_deg = np.degrees(
            mean_direction_rad
        )

        # ----------------------------------------------------
        # Resultant vector length
        # ----------------------------------------------------
        #
        # R close to 1  -> directions are consistent
        # R close to 0  -> directions are highly dispersed
        #

        R = np.sqrt(
            mean_sin**2 +
            mean_cos**2
        )

        # ----------------------------------------------------
        # Circular standard deviation
        # ----------------------------------------------------

        if R > 0:

            circular_std_rad = np.sqrt(
                -2 * np.log(R)
            )

            circular_std_deg = np.degrees(
                circular_std_rad
            )

        else:

            circular_std_deg = np.nan

        results.append({
            "Pedestrian": int(ped_id),
            "Mean Direction": mean_direction_deg,
            "Direction Consistency": R,
            "Circular Direction Std": circular_std_deg
        })

    return pd.DataFrame(results)

# ============================================================
# 6. PEDESTRIAN INTERACTION STATISTICS
# ============================================================

def interaction_stats(frames, ped_ids, x, y):

    unique_frames = np.unique(
        frames[~np.isnan(frames)]
    )

    results = []

    for frame in unique_frames:

        frame_mask = (
            (frames == frame)
            & ~np.isnan(ped_ids)
            & ~np.isnan(x)
            & ~np.isnan(y)
        )

        frame_ped_ids = ped_ids[frame_mask]
        frame_x = x[frame_mask]
        frame_y = y[frame_mask]

        # Need at least 2 pedestrians
        if len(frame_ped_ids) < 2:
            continue

        for i in range(len(frame_ped_ids)):

            ped_id = frame_ped_ids[i]

            dx = frame_x - frame_x[i]
            dy = frame_y - frame_y[i]

            distances = np.sqrt(
                dx**2 + dy**2
            )

            # Remove self-distance
            distances = distances[
                distances > 0
            ]

            if len(distances) == 0:
                continue

            nearest_distance = np.min(
                distances
            )

            average_distance = np.mean(
                distances
            )

            nearby_count = np.sum(
                distances <= NEARBY_DISTANCE
            )

            results.append({
                "Frame": frame,
                "Pedestrian": int(ped_id),
                "Nearest Distance": nearest_distance,
                "Average Neighbor Distance": average_distance,
                "Nearby Pedestrians": nearby_count
            })

    detailed_df = pd.DataFrame(results)

    if detailed_df.empty:
        return pd.DataFrame()

    summary_df = (
        detailed_df
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

    return summary_df


# ============================================================
# 7. CROWD DENSITY STATISTICS
# ============================================================

def density_stats(frames, ped_ids, x, y):

    unique_frames = np.unique(
        frames[~np.isnan(frames)]
    )

    

    results = []

    for frame in unique_frames:

        frame_mask = (
            (frames == frame)
            & ~np.isnan(ped_ids)
            & ~np.isnan(x)
            & ~np.isnan(y)
        )

        frame_ped_ids = ped_ids[frame_mask]
        frame_x = x[frame_mask]
        frame_y = y[frame_mask]

        number_of_pedestrians = len(
            frame_ped_ids
        )

        if number_of_pedestrians < 2:
            continue

        # ----------------------------------------------------
        # Calculate local density for every pedestrian
        # ----------------------------------------------------

        local_densities = []

        for i in range(number_of_pedestrians):

            dx = frame_x - frame_x[i]
            dy = frame_y - frame_y[i]

            distances = np.sqrt(
                dx**2 + dy**2
            )

            # Remove self
            distances = distances[
                distances > 0
            ]

            # Number of pedestrians within radius
            nearby_count = np.sum(
                distances <= DENSITY_RADIUS
            )

            # Area of circular neighborhood
            area = np.pi * (
                DENSITY_RADIUS ** 2
            )

            local_density = (
                nearby_count / area
            )

            local_densities.append(
                local_density
            )

        results.append({
            "Frame": frame,
            "Number of Pedestrians":
                number_of_pedestrians,

            "Average Local Density":
                np.mean(local_densities),

            "Maximum Local Density":
                np.max(local_densities)
        })

    density_df = pd.DataFrame(results)

    if density_df.empty:

        return {
            "Average Pedestrians/Frame": np.nan,
            "Maximum Pedestrians/Frame": np.nan,
            "Average Local Density": np.nan,
            "Maximum Local Density": np.nan
        }

    return {
        "Average Pedestrians/Frame":
            density_df[
                "Number of Pedestrians"
            ].mean(),

        "Maximum Pedestrians/Frame":
            density_df[
                "Number of Pedestrians"
            ].max(),

        "Average Local Density":
            density_df[
                "Average Local Density"
            ].mean(),

        "Maximum Local Density":
            density_df[
                "Maximum Local Density"
            ].max()
    }


# ============================================================
# 8. ANALYZE ONE FILE
# ============================================================

def analyze_file(file_path):

    frames, ped_ids, x, y = load_data(
        file_path
    )

    # Basic statistics
    basic = basic_stats(
        frames,
        ped_ids,
        x,
        y
    )

    # Trajectory
    trajectory = trajectory_stats(
        frames,
        ped_ids,
        x,
        y
    )

    # Speed
    speed = speed_stats(
        frames,
        ped_ids,
        x,
        y
    )

    # Direction
    direction = direction_stats(
        frames,
        ped_ids,
        x,
        y
    )

    # Interaction
    interaction = interaction_stats(
        frames,
        ped_ids,
        x,
        y
    )

    # Density
    density = density_stats(
        frames,
        ped_ids,
        x,
        y
    )

    # --------------------------------------------------------
    # Combine into one dataset-level summary
    # --------------------------------------------------------

    result = {
        "Dataset": file_path.stem,
        "Split": file_path.parent.parent.name,
        "Scene": file_path.parent.name,

        "Total Observations":
            basic["Total Observations"],

        "Unique Frames":
            basic["Unique Frames"],

        "Pedestrians":
            basic["Pedestrians"],

        "Average Trajectory Distance":
            trajectory[
                "Total Distance"
            ].mean()
            if not trajectory.empty else np.nan,

        "Average Trajectory Duration":
            trajectory[
                "Duration"
            ].mean()
            if not trajectory.empty else np.nan,

        "Average Speed":
            speed[
                "Average Speed"
            ].mean()
            if not speed.empty else np.nan,

        "Maximum Speed":
            speed[
                "Maximum Speed"
            ].max()
            if not speed.empty else np.nan,

        "Average Direction Consistency":
    direction[
        "Direction Consistency"
    ].mean()
    if not direction.empty else np.nan,

        "Average Circular Direction Std":
             direction[
                 "Circular Direction Std"
            ].mean()
            if not direction.empty else np.nan,

        "Average Nearest Distance":
            interaction[
                "Average_Nearest_Distance"
            ].mean()
            if not interaction.empty else np.nan,

        "Average Nearby Pedestrians":
            interaction[
                "Average_Nearby_Pedestrians"
            ].mean()
            if not interaction.empty else np.nan,

        **density
    }

    return result


# ============================================================
# 9. RUN ON ALL DATASETS
# ============================================================

def main():

    print("Searching for trajectory files...\n")

    files = []

    for split in [
        "train",
        "test",
        "validation"
    ]:

        split_path = DATA_ROOT / split

        if split_path.exists():

            files.extend(
                split_path.rglob("*.txt")
            )

    print(f"Found {len(files)} files.\n")

    all_results = []

    for file_path in files:

        print(
            f"Processing: {file_path}"
        )

        try:

            result = analyze_file(
                file_path
            )

            all_results.append(result)

        except Exception as e:

            print(
                f"ERROR: {file_path}"
            )

            print(e)

    # --------------------------------------------------------
    # Create comparison table
    # --------------------------------------------------------

    comparison_df = pd.DataFrame(
        all_results
    )

    # Sort
    comparison_df = comparison_df.sort_values(
        ["Split", "Scene", "Dataset"]
    )

    # Save
    output_file = (
        Path(__file__).parent
        / "dataset_comparison.csv"
    )

    comparison_df.to_csv(
        output_file,
        index=False
    )

    print("\n======================================")
    print("EDA COMPLETE")
    print("======================================")

    print(
        f"Datasets analyzed: "
        f"{len(comparison_df)}"
    )

    print(
        f"\nResults saved to:\n"
        f"{output_file}"
    )

    print("\nComparison:")
    print(comparison_df.to_string(index=False))


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":
    main()