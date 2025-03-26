import matplotlib
import slab
# matplotlib.use("QtAgg")
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import random

def make_sequence(
        azimuth_range=(-60, 60),
        elevation_range=(-50, 50),
        sector_size=(20, 20),  # (azimuth_size, elevation_size)
        points_per_sector=3,
        min_sector_distance=30
):
    """
    Generates uniformly random points within sectors while ensuring a minimum distance between successive sectors.

    Parameters:
    - azimuth_range: tuple (min_azimuth, max_azimuth)
    - elevation_range: tuple (min_elevation, max_elevation)
    - sector_size: tuple (azimuth_size, elevation_size) in degrees
    - min_sector_distance: minimum distance between successive sectors in sequence
    - points_per_sector: number of points per sector

    Returns:
    - points: List of (azimuth, elevation) tuples
    - selected_sectors: List of selected sector centers
    """
    azimuth_size, elevation_size = sector_size
    num_azimuth_sectors = (azimuth_range[1] - azimuth_range[0]) // azimuth_size
    num_elevation_sectors = (elevation_range[1] - elevation_range[0]) // elevation_size
    num_sectors = num_azimuth_sectors * num_elevation_sectors

    # Compute sector centers
    sector_centers = [
        (azimuth_range[0] + (i + 0.5) * azimuth_size, elevation_range[0] + (j + 0.5) * elevation_size)
        for i in range(num_azimuth_sectors) for j in range(num_elevation_sectors)
    ]
    random.shuffle(sector_centers)

    # Select sectors ensuring minimum distance constraint
    selected_sectors = []
    remaining_sectors = sector_centers[:]
    while len(selected_sectors) < num_sectors:
        if not selected_sectors:
            selected_sectors.append(remaining_sectors.pop(0))
        else:
            last_sector = selected_sectors[-1]
            valid_sectors = [
                sec for sec in remaining_sectors
                if np.linalg.norm(np.array(sec) - np.array(last_sector)) >= min_sector_distance
            ]
            if valid_sectors:
                selected_sector = valid_sectors.pop(0)
                selected_sectors.append(selected_sector)
                remaining_sectors.remove(selected_sector)
            else:
                break  # Stop if no valid sector is found

    # Generate random points within each selected sector
    points = [
        (
            np.random.uniform(sector[0] - azimuth_size / 2, sector[0] + azimuth_size / 2),
            np.random.uniform(sector[1] - elevation_size / 2, sector[1] + elevation_size / 2)
        )
        for sector in selected_sectors for _ in range(points_per_sector)
    ]
    sequence = slab.Trialsequence(points)
    sequence.sector_centers = sector_centers
    return sequence
#
#
# def plot_random_points(points, selected_sectors, azimuth_range, elevation_range, sector_size):
#     """Plots the generated random points and sector boundaries with grid lines."""
#     azimuth_size, elevation_size = sector_size
#     fig, ax = plt.subplots(figsize=(8, 6))
#     ax.set_xlim(azimuth_range)
#     ax.set_ylim(elevation_range)
#     ax.set_xlabel("Azimuth (°)")
#     ax.set_ylabel("Elevation (°)")
#     ax.set_title("Random Points in Sectors with Minimum Distance Constraint")
#
#     # Plot sector boundaries
#     for sec in selected_sectors:
#         rect = plt.Rectangle(
#             (sec[0] - azimuth_size / 2, sec[1] - elevation_size / 2),
#             azimuth_size, elevation_size, edgecolor='gray', facecolor='none', linestyle="--"
#         )
#         ax.add_patch(rect)
#
#     # Plot grid lines
#     azimuth_ticks = np.arange(azimuth_range[0], azimuth_range[1] + azimuth_size, azimuth_size)
#     elevation_ticks = np.arange(elevation_range[0], elevation_range[1] + elevation_size, elevation_size)
#     ax.set_xticks(azimuth_ticks)
#     ax.set_yticks(elevation_ticks)
#     ax.grid(True, linestyle="--", linewidth=0.5)
#
#     # Plot points
#     azimuths, elevations = zip(*points)
#     ax.scatter(azimuths, elevations, color='red', label="Random Points")
#
#     ax.legend()
#     plt.show()

# Example usage
# azimuth_range = (-50, 50)
# elevation_range = (-40, 40)
# sector_size = (10, 10)  # (azimuth_size, elevation_size)
# min_sector_distance = 30
# points_per_sector = 3
#
# _, points, selected_sectors = make_sequence(
#     azimuth_range, elevation_range, sector_size, min_sector_distance, points_per_sector
# )
# plot_random_points(points, selected_sectors, azimuth_range, elevation_range, sector_size)

# import matplotlib
# matplotlib.use("Qt5Agg")
# # matplotlib.use("TkAgg")