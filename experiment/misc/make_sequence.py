import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

def generate_points(field_azimuth, field_elevation, square_size, min_distance):
    az_min, az_max = field_azimuth
    el_min, el_max = field_elevation
    az_step, el_step = square_size

    # Compute square centers
    az_centers = np.arange(az_min + az_step / 2, az_max, az_step)
    el_centers = np.arange(el_min + el_step / 2, el_max, el_step)
    square_centers = np.array([[az, el] for az in az_centers for el in el_centers])

    # Generate one random point per square
    points = square_centers + np.random.uniform(-az_step / 2, az_step / 2, size=square_centers.shape)

    # Repeat each point 3 times
    repeated_points = np.repeat(points, 3, axis=0)

    # Enforce min-distance constraint
    return enforce_min_distance(repeated_points, min_distance), square_centers, (az_min, az_max, az_step), (el_min, el_max, el_step)

def enforce_min_distance(points, min_distance):
    np.random.shuffle(points)
    ordered_points = [points[0]]
    remaining_points = list(points[1:])

    while remaining_points:
        valid_indices = [i for i, p in enumerate(remaining_points)
                         if distance.euclidean(ordered_points[-1], p) >= min_distance]

        if valid_indices:
            ordered_points.append(remaining_points.pop(np.random.choice(valid_indices)))
        else:
            np.random.shuffle(remaining_points)

    return np.array(ordered_points)

def plot_with_grid(points, square_centers, az_range, el_range):
    az_min, az_max, az_step = az_range
    el_min, el_max, el_step = el_range

    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 1], alpha=0.6, label="Generated Points")
    plt.scatter(square_centers[:, 0], square_centers[:, 1], color="red", marker="x", s=100, label="Square Centers")

    # Set grid lines matching the squares
    plt.xticks(np.arange(az_min, az_max + az_step, az_step))
    plt.yticks(np.arange(el_min, el_max + el_step, el_step))
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.xlabel("Azimuth (°)")
    plt.ylabel("Elevation (°)")
    plt.title("Randomized Points with Grid Aligned to Squares")
    plt.legend()
    plt.show()

# Parameters
field_azimuth = (-45, 45)
field_elevation = (-45, 45)
square_size = (10, 10)
min_distance = 30

# Generate and plot points
points, square_centers, az_range, el_range = generate_points(field_azimuth, field_elevation, square_size, min_distance)
plot_with_grid(points, square_centers, az_range, el_range)
