import matplotlib
matplotlib.use('tkagg')
import numpy
import scipy
from matplotlib import pyplot as plt

def localization_accuracy(sequence):
    if sequence.this_n == -1 or sequence.n_remaining == len(sequence.data) or not sequence.data:
        return numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan
    # retrieve data
    loc_data = numpy.asarray(sequence.data)
    loc_data = loc_data.reshape(loc_data.shape[0], 2, 2)
    targets = loc_data[:, 1]  # [az, ele]
    responses = loc_data[:, 0]

    #  elevation gain, rmse, response variability
    elevation_gain, n = scipy.stats.linregress(targets[:, 1], responses[:, 1])[:2]
    azimuth_gain, n = scipy.stats.linregress(targets[:, 0], responses[:, 0])[:2]
    rmse = numpy.sqrt(numpy.mean(numpy.square(targets - responses), axis=0))
    az_rmse, ele_rmse = rmse[0], rmse[1]
    variability = compute_sector_precision(targets, responses, sequence.sector_centers, sequence.sector_size)
    az_sd, ele_sd = variability[0], variability[1]
    return elevation_gain, ele_rmse, ele_sd, azimuth_gain, az_rmse, az_sd

def compute_sector_precision(targets, responses, sector_centers, sector_size):
    """
    Estimates response precision per sector by aligning targets and measuring
    the spread (std) of aligned responses, then averaging over sectors.

    Parameters:
    - targets: Nx2 array (azimuth, elevation)
    - responses: Nx2 array (azimuth, elevation)
    - sector_centers: Mx2 array of sector center coordinates
    - sector_size: tuple (az_size, el_size)

    Returns:
    - per_sector_std: list of (azimuth_std, elevation_std) for each sector
    - mean_std: tuple of (mean_azimuth_std, mean_elevation_std)
    """
    az_size, el_size = sector_size
    per_sector_std = []

    for center in sector_centers:
        # Define bounds of the current sector
        az_min = center[0] - az_size / 2
        az_max = center[0] + az_size / 2
        el_min = center[1] - el_size / 2
        el_max = center[1] + el_size / 2

        # Get indices of targets in this sector
        in_sector = numpy.where(
            (targets[:, 0] >= az_min) & (targets[:, 0] < az_max) &
            (targets[:, 1] >= el_min) & (targets[:, 1] < el_max)
        )[0]

        if len(in_sector) >= 2:
            # Shift targets and responses so that all targets align at origin
            response_shift = responses[in_sector] - targets[in_sector]

            az_std = numpy.std(response_shift[:, 0])
            el_std = numpy.std(response_shift[:, 1])
            per_sector_std.append((az_std, el_std))

    # Compute mean std across sectors
    if per_sector_std:
        per_sector_std = numpy.array(per_sector_std)
        mean_std = tuple(numpy.mean(per_sector_std, axis=0))
    else:
        mean_std = (numpy.nan, numpy.nan)

    return mean_std

def plot_localization(sequence):
    """
    Plots representative mean responses by aligning targets,
    connects them in a grid, and shows trimmed sector center lines only across actual field.
    """
    if sequence.this_n == -1 or sequence.n_remaining == len(sequence.data) or not sequence.data:
        return numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan
    # retrieve data
    loc_data = numpy.asarray(sequence.data)
    loc_data = loc_data.reshape(loc_data.shape[0], 2, 2)
    targets = loc_data[:, 1]  # [az, ele]
    responses = loc_data[:, 0]
    sector_centers = sequence.sector_centers
    sector_size = sequence.sector_size
    az_size, el_size = sector_size
    mean_responses = []
    center_grid = {}

    eg, rmse, sd, *_ = localization_accuracy(sequence)

    for center in sector_centers:
        az_min = center[0] - az_size / 2
        az_max = center[0] + az_size / 2
        el_min = center[1] - el_size / 2
        el_max = center[1] + el_size / 2

        in_sector = numpy.where(
            (targets[:, 0] >= az_min) & (targets[:, 0] < az_max) &
            (targets[:, 1] >= el_min) & (targets[:, 1] < el_max)
        )[0]

        if len(in_sector) == 0:
            continue

        response_shift = responses[in_sector] - targets[in_sector]
        mean_shift = numpy.mean(response_shift, axis=0)
        representative_response = center + mean_shift

        mean_responses.append(representative_response)
        center_grid[tuple(center)] = representative_response

    mean_responses = numpy.array(mean_responses)

    # Axis setup
    az_vals = sorted(set([c[0] for c in sector_centers]))
    el_vals = sorted(set([c[1] for c in sector_centers]))
    az_min = min(az_vals) - az_size / 2
    az_max = max(az_vals) + az_size / 2
    el_min = min(el_vals) - el_size / 2
    el_max = max(el_vals) + el_size / 2

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect('equal')
    ax.set_xlim(az_min, az_max)
    ax.set_ylim(el_min, el_max)
    ax.set_xlabel("Azimuth (°)")
    ax.set_ylabel("Elevation (°)")
    ax.set_title(f"EG: {eg:.2f}, RMSE: {rmse:.2f}, SD: {sd:.2f}")
    ax.grid(False)

    # Draw trimmed sector center lines
    for x in az_vals:
        ax.plot([x, x], [min(el_vals), max(el_vals)], color='gray', linestyle='-', linewidth=1)
    for y in el_vals:
        ax.plot([min(az_vals), max(az_vals)], [y, y], color='gray', linestyle='-', linewidth=1)

    # Plot mean responses
    ax.plot(mean_responses[:, 0], mean_responses[:, 1], 'ko', markersize=6)

    # Connect mean responses in grid layout
    sector_lookup = {tuple(sc): center_grid[tuple(sc)] for sc in sector_centers if tuple(sc) in center_grid}
    for el in el_vals:
        row = [sector_lookup[(az, el)] for az in az_vals if (az, el) in sector_lookup]
        if len(row) > 1:
            ax.plot([p[0] for p in row], [p[1] for p in row], 'k-', linewidth=2)
    for az in az_vals:
        col = [sector_lookup[(az, el)] for el in el_vals if (az, el) in sector_lookup]
        if len(col) > 1:
            ax.plot([p[0] for p in col], [p[1] for p in col], 'k-', linewidth=2)

    plt.tight_layout()
    plt.show()