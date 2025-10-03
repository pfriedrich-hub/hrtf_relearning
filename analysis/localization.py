import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
import numpy
import scipy

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
        in_sector = numpy.where((targets[:, 0] >= az_min) & (targets[:, 0] < az_max) &
            (targets[:, 1] >= el_min) & (targets[:, 1] < el_max))[0]
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

def target_p(sequence, show=False, axis=None):
    # calculate target probabilities depending on polar error in the trial sequence
    # retrieve data
    loc_data = numpy.asarray(sequence.data)
    loc_data = loc_data.reshape(loc_data.shape[0], 2, 2)
    targets = loc_data[:, 1]  # [az, ele]
    responses = loc_data[:, 0]
    response_errors = numpy.zeros((len(sequence.sector_centers), 3))
    for idx, center in enumerate(sequence.sector_centers):
        in_sector = numpy.where(        # Get indices of targets in this sector
            (targets[:, 0] >= center[0] - sequence.sector_size[0] / 2)
            & (targets[:, 0] < center[0] + sequence.sector_size[0] / 2) &
            (targets[:, 1] >= center[1] - sequence.sector_size[1] / 2)
            & (targets[:, 1] < center[1] + sequence.sector_size[1] / 2))[0]
        rmse = numpy.sqrt(numpy.mean(numpy.square(targets[in_sector] - responses[in_sector]), axis=0))
        polar_error = rmse[1]  # use only polar error to calculate target probabilities
        response_errors[idx] = numpy.array((center[0], center[1], polar_error))
    target_p = response_errors[:, 2] / numpy.sum(response_errors[:, 2])
    response_errors = numpy.hstack((response_errors, numpy.expand_dims(target_p, axis=1)))
    if show:
        azimuths = numpy.unique(response_errors[:, 0])
        elevations = numpy.unique(response_errors[:, 1])
        if not axis:
            fig, axis = plt.subplots()
        p_grid = numpy.zeros((len(azimuths), len(elevations)))        # Create an empty z-grid
        for row in response_errors:        # Fill the z-grid by matching (x, y) to z
            x_idx = numpy.where(azimuths == row[0])[0][0]
            y_idx = numpy.where(elevations == row[1])[0][0]
            p_grid[y_idx, x_idx] = row[3]
        contour = axis.pcolormesh(azimuths, elevations, p_grid)
        cbar_levels = numpy.linspace(0, 0.1, 10)
        cax_pos = list(axis.get_position().bounds)  # (x0, y0, width, height)
        cax_pos[0] += 0.8  # x0
        cax_pos[2] = 0.012  # width
        cax = fig.add_axes(cax_pos)
        cbar = fig.colorbar(contour, cax, orientation="vertical", ticks=numpy.linspace(0, 0.1, 5))
        cax.set_title('Probability')
        axis.set_xlabel('Azimuth')
        axis.set_ylabel('Elevation')
        az_ticks = numpy.array((azimuths - sequence.sector_size[0] / 2, azimuths + sequence.sector_size[0] / 2)).T
        ele_ticks = numpy.array((elevations - sequence.sector_size[1] / 2, elevations + sequence.sector_size[1] / 2)).T
        axis.set_yticks(az_ticks)
        axis.set_xticks(ele_ticks)
        # axis.set_ylim(numpy.min(elevations) - 15, numpy.max(elevations) + 15)
        # axis.set_xlim(numpy.min(azimuths) - 15, numpy.max(azimuths) + 15)
        # localization_accuracy(sequence, show=True, plot_dim=2, binned=True)
    return response_errors

def plot_localization(sequence, report_stats=['elevation', 'azimuth'], filepath=None):
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
    az_size, el_size = sequence.sector_size
    eg, ele_rmse, ele_sd, ag, az_rmse, az_sd = localization_accuracy(sequence)

    mean_responses = []
    center_grid = {}
    # get targets and responses in each sector
    for center in sector_centers:
        az_min = center[0] - az_size / 2
        az_max = center[0] + az_size / 2
        el_min = center[1] - el_size / 2
        el_max = center[1] + el_size / 2
        in_sector = numpy.where((targets[:, 0] >= az_min) & (targets[:, 0] < az_max) &
            (targets[:, 1] >= el_min) & (targets[:, 1] < el_max))[0]
        if len(in_sector) == 0:
            continue
        # for each sector, calculate mean vector across target-response pairs
        response_shift = responses[in_sector] - targets[in_sector]
        mean_shift = numpy.mean(response_shift, axis=0)
        representative_response = center + mean_shift
        mean_responses.append(representative_response)
        center_grid[tuple(center)] = representative_response

    mean_responses = numpy.array(mean_responses)

    # Axis setup
    az_vals = sorted(set([c[0] for c in sector_centers]))
    el_vals = sorted(set([c[1] for c in sector_centers]))
    az_min = min(az_vals) - az_size - 5
    az_max = max(az_vals) + az_size + 5
    el_min = min(el_vals) - el_size - 5
    el_max = max(el_vals) + el_size + 5

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect('equal')
    ax.set_xlim(az_min, az_max)
    ax.set_ylim(el_min, el_max)
    ax.set_xlabel("Azimuth (°)")
    ax.set_ylabel("Elevation (°)")
    title = ''
    if 'elevation' in report_stats:
        title += f"EG: {eg:.2f}, RMSE: {ele_rmse:.2f}, SD: {ele_sd:.2f}"
    if 'azimuth' in report_stats:
        title += f"\n AG: {ag:.2f}, az RMSE: {az_rmse:.2f}, az SD: {az_sd:.2f}"
    ax.set_title(title)
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


    if filepath:
        if not filepath.exists():
            filepath.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath / f'{sequence.name}.png')

    plt.tight_layout()
    plt.show()