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

    # # elevations = numpy.unique(loc_data[:, 1, 1])
    # # azimuths = numpy.unique(loc_data[:, 1, 0])
    # # targets[:, 1] = loc_data[:, 1, 1]  # target elevations
    # # responses[:, 1] = loc_data[:, 0, 1]  # percieved elevations
    # # targets[:, 0] = loc_data[:, 1, 0]
    # # responses[:, 0] = loc_data[:, 0, 0]
    #
    # #
    # # variability = numpy.mean([numpy.std(responses[numpy.where(numpy.all(targets == target, axis=1))], axis=0)
    # #                 for target in numpy.unique(targets, axis=0)], axis=0)
    #
    #
    # # target ids
    # # right_ids = numpy.where(loc_data[:, 1, 0] > 0)
    # # left_ids = numpy.where(loc_data[:, 1, 0] < 0)
    # # mid_ids = numpy.where(loc_data[:, 1, 0] == 0)
    # # above_ids = numpy.where(loc_data[:, 1, 1] > 0)
    # # below_ids = numpy.where(loc_data[:, 1, 1] < 0)
    #
    # # mean perceived location for each target speaker
    # # i = 0
    # # mean_loc = numpy.zeros((45, 2, 2))
    # # for az_id, azimuth in enumerate(numpy.unique(targets[:, 0])):
    # #     for ele_id, elevation in enumerate(numpy.unique(targets[:, 1])):
    # #         [perceived_targets] = loc_data[numpy.where(numpy.logical_and(loc_data[:, 1, 1] == elevation,
    # #                       loc_data[:, 1, 0] == azimuth)), 0]
    # #         if perceived_targets.size != 0:
    # #             mean_perceived = numpy.mean(perceived_targets, axis=0)
    # #             mean_loc[i] = numpy.array(((azimuth, mean_perceived[0]), (elevation, mean_perceived[1])))
    # #             i += 1
    #
    # # # divide target space in 16 half overlapping sectors and get mean response for each sector
    # # mean_loc_binned = numpy.empty((0, 2, 2))
    # # for a in range(6):
    # #     for e in range(6):
    # #         tar_bin = loc_data[numpy.logical_or(loc_data[:, 1, 0] == azimuths[a],
    # #                                             loc_data[:, 1, 0] == azimuths[a+1])]
    # #         tar_bin = tar_bin[numpy.logical_or(tar_bin[:, 1, 1] == elevations[e],
    # #                                            tar_bin[:, 1, 1] == elevations[e+1])]
    # #         tar_bin[:, 1] = numpy.array((numpy.mean([azimuths[a], azimuths[a+1]]),
    # #                                      numpy.mean([elevations[e], elevations[e+1]])))
    # #         mean_tar_bin = numpy.mean(tar_bin, axis=0).T
    # #         mean_tar_bin[:, [0, 1]] = mean_tar_bin[:, [1, 0]]
    # #         mean_loc_binned = numpy.concatenate((mean_loc_binned, [mean_tar_bin]))
    # #
    #
    # # az_range = (-60, 60)
    # # ele_range = (-45, 45)
    # # az_range = (-52.5, 52.5)
    # # ele_range = (-37.5, 37.5)
    # # sector_size = 10
    #
    # # step 1: divide target space into half overlapping sectors with "sector_size":
    # # az_bins = numpy.arange(az_range[0], az_range[1] + sector_size, sector_size / 2)
    # # az_bins = numpy.sort(numpy.asarray([[az_bins[i], az_bins[i + 2]] for i in range(len(az_bins) - 2)]))
    #
    # #
    # #
    # #
    # #
    # # ele_bins = numpy.arange(ele_range[0], ele_range[1] + bin_size, bin_size)
    # #
    # # for az in az_bins:
    # #     for ele in ele_bins:
    # #
    # #
    # # # Compute the 2D histogram
    # # bin_counts, x_edges, y_edges, _ = scipy.stats.binned_statistic_2d(
    # #     responses[:, 0], responses[:, 1], None, statistic='count', bins=[x_bins, y_bins])
    # #
    # # # plot
    # # if show:
    # #     if not axis:
    # #         fig, axis = plt.subplots(1, 1)
    # #     elevation_ticks = numpy.unique(targets[:, 1])
    # #     azimuth_ticks = numpy.unique(targets[:, 0])
    # #     # axis.set_yticks(elevation_ticks)
    # #     # axis.set_ylim(numpy.min(elevation_ticks)-15, numpy.max(elevation_ticks)+15)
    # #     if plot_dim == 2:
    # #         # axis.set_xticks(azimuth_ticks)
    # #         # axis.set_xlim(numpy.min(azimuth_ticks)-15, numpy.max(azimuth_ticks)+15)
    # #         if show_single_responses:
    # #             axis.scatter(responses[:, 0], responses[:, 1], s=8, edgecolor='grey', facecolor='none')
    # #         if binned:
    # #             azimuths = numpy.unique(mean_loc_binned[:, 0, 0])
    # #             elevations = numpy.unique(mean_loc_binned[:, 1, 0])
    # #             mean_loc = mean_loc_binned
    # #             # azimuth_ticks = azimuths
    # #             # elevation_ticks = elevations
    # #         for az in azimuths:  # plot lines between target locations
    # #             [x] = mean_loc[numpy.where(mean_loc[:, 0, 0] == az), 0, 0]
    # #             [y] = mean_loc[numpy.where(mean_loc[:, 0, 0] == az), 1, 0]
    # #             axis.plot(x, y, color='black', linewidth=0.5)
    # #         for ele in elevations:
    # #             [x] = mean_loc[numpy.where(mean_loc[:, 1, 0] == ele), 0, 0]
    # #             [y] = mean_loc[numpy.where(mean_loc[:, 1, 0] == ele), 1, 0]
    # #             axis.plot(x, y, color='black', linewidth=0.5)
    # #         axis.scatter(mean_loc[:, 0, 1], mean_loc[:, 1, 1], color='black', s=25)
    # #         for az in azimuths:  # plot lines between mean perceived locations for each target
    # #             [x] = mean_loc[numpy.where(mean_loc[:, 0, 0] == az), 0, 1]
    # #             [y] = mean_loc[numpy.where(mean_loc[:, 0, 0] == az), 1, 1]
    # #             axis.plot(x, y, color='black')
    # #         for ele in elevations:
    # #             [x] = mean_loc[numpy.where(mean_loc[:, 1, 0] == ele), 0, 1]
    # #             [y] = mean_loc[numpy.where(mean_loc[:, 1, 0] == ele), 1, 1]
    # #             axis.plot(x, y, color='black')
    # #     elif plot_dim == 1:
    # #         axis.set_xticks(elevation_ticks)
    # #         axis.set_xlim(numpy.min(elevation_ticks)-15, numpy.max(elevation_ticks)+15)
    # #         axis.set_xlabel('target elevations')
    # #         axis.set_ylabel('perceived elevations')
    # #         axis.grid(visible=True, which='major', axis='both', linestyle='dashed', linewidth=0.5, color='grey')
    # #         axis.set_axisbelow(True)
    # #         # scatter plot with regression line (elevation gain)
    # #         axis.scatter(targets[:, 1][left_ids], responses[:, 1][left_ids], s=10, c='red', label='left')
    # #         axis.scatter(targets[:, 1][right_ids], responses[:, 1][right_ids], s=10, c='blue', label='right')
    # #         axis.scatter(targets[:, 1][mid_ids], responses[:, 1][mid_ids], s=10, c='black', label='middle')
    # #         x = numpy.arange(-55, 56)
    # #         y = elevation_gain * x + n
    # #         axis.plot(x, y, c='grey', linewidth=1, label='elevation gain %.2f' % elevation_gain)
    # #         plt.legend()
    # #     axis.set_yticks(elevation_ticks)
    # #     axis.set_ylim(numpy.min(elevation_ticks) - 15, numpy.max(elevation_ticks) + 15)
    # #     axis.set_xticks(azimuth_ticks)
    # #     axis.set_xlim(numpy.min(azimuth_ticks) - 15, numpy.max(azimuth_ticks) + 15)
    # #     axis.set_title('elevation gain: %.2f' % elevation_gain)
    # #     plt.show()
    #
    # # #  return EG, RMSE and Response Variability
    # return elevation_gain, ele_rmse, ele_sd, azimuth_gain, az_rmse, az_sd
