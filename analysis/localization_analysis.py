import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy
import scipy

def localization_accuracy(data, show=True, plot_dim=2, binned=True, axis=None, show_single_responses=True,
                          elevation='all', azimuth='all'):
    # retrieve data
    azimuths = numpy.unique(data[:, 0])
    elevations = numpy.unique(data[:, 1])
    targets = data[:, :2]  # [az, ele]
    responses = data[:, 2:]

    #  elevation gain, rmse, response variability
    elevation_gain, n = scipy.stats.linregress(targets[:, 1], responses[:, 1])[:2]
    rmse = numpy.sqrt(numpy.mean(numpy.square(targets - responses), axis=0))
    variability = numpy.mean([numpy.std(responses[numpy.where(numpy.all(targets == target, axis=1))], axis=0)
                    for target in numpy.unique(targets, axis=0)], axis=0)
    az_rmse, ele_rmse = rmse[0], rmse[1]
    az_sd, ele_sd = variability[0], variability[1]
    az_var, ele_var = az_sd ** 2, ele_sd ** 2


    # mean perceived location for each target speaker
    i = 0
    mean_loc = numpy.zeros((len(azimuths) * len(elevations), 2, 2))
    for az_id, azimuth in enumerate(azimuths):
        for ele_id, elevation in enumerate(elevations):
            [perceived_targets] = data[numpy.where(numpy.logical_and(data[:, 0] == azimuth, data[:, 1] == elevation)), 2:]
            if perceived_targets.size != 0:
                mean_perceived = numpy.mean(perceived_targets, axis=0)
                mean_loc[i] = numpy.array(((azimuth, mean_perceived[0]), (elevation, mean_perceived[1])))
                i += 1

    # divide target space in 16 half overlapping sectors and get mean response for each sector
    binned_data = numpy.empty((0, 4))
    for a in range(6):
        for e in range(6):
            # select for azimuth
            tar_bin = data[numpy.logical_or(data[:, 0] == azimuths[a], data[:, 0] == azimuths[a + 1])]
            # select for elevation
            tar_bin = tar_bin[numpy.logical_or(tar_bin[:, 1] == elevations[e], tar_bin[:, 1] == elevations[e + 1])]
            tar_bin[:, :2] = numpy.array((numpy.mean([azimuths[a], azimuths[a + 1]]),
                                         numpy.mean([elevations[e], elevations[e + 1]])))
            tar_bin = numpy.mean(tar_bin, axis=0)
            binned_data = numpy.vstack((binned_data, tar_bin))

    if show:
        if not axis:
            fig, axis = plt.subplots(1, 1)
        elevation_ticks = numpy.unique(targets[:, 1])
        azimuth_ticks = numpy.unique(targets[:, 0])
        # axis.set_yticks(elevation_ticks)
        # axis.set_ylim(numpy.min(elevation_ticks)-15, numpy.max(elevation_ticks)+15)
        if plot_dim == 2:
            # axis.set_xticks(azimuth_ticks)
            # axis.set_xlim(numpy.min(azimuth_ticks)-15, numpy.max(azimuth_ticks)+15)
            if show_single_responses:
                axis.scatter(responses[:, 0], responses[:, 1], s=8, edgecolor='grey', facecolor='none')
            if binned:
                azimuths = numpy.unique(binned_data[:, 0])
                elevations = numpy.unique(binned_data[:, 1])
                mean_loc = binned_data
                # azimuth_ticks = azimuths
                # elevation_ticks = elevations
            for az in azimuths:  # plot lines between target locations
                [x] = mean_loc[numpy.where(mean_loc[:, 0] == az), 0]
                [y] = mean_loc[numpy.where(mean_loc[:, 0] == az), 1]
                axis.plot(x, y, color='black', linewidth=0.5)
            for ele in elevations:
                [x] = mean_loc[numpy.where(mean_loc[:, 1] == ele), 0]
                [y] = mean_loc[numpy.where(mean_loc[:, 1] == ele), 1]
                axis.plot(x, y, color='black', linewidth=0.5)

            axis.scatter(mean_loc[:, 2], mean_loc[:, 3], color='black', s=25)
            for az in azimuths:  # plot lines between target locations
                [x] = mean_loc[numpy.where(mean_loc[:, 0] == az), 2]
                [y] = mean_loc[numpy.where(mean_loc[:, 0] == az), 3]
                axis.plot(x, y, color='black', linewidth=1)
            for ele in elevations:
                [x] = mean_loc[numpy.where(mean_loc[:, 1] == ele), 2]
                [y] = mean_loc[numpy.where(mean_loc[:, 1] == ele), 3]
                axis.plot(x, y, color='black', linewidth=1)

        elif plot_dim == 1:
            # target ids
            right_ids = numpy.where(data[:, 0] > 0)
            left_ids = numpy.where(data[:, 0] < 0)
            mid_ids = numpy.where(data[:, 0] == 0)
            # above_ids = numpy.where(loc_data[:, 1, 1] > 0)
            # below_ids = numpy.where(loc_data[:, 1, 1] < 0)
            axis.set_xticks(elevation_ticks)
            axis.set_xlim(numpy.min(elevation_ticks)-15, numpy.max(elevation_ticks)+15)
            axis.set_xlabel('target elevations')
            axis.set_ylabel('perceived elevations')
            axis.grid(visible=True, which='major', axis='both', linestyle='dashed', linewidth=0.5, color='grey')
            axis.set_axisbelow(True)
            # scatter plot with regression line (elevation gain)
            axis.scatter(targets[:, 1][left_ids], responses[:, 1][left_ids], s=10, c='red', label='left')
            axis.scatter(targets[:, 1][right_ids], responses[:, 1][right_ids], s=10, c='blue', label='right')
            axis.scatter(targets[:, 1][mid_ids], responses[:, 1][mid_ids], s=10, c='black', label='middle')
            x = numpy.arange(-55, 56)
            y = elevation_gain * x + n
            axis.plot(x, y, c='grey', linewidth=1, label='elevation gain %.2f' % elevation_gain)
            plt.legend()
        axis.set_yticks(elevation_ticks)
        axis.set_ylim(numpy.min(elevation_ticks) - 15, numpy.max(elevation_ticks) + 15)
        axis.set_xticks(azimuth_ticks)
        axis.set_xlim(numpy.min(azimuth_ticks) - 15, numpy.max(azimuth_ticks) + 15)
        axis.set_title('elevation gain: %.2f' % elevation_gain)
        plt.show()

    # #  return EG, RMSE and Response Variability
    return elevation_gain, ele_rmse, ele_sd, az_rmse, az_sd