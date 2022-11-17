import scipy
import slab
from pathlib import Path
import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy

data_dir = Path.cwd() / 'data' / 'localization_data' / 'pilot'
subject_id = 'paul_ears_free_21.10'

def localization_accuracy(subject_id, show=True, plot_dim=1):
    # calculate elevation gain
    sequence = slab.Trialsequence(conditions=47, n_reps=1)
    sequence.load_pickle(file_name=data_dir / subject_id)
    loc_data = numpy.asarray(sequence.data)
    loc_data = loc_data.reshape(loc_data.shape[0], 2, 2)
    elevations = numpy.unique(loc_data[:, 1, 1])
    azimuths = numpy.unique(loc_data[:, 1, 0])
    target_elevations = loc_data[:, 1, 1]  # target elevations
    perceived_elevations = loc_data[:, 0, 1]  # percieved elevations
    target_azimuths = loc_data[:, 1, 0]
    perceived_azimuths = loc_data[:, 0, 0]

    # target ids
    right_ids = numpy.where(loc_data[:, 1, 0] > 0)
    left_ids = numpy.where(loc_data[:, 1, 0] < 0)
    mid_ids = numpy.where(loc_data[:, 1, 0] == 0)
    # above_ids = numpy.where(loc_data[:, 1, 1] > 0)
    # below_ids = numpy.where(loc_data[:, 1, 1] < 0)

    # mean perceived location for each target speaker
    i = 0
    mean_loc_data = numpy.zeros((45, 2, 2))
    for az_id, azimuth in enumerate(numpy.unique(target_azimuths)):
        for ele_id, elevation in enumerate(numpy.unique(target_elevations)):
            [perceived_targets] = loc_data[numpy.where(numpy.logical_and(loc_data[:, 1, 1] == elevation,
                          loc_data[:, 1, 0] == azimuth)), 0]
            if perceived_targets.size != 0:
                mean_perceived = numpy.mean(perceived_targets, axis=0)
                mean_loc_data[i] = numpy.array(((azimuth, mean_perceived[0]), (elevation, mean_perceived[1])))
                i += 1

    elevation_gain, n = scipy.stats.linregress(target_elevations, perceived_elevations)[:2]
    rmse = numpy.sqrt(numpy.square(numpy.subtract(target_elevations, perceived_elevations)).mean())
    sd = numpy.mean([numpy.std(perceived_elevations[numpy.where(target_elevations == target)])
                for target in numpy.unique(target_elevations)])
    if show:
        fig, axis = plt.subplots(1, 1)
        elevation_ticks = numpy.unique(target_elevations)
        azimuth_ticks = numpy.unique(target_azimuths)
        axis.set_yticks(elevation_ticks)
        axis.set_ylim(numpy.min(elevation_ticks)-15, numpy.max(elevation_ticks)+15)
        if plot_dim == 2:
            axis.set_xticks(azimuth_ticks)
            axis.set_xlim(numpy.min(azimuth_ticks)-15, numpy.max(azimuth_ticks)+15)
            for az in azimuths:  # plot lines between mean perceived locations for each target
                [x] = mean_loc_data[numpy.where(mean_loc_data[:, 0, 0]==az), 0, 0]
                [y] = mean_loc_data[numpy.where(mean_loc_data[:, 0, 0]==az), 1, 0]
                axis.plot(x, y, color='black', linewidth=0.5)
            for ele in elevations:
                [x] = mean_loc_data[numpy.where(mean_loc_data[:, 1, 0] == ele), 0, 0]
                [y] = mean_loc_data[numpy.where(mean_loc_data[:, 1, 0] == ele), 1, 0]
                axis.plot(x, y, color='black', linewidth=0.5)
            axis.scatter(perceived_azimuths, perceived_elevations, s=8, edgecolor='grey', facecolor='none')
            axis.scatter(mean_loc_data[:, 0, 1], mean_loc_data[:, 1, 1], color='black', s=25)
            for az in azimuths:  # plot lines between mean perceived locations for each target
                [x] = mean_loc_data[numpy.where(mean_loc_data[:, 0, 0]==az), 0, 1]
                [y] = mean_loc_data[numpy.where(mean_loc_data[:, 0, 0]==az), 1, 1]
                axis.plot(x, y, color='black')
            for ele in elevations:
                [x] = mean_loc_data[numpy.where(mean_loc_data[:, 1, 0] == ele), 0, 1]
                [y] = mean_loc_data[numpy.where(mean_loc_data[:, 1, 0] == ele), 1, 1]
                axis.plot(x, y, color='black')
        if plot_dim == 1:
            axis.set_xticks(elevation_ticks)
            axis.set_xlim(numpy.min(elevation_ticks)-15, numpy.max(elevation_ticks)+15)
            axis.set_xlabel('target elevations')
            axis.set_ylabel('perceived elevations')
            axis.grid(visible=True, which='major', axis='both', linestyle='dashed', linewidth=0.5, color='grey')
            axis.set_axisbelow(True)
            # scatter plot with regression line (elevation gain)
            axis.scatter(target_elevations[left_ids], perceived_elevations[left_ids], s=10, c='red', label='left')
            axis.scatter(target_elevations[right_ids], perceived_elevations[right_ids], s=10, c='blue', label='right')
            axis.scatter(target_elevations[mid_ids], perceived_elevations[mid_ids], s=10, c='black', label='middle')
            x = numpy.arange(-55, 56)
            y = elevation_gain * x + n
            axis.plot(x, y, c='grey', linewidth=1, label='elevation gain %.2f' % elevation_gain)
            plt.legend()
        axis.set_title(str(subject_id))
        plt.show()
    return elevation_gain, rmse, sd

def trial_to_trial_performance(subject_id, show=True):
    sequence = slab.Trialsequence(conditions=47, n_reps=1)
    sequence.load_pickle(file_name=data_dir / subject_id)
    loc_data = numpy.asarray(sequence.data)
    loc_data = loc_data.reshape(loc_data.shape[0], 2, 2)
    target_elevations = loc_data[:, 1, 1]  # target elevations
    perceived_elevations = loc_data[:, 0, 1]  # percieved elevations
    # target ids
    right_ids = numpy.where(loc_data[:, 1, 0] > 0)
    left_ids = numpy.where(loc_data[:, 1, 0] < 0)
    mid_ids = numpy.where(loc_data[:, 1, 0] == 0)
    trial_error = numpy.abs(numpy.subtract(target_elevations, perceived_elevations))
    if show:
        x = numpy.arange(len(trial_error))
        m, n = scipy.stats.linregress(x, trial_error)[:2]
        y = m * x + n
        fig, axis = plt.subplots(1, 1)
        axis.set_title(str(subject_id))
        axis.plot(trial_error)
        axis.plot(x, y)
    return trial_error, m, n

if __name__ == "__main__":
    trial_to_trial_performance(subject_id, show=True)
    elevation_gain, rmse, sd = localization_accuracy(subject_id, show=True, plot_dim=1)
    elevation_gain, rmse, sd = localization_accuracy(subject_id, show=True, plot_dim=2)
    print('gain: %.2f\nrmse: %.2f\nsd: %.2f' % (elevation_gain, rmse, sd))

"""
# for azimuth:
az_x = loc_data[:, 1, 0]
az_y = loc_data[:, 0, 0]
bads_idx = numpy.where(az_y == None)
az_y = numpy.array(numpy.delete(az_y, bads_idx), dtype=numpy.float)
az_x = numpy.array(numpy.delete(az_x, bads_idx), dtype=numpy.float)
azimuth_gain = scipy.stats.linregress(az_x, az_y)[0]
"""