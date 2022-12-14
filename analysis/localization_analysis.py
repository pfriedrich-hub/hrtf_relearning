import scipy
import slab
from pathlib import Path
import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy

def localization_accuracy(sequence, show=True, plot_dim=1, binned=True, axis=None):
    # calculate elevation gain
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
    mean_loc = numpy.zeros((45, 2, 2))
    for az_id, azimuth in enumerate(numpy.unique(target_azimuths)):
        for ele_id, elevation in enumerate(numpy.unique(target_elevations)):
            [perceived_targets] = loc_data[numpy.where(numpy.logical_and(loc_data[:, 1, 1] == elevation,
                          loc_data[:, 1, 0] == azimuth)), 0]
            if perceived_targets.size != 0:
                mean_perceived = numpy.mean(perceived_targets, axis=0)
                mean_loc[i] = numpy.array(((azimuth, mean_perceived[0]), (elevation, mean_perceived[1])))
                i += 1

    # divide target space in 16 half overlapping sectors and get mean response for each sector
    mean_loc_binned = numpy.empty((0, 2, 2))
    for a in range(6):
        for e in range(6):
            tar_bin = loc_data[numpy.logical_or(loc_data[:, 1, 0] == azimuths[a],
                                                loc_data[:, 1, 0] == azimuths[a+1])]
            tar_bin = tar_bin[numpy.logical_or(tar_bin[:, 1, 1] == elevations[e],
                                               tar_bin[:, 1, 1] == elevations[e+1])]
            tar_bin[:, 1] = numpy.array((numpy.mean([azimuths[a], azimuths[a+1]]),
                                         numpy.mean([elevations[e], elevations[e+1]])))
            mean_tar_bin = numpy.mean(tar_bin, axis=0).T
            mean_tar_bin[:, [0, 1]] = mean_tar_bin[:, [1, 0]]
            mean_loc_binned = numpy.concatenate((mean_loc_binned, [mean_tar_bin]))

    elevation_gain, n = scipy.stats.linregress(target_elevations, perceived_elevations)[:2]
    rmse = numpy.sqrt(numpy.mean(numpy.square(target_elevations - perceived_elevations)))
    dev = numpy.abs(target_elevations - perceived_elevations) - \
          numpy.mean(numpy.abs(target_elevations - perceived_elevations))
    sd = numpy.sqrt(numpy.mean(numpy.square(dev)))

    # sd = numpy.sqrt(numpy.mean(numpy.abs(numpy.subtract(target_elevations, perceived_elevations))))
    # sd = numpy.std(numpy.abs(numpy.subtract(target_elevations, perceived_elevations)))
    # sd = numpy.mean([numpy.std(perceived_elevations[numpy.where(target_elevations == target)])
    #             for target in numpy.unique(target_elevations)])

    if show:
        if not axis:
            fig, axis = plt.subplots(1, 1)
        elevation_ticks = numpy.unique(target_elevations)
        azimuth_ticks = numpy.unique(target_azimuths)
        # axis.set_yticks(elevation_ticks)
        # axis.set_ylim(numpy.min(elevation_ticks)-15, numpy.max(elevation_ticks)+15)
        if plot_dim == 2:
            # axis.set_xticks(azimuth_ticks)
            # axis.set_xlim(numpy.min(azimuth_ticks)-15, numpy.max(azimuth_ticks)+15)
            axis.scatter(perceived_azimuths, perceived_elevations, s=8, edgecolor='grey', facecolor='none')
            if binned:
                azimuths = numpy.unique(mean_loc_binned[:, 0, 0])
                elevations = numpy.unique(mean_loc_binned[:, 1, 0])
                mean_loc = mean_loc_binned
                # azimuth_ticks = azimuths
                # elevation_ticks = elevations
            for az in azimuths:  # plot lines between target locations
                [x] = mean_loc[numpy.where(mean_loc[:, 0, 0]==az), 0, 0]
                [y] = mean_loc[numpy.where(mean_loc[:, 0, 0]==az), 1, 0]
                axis.plot(x, y, color='black', linewidth=0.5)
            for ele in elevations:
                [x] = mean_loc[numpy.where(mean_loc[:, 1, 0] == ele), 0, 0]
                [y] = mean_loc[numpy.where(mean_loc[:, 1, 0] == ele), 1, 0]
                axis.plot(x, y, color='black', linewidth=0.5)
            axis.scatter(mean_loc[:, 0, 1], mean_loc[:, 1, 1], color='black', s=25)
            for az in azimuths:  # plot lines between mean perceived locations for each target
                [x] = mean_loc[numpy.where(mean_loc[:, 0, 0]==az), 0, 1]
                [y] = mean_loc[numpy.where(mean_loc[:, 0, 0]==az), 1, 1]
                axis.plot(x, y, color='black')
            for ele in elevations:
                [x] = mean_loc[numpy.where(mean_loc[:, 1, 0] == ele), 0, 1]
                [y] = mean_loc[numpy.where(mean_loc[:, 1, 0] == ele), 1, 1]
                axis.plot(x, y, color='black')
        elif plot_dim == 1:
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
        axis.set_yticks(elevation_ticks)
        axis.set_ylim(numpy.min(elevation_ticks) - 15, numpy.max(elevation_ticks) + 15)
        axis.set_xticks(azimuth_ticks)
        axis.set_xlim(numpy.min(azimuth_ticks) - 15, numpy.max(azimuth_ticks) + 15)
        axis.set_title('elevation gain: %.2f' % elevation_gain)
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


"""
subject_id = 'nn'
condition = 'earmolds_1'
data_dir = Path.cwd() / 'data' / 'experiment' / 'bracket_1' / subject_id / condition
import datetime
date = datetime.datetime.now()

# file_name = 'localization_' + subject_id + '_' + condition + '_10.12_1 '#date.strftime('_%d.%m') + '_1_2'
file_name = 'localization_nn_earmolds_1_14.12_2'
sequence = slab.Trialsequence(conditions=45, n_reps=1)
sequence.load_pickle(file_name=data_dir / file_name)

# plot
# elevation_gain, rmse, sd = localization_accuracy(sequence, show=True, plot_dim=1)
elevation_gain, rmse, sd = localization_accuracy(sequence, show=True, plot_dim=2, binned=True)
print(file_name)
print('gain: %.2f\nrmse: %.2f\nsd: %.2f' % (elevation_gain, rmse, sd))
# plt.title(file_name)

# stitch incomplete sequences
from copy import deepcopy
sequence = deepcopy(sequence_1)
data_1 = sequence_1.data[:-sequence_1.n_remaining]
data_2 = sequence_2.data[:-sequence_2.n_remaining]
data = data_1 + data_2
sequence.data = data

### correct azimuth for >300°
for i, entry in enumerate(sequence.data):
    sequence.data[i][0][sequence.data[i][0] > 300] -= 360
    
for i, entry in enumerate(sequence.data):
    sequence.data[i][0][sequence.data[i][0] < 300] += 360
    
# save
sequence.save_pickle(data_dir / file_name, clobber=True)


# for azimuth:
az_x = loc_data[:, 1, 0]
az_y = loc_data[:, 0, 0]
bads_idx = numpy.where(az_y == None)
az_y = numpy.array(numpy.delete(az_y, bads_idx), dtype=numpy.float)
az_x = numpy.array(numpy.delete(az_x, bads_idx), dtype=numpy.float)
azimuth_gain = scipy.analysis.linregress(az_x, az_y)[0]
"""