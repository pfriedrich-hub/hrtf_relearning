import scipy
import slab
from pathlib import Path
import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy
import copy

def get_localization_data(path, conditions):
    subject_dir_list = list(path.iterdir())
    loc_dict = {}
    loc_dict['files'] = {}
    for condition in conditions:
        loc_dict[condition] = {}
        loc_dict['files'][condition] = {}
        for subj_idx, subject_path in enumerate(subject_dir_list):
            subject_dir = subject_path / condition
            loc_dict[condition][subject_path.name] = []
            loc_dict['files'][condition][subject_path.name] = []
            # iterate over localization accuracy files
            for file_name in sorted(list(subject_dir.iterdir())):
                if file_name.is_file() and file_name.suffix != '.sofa':
                    sequence = slab.Trialsequence(conditions=45, n_reps=3)
                    sequence.load_pickle(file_name=file_name)
                    loc_dict[condition][subject_path.name].append(sequence)
                    loc_dict['files'][condition][subject_path.name].append(file_name.name)
    return loc_dict

def localization_accuracy(sequence, show=True, plot_dim=1, binned=True, axis=None):
    # retrieve data
    loc_data = numpy.asarray(sequence.data)
    loc_data = loc_data.reshape(loc_data.shape[0], 2, 2)
    targets = loc_data[:, 1]  # [az, ele]
    responses = loc_data[:, 0]
    elevations = numpy.unique(loc_data[:, 1, 1])
    azimuths = numpy.unique(loc_data[:, 1, 0])
    targets[:, 1] = loc_data[:, 1, 1]  # target elevations
    responses[:, 1] = loc_data[:, 0, 1]  # percieved elevations
    targets[:, 0] = loc_data[:, 1, 0]
    responses[:, 0] = loc_data[:, 0, 0]
    #  elevation gain, rmse, response variability
    elevation_gain, n = scipy.stats.linregress(targets[:, 1], responses[:, 1])[:2]
    rmse = numpy.sqrt(numpy.mean(numpy.square(targets - responses), axis=0))
    variability = numpy.mean([numpy.std(responses[numpy.where(numpy.all(targets == target, axis=1))], axis=0)
                    for target in numpy.unique(targets, axis=0)], axis=0)

    az_rmse, ele_rmse = rmse[0], rmse[1]
    az_var, ele_var = variability[0], variability[1]

    # target ids
    right_ids = numpy.where(loc_data[:, 1, 0] > 0)
    left_ids = numpy.where(loc_data[:, 1, 0] < 0)
    mid_ids = numpy.where(loc_data[:, 1, 0] == 0)
    # above_ids = numpy.where(loc_data[:, 1, 1] > 0)
    # below_ids = numpy.where(loc_data[:, 1, 1] < 0)

    # mean perceived location for each target speaker
    i = 0
    mean_loc = numpy.zeros((45, 2, 2))
    for az_id, azimuth in enumerate(numpy.unique(targets[:, 0])):
        for ele_id, elevation in enumerate(numpy.unique(targets[:, 1])):
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
            axis.scatter(responses[:, 0], responses[:, 1], s=8, edgecolor='grey', facecolor='none')
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
    #  return EG, RMSE and Response Variability
    return elevation_gain, ele_rmse, ele_var, az_rmse, az_var

def trial_to_trial_performance(subject_id, show=True):
    sequence = slab.Trialsequence(conditions=47, n_reps=1)
    sequence.load_pickle(file_name=data_dir / subject_id)
    loc_data = numpy.asarray(sequence.data)
    loc_data = loc_data.reshape(loc_data.shape[0], 2, 2)
    targets[:, 1] = loc_data[:, 1, 1]  # target elevations
    responses[:, 1] = loc_data[:, 0, 1]  # percieved elevations
    # target ids
    right_ids = numpy.where(loc_data[:, 1, 0] > 0)
    left_ids = numpy.where(loc_data[:, 1, 0] < 0)
    mid_ids = numpy.where(loc_data[:, 1, 0] == 0)
    trial_error = numpy.abs(numpy.subtract(targets[:, 1], responses[:, 1]))
    if show:
        x = numpy.arange(len(trial_error))
        m, n = scipy.stats.linregress(x, trial_error)[:2]
        y = m * x + n
        fig, axis = plt.subplots(1, 1)
        axis.set_title(str(subject_id))
        axis.plot(trial_error)
        axis.plot(x, y)
    return trial_error, m, n

def get_target_proabilities(sequence, show=False, axis=None):
    # calculate target probabilities depending on localization error
    # retrieve data
    loc_data = numpy.asarray(sequence.data)
    loc_data = loc_data.reshape(loc_data.shape[0], 2, 2)
    responses = loc_data[:, 0]
    azimuths = numpy.unique(loc_data[:, 1, 0])
    elevations = numpy.unique(loc_data[:, 1, 1])
    targets = []
    for az in azimuths:  # this makes no sense but is in alignment with ff speaker table
        for ele in -numpy.sort(-elevations):
            targets.append([az, ele])
    targets = numpy.asarray(targets)
    targets = numpy.delete(targets, [0, 6, 24, -1, -7], axis=0)
    # targets = numpy.unique(loc_data[:, 1], axis=0)
    # mean response error each target speaker
    response_error = numpy.zeros((len(targets), 3))
    for idx, target in enumerate(targets):
        [perceived_targets] = loc_data[numpy.where(numpy.all(loc_data[:, 1] == target, axis=1)), 0]
        mean_response = numpy.mean(perceived_targets, axis=0)
        error = numpy.linalg.norm(target - mean_response)
        response_error[idx] = numpy.append(target, error)
    target_p = numpy.expand_dims(response_error[:, 2], axis=1) / numpy.sum(response_error[:, 2])
    response_error = numpy.hstack((response_error, target_p))
    if show:
        elevations = numpy.unique(loc_data[:, 1, 1])
        azimuths = numpy.unique(loc_data[:, 1, 0])
        img = numpy.zeros((len(elevations), len(azimuths)))
        for target in targets:
            az_idx = numpy.where(azimuths == target[0])[0][0]
            ele_idx = numpy.where(elevations == target[1])[0][0]
            img[ele_idx][az_idx] = response_error[numpy.all(response_error[:, :2] == target, axis=1), 3]
        img[img == 0] = None
        if not axis:
            fig, axis = plt.subplots()
        cbar_levels = numpy.linspace(0, 0.1, 10)
        contour = axis.pcolormesh(azimuths, elevations, img)
        cax_pos = list(axis.get_position().bounds)  # (x0, y0, width, height)
        cax_pos[0] += 0.8  # x0
        cax_pos[2] = 0.012  # width
        cax = fig.add_axes(cax_pos)
        cbar = fig.colorbar(contour, cax, orientation="vertical", ticks=numpy.linspace(0, 0.1, 5))
        cax.set_title('Probability')
        axis.set_xlabel('Azimuth')
        axis.set_ylabel('Elevation')
        axis.set_yticks(elevations)
        axis.set_ylim(numpy.min(elevations) - 15, numpy.max(elevations) + 15)
        axis.set_xticks(azimuths)
        axis.set_xlim(numpy.min(azimuths) - 15, numpy.max(azimuths) + 15)
        localization_accuracy(sequence, show=True, plot_dim=2, binned=True)
    return response_error

def load_latest(subject_dir):
    file_list = []
    for file_name in sorted(list(subject_dir.iterdir())):
        if file_name.is_file() and file_name.suffix != '.sofa':
            file_list.append(file_name)
            file_list.sort()
    sequence = slab.Trialsequence(conditions=45, n_reps=3)
    sequence.load_pickle(file_name=file_list[-1])
    print(f'Loaded {file_list[-1].name}')
    return(sequence)

""" # ----------- plot and work on localization data ------------- #
from pathlib import Path
from copy import deepcopy
import slab

subject_id = 'sm'
condition = 'Earmolds Week 2'
data_dir = Path.cwd() / 'data' / 'experiment' / 'bracket_2' / subject_id / condition

file_name = 'localization_sm_Earmolds Week 2_29.01_1'
sequence = slab.Trialsequence(conditions=45, n_reps=1)
sequence.load_pickle(file_name=data_dir / file_name)
# plot
elevation_gain, rmse, sd, _, _, = localization_accuracy(sequence, show=True, plot_dim=2, binned=True)
print('elevation_gain: %.2f\nrmse: %.2f\nsd: %.2f' % (elevation_gain, rmse, sd))
# plt.title(file_name)


#--------- stitch incomplete sequences ------------------#

filename_1 = 'localization_sm_Earmolds Week 1_29.01'
filename_2 = 'localization_sm_Earmolds Week 1_29.01_1'
sequence_1 = slab.Trialsequence(conditions=45, n_reps=1)
sequence_2 = deepcopy(sequence_1)
sequence_1.load_pickle(file_name=data_dir / filename_1)
sequence_2.load_pickle(file_name=data_dir / filename_2)
data_1 = sequence_1.data[:-sequence_1.n_remaining]
data_2 = sequence_2.data[:-sequence_2.n_remaining]
data = data_1 + data_2
sequence = sequence_1
file_name = filename_1
sequence.data = data

#  save
sequence.save_pickle(data_dir / file_name, clobber=True)

# ----------- correct azimuth for >300° ---------- #

for i, entry in enumerate(sequence.data):
    sequence.data[i][0][sequence.data[i][0] > 300] -= 360
    
for i, entry in enumerate(sequence.data):
    sequence.data[i][0][sequence.data[i][0] < 300] += 360
    
# -------------- save ------------------#

sequence.save_pickle(data_dir / file_name, clobber=True)


# for azimuth:
az_x = loc_data[:, 1, 0]
az_y = loc_data[:, 0, 0]
bads_idx = numpy.where(az_y == None)
az_y = numpy.array(numpy.delete(az_y, bads_idx), dtype=numpy.float)
az_x = numpy.array(numpy.delete(az_x, bads_idx), dtype=numpy.float)
azimuth_gain = scipy.analysis.linregress(az_x, az_y)[0]
"""