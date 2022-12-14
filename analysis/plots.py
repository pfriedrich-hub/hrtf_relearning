
""" -------  plot localization accuracy of all participants ------ """
from analysis.localization_analysis import localization_accuracy
from pathlib import Path
import matplotlib.pyplot as plt
import slab

# get path for each subject data folder
subject_dir_list = list((Path.cwd() / 'data' / 'experiment' / 'bracket_1').iterdir())
condition = 'earmolds_1'
fig, axis = plt.subplots(6, len(subject_dir_list), sharex=True, sharey=True)

for subj_idx, subject_path in enumerate(subject_dir_list):
    subject_dir = subject_path / condition
    file_idx = 0
    for file_name in sorted(list(subject_dir.iterdir())):
        if file_name.is_file() and not file_name.suffix == '.sofa':
            sequence = slab.Trialsequence(conditions=45, n_reps=1)
            sequence.load_pickle(file_name=file_name)
            elevation_gain, rmse, sd = localization_accuracy(sequence, show=True, plot_dim=2,
                                        binned=True, axis=axis[file_idx, subj_idx])
            axis[file_idx, 0].set_ylabel('Day %i' % int(file_idx+1))
            axis[file_idx, 0].yaxis.set_label_coords(-0.5, 0.5)
            file_idx += 1
fig.text(0.5, 0.07, 'Response azimuth (deg)', ha='center')
fig.text(0.08, 0.5, 'Response elevation (deg)', va='center', rotation='vertical')
axis[0,0].set_xticks(axis[0,0].get_xticks().astype('int'))
for idx, i in enumerate(range(2, 10, 2)):
    fig.text(i/10, 0.95, subject_dir_list[idx].name)


""" Plot Images, Differences, Correlation of HRTFs """
from pathlib import Path
import numpy
import scipy
import matplotlib.pyplot as plt
import analysis.hrtf_analysis as hrtf_analysis
path = Path.cwd() / 'data' / 'experiment' / 'bracket_1'
condition = 'ears_free'

conditions = ['ears_free', 'earmolds', 'earmolds_1']
hrtf_dict = {'min': [], 'max': [], 'difference': {}, 'correlation': {}}
compare = [['ears_free', 'earmolds'], ['ears_free', 'earmolds_1'], ['earmolds', 'earmolds_1']]
diff_conditions = ['Difference Ears Free - Mold 1',
              'Difference Ears Free - Mold 2', 'Difference Mold 1 - Mold 2']
corr_conditions = ['Correlation Ears Free - Mold 1',
              'Correlation Ears Free - Mold 2', 'Correlation Mold 1 - Mold 2']
n_bins = 150
xlim = (4000, 16000)
# baseline HRTFs and return average for each condition
for condition in conditions:
    # get HRTFs from one condition
    hrtf_list = hrtf_analysis.list_hrtfs(path, condition)
    # process HRTFs
    for idx, hrtf in enumerate(hrtf_list):
        hrtf = hrtf_analysis.baseline_hrtf(hrtf, bandwidth=(4000, 16000))
        hrtf = hrtf.diffuse_field_equalization()
        hrtf_list[idx] = hrtf
    # average HRTFs
    avg = hrtf_analysis.average_hrtf(hrtf_list)
    # save to HRTF-dictionary
    src = avg.cone_sources(0)
    hrtf_dict[condition] = avg
    hrtf_dict['min'].append(avg.tfs_from_sources(src, n_bins).min())
    hrtf_dict['max'].append(avg.tfs_from_sources(src, n_bins).max())

# get difference HRTFs
for i in range(3):
    hrtf_dict['difference'][diff_conditions[i]] = hrtf_analysis.hrtf_difference(hrtf_dict[compare[i][0]], hrtf_dict[compare[i][1]])
# get min and max values for img cbar scaling etc
for condition in diff_conditions:
    hrtf_dict['min'].append(hrtf_dict['difference'][condition].tfs_from_sources(src, n_bins).min())
    hrtf_dict['max'].append(hrtf_dict['difference'][condition].tfs_from_sources(src, n_bins).max())
z_min = numpy.floor(numpy.min(hrtf_dict['min'])) - 1
z_max = numpy.ceil(numpy.max(hrtf_dict['max']))

# plot
fig, axis = plt.subplots(2, 3, sharey=True, figsize=(13, 8))
title_list = [['Ears Free', 'Week 1 Molds', 'Week 2 Molds'], diff_conditions, corr_conditions]
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05)
cbar = False
for i in range(3):
    if i == 2:
        cbar = True
    # plot HRTF
    hrtf_analysis.plot_hrtf_image(hrtf_dict[conditions[i]], n_bins=n_bins,
                                  bandwidth=xlim, axis=axis[0, i], z_min=z_min, z_max=z_max, cbar=cbar)
    axis[0, i].set_title(title_list[0][i])
    # plot HRTF differences
    hrtf_analysis.plot_hrtf_image(hrtf_dict['difference'][diff_conditions[i]], n_bins=n_bins,
                                  bandwidth=xlim, axis=axis[1, i], z_min=z_min, z_max=z_max, cbar=cbar)
    axis[1, i].set_title(title_list[1][i])
fig.text(0.5, 0.04, 'Frequency (kHz)', ha='center', size=13)
fig.text(0.07, 0.5, 'Elevation (degrees)', va='center', rotation='vertical', size=13)

# compute and plot HRTF correlation
correlation = []
fig, axis = plt.subplots(1, 3, sharey=True, figsize=(12, 4))
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05)
cbar = False
for i in range(3):
    if i == 2:
        cbar = True
    correlation.append(hrtf_analysis.hrtf_correlation(hrtf_dict[compare[i][0]], hrtf_dict[compare[i][1]],
                            show=True, axis=axis[i], bandwidth=xlim, cbar=cbar, n_bins=n_bins))
    axis[i].set_title(title_list[2][i])
fig.text(0.5, 0.02, 'Elevation (degrees)', ha='center', size=13)
fig.text(0.07, 0.5, 'Elevation (degrees)', va='center', rotation='vertical', size=13)


""" Plot mean VSI across bands """
vsi_list = numpy.asarray(vsi_list)
fig, axis = plt.subplots()
axis.plot(numpy.mean(vsi_list, axis=0), c='k')
axis.set_xticks([0, 1, 2, 3, 4])
bandwidths = numpy.array(((4, 8), (4.8, 9.5), (5.7, 11.3), (6.7, 13.5), (8, 16))) * 1000
labels = [item.get_text() for item in axis.get_xticklabels()]
for idx, band in enumerate(bandwidths / 1000):
    labels[idx] = '%.1f - %.1f' % (band[0], band[1])
err = scipy.stats.sem(vsi_list, axis=0)
axis.errorbar(axis.get_xticks(), numpy.mean(vsi_list, axis=0), capsize=3,
              yerr=err, fmt="o", c='0.6', elinewidth=0.5, markersize=3)
axis.set_xticklabels(labels)
axis.set_xlabel('Frequency bands (kHz)')
axis.set_ylabel('VSI')

"""
# plot individual hrtf of participants
for subj_idx, subj_hrtf in enumerate(hrtf_list):
    fig, axis = plt.subplots()
    subj_hrtf.plot_tf(n_bins=300, kind='image', xlim=freq_range, sourceidx=hrtf.cone_sources(0), axis=axis)
    axis.set_title(file_list[subj_idx])


""" compare free vs mold """
fig, axis = plt.subplots(2, 2)
# hrtf_free, hrtf_mold = get_hrtfs(data_dir)
src_free = hrtf_free.cone_sources(0, full_cone=True)
src_mold = hrtf_mold.cone_sources(0, full_cone=True)
# waterfall and vsi
plot_vsi(hrtf_free, src_free, n_bins, axis=axis[0, 0])
plot_vsi(hrtf_mold, src_mold, n_bins, axis=axis[0, 1])
hrtf_free.plot_tf(src_free, n_bins=n_bins, kind='waterfall', axis=axis[1, 0])
hrtf_mold.plot_tf(src_mold, n_bins=n_bins, kind='waterfall', axis=axis[1, 1])
axis[0, 0].set_title('ears free')
axis[0, 1].set_title('mold')

# cross correlation
# plot_hrtf_correlation(hrtf_free, hrtf_mold, src)"""

""" Plot DTF correlation """
from hrtf_analysis import dtf_correlation
def plot_correlation(hrtf_free, hrtf_mold, sources):
    # compare heatmap of hrtf free and with mold
    fig, axis = plt.subplots(2, 2, sharey=True)
    hrtf_free.plot_tf(sources, n_bins=96, kind='image', ear='left', xlim=(4000, 12000), axis=axis[0, 0])
    hrtf_mold.plot_tf(sources, n_bins=96, kind='image', ear='left', xlim=(4000, 12000), axis=axis[0, 1])
    fig.text(0.3, 0.9, 'Ear Free', ha='center')
    fig.text(0.7, 0.9, 'With Mold', ha='center')
    # plot hrtf autocorrelation free
    corr_mtx, cbar_1 = dtf_correlation(hrtf_free, hrtf_free, show=True, bandwidth=None,
                                         n_bins=96, axis=axis[1, 0])
    # plot hrtf correlation free vs mold
    cross_corr_mtx, cbar_2 = dtf_correlation(hrtf_free, hrtf_mold, show=True, bandwidth=None,
                                               n_bins=96, axis=axis[1, 1])
    fig.text(0.3, 0.5, 'Autocorrelation Ear Free', ha='center')
    fig.text(0.7, 0.5, 'Correlation Free vs. Mold', ha='center')
    cbar_1.remove()"""
