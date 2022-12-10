import scipy
import slab
from pathlib import Path
data_dir = Path.cwd() / 'data'
import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy

# elevation gain
subject = 'jakab_mold_1.0_12_Sep'
# subj_id = subject + '_mold_1_01_Jul'
# subj_id = subject + '_no_mold_01_Jul'

plt.figure()
sequence = slab.Trialsequence(conditions=45, n_reps=1)
sequence.load_pickle(file_name=data_dir / 'localization_data' / subject)
loc_data = numpy.asarray(sequence.data)
loc_data = loc_data.reshape(loc_data.shape[0], 2, 2)
ele_x = loc_data[:, 1, 1]  # target elevations
ele_y = loc_data[:, 0, 1]  # percieved elevations
bads_idx = numpy.where(ele_y == None)
ele_y = numpy.array(numpy.delete(ele_y, bads_idx), dtype='float')
ele_x = numpy.array(numpy.delete(ele_x, bads_idx), dtype='float')
plt.scatter(ele_x, ele_y)
elevation_gain = scipy.stats.linregress(ele_x, ele_y)[0]
plt.title(subject + '; elevation_gain %.2f' % elevation_gain)
plt.show()

az_x = loc_data[:, 1, 0]
az_y = loc_data[:, 0, 0]
bads_idx = numpy.where(az_y == None)
az_y = numpy.array(numpy.delete(az_y, bads_idx), dtype=numpy.float)
az_x = numpy.array(numpy.delete(az_x, bads_idx), dtype=numpy.float)
azimuth_gain = scipy.stats.linregress(az_x, az_y)[0]

#  DTF correlation matrix
data_dir = Path.cwd() / 'data'
filename1 = 'kemar_free.sofa'
filename2 = 'kemar_mold_1.sofa'
hrtf_1 = slab.HRTF(data_dir / 'hrtfs' / filename1)
hrtf_2 = slab.HRTF(data_dir / 'hrtfs' / filename1)

sources = hrtf_1.cone_sources(0)
tfs_1 = hrtf_1.tfs_from_sources(sources, n_bins=200)
tfs_2 = hrtf_2.tfs_from_sources(sources, n_bins=200)

n_sources = len(sources)
corr_mtx = numpy.zeros((n_sources, n_sources))
for i in range(n_sources):
    for j in range(n_sources):
        corr_mtx[i, j] = numpy.corrcoef(tfs_1[:, i], tfs_2[:, j])[1, 0]


# plot correlation matrix
fig, axis = plt.subplots()
contour = axis.contourf(hrtf_1.sources.vertical_polar[sources, 1], hrtf_2.sources.vertical_polar[sources, 1], corr_mtx,
                        cmap=None, levels=10)
ax, _ = matplotlib.colorbar.make_axes(plt.gca())
cbar = matplotlib.colorbar.ColorbarBase(ax, cmap=None, ticks=numpy.arange(0, 1.1, .1),
                       norm=matplotlib.colors.Normalize(vmin=0, vmax=1), label='Correlation Coefficient')
axis.set_ylabel('HRTF 1 Elevation (degrees)')
axis.set_xlabel('HRTF 2 Elevation (degrees)')


# power analysis
from statsmodels.stats.power import tt_solve_power
import numpy
# power: (= 1 - beta, fehler 2. art) - wahrscheinlichkeit die nullhypothese richtigerweise zu verwerfen
# alpha: (fehler 1. art) - wahrscheinlichkeit, die nullhypothese fälschlicherweise zu verwerfen
# effect_size: cohen's d
x1 = 0
x2 = 0
std1 = 0
std2 = 0
d = (numpy.mean(x1) - numpy.mean(x2)) / numpy.sqrt(((std1**2)+(std2**2))/2)
# calculate N
N = tt_solve_power(power=0,effect_size=0,alpha=0)

# plot localization accuracy of all participants
from analysis.localization_analysis import localization_accuracy
from pathlib import Path
import matplotlib.pyplot as plt
import slab

# get path for each subject data folder
subject_dir_list = list((Path.cwd() / 'data' / 'experiment' / 'bracket_1').iterdir())
fig, axis = plt.subplots(6, len(subject_dir_list), sharex=True, sharey=True)
condition = 'earmolds'
for subj_idx, subject_path in enumerate(subject_dir_list):
    subject_dir = subject_path / condition
    # iterate over localization accuracy files
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


# remove invalid values, this is redundant for meta motion head tracking
# bads_idx = numpy.where(ele_y == None)
# ele_y = numpy.array(numpy.delete(ele_y, bads_idx), dtype='float')
# ele_x = numpy.array(numpy.delete(ele_x, bads_idx), dtype='float')
"""



