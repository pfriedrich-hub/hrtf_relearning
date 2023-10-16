"""
ole_test hrtf surface variance
"""

import analysis.hrtf_analysis as hrtf_analysis
import analysis.localization_analysis as loc_analysis
import analysis.plot.localization_plot as loc_plot
import analysis.plot.hrtf_plot as hrtf_plot
from pathlib import Path
import numpy
from matplotlib import pyplot as plt
data_path = path=Path.cwd() / 'data' / 'experiment' / 'master'
w2_exclude=['cs', 'lm', 'lk']  # these subjects did not complete Week 2 of the experiment
localization_dataframe = loc_analysis.get_localization_dataframe(path, w2_exclude)

exclude = ['svm']
hrtf_dataframe = hrtf_analysis.get_hrtf_df(path, processed=False, exclude=exclude)
hrtf_dataframe = hrtf_analysis.process_hrtfs(hrtf_dataframe, filter=None, baseline=True, write=False)
hrtf_stats = loc_analysis.localization_hrtf_df(localization_dataframe, hrtf_dataframe)

for subject in hrtf_stats.subject:
    subject_data = hrtf_stats[hrtf_stats['subject'] == subject]
    hrtf_ef = hrtf_analysis.erb_filter_hrtf(subject_data['EF hrtf'].iloc[0], return_bins=True)[2]
    hrtf_m1 = hrtf_analysis.erb_filter_hrtf(subject_data['M1 hrtf'].iloc[0], return_bins=True)[2]
    hrtf_diff = hrtf_analysis.hrtf_difference(hrtf_ef, hrtf_m1)

    hrtf_ef.plot_tf(hrtf_ef.cone_sources(0), kind='surface', n_bins=83, ear='left')
    hrtf_m1.plot_tf(hrtf_ef.cone_sources(0), kind='surface', n_bins=83, ear='left')

    fig, axes = plt.subplots(2, 2)
    hrtf_diff.plot_tf(hrtf_ef.cone_sources(0), kind='waterfall', n_bins=83, ear='left', axis=axes[0])
    hrtf_diff.plot_tf(hrtf_ef.cone_sources(0), kind='waterfall', n_bins=83, ear='right', axis=axes[1])
    axes[0].set_title('left')
    axes[1].set_title('right')
    loc_plot.localization_plot(to_plot=subject, axes=)
    fig.title('subject')

    dtf_data = hrtf_diff.tfs_from_sources(sources=hrtf_diff.cone_sources(0), ear='both', n_bins=None)

    plt.figure()
    for i in range(6):
        plt.plot(dtf_data[i])


    numpy.var(dtf_data)

    hrtf_analysis.spectral_difference(hrtf_ef, hrtf_m1, bandwidth=(4000, 16000))



























import analysis.localization_analysis as loc_analysis
import analysis.hrtf_analysis as hrtf_analysis
import analysis.plot.localization_plot as loc_plot
import analysis.plot.hrtf_plot as hrtf_plot
from pathlib import Path
import scipy.stats
import pandas
import numpy
from matplotlib import pyplot as plt
pandas.set_option('display.max_rows', None, 'display.max_columns', None, 'display.precision', 5,
                  'display.expand_frame_repr', False)

path = Path.cwd() / 'data' / 'experiment' / 'master'
w2_exclude=['cs', 'lm', 'lk']
localization_dataframe = loc_analysis.get_localization_dataframe(path, w2_exclude)
processed = True
bands = [(4000, 5700), (5700, 8000), (8000, 11300), (11300, 16000)]  # to modify
n_bins=None # no big difference on the average
equalize=False  # results in implausible vsi?
condition = 'Ears Free'
subject = 'jl'

hrtf_dataframe = hrtf_analysis.get_hrtf_df(path, processed=processed)
hrtf_stats = loc_analysis.localization_hrtf_df(localization_dataframe, hrtf_dataframe)

# subj hrtf
hrtf = hrtf_dataframe[hrtf_dataframe['subject']==subject][hrtf_dataframe['condition']==condition]['hrtf'].item()

# overview plots
hrtf_plot.plot_hrtf_overview(hrtf_dataframe, subject, condition, bands, n_bins, equalize)
# hrtf_analysis.mean_vsi_across_bands(hrtf_dataframe, condition, bands, n_bins, equalize, show=True)
# hrtf_plot.plot_average(hrtf_dataframe, condition=condition, equalize=True, kind='image')  #todo fix copy error

# subject plots


# ----- TEST VSI ------'
# plot vsi across overlapping octave bands
# bands = [(3500, 7000), (4200, 8300), (4900, 9900), (5900, 11800), (7000, 14000)]
# bands = [(3500, 7000), (4200, 8500), (5100, 10200), (6200, 12400), (7500, 15000)]
# bands = [(4000, 8000), (4800, 9500), (5700, 11300), (6700, 13500), (8000, 16000)]
# bands = [(4000, 8000), (4800, 9500), (5700, 11300), (6700, 13500)]
# non overlapping
# bands = [(3500, 5000), (5000, 7200), (7200, 10400), (10400, 15000)]



#todo try:
# I
# load processed and check vsi for a single hrtf and band,
# see why auto correlation sometimes increases when there are notches
# II
# try different preprocessing steps:
# smoothing:
# filter hrtf with lp tresh < 1500 hz
# use binning to smoothe
# triangular filterbank on recordings? / cosine filterbank does not have much of an effect





# elevation gain
subject = 'jakab_mold_1.0_12_Sep'
# subj_id = subject + '_mold_1_01_Jul'
# subj_id = subject + '_no_mold_01_Jul'

plt.figure()
sequence = slab.Trialsequence(conditions=47, n_reps=1)
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
data_dir = Path.cwd() / 'final_data'
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

"""
# remove invalid values, this is redundant for meta motion head tracking
# bads_idx = numpy.where(ele_y == None)
# ele_y = numpy.array(numpy.delete(ele_y, bads_idx), dtype='float')
# ele_x = numpy.array(numpy.delete(ele_x, bads_idx), dtype='float')




