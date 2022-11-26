import slab
from pathlib import Path
import os
import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from analysis.localization_analysis import localization_accuracy
import analysis.hrtf_analysis as hrtf_analysis

subject_id = 'paul'
condition = 'ears_free'
bracket = 'bracket_1'
# week = 'week_1'
data_dir = Path.cwd() / 'data' / 'experiment' / 'bracket_1' / subject_id / condition


""" plot HRTF and VSI across bands"""
sofa_name = 'kemar_ears_free_26.11.sofa'
# plot settings
plot_bins = 2400
plot_ear = 'left'
dfe = False  # apply diffuse field equalization
hrtf = slab.HRTF(data_dir / sofa_name)
sources = list(range(hrtf.n_sources - 1, -1, -1))  # get sources to plot, works for 0°/+/-17.5° cone

# plot
fig, axis = plt.subplots(2, 1)
hrtf_analysis.plot_tf(hrtf, sources, plot_bins, kind='waterfall', axis=axis[0], ear=plot_ear, xlim=(4000, 16000),
                      dfe=dfe)
hrtf_analysis.vsi_across_bands(hrtf, sources, n_bins=plot_bins, axis=axis[1], dfe=dfe)
axis[0].set_title(subject_id)


""" plot localization accuracy """
file_name = 'localization_paul_26.11'
sequence = slab.Trialsequence()
sequence.load_pickle(file_name=data_dir / file_name)
elevation_gain, rmse, sd = localization_accuracy(sequence, show=True, plot_dim=1)
elevation_gain, rmse, sd = localization_accuracy(sequence, show=True, plot_dim=2, binned=False)
print('gain: %.2f\nrmse: %.2f\nsd: %.2f' % (elevation_gain, rmse, sd))





"""

# get all HRTFs from a folder
hrtfs = []
for file in data_dir.iterdir():
    if file.suffix == '.sofa':
        hrtfs.append(slab.HRTF(file))
hrtf = hrtfs[0]  # assume no need for mutliple measurements for one condition, read 1st hrtf found
sources = list(range(hrtf.n_sources - 1, -1, -1))  # get sources to plot, works for 0°/+/-17.5° cone


group_stats = False
duration = 5  # duration of learning in days / samples

# # plot trial to trial performance in the accuracy test over days
# def plot_trial_accuracy():
#
# if __name__ == '__main__':
#
#     n_bins = 96
#     low_freq = 4000
#     high_freq = 16000
#     sources = hrtf.cone_sources(0)
#     fig, axis = plt.subplots(2, 1)
#     hrtf.plot_tf(sources, n_bins=n_bins, kind='waterfall', axis=axis[0], xlim=(low_freq, high_freq))
#     plot_vsi(hrtf, sources, n_bins=n_bins, axis=axis[1])
#     axis[0].set_title(filename)
#     hrtf.plot_tf(hrtf.cone_sources(0), xlim=(low_freq, high_freq), ear='left')
"""