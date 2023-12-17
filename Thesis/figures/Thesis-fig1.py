import analysis.statistics.stats_df as stats_df
import analysis.plot.spectral_behavior_collection as sp_beh_plot
import analysis.plot.hrtf_plot as hrtf_plot
import analysis.build_dataframe as build_df
from misc.unit_conversion import cm2in
from pathlib import Path
import scipy.stats
import pandas
import numpy
from matplotlib import pyplot as plt
import analysis.build_dataframe as get_df
pandas.set_option('display.max_rows', None, 'display.max_columns', None, 'display.precision', 5,
                  'display.expand_frame_repr', False)
path = Path.cwd() / 'data' / 'experiment' / 'master'
plot_path = Path('/Users/paulfriedrich/Desktop/HRTF relearning/Thesis/Results/figures')
subpl_labels = ['A', 'B', 'C', 'D', 'E', 'F']
hrtf_df = build_df.get_hrtf_df(path, processed=True)
main_df = get_df.main_dataframe(Path.cwd() / 'data' / 'experiment' / 'master', processed_hrtf=True)
main_df = stats_df.add_hrtf_stats(main_df, bandwidth=(3700, 12900))

# topic 2 spectral effect of molds
# figure box 2
plt.rcParams.update({'axes.spines.right': True, 'axes.spines.top': True,'axes.titlesize':8,
                     'xtick.labelsize': 8,'ytick.labelsize': 8, 'axes.labelsize': 8})
figsize = [17, 10]
figsize[0], figsize[1] = cm2in(figsize[0]), cm2in(figsize[1])
fig, axes = plt.subplots(2,3, figsize=(figsize[0], figsize[1]), sharex=False)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=0.3)
hrtf_plot.hrtf_compare(hrtf_df, axes=axes[0], average_ears=True, hrtf_diff=False, zlim=(-12,8), figsize=[14,5.2])
hrtf_plot.compare_spectral_change_p(main_df, axes=axes[1], bandwidth=(4000, 16000),  figsize=[14,5.2])
axes[0,1].set_xlabel('')

plt.savefig(plot_path / 'fig2' / 'fig2.svg', format='svg', bbox_inches='tight')


# A-F: spectral profile of participants ears + boxplot vsi across conditions
# and spectral change p + boxplot vsi dissimilarity
sp_beh_plot.boxplot_vsi(main_df, axis=axes[0, 1])
sp_beh_plot.boxplot_vsi_dis(main_df, axis=axes[1, 0])

# transition from spectrum back to vsis
# vsi l vs r ears -> overview on vsis and mold impact
# -> behavioral effects of molds
# vsi dissimilarity and behavior
# vsi dissimilarity in physiological range

# topic 3 adaptation
# learning plot

# ---- figure box 1 ----- #
figsize = [11, 26]
plt.rcParams.update({'axes.spines.right': False, 'axes.spines.top': False})
figsize[0], figsize[1] = cm2in(figsize[0]), cm2in(figsize[1])
fig, axes = plt.subplots(3,1, figsize=(figsize[0], figsize[1]), gridspec_kw={'height_ratios': [1.4, 1, 1]})
# vsi across bands
hrtf_plot.plot_mean_vsi_across_bands(hrtf_df, condition='Ears Free', bands=None, axis=axes[0], ear_idx=[0], figsize=(14,8))
# ef rmse vsi
sp_beh_plot.ef_vsi_th(main_df, measure='vertical RMSE', axis=axes[1], figsize=(14,8))
# L/R vsi across conditions
sp_beh_plot.th_vsi_l_r(main_df, axis=axes[2], figsize=(12,8))
for ax_id, ax in enumerate(axes):
    ax.annotate(subpl_labels[ax_id], (-.13, 1.05), c='k', weight='bold', xycoords='axes fraction')

# alternative
figsize = [15, 15]
plt.rcParams.update({'axes.spines.right': False, 'axes.spines.top': False})
figsize[0], figsize[1] = cm2in(figsize[0]), cm2in(figsize[1])
fig = plt.figure(figsize=(figsize[0], figsize[1]))
ax1 = plt.subplot2grid(shape=(3, 2), loc=(0, 0), colspan=2, rowspan=2)
ax2 = plt.subplot2grid(shape=(3, 2), loc=(2, 0), colspan=1)
ax3 = plt.subplot2grid(shape=(3, 2), loc=(2, 1), rowspan=2)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.3)
# vsi across bands
hrtf_plot.plot_mean_vsi_across_bands(hrtf_df, condition='Ears Free', bands=None, axis=ax1, ear_idx=[0], figsize=(14,8))
# ef rmse vsi
sp_beh_plot.ef_vsi_th(main_df, measure='vertical RMSE', axis=ax2, figsize=(14,8))
# L/R vsi across conditions
sp_beh_plot.th_vsi_l_r(main_df, axis=ax3, figsize=(12,8))
for ax_id, ax in enumerate([ax1, ax2, ax3]):
    ax.annotate(subpl_labels[ax_id], (-.13, 1.05), c='k', weight='bold', xycoords='axes fraction')
plt.savefig(plot_path / 'fig1' / 'fig1.svg', format='svg', bbox_inches='tight')
