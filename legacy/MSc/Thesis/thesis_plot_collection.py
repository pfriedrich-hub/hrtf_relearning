import old.MSc.analysis.plot.spectral_behavior_collection as sp_beh_plot
import old.MSc.analysis.plot.hrtf_plot as hrtf_plot
import old.MSc.analysis.plot.localization_plot as loc_plot
import old.MSc.analysis.plot.elevation_gain_learning as ele_learn
import old.MSc.analysis.statistics.stats_df as stats_df
import old.MSc.analysis.build_dataframe as build_df
from MSc.misc import cm2in
from old.MSc.misc.rcparams import set_rcParams
from pathlib import Path
import pandas
import numpy
from matplotlib import pyplot as plt
import old.MSc.analysis.build_dataframe as get_df
pandas.set_option('display.max_rows', None, 'display.max_columns', None, 'display.precision', 5,
                  'display.expand_frame_repr', False)
path = Path.cwd() / 'data' / 'experiment' / 'master'
plot_path = Path('/Users/paulfriedrich/Desktop/HRTF relearning/Thesis/Results/figures')
subpl_labels = ['A', 'B', 'C', 'D', 'E', 'F']
set_rcParams() # plot parameters
hrtf_df = build_df.get_hrtf_df(path, processed=True)
main_df = get_df.main_dataframe(Path.cwd() / 'data' / 'experiment' / 'master', processed_hrtf=True)
main_df = stats_df.add_hrtf_stats(main_df, bandwidth=(5700, 11300), vsi_dis_bw=(5700, 13500))  # bandwidth of analyses

# --- free ears --- #
# sample images 'lm' left and right - figure box
figsize = [10, 5]
figsize[0], figsize[1] = cm2in(figsize[0]), cm2in(figsize[1])
fig, axes = plt.subplots(1,3, figsize=(figsize[0], figsize[1]), gridspec_kw={'width_ratios': [14, 14, 1]}
    ,constrained_layout=True)
hrtf_plot.l_r_image(hrtf=main_df.iloc[5]['EF hrtf'], axes=axes, cbar_axis=axes[2])
# plt.savefig(plot_path / 'fig1' / 'fig1.svg', format='svg')

# free ears - spectral properties and behavior - figure box
plt.rcParams.update({'axes.spines.right': False, 'axes.spines.top': False})
figsize = [10, 13]
figsize[0], figsize[1] = cm2in(figsize[0]), cm2in(figsize[1])
fig = plt.figure(figsize=(figsize[0], figsize[1]), constrained_layout=True)
ax1 = plt.subplot2grid(shape=(3, 2), loc=(0, 0), colspan=2, rowspan=2)
ax2 = plt.subplot2grid(shape=(3, 2), loc=(2, 0), colspan=1)
ax3 = plt.subplot2grid(shape=(3, 2), loc=(2, 1), rowspan=2)
# fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.5)
# vsi across bands
hrtf_plot.plot_mean_vsi_across_bands(hrtf_df, condition='Ears Free', bands=None, axis=ax1, ear_idx=[0])
# L/R vsi
sp_beh_plot.th_vsi_ef_l_r(main_df, axis=ax2)
# ef rmse vsi
main_df = stats_df.add_hrtf_stats(main_df, bandwidth=(5700, 13500))  # edit dev bw for next line
sp_beh_plot.ef_vsi_th(main_df, measure='vertical RMSE', axis=ax3)
main_df = stats_df.add_hrtf_stats(main_df, bandwidth=(5700, 11300), vsi_dis_bw=(5700, 13500))
for ax_id, ax in enumerate([ax1, ax2, ax3]):
    ax.annotate(subpl_labels[ax_id], (.05, .9), c='k', weight='bold', xycoords='axes fraction')
# plt.savefig(plot_path / 'fig2' / 'fig2.svg', format='svg')

# --- spectral change --- #
# spectra and probability - figure box
plt.rcParams.update({'axes.spines.right': True, 'axes.spines.top': True})
figsize = [15, 10]
figsize[0], figsize[1] = cm2in(figsize[0]), cm2in(figsize[1])
# fig, axes = plt.subplots(2,3, figsize=(figsize[0], figsize[1]), sharex=False)
fig, axes = plt.subplots(2,4, figsize=(figsize[0], figsize[1]), gridspec_kw={'width_ratios': [14, 14, 14, 1]}
    ,constrained_layout=True)
hrtf_plot.hrtf_compare(hrtf_df, axes=axes[0], cbar_axis=axes[0, -1], average_ears=True, hrtf_diff=False, zlim=(-12, 8), figsize=[14, 5.2])
hrtf_plot.compare_spectral_change_p(main_df, axes=axes[1], cbar_axis=axes[1, -1], bandwidth=(4000, 16000), figsize=[14, 5.2])
axes[0,1].set_xlabel('')
axes[0,-1].set_yticks(numpy.arange(-8, 6, 4))
# plt.savefig(plot_path / 'fig3' / 'fig3.svg', format='svg')

# acoustic effects on vsi / vsi dissimilarity figure box
# M1/M2 VSI overview and spectral change box plots - figure box
plt.rcParams.update({'axes.spines.right': False, 'axes.spines.top': False})
figsize = [11, 11.05]
figsize[0], figsize[1] = cm2in(figsize[0]), cm2in(figsize[1])
fig, axes = plt.subplots(2,2, figsize=(figsize[0], figsize[1]), gridspec_kw={'width_ratios': [1.2, 1]},
                         layout='constrained')
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)
sp_beh_plot.th_vsi_m_l_r(main_df, axis=axes[0, 0], figsize=(12,8))
sp_beh_plot.th_boxplot_vsi(main_df, axis=axes[0, 1])
sp_beh_plot.th_scatter_perm_vsi_dis(main_df, bandwidth=(5700, 13500), axis=axes[1, 0])
sp_beh_plot.th_boxplot_vsi_dis(main_df, axis=axes[1, 1])
for ax_id, ax in enumerate(axes.flatten()):
    ax.annotate(subpl_labels[ax_id], (-.2, 1.05), c='k', weight='bold', xycoords='axes fraction')
# plt.savefig(plot_path / 'fig4' / 'fig4.svg', format='svg')

# behavioral impact and adaptation
plt.rcParams.update({'axes.spines.right': False, 'axes.spines.top': False})
fig, axes = ele_learn.learning_plot(to_plot='average', path=path, w2_exclude = ['cs', 'lm', 'lk'], figsize=(18, 11))
# plt.savefig(plot_path / 'fig5' / 'fig5.svg', format='svg')

# acoustic effect on behavioral impact - figure box
plt.rcParams.update({'axes.spines.right': False, 'axes.spines.top': False})
figsize = [12, 5]
figsize[0], figsize[1] = cm2in(figsize[0]), cm2in(figsize[1])
fig, axes = plt.subplots(1,3, figsize=(figsize[0], figsize[1]), sharey=True, layout='constrained')
sp_beh_plot.th_d0dr_vsi_dis(main_df, measure='vertical RMSE', axis=axes[0])
sp_beh_plot.th_d5dr_vsi_dis(main_df, measure='vertical RMSE', axis=axes[1])
sp_beh_plot.th_d5dr_vsi_dis_m1m2(main_df, measure='vertical RMSE', axis=axes[2])
for ax_id, ax in enumerate(axes):
    ax.annotate(subpl_labels[ax_id], (.05, .9), c='k', weight='bold', xycoords='axes fraction')
axes[1].set_ylabel('')
axes[2].set_ylabel('')
axes[0].set_xlabel('')
axes[2].set_xlabel('')
# plt.savefig(plot_path / 'fig6' / 'fig6.svg', format='svg')

# evolution of response pattern
plt.rcParams.update({'axes.spines.right': True, 'axes.spines.top': True})
fig, axis = loc_plot.response_evolution(to_plot='average', figsize=(14,7.5))
# plt.savefig(plot_path / 'fig7' / 'fig7.svg', format='svg')


# deprecated
# VSI dissimilarity EF / M1M2 vs EF / EF
plt.rcParams.update({'axes.spines.right': False, 'axes.spines.top': False})
figsize = [11.8, 6]
figsize[0], figsize[1] = cm2in(figsize[0]), cm2in(figsize[1])
fig, axes = plt.subplots(1,2, figsize=(figsize[0], figsize[1]), layout='constrained', gridspec_kw={'width_ratios': [1.2, 1]})
sp_beh_plot.th_scatter_perm_vsi_dis(main_df, bandwidth=(5700, 13500), axis=axes[0])
sp_beh_plot.th_boxplot_vsi_dis(main_df, axis=axes[1])
for ax_id, ax in enumerate(axes):
    ax.annotate(subpl_labels[ax_id], (.05, .9), c='k', weight='bold', xycoords='axes fraction')
# plt.savefig(plot_path / 'fig5' / 'fig5.svg', format='svg')
