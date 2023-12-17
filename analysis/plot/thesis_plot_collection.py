import analysis.plot.spectral_behavior_collection as sp_beh_plot
import analysis.plot.elevation_learning as ele_learn
import analysis.statistics.stats_df as stats_df
import analysis.plot.hrtf_plot as hrtf_plot
import analysis.build_dataframe as build_df
from misc.unit_conversion import cm2in
from misc.rcparams import set_rcParams
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
set_rcParams() # plot parameters
hrtf_df = build_df.get_hrtf_df(path, processed=True)
main_df = get_df.main_dataframe(Path.cwd() / 'data' / 'experiment' / 'master', processed_hrtf=True)
main_df = stats_df.add_hrtf_stats(main_df, bandwidth=(3700, 12900))

# --- free ears --- #
# sample images lm left and right - figure box
figsize = [12, 5]
figsize[0], figsize[1] = cm2in(figsize[0]), cm2in(figsize[1])
fig, axes = plt.subplots(1,2, figsize=(figsize[0], figsize[1]), sharey=True)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.03, hspace=0.3)
hrtf_plot.hrtf_image(hrtf=main_df.iloc[5]['EF hrtf'], chan=0, axis=axes[0], cbar=False, labels=True)
hrtf_plot.hrtf_image(hrtf=main_df.iloc[5]['EF hrtf'], chan=1, axis=axes[1], labels=True)
axes[1].set_ylabel('')
axes[0].set_title('Left ear')
axes[1].set_title('Right ear')
plt.savefig(plot_path / 'fig1' / 'fig1.svg', format='svg')

# free ears - spectral properties and behavior - figure box
plt.rcParams.update({'axes.spines.right': False, 'axes.spines.top': False})
figsize = [12, 15]
figsize[0], figsize[1] = cm2in(figsize[0]), cm2in(figsize[1])
fig = plt.figure(figsize=(figsize[0], figsize[1]))
ax1 = plt.subplot2grid(shape=(3, 2), loc=(0, 0), colspan=2, rowspan=2)
ax2 = plt.subplot2grid(shape=(3, 2), loc=(2, 0), colspan=1)
ax3 = plt.subplot2grid(shape=(3, 2), loc=(2, 1), rowspan=2)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.5)
# vsi across bands
hrtf_plot.plot_mean_vsi_across_bands(hrtf_df, condition='Ears Free', bands=None, axis=ax1, ear_idx=[0])
# L/R vsi across conditions
sp_beh_plot.th_vsi_ef_l_r(main_df, axis=ax2)
# ef rmse vsi
sp_beh_plot.ef_vsi_th(main_df, measure='vertical RMSE', axis=ax3)
for ax_id, ax in enumerate([ax1, ax2, ax3]):
    ax.annotate(subpl_labels[ax_id], (.05, .9), c='k', weight='bold', xycoords='axes fraction')
plt.savefig(plot_path / 'fig2' / 'fig2.svg', format='svg')

# --- spectral change --- #
# spectra and probability - figure box
plt.rcParams.update({'axes.spines.right': True, 'axes.spines.top': True})
figsize = [16, 10]
figsize[0], figsize[1] = cm2in(figsize[0]), cm2in(figsize[1])
fig, axes = plt.subplots(2,3, figsize=(figsize[0], figsize[1]), sharex=False)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=0.3)
hrtf_plot.hrtf_compare(hrtf_df, axes=axes[0], average_ears=True, hrtf_diff=False, zlim=(-12,8), figsize=[14,5.2])
hrtf_plot.compare_spectral_change_p(main_df, axes=axes[1], bandwidth=(4000, 16000),  figsize=[14,5.2])
axes[0,1].set_xlabel('')
plt.savefig(plot_path / 'fig3' / 'fig3.svg', format='svg')

# M1/M2 VSI overview and spectral change box plots - figure box
plt.rcParams.update({'axes.spines.right': False, 'axes.spines.top': False})
figsize = [16, 5]
figsize[0], figsize[1] = cm2in(figsize[0]), cm2in(figsize[1])
fig, axes = plt.subplots(1,3, figsize=(figsize[0], figsize[1]), sharey=False)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)
sp_beh_plot.th_vsi_m_l_r(main_df, axis=axes[0], figsize=(12,8))
sp_beh_plot.th_boxplot_vsi(main_df, axis=axes[1])
sp_beh_plot.th_boxplot_vsi_dis(main_df, axis=axes[2])
for ax_id, ax in enumerate(axes):
    ax.annotate(subpl_labels[ax_id], (.05, .9), c='k', weight='bold', xycoords='axes fraction')
plt.savefig(plot_path / 'fig4' / 'fig4.svg', format='svg')

# behavioral impact - figure box
figsize = [16, 5]
figsize[0], figsize[1] = cm2in(figsize[0]), cm2in(figsize[1])
fig, axes = plt.subplots(1,3, figsize=(figsize[0], figsize[1]), sharey=True)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.3)
sp_beh_plot.th_d0dr_vsi_dis(main_df, measure='vertical RMSE', axis=axes[0])
sp_beh_plot.th_d5dr_vsi_dis(main_df, measure='vertical RMSE', axis=axes[1])
sp_beh_plot.th_d5dr_vsi_dis_m1m2(main_df, measure='vertical RMSE', axis=axes[2])
for ax_id, ax in enumerate(axes):
    ax.annotate(subpl_labels[ax_id], (.05, .9), c='k', weight='bold', xycoords='axes fraction')
axes[1].set_ylabel('')
axes[2].set_ylabel('')
plt.savefig(plot_path / 'fig5' / 'fig5.svg', format='svg')

# adaptation
plt.rcParams.update({'axes.spines.right': False, 'axes.spines.top': False})
fig, axes = ele_learn.learning_plot(to_plot='average', path=path, w2_exclude = ['cs', 'lm', 'lk'], figsize=(20, 15))
plt.savefig(plot_path / 'fig6' / 'fig6.svg', format='svg')
