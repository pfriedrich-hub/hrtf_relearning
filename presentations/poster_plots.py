# import analysis.plot.spectral_behavior_collection as sp_beh_plot
import analysis.plot.hrtf_plot as hrtf_plot
import analysis.plot.localization_plot as loc_plot
# import analysis.plot.elevation_learning as ele_learn
import analysis.plot.elevation_gain_learning as eg_learn
# import analysis.statistics.stats_df as stats_df
# import analysis.build_dataframe as build_df
import misc.rcparams as rcparams
from pathlib import Path
# import scipy.stats
# import pandas
# import numpy
from matplotlib import pyplot as plt
import analysis.build_dataframe as get_df
# pandas.set_option('display.max_rows', None, 'display.max_columns', None, 'display.precision', 5,
#                   'display.expand_frame_repr', False)
path = Path.cwd() / 'data' / 'experiment' / 'master'
plot_path = Path('/Users/paulfriedrich/Desktop/HRTF relearning/poster/figures')
# subpl_labels = ['A', 'B', 'C', 'D', 'E', 'F']
# hrtf_df = build_df.get_hrtf_df(path, processed=True)
main_df = get_df.main_dataframe(Path.cwd() / 'data' / 'experiment' / 'master', processed_hrtf=True)
# main_df = stats_df.add_hrtf_stats(main_df, bandwidth=(5700, 11300), vsi_dis_bw=(5700, 13500))  # bandwidth of analyses

# main result: figure box: A - learning curve on elevation gain, B - precision and accuracy
rcparams.set_rcParams() # plot parameters
plt.rcParams.update({'axes.spines.right': False, 'axes.spines.top': False})
fig, axes = eg_learn.learning_plot(to_plot='average', path=path, w2_exclude = ['cs', 'lm', 'lk'], figsize=(15, 8))
plt.savefig(plot_path / 'eg_learning.svg', format='svg')

# # A - spectral image ef, B and C - spectral change probability m1 and m2
# rcparams.set_rcParams_poster() # plot parameters
# plt.rcParams.update({'axes.spines.right': True, 'axes.spines.top': True})
# figsize = [42, 13]
# hrtf_plot.spectral_overview(main_df, cm_figsize=figsize)
# plt.savefig(plot_path / 'spectral_overview.svg', format='svg')

# rcparams.set_rcParams() # plot parameters
# # evolution of response pattern
# plt.rcParams.update({'axes.spines.right': True, 'axes.spines.top': True})
# fig, axis = loc_plot.response_evolution(to_plot='average', figsize=(14,7.5))
# # plt.savefig(plot_path / 'fig7' / 'fig7.svg', format='svg')