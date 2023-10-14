import analysis.statistics.stats_df as stats_df
import analysis.plot.hrtf_plot as hrtf_plot
from analysis.plot import stats_plot_collection as stats_plot
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path
from matplotlib import pyplot as plt
main_df = stats_df.main_dataframe(Path.cwd() / 'data' / 'experiment' / 'master')

# bandwidth = (5700, 8000)
# bandwidth = (5700, 11300)  # 2015 - strong differences in vsi and spectral str across conditions (shift of spectral features outside bandwidth?)
# bandwidth = (3000, 15000) #
bandwidth = (3700, 12900) # 1999, good option for vsi efm1m2 spectral strength - phys. plausable

"""  --- compare spectral features left and right ear across condtions ---  """

main_df = stats_df.add_l_r_comparison(main_df, bandwidth)

# VSI / spectral strength across conditions
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
stats_plot.vsi_l_r(main_df, axis=axes[0, 0])
stats_plot.sp_str_l_r(main_df, axis=axes[0, 1])
stats_plot.boxplot_vsi(main_df, axis=axes[1, 0])
stats_plot.boxplot_sp_str(main_df, axis=axes[1, 1])
fig.suptitle('left vs right ear spectral features')

# VSI dissimilarity / spectral difference across conditions
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
stats_plot.scatter_vsi_dis_l_r(main_df, axis=axes[0, 0])
stats_plot.scatter_sp_dif_l_r(main_df, axis=axes[0, 1])
stats_plot.boxplot_vsi_dis(main_df, axis=axes[1, 0])
stats_plot.boxplot_sp_dif(main_df, axis=axes[1, 1])
fig.suptitle('VSI dissimilarity / spectral difference across conditions')

# VSI dissimilarity / spectral difference between participants' free ears
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
stats_plot.scatter_perm_vsi_dis(main_df, bandwidth, axis=axes[0])
stats_plot.scatter_perm_sp_dif(main_df, bandwidth, axis=axes[1])
fig.suptitle("VSI dissimilarity / spectral difference between participants' free ears")


# probability maps of spectral change induced by molds
fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
hrtf_plot.plot_spectral_change_p(main_df, condition=0, bandwidth=(4000, 16000), axis=axes[0], cbar=False)
hrtf_plot.plot_spectral_change_p(main_df, condition=1, bandwidth=(4000, 16000), axis=axes[1], cbar=False)
hrtf_plot.plot_spectral_change_p(main_df, condition=2, bandwidth=(4000, 16000), axis=axes[2], cbar=True)
axes[0].set_title('EF M1')
axes[1].set_title('EF M2')
axes[2].set_title('M1 M2')
