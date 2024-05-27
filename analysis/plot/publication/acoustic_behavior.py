import analysis.statistics.stats_df as stats_df
from matplotlib import pyplot as plt
from misc.unit_conversion import cm2in
import analysis.plot.spectral_behavior_collection as sp_beh_plot
from misc.rcparams import set_rcParams

def acoustic_behavior(main_df, figsize):
    set_rcParams()
    plt.rcParams.update({'axes.spines.right': False, 'axes.spines.top': False, 'axes.labelsize': '8',
                         'ytick.labelsize': '8', 'xtick.labelsize': '8', 'boxplot.medianprops.color': 'black',
                         'boxplot.medianprops.linewidth': .5})
    dpi = 264
    fig_widht, fig_height = cm2in(figsize[0]), cm2in(figsize[1])
    main_df = stats_df.add_hrtf_stats(main_df, bandwidth=(5700, 11300), vsi_dis_bw=(5700, 13500))
    # acoustic effect on behavioral impact - figure box
    fig, axes = plt.subplots(1,3, figsize=(fig_widht, fig_height), sharey=True, layout='constrained')
    sp_beh_plot.th_d0dr_vsi_dis(main_df, measure='vertical RMSE', axis=axes[0])
    sp_beh_plot.th_d5dr_vsi_dis(main_df, measure='vertical RMSE', axis=axes[1])
    sp_beh_plot.th_d5dr_vsi_dis_m1m2(main_df, measure='vertical RMSE', axis=axes[2])
    axes[0].annotate('A', (-.1, 1.1), c='k', weight='bold', xycoords='axes fraction')
    axes[1].annotate('B', (-.1, 1.1), c='k', weight='bold', xycoords='axes fraction')
    axes[2].annotate('C', (-.1, 1.1), c='k', weight='bold', xycoords='axes fraction')
    axes[1].set_ylabel('')
    axes[2].set_ylabel('')
    axes[0].set_xlabel('')
    axes[2].set_xlabel('')
