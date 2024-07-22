import MSc.analysis.statistics.stats_df as stats_df
from matplotlib import pyplot as plt
from MSc.misc.unit_conversion import cm2in
import MSc.analysis.plot.spectral_behavior_collection as sp_beh_plot
from MSc.misc.rcparams import set_rcParams


def mold_vsi(main_df, figsize):
    set_rcParams()
    plt.rcParams.update({'axes.spines.right': False, 'axes.spines.top': False, 'axes.labelsize': '8',
                         'ytick.labelsize': '8', 'xtick.labelsize': '8', 'boxplot.medianprops.color': 'black',
                         'boxplot.medianprops.linewidth': .5})
    dpi = 264
    fig_widht, fig_height = cm2in(figsize[0]), cm2in(figsize[1])
    main_df = stats_df.add_hrtf_stats(main_df, bandwidth=(5700, 11300), vsi_dis_bw=(5700, 13500))
    fig, axes = plt.subplots(2, 2, figsize=(fig_widht, fig_height), gridspec_kw={'width_ratios': [1.2, 1]},
                             layout='constrained', dpi=dpi)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)
    sp_beh_plot.vsi_m_l_r_pub(main_df, axis=axes[0, 0])
    sp_beh_plot.boxplot_vsi_pub(main_df, axis=axes[0, 1])
    sp_beh_plot.scatter_perm_vsi_dis_pub(main_df, bandwidth=(5700, 13500), axis=axes[1, 0])
    sp_beh_plot.boxplot_vsi_dis_pub(main_df, axis=axes[1, 1])
    axes[0, 0].annotate('A', (-.4, 1.1), c='k', weight='bold', xycoords='axes fraction')
    axes[0, 1].annotate('B', (-.4, 1.1), c='k', weight='bold', xycoords='axes fraction')
    axes[1, 0].annotate('C', (-.4, 1.1), c='k', weight='bold', xycoords='axes fraction')
    axes[1, 1].annotate('D', (-.4, 1.1), c='k', weight='bold', xycoords='axes fraction')
