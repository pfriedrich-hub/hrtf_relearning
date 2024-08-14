import MSc.analysis.statistics.stats_df as stats_df
from matplotlib import pyplot as plt
from MSc.misc.unit_conversion import cm2in
import MSc.analysis.plot.spectral_behavior_collection as sp_beh_plot
from MSc.misc.rcparams import set_rcParams

def acoustic_behavior(main_df, figsize):
    dpi = 264
    fs = 8  # label fontsize
    lw = .5
    plt.rcParams.update({'font.family':'Helvetica', 'axes.spines.right': False, 'axes.spines.top': False, 'axes.labelsize': fs,
                         'ytick.labelsize': fs, 'xtick.labelsize': fs, 'boxplot.medianprops.color': 'black',
                         'boxplot.medianprops.linewidth': .5})

    fig_widht, fig_height = cm2in(figsize[0]), cm2in(figsize[1])
    main_df = stats_df.add_hrtf_stats(main_df, bandwidth=(5700, 11300), vsi_dis_bw=(5700, 13500))
    # acoustic effect on behavioral impact - figure box
    fig, axes = plt.subplots(1,3, figsize=(fig_widht, fig_height), sharey=True, layout='constrained', dpi=dpi)
    sp_beh_plot.th_d0dr_vsi_dis(main_df, measure='vertical RMSE', axis=axes[0])
    sp_beh_plot.th_d5dr_vsi_dis(main_df, measure='vertical RMSE', axis=axes[1])
    sp_beh_plot.th_d5dr_vsi_dis_m1m2(main_df, measure='vertical RMSE', axis=axes[2])
    axes[0].annotate('A', (.05, .89), c='k', weight='bold', xycoords='axes fraction')
    axes[1].annotate('B', (.05, .89), c='k', weight='bold', xycoords='axes fraction')
    axes[2].annotate('C', (.05, .89), c='k', weight='bold', xycoords='axes fraction')
    axes[1].set_ylabel('')
    axes[2].set_ylabel('')
    axes[0].set_xlabel('')
    axes[2].set_xlabel('')
    for ax in axes:
        ax.set_box_aspect(1)
    axes[0].set_title('Free vs Mold 1', fontsize=fs)
    axes[1].set_title('Free vs Mold 2', fontsize=fs)
    axes[2].set_title('Mold 1 vs Mold 2', fontsize=fs)
