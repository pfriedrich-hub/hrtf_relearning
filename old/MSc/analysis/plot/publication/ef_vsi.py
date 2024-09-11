from matplotlib import pyplot as plt
import old.MSc.analysis.plot.hrtf_plot as hrtf_plot
from old.MSc.misc.unit_conversion import cm2in
import old.MSc.analysis.plot.spectral_behavior_collection as sp_beh_plot
import old.MSc.analysis.statistics.stats_df as stats_df

def ef_vsi(hrtf_df, main_df, figsize):
    plt.rcParams.update({'font.family':'Helvetica',
                         'axes.spines.right': False, 'axes.spines.top': False, 'axes.labelsize': '8'})
    dpi = 264
    fig_widht, fig_height = cm2in(figsize[0]), cm2in(figsize[1])
    fig = plt.figure(figsize=(fig_widht, fig_height), constrained_layout=True, dpi=dpi)
    ax1 = plt.subplot2grid(shape=(3, 2), loc=(0, 0), colspan=2, rowspan=2)
    ax2 = plt.subplot2grid(shape=(3, 2), loc=(2, 0), colspan=1)
    ax3 = plt.subplot2grid(shape=(3, 2), loc=(2, 1), rowspan=2)
    # fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.5)
    # vsi across bands
    hrtf_plot.plot_mean_vsi_across_bands(hrtf_df, condition='Ears Free', bands=None, axis=ax1, ear_idx=[0])
    # L/R vsi
    main_df = stats_df.add_hrtf_stats(main_df, bandwidth=(5700, 11300), vsi_dis_bw=(5700, 13500))
    sp_beh_plot.vsi_ef_l_r_pub(main_df, axis=ax2)
    # ef rmse vsi
    main_df = stats_df.add_hrtf_stats(main_df, bandwidth=(5700, 13500))  # edit dev bw for next line
    sp_beh_plot.ef_vsi_pub(main_df, measure='vertical RMSE', axis=ax3)
    ax1.annotate('A', (-.2, 1.005), c='k', weight='bold', xycoords='axes fraction')
    ax2.annotate('B', (-.55, 1.01), c='k', weight='bold', xycoords='axes fraction')
    ax3.annotate('C', (-.55, 1.01), c='k', weight='bold', xycoords='axes fraction')
    # plt.tight_layout(pad=.5, h_pad=None, w_pad=None, rect=None)
