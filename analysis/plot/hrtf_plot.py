import matplotlib.ticker
import analysis.hrtf_analysis as hrtf_analysis
import analysis.processing.hrtf_processing as hrtf_processing
import analysis.statistics.stats_df as stats_df
import numpy
import matplotlib.colors as colors
from matplotlib import pyplot as plt
import copy
import cmasher as cmr
from misc.unit_conversion import cm2in


def compare_spectral_change_p(main_df, axis=None, bandwidth=(4000, 16000), width=18, height=6):
    width = cm2in(width)
    height = cm2in(height)
    labelsize=8
    if not axis:
        fig, axes = plt.subplots(1, 3, sharey=False, sharex=True, figsize=(width, height), subplot_kw=dict(box_aspect=1))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05)
    threshold = None  # calculate threshold as rms between participants free ears dtfs
    plot_spectral_change_p(main_df, 0, threshold, bandwidth, axes[0], False)
    plot_spectral_change_p(main_df, 1, threshold, bandwidth, axes[1], False)
    plot_spectral_change_p(main_df, 2, threshold, bandwidth, axes[2], True)
    axes[1].set_yticks([])
    axes[2].set_yticks([])
    # add labels
    axes[1].set_xlabel('Frequency (kHz)', size=labelsize)
    axes[0].set_ylabel('Elevation (degrees)', size=labelsize)

    axes[0].set_title('Free ears vs Molds 1', size=labelsize)
    axes[1].set_title('Free ears vs Molds 2', size=labelsize)
    axes[2].set_title('Molds 1 vs Molds 2', size=labelsize)

    subpl_labels = ['A', 'B', 'C']
    for ax_id, ax in enumerate(axes):
        label_x = axes[ax_id].get_position().x0 + 0.005
        label_y = axes[ax_id].get_position().y1 - 0.07  # todo should be dynamic
        fig.text(x=label_x, y=label_y, s=subpl_labels[ax_id], size=labelsize+6, c='w', weight='bold')

    return fig, axis

def hrtf_compare(hrtf_df, axis=None, average_ears=True, hrtf_diff=False, zlim=(-11,6), width=12, height=4):
    width = cm2in(width)
    height = cm2in(height)
    labelsize = 8
    """
    plot mean subject dtfs for all conditions
    """
    bandwidth = (4000, 16000)
    hrtf_dict = dict()
    # get hrtfs
    hrtf_dict['hrtf_ef'] = hrtf_processing.average_hrtf(list(hrtf_df[hrtf_df['condition'] == 'Ears Free']['hrtf']),
                                                        average_ears)
    hrtf_dict['hrtf_m1'] = hrtf_processing.average_hrtf(
        list(hrtf_df[hrtf_df['condition'] == 'Earmolds Week 1']['hrtf']), average_ears)
    hrtf_dict['hrtf_m2'] = hrtf_processing.average_hrtf(
        list(hrtf_df[hrtf_df['condition'] == 'Earmolds Week 2']['hrtf']), average_ears)
    if hrtf_diff:
        hrtf_dict['hrtf_ef'] = hrtf_analysis.hrtf_difference(hrtf_dict['hrtf_ef'], hrtf_dict['hrtf_m1'])
        hrtf_dict['hrtf_m1'] = hrtf_analysis.hrtf_difference(hrtf_dict['hrtf_ef'], hrtf_dict['hrtf_m2'])
        hrtf_dict['hrtf_m2'] = hrtf_analysis.hrtf_difference(hrtf_dict['hrtf_m1'], hrtf_dict['hrtf_m2'])


    # --- plot ---- #
    # if not n_bins:
    n_bins = hrtf_dict['hrtf_ef'][0].n_frequencies
    # get amplitude range across DTFs for common color bar
    frequencies = hrtf_dict['hrtf_ef'][0].frequencies
    frequencies = numpy.linspace(0, frequencies[-1], n_bins)
    freqidx = numpy.logical_and(frequencies > bandwidth[0], frequencies < bandwidth[1])
    dtfs = []
    for hrtf in hrtf_dict.values():
        sources = hrtf.cone_sources(0)
        dtfs.append(hrtf.tfs_from_sources(sources, n_bins)[:, freqidx])

    if not zlim:
        z_min, z_max = numpy.floor(numpy.min(dtfs)), numpy.ceil(numpy.max(dtfs))
    else:
        z_min, z_max = zlim[0], zlim[1]

    if not axis:
        fig, axes = plt.subplots(1, 3, sharey=False, sharex=True, figsize=(width, height), subplot_kw=dict(box_aspect=1))

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05)

    # plot
    hrtf_image(hrtf_dict['hrtf_ef'], bandwidth, n_bins, axes[0], z_min=z_min, z_max=z_max, cbar=False)
    hrtf_image(hrtf_dict['hrtf_m1'], bandwidth, n_bins, axes[1], z_min=z_min, z_max=z_max, cbar=False)
    hrtf_image(hrtf_dict['hrtf_m2'], bandwidth, n_bins, axes[2], z_min=z_min, z_max=z_max, cbar=True)

    # remove tick labels in rows
    axes[1].set_yticks([])
    axes[2].set_yticks([])

    # add labels
    axes[1].set_xlabel('Frequency (kHz)', size=labelsize)
    axes[0].set_ylabel('Elevation (degrees)', size=labelsize)
    #
    # fig.text(0.5, 0.01, 'Frequency (kHz)', ha='center', size=labelsize)
    # fig.text(0.075, 0.5, 'Elevation (degrees)', va='center', rotation='vertical', size=labelsize)

    # add titles
    if not hrtf_diff:
        axes[0].set_title('Free ears', size=labelsize)
        axes[1].set_title('Molds 1', size=labelsize)
        axes[2].set_title('Molds 2', size=labelsize)
    else:
        axes[0].set_title('Difference Ears Free / Earmolds 1', size=labelsize)
        axes[1].set_title('Difference Ears Free / Earmolds 2', size=labelsize)
        axes[2].set_title('Difference Earmolds 1 / Earmolds 2', size=labelsize)

    subpl_labels = ['A', 'B', 'C']
    for ax_id, ax in enumerate(axes):
        label_x = axes[ax_id].get_position().x0 + 0.01 # left
        label_y = axes[ax_id].get_position().y1 - 0.1  # down # todo should be dynamic
        fig.text(x=label_x, y=label_y, s=subpl_labels[ax_id], size=labelsize+6, c='w', weight='bold')
    plt.show()
    return fig, axes

def plot_spectral_change_p(main_df, condition, threshold, bandwidth=(4000, 16000), axis=None, cbar=True):
    labelsize=8
    change_p, _ = stats_df.spectral_change_p(main_df, threshold)
    change_p = change_p[condition]
    src = main_df['EF hrtf'][0].cone_sources(0)
    elevations = main_df['EF hrtf'][0].sources.vertical_polar[src, 1]
    frequencies = main_df['EF hrtf'][0][0].frequencies
    ticks = [str(x) for x in (numpy.arange(4000, 16000 + 1, 4000) / 1000).astype('int')]
    if not axis:
        fig, axis = plt.subplots()
    else:
        fig = axis.get_figure()
    freq_idx = numpy.logical_and(frequencies >= bandwidth[0], frequencies <= bandwidth[1])
    cbar_levels = numpy.linspace(0, 1, 100)
    cmap = plt.get_cmap('cmr.rainforest_r')
    # cmap = 'RdYlBu'
    # cmap = plt.get_cmap('viridis_r')
    cmap = truncate_colormap(cmap, minval=0.05, maxval=0.7, n=512)
    contour = axis.contourf(frequencies[freq_idx], elevations, change_p[:, freq_idx],
                        cmap=cmap, origin='upper', levels=cbar_levels)
    axis.tick_params(axis='both', direction="in", bottom=True, top=True, left=True, right=True, reset=False,
                           labelsize=labelsize, width=1.5, length=2)
    # xticks
    axis.tick_params(axis='both', direction="in", bottom=True, top=True, left=True, right=True,
                           labelsize=labelsize, width=1.5, length=2)
    x_tick_pos = [(x) for x in (numpy.arange(bandwidth[0], bandwidth[1] + 1, 4000)).astype('int')]
    x_tick_labels = [str((x / 1000).astype('int')) for x in x_tick_pos]
    axis.set_xticklabels(x_tick_labels)
    x_ticks = axis.set_xticks(x_tick_pos)
    x_ticks[0]._apply_params(width=0)
    x_ticks[-1]._apply_params(width=0)
    try:
        axis.get_xticklabels()[0].set_horizontalalignment('left')
        axis.get_xticklabels()[-1].set_horizontalalignment('right')
    except IndexError:
        pass
     # this has to run
    axis.set_yticks(numpy.linspace(-30, 30, 5))
    axis.set_xticks(numpy.linspace(4000, 16000, 4))
    axis.set_xticklabels(ticks)

    if cbar:
        cbar_ticks = numpy.arange(0, 1.1, .2)
        cax_pos = list(axis.get_position().bounds)  # (x0, y0, width, height)
        cax_pos[0] = 0.92 # x0
        cax_pos[2] = 0.015  # width
        cax = fig.add_axes(cax_pos)
        cbar = fig.colorbar(contour, cax, orientation="vertical", ticks=cbar_ticks)
        cax.tick_params(axis='both', direction="in", bottom=True, top=True, left=True, right=True,
                               labelsize=labelsize, width=1.5, length=2)
        cax.set_title('p', size=labelsize)
    return axis

def hrtf_image(hrtf, bandwidth=(4000, 16000), n_bins=None, axis=None, chan=0, z_min=None, z_max=None, cbar=True,
               labels=False):
    labelsize=8
    src = hrtf.cone_sources(0)
    elevations = hrtf.sources.vertical_polar[src, 1]
    if n_bins is None:
        n_bins = hrtf.data[0].n_frequencies
    img = numpy.zeros((n_bins, len(src)))
    for idx, source in enumerate(src):
        filt = hrtf[source]
        freqs, h = filt.tf(channels=chan, n_bins=n_bins, show=False)
        img[:, idx] = h.flatten()
    img[img < -40] = -40  # clip at -40 dB transfer
    freq_idx = numpy.logical_and(freqs >= bandwidth[0], freqs <= bandwidth[1])
    if not z_min:
        z_min = numpy.floor(numpy.min(img[freq_idx]))
    if not z_max:
        z_max = numpy.ceil(numpy.max(img[freq_idx]))
    cbar_levels = numpy.linspace(z_min, z_max, 100)
    # plot
    if not axis:
        fig, axis = plt.subplots()
    else:
        fig = axis.get_figure()
    cmap = plt.get_cmap('cmr.rainforest_r')
    # cmap = 'RdYlBu'
    # cmap = plt.get_cmap('viridis_r')
    cmap = truncate_colormap(cmap, minval=0.05, maxval=0.7, n=512)
    contour = axis.contourf(freqs[freq_idx], elevations, img.T[:, freq_idx],
                        cmap=cmap, origin='upper', levels=cbar_levels)
    axis.tick_params(axis='both', direction="in", bottom=True, top=True, left=True, right=True, reset=False,
                           labelsize=labelsize, width=1.5, length=2)
    # xticks
    x_tick_pos = [(x) for x in (numpy.arange(bandwidth[0], bandwidth[1] + 1, 4000)).astype('int')]
    x_tick_labels = [str((x / 1000).astype('int')) for x in x_tick_pos]
    axis.set_xticklabels(x_tick_labels)
    x_ticks = axis.set_xticks(x_tick_pos)
    x_ticks[0]._apply_params(width=0)
    x_ticks[-1]._apply_params(width=0)
    try:
        axis.get_xticklabels()[0].set_horizontalalignment('left')
        axis.get_xticklabels()[-1].set_horizontalalignment('right')
    except IndexError:
        pass
     # this has to run simultaneously to plot creation to work
    # yticks
    axis.set_yticks(numpy.linspace(-30, 30, 5))
    if cbar:
        cbar_ticks = numpy.arange(z_min, z_max, 4)[1:]
        cax_pos = list(axis.get_position().bounds)  # (x0, y0, width, height)
        cax_pos[0] = 0.92 # x0
        cax_pos[2] = 0.015  # width
        cax = fig.add_axes(cax_pos)
        cbar = fig.colorbar(contour, cax, orientation="vertical", ticks=cbar_ticks)
        cax.tick_params(axis='both', direction="in", bottom=True, top=True, left=True, right=True,
                               labelsize=labelsize, width=1.5, length=2)
        cax.set_title('dB', size=labelsize)
    if labels:
        axis.set_xlabel('Frequency (kHz)', size=labelsize)
        axis.set_ylabel('Elevation (degrees)', size=labelsize)
    return axis


def plot_average(hrtf_df, condition='Ears Free', kind='image'):
    """ plot average hrtf for each condition """
    df = copy.deepcopy(hrtf_df)
    hrtf_list = list(df[df['condition'] == condition]['hrtf'])
    mean_hrtf = hrtf_processing.average_hrtf(hrtf_list)
    # if equalize:
    #     mean_hrtf = mean_hrtf.diffuse_field_equalization()
    if kind == 'image':
        axis = hrtf_image(mean_hrtf, bandwidth=(4000, 16000), n_bins=300)
        cbar = axis.get_figure().get_axes()[1]
        cbar_pos = cbar.get_position()  # (x0, y0, width, height)
        cbar_pos.x1 = 0.94
        cbar_pos.x0 = 0.92
        cbar.set_position(cbar_pos)
    elif kind == 'waterfall':
        mean_hrtf.plot_tf(mean_hrtf.cone_sources())
    return axis


def plot_vsi_across_bands(vsi_across_bands, bands, axis=None):
    if not bands:
        bands = [(4000, 5700), (5700, 8000), (8000, 11300), (11300, 16000)]
    if not axis:
        fig, axis = plt.subplots()
    axis.plot(vsi_across_bands, c='k', lw=0.5)
    axis.set_xticks(numpy.arange(len(bands)))
    labels = [item.get_text() for item in axis.get_xticklabels()]
    for idx, band in enumerate(numpy.asarray(bands) / 1000):
        labels[idx] = '%.1f - %.1f' % (band[0], band[1])
    axis.set_xticklabels(labels)
    axis.set_yticks(numpy.arange(0.1, 1.2, 0.1))
    axis.set_xlabel('Frequency bands (kHz)')
    axis.set_ylabel('VSI')

def plot_vsi_dissimilarity_across_bands(vsi_dissimilarity_across_bands, bands, axis=None):
    if not bands:
        bands = [(4000, 5700), (5700, 8000), (8000, 11300), (11300, 16000)]
    if not axis:
        fig, axis = plt.subplots()
    axis.plot(vsi_dissimilarity_across_bands, c='k', lw=0.5)
    axis.set_xticks(numpy.arange(len(bands)))
    labels = [item.get_text() for item in axis.get_xticklabels()]
    for idx, band in enumerate(numpy.asarray(bands) / 1000):
        labels[idx] = '%.1f - %.1f' % (band[0], band[1])
    axis.set_xticklabels(labels)
    axis.set_yticks(numpy.arange(0.1, 1.2, 0.1))
    axis.set_xlabel('Frequency bands (kHz)')
    axis.set_ylabel('VSI Dissimilarity')


def plot_spectral_strength_across_bands(spectral_strength_across_bands, bands, axis=None):
    if not bands:
        bands = [(4000, 5700), (5700, 8000), (8000, 11300), (11300, 16000)]
    if not axis:
        fig, axis = plt.subplots()
    axis.plot(spectral_strength_across_bands, c='k', lw=0.5)
    axis.set_xticks(numpy.arange(len(bands)))
    labels = [item.get_text() for item in axis.get_xticklabels()]
    for idx, band in enumerate(numpy.asarray(bands) / 1000):
        labels[idx] = '%.1f - %.1f' % (band[0], band[1])
    axis.set_xticklabels(labels)
    axis.set_yticks(numpy.arange(0, 60, 10))
    axis.set_xlabel('Frequency bands (kHz)')
    axis.set_ylabel('spectral strength')

def plot_hrtfs(hrtf_df, condition='Ears Free'):
    hrtf_list = list(hrtf_df[hrtf_df['condition'] == condition]['hrtf'])
    for subj_idx, hrtf in enumerate(hrtf_list):
        hrtf.plot_tf(hrtf.cone_sources())
        subject_id = list(hrtf_df['subject'].unique())[subj_idx]
        plt.title(subject_id)
        plt.savefig(f'/Users/paulfriedrich/Desktop/hrtf_relearning/data/figures/hrtfs/{subject_id}')


def subj_hrtf_vsi(hrtf_df, to_plot='all', condition='Ears Free', bands=None):
    """
    plot hrtfs and vsi across bands for all subjects
    """
    if not bands:
        bands = [(4000, 5700), (5700, 8000), (8000, 11300), (11300, 16000)]  # to modify
    if to_plot != 'all':
        hrtf_df = hrtf_df[hrtf_df['subject'] == to_plot]
    for index, row in hrtf_df[hrtf_df['condition'] == condition].iterrows():
        fig, ax = plt.subplots(2, 1, figsize=(5, 7))
        fig.suptitle(row['subject'])
        hrtf = copy.deepcopy(row['hrtf'])
        hrtf.plot_tf(hrtf.cone_sources(0), axis=ax[0], xlim=(bands[0][0], bands[-1][1]))
        plot_vsi_across_bands(hrtf, bands, axis=ax[1])

def plot_correlation_matrix(correlation_matrix, axis=None, c_bar=True, tiles=False):
    labelsize=8
    if axis is None:
        fig, axis = plt.subplots()
    else:
        fig = axis.get_figure()
    cbar_levels = numpy.linspace(-1, 1, 20)
    if tiles:
        contour = axis.imshow(correlation_matrix, cmap='viridis', vmin=cbar_levels.min(), vmax=cbar_levels.max())
    else:
        contour = axis.contourf(numpy.arange(7),
                                numpy.arange(6, -1, -1), correlation_matrix,
                                cmap='viridis', levels=cbar_levels)
    labels = ['-37.5', '-25.0', '-12.5', '0.0', '12.5', '25.0', '37.5']
    axis.set_xticklabels(labels)
    labels[0] = None
    axis.set_yticklabels(labels)
    axis.tick_params(axis='both', direction="in", bottom=True, top=True, left=True, right=True,
                     labelsize=labelsize, width=1.5, length=2)
    axis.set_ylabel('Elevation (degrees)')
    axis.set_xlabel('Elevation (degrees)')
    if c_bar:
        cbar_ticks = numpy.linspace(-1, 1, 11)
        cax_pos = list(axis.get_position().bounds)  # (x0, y0, width, height)
        cax_pos[0] = 0.92  # x0
        cax_pos[2] = 0.015  # width
        cax = fig.add_axes(cax_pos)
        cbar = fig.colorbar(contour, cax, orientation="vertical", ticks=cbar_ticks)
        cax.tick_params(axis='both', direction="in", bottom=True, top=True, left=True, right=True,
                        labelsize=labelsize, width=1.5, length=2)


def hrtf_overview(hrtf_df, to_plot='average', n_bins=None, dfe=False, axis=None, average_ears=True):
    """
    to_plot: 'average' or subject id
    """
    labelsize=8
    bandwidth = (4000, 16000)
    hrtf_dict = dict()
    # get hrtfs
    if to_plot == 'average':
        hrtf_dict['hrtf_ef'] = hrtf_processing.average_hrtf(list(hrtf_df[hrtf_df['condition'] == 'Ears Free']['hrtf']), average_ears)
        hrtf_dict['hrtf_m1'] = hrtf_processing.average_hrtf(list(hrtf_df[hrtf_df['condition'] == 'Earmolds Week 1']['hrtf']), average_ears)
        hrtf_dict['hrtf_m2'] = hrtf_processing.average_hrtf(list(hrtf_df[hrtf_df['condition'] == 'Earmolds Week 2']['hrtf']), average_ears)
    else:
        hrtf_dict['hrtf_ef'] = list(hrtf_df[hrtf_df['condition'] == 'Ears Free'][hrtf_df['subject'] == to_plot]['hrtf'])[0]
        hrtf_dict['hrtf_m1'] = list(hrtf_df[hrtf_df['condition'] == 'Earmolds Week 1'][hrtf_df['subject'] == to_plot]['hrtf'])[0]
        hrtf_dict['hrtf_m2'] = list(hrtf_df[hrtf_df['condition'] == 'Earmolds Week 2'][hrtf_df['subject'] == to_plot]['hrtf'])[0]

    # get difference DTFs
    hrtf_dict['diff_ef_m1'] = hrtf_analysis.hrtf_difference(hrtf_dict['hrtf_ef'], hrtf_dict['hrtf_m1'])
    hrtf_dict['diff_ef_m2'] = hrtf_analysis.hrtf_difference(hrtf_dict['hrtf_ef'], hrtf_dict['hrtf_m2'])
    hrtf_dict['diff_m1_m2'] = hrtf_analysis.hrtf_difference(hrtf_dict['hrtf_m1'], hrtf_dict['hrtf_m2'])
    # hrtf_dict['diff_diff_ef_m1_diff_ef_m2'] = hrtf_analysis.hrtf_difference(diff_ef_m1, diff_ef_m2)

    # apply diffuse field equalization
    if dfe:
        for key in hrtf_dict:
            hrtf_dict[key] = hrtf_dict[key].diffuse_field_equalization()

    # --- plot ---- #
    if not n_bins:
        n_bins = hrtf_dict['hrtf_ef'][0].n_frequencies
    # get amplitude range across DTFs for common color bar
    frequencies = hrtf_dict['diff_ef_m1'][0].frequencies
    frequencies = numpy.linspace(0, frequencies[-1], n_bins)
    freqidx = numpy.logical_and(frequencies > bandwidth[0], frequencies < bandwidth[1])
    dtfs = []
    for hrtf in hrtf_dict.values():
        sources = hrtf.cone_sources(0)
        dtfs.append(hrtf.tfs_from_sources(sources, n_bins)[:, freqidx])
    z_min, z_max = numpy.floor(numpy.min(dtfs)), numpy.ceil(numpy.max(dtfs))
    if not axis:
        fig, axes = plt.subplots(2, 3, sharey=False, sharex=True, figsize=(12, 8), subplot_kw=dict(box_aspect=1))
    axes[0,0].set_box_aspect(1)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05)
    # move image of free ears hrtf to midline
    axes[0][0].remove()
    pos = axes[1][0].get_position()
    pos.y0 = 0.32
    pos.y1 = 0.67
    pos.x0 -= 0.03
    pos.x1 -= 0.03
    axes[1][0].set_position(pos)

    # plot
    hrtf_image(hrtf_dict['hrtf_ef'], bandwidth, n_bins, axes[1, 0], z_min=z_min, z_max=z_max, cbar=False)
    hrtf_image(hrtf_dict['hrtf_m1'], bandwidth, n_bins, axes[0, 1], z_min=z_min, z_max=z_max, cbar=False)
    hrtf_image(hrtf_dict['hrtf_m2'], bandwidth, n_bins, axes[1, 1], z_min=z_min, z_max=z_max, cbar=False)
    hrtf_image(hrtf_dict['diff_ef_m1'], bandwidth, n_bins, axes[0, 2], z_min=z_min, z_max=z_max, cbar=False)
    hrtf_image(hrtf_dict['diff_ef_m2'], bandwidth, n_bins, axes[1, 2], z_min=z_min, z_max=z_max, cbar=True)

    # plot m1 m2 diff
    fig1, axes1 = plt.subplots(1,1)
    hrtf_image(hrtf_dict['diff_m1_m2'], bandwidth, n_bins, axis=axes1, z_min=z_min, z_max=z_max, cbar=True)
    axes1.set_title('Difference M1 / M2')

    # remove tick labels in rows
    axes[0, 2].set_yticks([])
    axes[1, 2].set_yticks([])

    # move colorbar
    cbar = fig.get_axes()[5]
    cbar_pos = cbar.get_position()
    cbar_pos.y0 = 0.109
    cbar_pos.y1 = 0.88
    cbar.set_position(cbar_pos)

    # add labels
    fig.text(0.5, 0.04, 'Frequency (kHz)', ha='center', size=labelsize)
    fig.text(0.03, 0.5, 'Elevation (degrees)', va='center', rotation='vertical', size=labelsize)

    # add titles
    axes[1, 0].set_title('Ears Free')
    axes[0, 1].set_title('M1')
    axes[1, 1].set_title('M2')
    axes[0, 2].set_title('Difference Ears Free / M1')
    axes[1, 2].set_title('Difference Ears Free / M2')
    return fig, axis

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(numpy.linspace(minval, maxval, n)))
    return new_cmap

"""
# ---- deprecated ---- #
def plot_vsi_across_bands_old(hrtf, bands, n_bins, axis=None):
    vsi_across_bands = hrtf_analysis.vsi_across_bands_old(hrtf, bands, n_bins)
    if not bands:
        bands = [(4000, 5700), (5700, 8000), (8000, 11300), (11300, 16000)]
    if not axis:
        fig, axis = plt.subplots()
    axis.plot(vsi_across_bands, c='k', lw=0.5)
    axis.set_xticks(numpy.arange(len(bands)))
    labels = [item.get_text() for item in axis.get_xticklabels()]
    for idx, band in enumerate(numpy.asarray(bands) / 1000):
        labels[idx] = '%.1f - %.1f' % (band[0], band[1])
    axis.set_xticklabels(labels)
    axis.set_yticks(numpy.arange(0.1, 1.2, 0.1))
    axis.set_xlabel('Frequency bands (kHz)')
    axis.set_ylabel('VSI')

def hrtf_correlation(hrtf_df, to_plot='average', n_bins=None, dfe=True, axis=None):
    n_bins = 150
    bandwidth = (4000, 16000)
    hrtf_dict = dict()
    # get hrtfs
    if to_plot == 'average':
        hrtf_dict['hrtf_ef'] = hrtf_analysis.average_hrtf(list(hrtf_df[hrtf_df['condition'] == 'Ears Free']['hrtf']))
        hrtf_dict['hrtf_m1'] = hrtf_analysis.average_hrtf(list(hrtf_df[hrtf_df['condition'] == 'Earmolds Week 1']['hrtf']))
        hrtf_dict['hrtf_m2'] = hrtf_analysis.average_hrtf(list(hrtf_df[hrtf_df['condition'] == 'Earmolds Week 2']['hrtf']))
    else:
        hrtf_dict['hrtf_ef'] = hrtf_df[hrtf_df['condition'] == 'Ears Free'][hrtf_df['subject'] == to_plot]['hrtf']
        hrtf_dict['hrtf_m1'] = hrtf_df[hrtf_df['condition'] == 'Earmolds Week 1'][hrtf_df['subject'] == to_plot]['hrtf']
        hrtf_dict['hrtf_m2'] = hrtf_df[hrtf_df['condition'] == 'Earmolds Week 2'][hrtf_df['subject'] == to_plot]['hrtf']

    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(12, 4))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05)
    corr_ef_m1 = hrtf_analysis.hrtf_correlation(hrtf_dict['hrtf_ef'], hrtf_dict['hrtf_m1'], show=True, axis=axes[0],
                                                bandwidth=bandwidth, cbar=False, n_bins=n_bins)
    corr_ef_m2= hrtf_analysis.hrtf_correlation(hrtf_dict['hrtf_ef'], hrtf_dict['hrtf_m2'], show=True, axis=axes[1],
                                               bandwidth=bandwidth, cbar=False, n_bins=n_bins)
    corr_m1_m2= hrtf_analysis.hrtf_correlation(hrtf_dict['hrtf_m1'], hrtf_dict['hrtf_m2'], show=True, axis=axes[2],
                                               bandwidth=bandwidth, cbar=True, n_bins=n_bins)

    cbar = False
    for i in range(3):
        if i == 2:
            cbar = True
        correlation.append(hrtf_correlation(dict[compare[i][0]], dict[compare[i][1]], show=True, axis=axis[i],
                                            bandwidth=bandwidth, cbar=cbar, n_bins=n_bins))
        axis[i].set_title(title_list[2][i])
        vsi_dis = vsi_dissimilarity(dict[compare[i][0]], dict[compare[i][1]], bandwidth)
        axis[i].text(-30, 30, 'VSI dissimilarity: %.2f' % vsi_dis, size=labelsize)
    fig.text(0.5, 0.02, 'Elevation (degrees)', ha='center', size=labelsize)
    fig.text(0.07, 0.5, 'Elevation (degrees)', va='center', rotation='vertical', size=labelsize)
    if title:
        fig.suptitle(title)
    return fig, axis
# def subj_hrtf_vsi_dis(hrtf_df, to_plot='all', conditions=('Ears Free', 'Earmolds Week 1'), bands=None)
#     
#     plot hrtfs and vsi dissimilarity across bands for all subjects
#     
#     if not bands:
#         bands = [(4000, 5700), (5700, 8000), (8000, 11300), (11300, 16000)]  # to modify
#     if to_plot != 'all':
#         hrtf_df = hrtf_df[hrtf_df['subject'] == to_plot]
#     for index, row in hrtf_df:
#         a=row
#
#     for index, row in hrtf_df[hrtf_df['condition'] == conditions[0]].iterrows():
#         fig, ax = plt.subplots(3, 1, figsize=(5, 7))
#         fig.suptitle(row['subject'])
#         hrtf = copy.deepcopy(row['hrtf'])
#         hrtf.plot_tf(hrtf.cone_sources(0), axis=ax[0], xlim=(bands[0][0], bands[-1][1]))
#         plot_vsi_across_bands(hrtf, bands, axis=ax[1])

"""