from pathlib import Path
import analysis.hrtf_analysis as hrtf_analysis


def plot_hrtf_diff(hrtf_df, to_plot='average', dfe=True, axis=None):
    """
    to_plot: 'average' or subject id
    """
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
    # get difference DTFs
    hrtf_dict['diff_ef_m1'] = hrtf_analysis.hrtf_difference(hrtf_ef, hrtf_m1)
    hrtf_dict['diff_ef_m2'] = hrtf_analysis.hrtf_difference(hrtf_ef, hrtf_m2)
    hrtf_dict['diff_m1_m2'] = hrtf_analysis.hrtf_difference(hrtf_m1, hrtf_m2)
    # hrtf_dict['diff_diff_ef_m1_diff_ef_m2'] = hrtf_analysis.hrtf_difference(diff_ef_m1, diff_ef_m2)

    # apply diffuse field equalization
    if dfe:
        for key in hrtf_dict:
            hrtf_dict[key] = hrtf_dict[key].diffuse_field_equalization()

    # --- plot ---- #
    # get amplitude range across DTFs for common color bar
    frequencies = diff_ef_m1[0].frequencies
    frequencies = numpy.linspace(0, frequencies[-1], n_bins)
    freqidx = numpy.logical_and(frequencies > bandwidth[0], frequencies < bandwidth[1])
    dtfs = []
    for hrtf in hrtf_dict.values():
        dtfs.append(hrtf.tfs_from_sources(src_idx, n_bins)[:, freqidx])
    z_min, z_max = numpy.floor(numpy.min(dtfs)), numpy.ceil(numpy.max(dtfs))

    if not axis:
        fig, axes = plt.subplots(2, 3, sharey=False, sharex=True, figsize=(13, 8))
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
    fig.text(0.5, 0.04, 'Frequency (kHz)', ha='center', size=13)
    fig.text(0.03, 0.5, 'Elevation (degrees)', va='center', rotation='vertical', size=13)

    # add titles
    axes[1, 0].set_title('Ears Free')
    axes[0, 1].set_title('M1')
    axes[1, 1].set_title('M2')
    axes[0, 2].set_title('Difference Ears Free / M1')
    axes[1, 2].set_title('Difference Ears Free / M2')
    return fig, axis


def hrtf_image(hrtf, bandwidth=(4000, 16000), n_bins=300, axis=None, z_min=None, z_max=None, cbar=True):
    src = hrtf.cone_sources(0)
    elevations = hrtf.sources.vertical_polar[src, 1]
    ticks = [str(x) for x in (numpy.arange(4000, 16000 + 1, 4000) / 1000).astype('int')]
    img = numpy.zeros((n_bins, len(src)))
    if not axis:
        fig, axis = plt.subplots()
    else:
        fig = axis.get_figure()
    for idx, source in enumerate(src):
        filt = hrtf[source]
        freqs, h = filt.tf(channels=0, n_bins=n_bins, show=False)
        img[:, idx] = h.flatten()
    img[img < -40] = -40  # clip at -40 dB transfer
    freq_idx = numpy.logical_and(freqs >= bandwidth[0], freqs <= bandwidth[1])
    if not z_min:
        z_min = numpy.floor(numpy.min(img[freq_idx]))
    if not z_max:
        z_max = numpy.ceil(numpy.max(img[freq_idx]))
    cbar_levels = numpy.linspace(z_min, z_max, 100)
    contour = axis.contourf(freqs[freq_idx], elevations, img.T[:, freq_idx],
                        cmap='RdYlBu', origin='upper', levels=cbar_levels)
    axis.set_yticks(numpy.linspace(-30, 30, 5))
    axis.set_xticks(numpy.linspace(4500, 15500, 4))
    axis.set_xticklabels(ticks)
    axis.tick_params(axis='both', direction="in", bottom=True, top=True, left=True, right=True,
                           labelsize=13, width=1.5, length=2)
    if cbar:
        cbar_ticks = numpy.arange(z_min, z_max, 4)
        cax_pos = list(axis.get_position().bounds)  # (x0, y0, width, height)
        cax_pos[0] += 0.26  # x0
        cax_pos[2] = 0.012  # width
        cax = fig.add_axes(cax_pos)
        cbar = fig.colorbar(contour, cax, orientation="vertical", ticks=cbar_ticks)
        cax.tick_params(axis='both', direction="in", bottom=True, top=True, left=True, right=True,
                               labelsize=13, width=1.5, length=2)
        cax.set_title('dB')
    return axis


def plot_hrtf_corr(hrtf_df, to_plot='average', dfe=True, axis=None):
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
        axis[i].text(-30, 30, 'VSI dissimilarity: %.2f' % vsi_dis, size=10)
    fig.text(0.5, 0.02, 'Elevation (degrees)', ha='center', size=13)
    fig.text(0.07, 0.5, 'Elevation (degrees)', va='center', rotation='vertical', size=13)
    if title:
        fig.suptitle(title)
    return fig, axis

def plot_average(hrtf_df, condition='Ears Free'):
    hrtf_list = list(hrtf_df[hrtf_df['condition'] == condition]['hrtf'])
    mean_hrtf = hrtf_analysis.average_hrtf(hrtf_list)
    mean_hrtf = mean_hrtf.diffuse_field_equalization()
    axis = hrtf_image(mean_hrtf)
    cbar = axis.get_figure().get_axes()[1]
    cbar_pos = cbar.get_position()  # (x0, y0, width, height)
    cbar_pos.x1 = 0.94
    cbar_pos.x0 = 0.92
    cbar.set_position(cbar_pos)


def plot_vsi_across_bands(vsi_across_bands, axis=None):
    if not axis:
        fig, axis = plt.subplots()
    axis.plot(vsi_across_bands, c='k')
    axis.set_xticks([0, 1, 2, 3, 4])
    labels = [item.get_text() for item in axis.get_xticklabels()]
    bandwidths = numpy.array(((4, 8), (4.8, 9.5), (5.7, 11.3), (6.7, 13.5), (8, 16))) * 1000
    for idx, band in enumerate(bandwidths / 1000):
        labels[idx] = '%.1f - %.1f' % (band[0], band[1])
    axis.set_xticklabels(labels)
    axis.set_xlabel('Frequency bands (kHz)')
    axis.set_ylabel('VSI')
