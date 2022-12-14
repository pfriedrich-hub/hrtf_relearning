import os
import slab
from pathlib import Path
import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy
import copy

# todo: fix problem of resolution downscaling with frequency for vsi:
#  for example kemar vsi with 0.1s looks better than for 0.05 seconds (higher frequencies have to
#  few samples here to be represented correctly in vsi)

def list_hrtfs(path, condition):
    subject_dir_list = list(path.iterdir())
    hrtf_list = []
    file_list = []
    for subj_idx, subject_path in enumerate(subject_dir_list):
        subject_dir = subject_path / condition
        # iterate over localization accuracy files
        for file_name in sorted(list(subject_dir.iterdir())):
            if file_name.is_file() and file_name.suffix == '.sofa':
                hrtf_list.append(slab.HRTF(file_name))
                file_list.append(file_name.name)
    return hrtf_list

def baseline_hrtf(hrtf, bandwidth=(3000, 17000)):
    "Center transfer functions around 0"
    hrtf_out = copy.deepcopy(hrtf)
    for src_idx, tf in enumerate(hrtf_out):
        db_data = 20 * numpy.log10(tf.data)
        # set values outside of freq_range to mean of data in freq range
        in_range = db_data[numpy.logical_and(tf.frequencies > bandwidth[0],
                                 tf.frequencies < bandwidth[1])]
        mean = numpy.mean(in_range, axis=0)
        db_data[numpy.logical_or(tf.frequencies < bandwidth[0],
                                 tf.frequencies > bandwidth[1])] = mean
        # center data around zero
        db_data -= mean
        tf_data = 10 ** (db_data/20)
        hrtf_out[src_idx].data = tf_data
    return hrtf_out

def average_hrtf(hrtf_list):
    tf_data = numpy.zeros((hrtf_list[0].n_sources, len(hrtf_list), hrtf_list[0][0].n_samples, 2))
    for hrtf_idx, hrtf in enumerate(hrtf_list):
        for src_idx, tf in enumerate(hrtf.data):
            tf_data[src_idx, hrtf_idx] = tf.data
    tf_data = numpy.mean(tf_data, axis=1)
    for src_idx, tf_data in enumerate(tf_data):
        hrtf[src_idx].data = tf_data
    return hrtf

def hrtf_difference(hrtf_1, hrtf_2):
    hrtf_1 = copy.deepcopy(hrtf_1)
    hrtf_2 = copy.deepcopy(hrtf_2)
    hrtf_diff = copy.deepcopy(hrtf_1)
    if not hrtf_1.n_elevations == hrtf_2.n_elevations:
        print('HRTFs must have same number of sources!')
    for src_idx in range(hrtf_1.n_elevations):
        hrtf_1_db = 20 * numpy.log10(hrtf_1[src_idx].data)
        hrtf_2_db = 20 * numpy.log10(hrtf_2[src_idx].data)
        hrtf_diff_db = numpy.subtract(hrtf_2_db, hrtf_1_db)
        hrtf_diff[src_idx].data = 10 ** (hrtf_diff_db/20)
    return hrtf_diff

def plot_hrtf_image(hrtf, bandwidth=(4000, 16000), n_bins=300, axis=None, z_min=-10, z_max=8, cbar = False):
    src = hrtf.cone_sources(0)
    elevations = hrtf.sources.vertical_polar[src, 1]
    ticks = [str(x) for x in (numpy.arange(4000, 16000 + 1, 4000) / 1000).astype('int')]
    cbar_levels = numpy.linspace(z_min, z_max, 100)
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

def vsi_across_bands(hrtf, cone=0, n_bins=300, show=True, axis=None):
    # calculate vsi across 1/2 octave frequency bands
    sources = hrtf.cone_sources(cone)
    dtfs = hrtf.tfs_from_sources(sources, n_bins)
    frequencies = numpy.linspace(0, hrtf[0].frequencies[-1], n_bins)
    bandwidths = numpy.array(((4, 8), (4.8, 9.5), (5.7, 11.3), (6.7, 13.5), (8, 16))) * 1000
    vsi = numpy.zeros(len(bandwidths))
    # extract vsi for each band
    for idx, bw in enumerate(bandwidths):
        dtf_band = dtfs[numpy.logical_and(frequencies >= bw[0], frequencies <= bw[1])]
        sum_corr = 0
        n = 0
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                sum_corr += numpy.corrcoef(dtf_band[:, i], dtf_band[:, j])[1, 0]
                n += 1
        vsi[idx] = 1 - sum_corr / n
    if show:
        if not axis:
            fig, axis = plt.subplots()
        axis.plot(vsi, c='k')
        axis.set_xticks([0, 1, 2, 3, 4])
        labels = [item.get_text() for item in axis.get_xticklabels()]
        for idx, band in enumerate(bandwidths / 1000):
            labels[idx] = '%.1f - %.1f' % (band[0], band[1])
        axis.set_xticklabels(labels)
        axis.set_xlabel('Frequency bands (kHz)')
        axis.set_ylabel('VSI')
    return vsi

def hrtf_correlation(hrtf_1, hrtf_2, show=False, bandwidth=(4000, 16000), n_bins=300, axis=None, cbar=True):
    # get sources and dtfs
    sources = hrtf_1.cone_sources(0)
    dtf_1 = hrtf_1.tfs_from_sources(sources, n_bins)
    dtf_2 = hrtf_2.tfs_from_sources(sources, n_bins)
    if bandwidth:  # cap dtf to bandwidth
        frequencies = numpy.linspace(0, hrtf_1[0].frequencies[-1], n_bins)
        dtf_1 = dtf_1[numpy.logical_and(frequencies >= bandwidth[0], frequencies <= bandwidth[1])]
        dtf_2 = dtf_2[numpy.logical_and(frequencies >= bandwidth[0], frequencies <= bandwidth[1])]
    # calculate correlation coefficients
    n_sources = len(sources)
    corr_mtx = numpy.zeros((n_sources, n_sources))
    for i in range(n_sources):
        for j in range(n_sources):
            corr_mtx[i, j] = numpy.corrcoef(dtf_1[:, i], dtf_2[:, j])[1, 0]
    # plot correlation matrix
    if show:
        if axis is None:
            fig, axis = plt.subplots()
        else:
            fig = axis.get_figure()
        cbar_levels = numpy.linspace(-1, 1, 100)
        contour = axis.contourf(hrtf_1.sources.vertical_polar[sources, 1],
                                hrtf_2.sources.vertical_polar[sources, 1], corr_mtx,
                                cmap='viridis', levels=cbar_levels)
        axis.set_xticks(numpy.linspace(-30, 30, 5))
        axis.set_yticks(numpy.linspace(-30, 30, 5))
        axis.tick_params(axis='both', direction="in", bottom=True, top=True, left=True, right=True,
                         labelsize=13, width=1.5, length=2)
        if cbar:
            cbar_ticks = numpy.linspace(-1, 1, 11)
            cax_pos = list(axis.get_position().bounds)  # (x0, y0, width, height)
            cax_pos[0] += 0.26  # x0
            cax_pos[2] = 0.012  # width
            cax = fig.add_axes(cax_pos)
            cbar = fig.colorbar(contour, cax, orientation="vertical", ticks=cbar_ticks)
            cax.tick_params(axis='both', direction="in", bottom=True, top=True, left=True, right=True,
                            labelsize=13, width=1.5, length=2)
    return corr_mtx

def vsi_dissimilarity(hrtf_1, hrtf_2, bandwidth):
    # get correlation matrices
    correlation_free_v_mold = hrtf_correlation(hrtf_1, hrtf_2, bandwidth=bandwidth)
    autocorrelation_free = hrtf_correlation(hrtf_1, hrtf_1, bandwidth=bandwidth)
    # VSI dissimilarity: euclidean distance between the matrices
    vsi_dissimilarity = numpy.linalg.norm(correlation_free_v_mold - autocorrelation_free)
    return vsi_dissimilarity

"""   
subject_id = 'nn'
condition = 'earmolds'
data_dir = Path.cwd() / 'data' / 'experiment' / 'bracket_1' / subject_id / condition
import datetime
date = datetime.datetime.now()

hrtf = slab.HRTF(data_dir / str(subject_id + '_' + condition + '_05.12.sofa')) #date.strftime('_%d.%m'))) + '.sofa')

dfe = False  # whether to use diffuse field equalization to plot hrtf and compute vsi
plot_bins = 2400  # number of bins also used to calculate vsi across bands (use 80 to minimize´frequency-resolution dependend vsi change)
plot_ear = 'left'  # ear for which to plot HRTFs
sources = list(range(hrtf.n_sources-1, -1, -1))  # works for 0°/+/-17.5° cone
fig, axis = plt.subplots(2, 1)
plot_tf(hrtf, sources, plot_bins, kind='waterfall', axis=axis[0], ear=plot_ear, xlim=(4000, 16000), dfe=dfe)
vsi_across_bands(hrtf, sources, n_bins=plot_bins, axis=axis[1], dfe=dfe)
axis[0].set_title(subject_id + ' ' + condition)
# hrtf.plot_tf(sources, xlim=(low_freq, high_freq), ear=plot_ear)
# hrtf.plot_tf(sources, xlim=(4000, 16000), ear=plot_ear)
"""