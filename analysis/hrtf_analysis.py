import os
import slab
from pathlib import Path
import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy
import copy
import scipy

# todo: fix problem of resolution downscaling with frequency for vsi:
#  for example kemar vsi with 0.1s looks better than for 0.05 seconds (higher frequencies have to
#  few samples here to be represented correctly in vsi) - temporary fix: increase samplerate to 96 kHz

def get_hrtfs(path, subject_list, conditions, smoothe=True, baseline=True, bandwidth=(4000, 16000), dfe=True):
    hrtf_dict = {}
    for condition in conditions:
        hrtf_dict[condition] = {}
        for subj_idx, subject_path in enumerate(subject_list):
            if smoothe:
                subject_dir = Path(path / subject_path / condition / 'processed_hrtf')
            elif not smoothe:
                subject_dir = Path(path/ subject_path / condition)
            # create subject dir if it doesnt exist
            if not subject_dir.exists():
                subject_dir.mkdir()
            for file_name in sorted(list(subject_dir.iterdir())):
                if file_name.is_file() and file_name.suffix == '.sofa':
                    hrtf = slab.HRTF(file_name)
                    if baseline:
                        hrtf = baseline_hrtf(hrtf, bandwidth=bandwidth)
                    if dfe:
                        hrtf = hrtf.diffuse_field_equalization()
                    hrtf_dict[condition][subject_path] = hrtf
        hrtf_dict[condition]['average'] = average_hrtf(list(hrtf_dict[condition].values()))
    return hrtf_dict

def write_processed_hrtf(hrtf_dict, path, dir_name='processed_hrtf'):
    subject_dir_list = list(path.iterdir())
    for condition in hrtf_dict.keys():
        for subj_idx, subject_path in enumerate(subject_dir_list):
            if subject_path.name in hrtf_dict[condition].keys():
                Path.mkdir(subject_path / condition / dir_name, exist_ok=True)
                hrtf_dict[condition][subject_path.name].write_sofa(subject_path / condition /
                                                    dir_name / str(condition + '_processed.sofa'))

def baseline_hrtf(hrtf, bandwidth=(3000, 17000)):
    "Center transfer functions around 0"
    hrtf_out = copy.deepcopy(hrtf)
    sources = hrtf_out.cone_sources(0)
    frequencies = hrtf[0].frequencies
    tf_data = hrtf_out.tfs_from_sources(sources, n_bins=len(frequencies), ear='both')
    in_range = tf_data[:, numpy.logical_and(frequencies > bandwidth[0], frequencies < bandwidth[1])]
    tf_data -= numpy.mean(numpy.mean(in_range, axis=1), axis=0)
    tf_data[:, numpy.logical_or(frequencies < bandwidth[0], frequencies > bandwidth[1])] = 0
    tf_data = 10 ** (tf_data / 20)
    for idx, source in enumerate(sources):
        hrtf_out[source].data = tf_data[idx]
    return hrtf_out

def smoothe_hrtf(hrtf, high_cutoff=1500):
    hrtf_out = copy.deepcopy(hrtf)
    filt = slab.Filter.band(kind='lp', frequency=high_cutoff, samplerate=hrtf.samplerate,
                            length=hrtf[0].n_samples, fir=True)
    for tf in hrtf_out:
        tf_data = 20 * numpy.log10(tf.data)
        to_filter = slab.Sound(tf_data, samplerate=tf.samplerate)
        filtered = filt.apply(to_filter)
        tf_data = 10 ** (filtered.data/20)
        tf.data = tf_data
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

def vsi_dissimilarity(hrtf_1, hrtf_2, bandwidth):
    # get correlation matrices
    correlation = hrtf_correlation(hrtf_1, hrtf_2, bandwidth=bandwidth)
    autocorrelation = hrtf_correlation(hrtf_1, hrtf_1, bandwidth=bandwidth)
    # VSI dissimilarity: euclidean distance between the matrices
    vsi_dissimilarity = numpy.sqrt(numpy.mean((correlation - autocorrelation)**2))
    return vsi_dissimilarity

def vsi_across_bands(hrtf, cone=0, n_bins=300, show=True, axis=None):
    # calculate vsi across 1/2 octave frequency bands
    sources = hrtf.cone_sources(cone)
    dtfs = hrtf.tfs_from_sources(sources, n_bins)
    frequencies = numpy.linspace(0, hrtf[0].frequencies[-1], n_bins)
    bandwidths = numpy.array(((4, 8), (4.8, 9.5), (5.7, 11.3), (6.7, 13.5), (8, 16))) * 1000
    vsi = numpy.zeros(len(bandwidths))
    # extract vsi for each band
    for idx, bw in enumerate(bandwidths):
        dtf_band = dtfs[:, numpy.logical_and(frequencies >= bw[0], frequencies <= bw[1])]
        sum_corr = 0
        n = 0
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                sum_corr += numpy.corrcoef(dtf_band[i].flatten(), dtf_band[j].flatten())[1, 0]
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

def mean_vsi_across_bands(hrtf_dict, show=True):
    vsi_mtx = numpy.zeros((len(hrtf_dict.keys()), 2, 5))
    for c_idx, condition in enumerate(hrtf_dict.keys()):
        vsi_list = []
        subj_dict = hrtf_dict[condition]
        for subj in subj_dict.keys():
            vsi_list.append(vsi_across_bands(hrtf_dict[condition][subj], n_bins=4884, show=False))
            vsi_mean = numpy.mean(vsi_list, axis=0)
            vsi_se = scipy.stats.sem(vsi_list, axis=0)
            vsi_mtx[c_idx] = numpy.array((vsi_mean, vsi_se))
        if show:
            bandwidths = numpy.array(((4, 8), (4.8, 9.5), (5.7, 11.3), (6.7, 13.5), (8, 16))) * 1000
            fig, axis = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(14, 4))
            fig.subplots_adjust(left=0.05, bottom=None, right=0.95, top=None, wspace=0.1)
            for i in range(3):
                ax = axis[i]
                ax.plot(vsi_mtx[i, 0], c='black', linewidth=0.5)
                ax.set_xticks([0, 1, 2, 3, 4])
                ax.errorbar(ax.get_xticks(), vsi_mtx[i, 0], yerr=vsi_mtx[i, 1],
                            fmt="o", c='black', elinewidth=0.5, markersize=4, capsize=2, fillstyle='none')
                labels = [item.get_text() for item in ax.get_xticklabels()]
                for idx, band in enumerate(bandwidths / 1000):
                    labels[idx] = '%.1f - %.1f' % (band[0], band[1])
                ax.set_xticklabels(labels)
                ax.set_xlabel('Frequency bands (kHz)')
                ax.set_title(list(hrtf_dict.keys())[i])
            axis[0].set_ylabel('VSI')

def hrtf_images(plot_dict, n_bins, bandwidth, title=None, plot='image'):
    # input is a dictionary with keys = condition and value = HRTF, 3 conditions
    dict = copy.deepcopy(plot_dict)
    conditions = list(dict.keys())
    diff_conditions = ['Difference Ears Free - Mold 1',
                       'Difference Ears Free - Mold 2', 'Difference Mold 1 - Mold 2']
    corr_conditions = ['Correlation Ears Free - Mold 1',
                       'Correlation Ears Free - Mold 2', 'Correlation Mold 1 - Mold 2']
    compare = [['Ears Free', 'Earmolds Week 1'], ['Ears Free', 'Earmolds Week 2'],
               ['Earmolds Week 1', 'Earmolds Week 2']]
    # 0° az cone sources
    src_idx = dict[conditions[0]].cone_sources(0)
    # get difference HRTFs
    dict['difference'], dict['min'], dict['max'] = {}, [], []
    for i in range(3):
        dict['difference'][diff_conditions[i]] = hrtf_difference(dict[compare[i][0]], dict[compare[i][1]])
    # get min and max values for img cbar scaling etc
    frequencies = dict[conditions[0]][0].frequencies
    frequencies = numpy.linspace(0, frequencies[-1], n_bins)
    freqidx = numpy.logical_and(frequencies > bandwidth[0], frequencies < bandwidth[1])
    for condition in conditions:
        dict['min'].append(dict[condition].tfs_from_sources(src_idx, n_bins)[:, freqidx].min())
        dict['max'].append(dict[condition].tfs_from_sources(src_idx, n_bins)[:, freqidx].max())
    for condition in diff_conditions:
        dict['min'].append(dict['difference'][condition].tfs_from_sources(src_idx, n_bins)[:, freqidx].min())
        dict['max'].append(dict['difference'][condition].tfs_from_sources(src_idx, n_bins)[:, freqidx].max())
    z_min = numpy.floor(numpy.min(dict['min'])) - 1
    z_max = numpy.ceil(numpy.max(dict['max']))
    title_list = [['Ears Free', 'Week 1 Molds', 'Week 2 Molds'], diff_conditions, corr_conditions]

    # plot
    if plot == 'image':
        fig, axis = plt.subplots(2, 3, sharey=True, figsize=(13, 8))
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05)
        cbar = False
        for i in range(3):
            if i == 2:
                cbar = True
            # plot HRTF
            hrtf_image(dict[conditions[i]], n_bins=n_bins,
                                          bandwidth=bandwidth, axis=axis[0, i], z_min=z_min, z_max=z_max, cbar=cbar)
            axis[0, i].set_title(title_list[0][i])
            # plot HRTF differences
            hrtf_image(dict['difference'][diff_conditions[i]], n_bins=n_bins,
                                          bandwidth=bandwidth, axis=axis[1, i], z_min=z_min, z_max=z_max, cbar=cbar)
            axis[1, i].set_title(title_list[1][i])
        fig.text(0.5, 0.04, 'Frequency (kHz)', ha='center', size=13)
        fig.text(0.07, 0.5, 'Elevation (degrees)', va='center', rotation='vertical', size=13)
        if title:
            fig.suptitle(title)
        return fig, axis

    elif plot == 'correlation': # compute and plot HRTF correlation
        correlation = []
        fig, axis = plt.subplots(1, 3, sharey=True, figsize=(12, 4))
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05)
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
    else:
        print('plot argument can be "image" or "correlation"')

def hrtf_correlation(hrtf_1, hrtf_2, show=False, bandwidth=(4000, 16000), n_bins=300, axis=None, cbar=True):
    # get sources and dtfs
    sources = hrtf_1.cone_sources(0)
    dtf_1 = hrtf_1.tfs_from_sources(sources, n_bins)
    dtf_2 = hrtf_2.tfs_from_sources(sources, n_bins)
    # cap frequencies by bandwidth
    frequencies = numpy.linspace(0, hrtf_1[0].frequencies[-1], n_bins)
    freq_idx = numpy.logical_and(frequencies >= bandwidth[0], frequencies <= bandwidth[1])
    dtf_1 = dtf_1[:, freq_idx]
    dtf_2 = dtf_2[:, freq_idx]
    # calculate correlation coefficients
    n_sources = len(sources)
    corr_mtx = numpy.zeros((n_sources, n_sources))
    for i in range(n_sources):
        for j in range(n_sources):
            corr_mtx[i, j] = numpy.corrcoef(dtf_1[i].flatten(), dtf_2[j].flatten())[1, 0]
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


"""   
subject_id = 'ma'
condition = 'Earmolds Week 2'
data_dir = Path.cwd() / 'data' / 'experiment' / 'bracket_1' / subject_id / condition
file_name = 'ma_Earmolds Week 2_10.12.sofa'
hrtf = slab.HRTF(data_dir / file_name)
bandwidth = (4000, 16000)
n_bins = 300
hrtf = baseline_hrtf(hrtf, bandwidth=bandwidth)
hrtf = hrtf.diffuse_field_equalization()

# image
axis = plot_hrtf_image(hrtf, bandwidth, n_bins)
ax = axis.figure.get_axes()[1]
ax_pos = list(ax.get_position().bounds)
ax_pos[0] = 0.92
ax.set_position(ax_pos) # move cbar

# waterfall
hrtf.plot_tf(sourceidx=hrtf.cone_sources(0), xlim=bandwidth, n_bins=n_bins)


plot_bins = 2400  # number of bins also used to calculate vsi across bands (use 80 to minimize´frequency-resolution dependend vsi change)
plot_ear = 'left'  # ear for which to plot HRTFs
sources = list(range(hrtf.n_sources-1, -1, -1))  # works for 0°/+/-17.5° cone
fig, axis = plt.subplots(2, 1)
plot_tf(hrtf, sources, plot_bins, kind='waterfall', axis=axis[0], ear=plot_ear, xlim=(4000, 16000), dfe=dfe)
vsi_across_bands(hrtf, sources, n_bins=plot_bins, axis=axis[1], dfe=dfe)
axis[0].set_title(subject_id + ' ' + condition)
# hrtf.plot_tf(sources, xlim=(low_freq, high_freq), ear=plot_ear)
# hrtf.plot_tf(sources, xlim=(4000, 16000), ear=plot_ear
 """