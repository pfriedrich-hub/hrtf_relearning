import slab
from pathlib import Path
import matplotlib
import analysis.plot.hrtf_plot as hrtf_plot
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy
import copy
import scipy
import pandas
pandas.set_option('display.max_rows', 1000, 'display.max_columns', 1000, 'display.width', 1000)

# todo: fix problem of resolution downscaling with frequency for vsi:
#  for example kemar vsi with 0.1s looks better than for 0.05 seconds (higher frequencies have to
#  few samples here to be represented correctly in vsi) - temporary fix: increase samplerate to 96 kHz

def get_hrtf_df(path=Path.cwd() / 'data' / 'experiment' / 'master', processed=True, exclude=[]):
    subject_paths = list(path.iterdir())
    hrtf_df = pandas.DataFrame({'subject': [], 'filename': [], 'condition': [], 'hrtf': []})
    conditions = ['Ears Free', 'Earmolds Week 1', 'Earmolds Week 2']
    for subject_path in subject_paths:
        subject = subject_path.name
        if subject not in exclude:
            for condition in conditions:
                if processed:
                    condition_path = subject_path / condition / 'processed_hrtf'
                else:
                    condition_path = subject_path / condition
                for file_name in sorted(list(condition_path.iterdir())):
                    if file_name.is_file() and file_name.suffix == '.sofa':
                        hrtf = slab.HRTF(file_name)
                        new_row = [subject, file_name.name, condition, hrtf]
                        hrtf_df.loc[len(hrtf_df)] = new_row
    # hrtf_df.to_csv('/Users/paulfriedrich/projects/hrtf_relearning/data/experiment/data.csv')
    return hrtf_df

# ----- HRTF processing ----- #
def process_hrtfs(hrtf_dataframe, filter='erb', baseline=True, dfe=False, write=False):
    path = Path.cwd() / 'data' / 'experiment' / 'master'
    for index, row in hrtf_dataframe.iterrows():
        if write:
            processed_path = Path(path / row['subject'] / row['condition'] / 'processed_hrtf')
            if not processed_path.exists():
                processed_path.mkdir()
        hrtf = copy.deepcopy(row['hrtf'])
        if filter == 'scepstral':
            hrtf = scepstral_filter_hrtf(hrtf, high_cutoff=1500)
        elif filter == 'erb':
            hrtf = erb_filter_hrtf(hrtf, kind='cosine', low_cutoff=4000, high_cutoff=16000, bandwidth=0.0286,
                                   pass_bands=True, return_bins=False)
        if dfe:
            hrtf = hrtf.diffuse_field_equalization()  # not on the subject level
        if baseline:
            hrtf = baseline_hrtf(hrtf, bandwidth=(4000, 16000))  # baseline should be done after smoothing / dfe
        if write:
            hrtf.write_sofa(filename=processed_path / str('proccessed_' + row['filename']))
        print('processed ' + row['filename'])
        hrtf_dataframe.hrtf[index] = hrtf
    return hrtf_dataframe

def baseline_hrtf(hrtf, bandwidth=(3000, 16000)): #todo doesnt work yet (increases corrleation)
    "Center transfer functions around 0"
    hrtf_out = copy.deepcopy(hrtf)
    sources = hrtf_out.cone_sources(0)
    frequencies = hrtf[0].frequencies
    tf_data = hrtf_out.tfs_from_sources(sources, n_bins=len(frequencies), ear='both')
    in_range = tf_data[:, numpy.logical_and(frequencies > bandwidth[0], frequencies < bandwidth[1])]
    tf_data -= numpy.mean(numpy.mean(in_range, axis=1), axis=0)  # subtract mean for left and right ear separately
    # tf_data[:, numpy.logical_or(frequencies < bandwidth[0], frequencies > bandwidth[1])] = 0
    tf_data = 10 ** (tf_data / 20)
    for idx, source in enumerate(sources):
        hrtf_out[source].data = tf_data[idx]
    return hrtf_out

def erb_filter_hrtf(hrtf, kind='cosine', low_cutoff=4000, high_cutoff=16000, bandwidth=0.0286,
                    pass_bands=True, return_bins=False):
    """
    smoothe a transfer function by applying an erb-spaced triangular filterbank:
    compute a weighted sum of the energy in a range of FFT bins to get a number which can be interpreted
    as the energy measured at the output of a band-pass filter of a given center frequency/width
    """
    hrtf_out = copy.deepcopy(hrtf)
    hrtf_freqs = hrtf[0].frequencies
    n_freqs = hrtf[0].n_frequencies
    center_freqs_erb, oct_bandwidth, erb_spacing = slab.Filter._center_freqs(
        low_cutoff=low_cutoff, high_cutoff=high_cutoff, bandwidth=bandwidth, pass_bands=pass_bands)
    tf_freqs_erb = slab.Filter._freq2erb(hrtf_freqs)
    center_freqs = slab.Filter._erb2freq(center_freqs_erb)
    n_bins = len(center_freqs_erb)
    windows = numpy.zeros((n_freqs, n_bins))
    dtf_binned = numpy.zeros((n_bins))
    hrtf_binned = numpy.zeros((hrtf_out.n_elevations, n_bins, 2))
    for dtf_idx, dtf in enumerate(hrtf_out):
        for chan_idx in range(dtf.n_channels):
            for bin_id in range(n_bins):
                l = center_freqs_erb[bin_id] - erb_spacing
                h = center_freqs_erb[bin_id] + erb_spacing
                window_size = ((tf_freqs_erb > l) & (tf_freqs_erb < h)).sum()  # width of the triangular window
                if kind == 'triangular':
                    window = scipy.signal.windows.triang(window_size, sym=True) #todo peak vals not always 1
                elif kind == 'cosine':
                    window = scipy.signal.windows.cosine(window_size, sym=True)
                windows[(tf_freqs_erb > l) & (tf_freqs_erb < h), bin_id] = window
                weighted_sum = numpy.sum(dtf.data[:, chan_idx] * windows[:, bin_id]) / window_size  # normalize by window size
                dtf_binned[bin_id] = weighted_sum
            hrtf_binned[dtf_idx, :, chan_idx] = dtf_binned
            hrtf_out[dtf_idx].data[:, chan_idx] = numpy.interp(hrtf_freqs, center_freqs, dtf_binned) # interpolate
    if return_bins:
        return hrtf_binned, center_freqs, hrtf_out
    else:
        return hrtf_out

def scepstral_filter_hrtf(hrtf, high_cutoff=1500):
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
    list = copy.deepcopy(hrtf_list)
    tf_data = numpy.zeros((hrtf_list[0].n_sources, len(hrtf_list), hrtf_list[0][0].n_samples, 2))
    for hrtf_idx, hrtf in enumerate(hrtf_list):
        for src_idx, tf in enumerate(hrtf.data):
            tf_data[src_idx, hrtf_idx] = tf.data
    tf_data = numpy.mean(tf_data, axis=1)
    # dtf = copy.deepcopy(hrtf)
    for src_idx, tf_data in enumerate(tf_data):
        hrtf[src_idx].data = tf_data
    return hrtf


# ------ HRTF analysis ------ #

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

# ----- VSI (trapeau und schönwiesner 2015) ---- #
def vsi(hrtf, bandwidth=(4000, 16000), ear_idx=[0, 1]):
    corr_mtx = hrtf_correlation(hrtf, hrtf, bandwidth, ear_idx, show=False)
    corr_mtx = mtx_remove_main_diag(corr_mtx)
    vsi = 1 - numpy.mean(corr_mtx)
    return vsi

def hrtf_correlation(hrtf_1, hrtf_2, bandwidth=(4000, 16000), ear_idx=[0, 1], show=False, axis=None, c_bar=True):
    # ear_idx: [0, 1] == both ears, [0] == left ear, [1] == right ear
    freqs = hrtf_1[0].frequencies
    freq_idx = numpy.logical_and(freqs >= bandwidth[0], freqs <= bandwidth[1])
    dtfs_1 = hrtf_1.tfs_from_sources([0, 1, 2, 3, 4, 5, 6], n_bins=len(freqs), ear='both')[:, freq_idx]
    dtfs_2 = hrtf_2.tfs_from_sources([0, 1, 2, 3, 4, 5, 6], n_bins=len(freqs), ear='both')[:, freq_idx]
    corr_mtx = numpy.zeros((len(ear_idx), hrtf_1.n_sources, hrtf_1.n_sources))
    sources = hrtf_1.cone_sources(0)
    for ear_id in ear_idx:
        for i, source_idx_i in enumerate(range(hrtf_1.n_sources)):  # decreasing elevation
            for j, source_idx_j in enumerate(sources):  # increasing elevation
                # print(f'write correlation coefficient of hrtf_1 at {hrtf_1.sources.vertical_polar[source_idx_i, 1]} and '
                #       f'hrtf_2 at {hrtf_1.sources.vertical_polar[source_idx_j, 1]} to position {(i, j)}')
                corr_mtx[-ear_id, i, j] = numpy.corrcoef(dtfs_1[source_idx_i, :, ear_id],
                                                        dtfs_2[source_idx_j, :, ear_id])[1, 0]
    corr_mtx = numpy.mean(corr_mtx, axis=0)  # average left and right ear values if both ears are used
    if show:
        if not axis:
            fig, axis = plt.subplots()
        hrtf_plot.plot_correlation_matrix(corr_mtx, axis, c_bar)
    return corr_mtx

def vsi_across_bands(hrtf, bands=None, show=False, axis=None, ear_idx=[0,1]):
    if bands is None:  # calculate vsi across 5 octave frequency bands
        bands = [(4000, 8000), (4800, 9500), (5700, 11300), (6700, 13500), (8000, 16000)]
    vsis = numpy.zeros(len(bands))
    for i, bandwidth in enumerate(bands):
        vsis[i] = vsi(hrtf, bandwidth, ear_idx)
    if show:
        if not axis:
            fig, axis = plt.subplots()
        hrtf_plot.plot_vsi_across_bands(vsis, bands, axis=axis)
    return vsis

def vsi_dissimilarity(hrtf_1, hrtf_2, bandwidth=(4000, 16000), ear_idx=[0, 1]):
    """ compute dissimilarity between sets of DTFs"""
    dtf1 = copy.deepcopy(hrtf_1)
    dtf2 = copy.deepcopy(hrtf_2)
    correlation_mtx = hrtf_correlation(dtf1, dtf2, bandwidth, ear_idx)
    autocorrelation_mtx = hrtf_correlation(dtf1, dtf1, bandwidth, ear_idx)
    # VSI dissimilarity: euclidean distance between the matrices
    vsi_dissimilarity = numpy.sqrt(numpy.mean((correlation_mtx - autocorrelation_mtx)**2))
    return vsi_dissimilarity

def mtx_remove_main_diag(corr_mtx):
    mask = numpy.ones(corr_mtx.shape, dtype=bool)
    mask[numpy.diag_indices(7)] = False
    mask = numpy.flipud(mask)
    return corr_mtx[mask]

def mean_vsi_across_bands(hrtf_dataframe, condition='Ears Free', bands=None, show=False, axis=None, ear_idx=[0,1]):
    if bands is None:  # calculate vsi across 5 octave frequency bands
        bands = [(4000, 8000), (4800, 9500), (5700, 11300), (6700, 13500), (8000, 16000)]
    vsis = []
    for hrtf in list(hrtf_dataframe[hrtf_dataframe['condition'] == condition]['hrtf']):
        vsis.append(vsi_across_bands(hrtf, bands, ear_idx))
    mean_vsi_across_bands = numpy.mean(vsis, axis=0)
    if show:
        if not axis:
            fig, axis = plt.subplots()
        hrtf_plot.plot_vsi_across_bands(mean_vsi_across_bands, bands, axis=axis)
        err = scipy.stats.sem(vsis, axis=0)
        axis.errorbar(axis.get_xticks(), numpy.mean(vsis, axis=0), capsize=3,
                       yerr=err, fmt="o", c='k', elinewidth=0.5, markersize=3)
    return mean_vsi_across_bands


# ----- spectral strength (middlebrooks 1999) ---- #
def spectral_strength(hrtf, bandwidth=(3700, 12900), ear='both'):
    freqs = hrtf[0].frequencies
    freq_idx = numpy.logical_and(freqs >= bandwidth[0], freqs <= bandwidth[1])
    dtfs = hrtf.tfs_from_sources(hrtf.cone_sources(0), n_bins=len(freqs), ear=ear)[:, freq_idx]
    dtf_variance = numpy.var(dtfs, axis=1)
    spectral_strength = numpy.mean(dtf_variance)
    return spectral_strength

def spectral_difference(hrtf_1, hrtf_2, bandwidth=(4000, 16000), ear='both'):
    freqs = hrtf_1[0].frequencies
    freq_idx = numpy.logical_and(freqs >= bandwidth[0], freqs <= bandwidth[1])
    hrtf_diff = hrtf_difference(hrtf_1, hrtf_2)
    difference_spectrum = hrtf_diff.tfs_from_sources(hrtf_1.cone_sources(0), n_bins=len(freqs), ear=ear)[:, freq_idx]
    dtf_variance = numpy.var(difference_spectrum, axis=1)
    spectral_difference = numpy.mean(dtf_variance)
    return spectral_difference

def spectral_strength_across_bands(hrtf, bands=None, show=False, axis=None, ear='both'):
    if bands is None:  # calculate vsi across 5 octave frequency bands
        bands = [(4000, 8000), (4800, 9500), (5700, 11300), (6700, 13500), (8000, 16000)]
    sp_str = numpy.zeros(len(bands))
    for i, bandwidth in enumerate(bands):
        sp_str[i] = spectral_strength(hrtf, bandwidth, ear)
    if show:
        if not axis:
            fig, axis = plt.subplots()
        hrtf_plot.plot_spectral_strength_across_bands(sp_str, bands, axis=axis)
    return sp_str

def mean_spectral_strength_across_bands(hrtf_dataframe, condition='Ears Free', bands=None, show=False, axis=None, ear='both'):
    if bands is None:  # calculate vsi across 5 octave frequency bands
        bands = [(4000, 8000), (4800, 9500), (5700, 11300), (6700, 13500), (8000, 16000)]
    sp_str = []
    for hrtf in list(hrtf_dataframe[hrtf_dataframe['condition'] == condition]['hrtf']):
        sp_str.append(spectral_strength_across_bands(hrtf, bands, ear))
    mean_sp_str_across_bands = numpy.mean(sp_str, axis=0)
    if show:
        if not axis:
            fig, axis = plt.subplots()
        hrtf_plot.plot_spectral_strength_across_bands(mean_sp_str_across_bands, bands, axis=axis)
        err = scipy.stats.sem(sp_str, axis=0)
        axis.errorbar(axis.get_xticks(), numpy.mean(sp_str, axis=0), capsize=3,
                      yerr=err, fmt="o", c='k', elinewidth=0.5, markersize=3)
    return mean_sp_str_across_bands

# ---- helper ----- #
def load_hrtf(subject_id, condition='Ears Free', processed=False):
    hrtf_df = get_hrtf_df(processed=processed)
    # load hrtf
    # subject = random.choice(hrtf_df['subject'].unique())
    hrtf = hrtf_df[hrtf_df['subject'] == subject_id][hrtf_df['condition'] == condition]['hrtf'].values[0]
    return hrtf

# ---- deprecated ---- #

def mean_vsi_across_bands_old(hrtf_dataframe, condition='Ears Free', bands=None, show=False, axis=None, n_bins=None):
    if bands is None:  # calculate vsi across 5 octave frequency bands
        bands = [(4000, 8000), (4800, 9500), (5700, 11300), (6700, 13500), (8000, 16000)]
    vsis = []
    for hrtf in list(hrtf_dataframe[hrtf_dataframe['condition'] == condition]['hrtf']):
        vsis.append(vsi_across_bands_old(hrtf, bands, n_bins=n_bins, equalize=False))
    mean_vsi_across_bands = numpy.mean(vsis, axis=0)
    if show:
        if not axis:
            fig, axis = plt.subplots()
        axis.plot(mean_vsi_across_bands, c='k')
        axis.set_xticks(numpy.arange(len(bands)))
        labels = [item.get_text() for item in axis.get_xticklabels()]
        for idx, band in enumerate(numpy.asarray(bands) / 1000):
            labels[idx] = '%.1f - %.1f' % (band[0], band[1])
        axis.set_xticklabels(labels)
        axis.set_yticks(numpy.arange(0.1, 1.2, 0.1))
        axis.set_xlabel('Frequency bands (kHz)')
        axis.set_ylabel('VSI')
    return mean_vsi_across_bands


def vsi_across_bands_old(hrtf, bands=None, n_bins=None, equalize=False):
    """
    calculate vsi across frequency bands
    args:
    bands: list of tuples
    """
    dtf = copy.deepcopy(hrtf)
    if n_bins is None:
        n_bins = hrtf.data[0].n_frequencies
    if bands is None:   # calculate vsi across 5 octave frequency bands
        bands = [(4000, 8000), (4800, 9500), (5700, 11300),(6700, 13500), (8000, 16000)]
    sources = hrtf.cone_sources()
    frequencies = numpy.linspace(0, hrtf[0].frequencies[-1], n_bins)
    vsi = numpy.zeros(len(bands))
    if equalize: # apply diffuse field equalization
        dtf = dtf.diffuse_field_equalization()
    dtfs = dtf.tfs_from_sources(sources, n_bins, ear='left')
    # extract vsi for each band
    for idx, bw in enumerate(bands):
        freq_idx = numpy.logical_and(frequencies >= bw[0], frequencies <= bw[1])
        dtf_band = dtfs[:, freq_idx]
        #todo finish work here
        # plt.figure()
        # for band in dtf_band:
        #     plt.plot(frequencies[freq_idx], band)
        sum_corr = 0
        n = 0
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                sum_corr += numpy.corrcoef(dtf_band[i].flatten(), dtf_band[j].flatten())[1, 0]
                n += 1

        # plt.title(sum_corr / n)

        vsi[idx] = 1 - sum_corr / n
    return vsi

def vsi_old(hrtf, bandwidth=(4000, 16000), n_bins=None, sources=None, equalize=False):
    if n_bins is None:
        n_bins = hrtf.data[0].n_frequencies
    if sources is None:
        sources = hrtf.cone_sources()
    frequencies = hrtf[0].frequencies
    if equalize:
        dtf = hrtf.diffuse_field_equalization()
        tfs = dtf.tfs_from_sources(sources=sources, n_bins=n_bins)
    else:
        tfs = hrtf.tfs_from_sources(sources=sources, n_bins=n_bins)
    # only use tfs within bandwidth for correlation
    tfs = tfs[:, numpy.logical_and(frequencies >= bandwidth[0], frequencies <= bandwidth[1])]
    sum_corr = 0
    n = 0
    for i in range(len(sources)):
        for j in range(i+1, len(sources)):
            sum_corr += numpy.corrcoef(tfs[i].flatten(), tfs[j].flatten())[1, 0]
            n += 1
    return 1 - sum_corr / n

def hrtf_correlation_old(hrtf_1, hrtf_2, bandwidth=(4000, 16000), n_bins=None, show=False, axis=None, cbar=True):
    if n_bins is None:
        n_bins = hrtf_1.data[0].n_frequencies
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
        contour = axis.matshow(corr_mtx)
        axis.set_xticks(numpy.linspace(-30, 30, 5))
        axis.set_yticks(numpy.linspace(-30, 30, 5))
        axis.tick_params(axis='both', direction="in", bottom=True, top=True, left=True, right=True,
                         labelsize=13, width=1.5, length=2)
        if cbar:
            cax_pos = list(axis.get_position().bounds)  # (x0, y0, width, height)
            cax_pos[0] += 0.26  # x0
            cax_pos[2] = 0.012  # width
            cax = fig.add_axes(cax_pos)
            cbar = fig.colorbar(contour, cax, orientation="vertical", ticks=cbar_ticks)
            cax.tick_params(axis='both', direction="in", bottom=True, top=True, left=True, right=True,
                            labelsize=13, width=1.5, length=2)
    return corr_mtx

def get_hrtf_dict(path, subject_list, conditions, smoothe=True, baseline=True, bandwidth=(4000, 16000), dfe=True):
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
                    if dfe:
                        hrtf = hrtf.diffuse_field_equalization()
                    if baseline:
                        hrtf = baseline_hrtf(hrtf, bandwidth=bandwidth)
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


# def mean_vsi_across_bands(hrtf_dict, show=True):
#     vsi_mtx = numpy.zeros((len(hrtf_dict.keys()), 2, 5))
#     for c_idx, condition in enumerate(hrtf_dict.keys()):
#         vsi_list = []
#         subj_dict = hrtf_dict[condition]
#         for subj in subj_dict.keys():
#             vsi_list.append(vsi_across_bands(hrtf_dict[condition][subj], n_bins=4884, show=False))
#             vsi_mean = numpy.mean(vsi_list, axis=0)
#             vsi_se = scipy.statistics.sem(vsi_list, axis=0)
#             vsi_mtx[c_idx] = numpy.array((vsi_mean, vsi_se))
#         if show:
#             bandwidths = numpy.array(((4, 8), (4.8, 9.5), (5.7, 11.3), (6.7, 13.5), (8, 16))) * 1000
#             fig, axis = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(14, 4))
#             fig.subplots_adjust(left=0.05, bottom=None, right=0.95, top=None, wspace=0.1)
#             for i in range(3):
#                 ax = axis[i]
#                 ax.plot(vsi_mtx[i, 0], c='black', linewidth=0.5)
#                 ax.set_xticks([0, 1, 2, 3, 4])
#                 ax.errorbar(ax.get_xticks(), vsi_mtx[i, 0], yerr=vsi_mtx[i, 1],
#                             fmt="o", c='black', elinewidth=0.5, markersize=4, capsize=2, fillstyle='none')
#                 labels = [item.get_text() for item in ax.get_xticklabels()]
#                 for idx, band in enumerate(bandwidths / 1000):
#                     labels[idx] = '%.1f - %.1f' % (band[0], band[1])
#                 ax.set_xticklabels(labels)
#                 ax.set_xlabel('Frequency bands (kHz)')
#                 ax.set_title(list(hrtf_dict.keys())[i])
#             axis[0].set_ylabel('VSI')
