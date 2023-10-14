import analysis.plot.hrtf_plot as hrtf_plot
import analysis.processing.hrtf_processing as hrtf_processing
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# matplotlib.use('TkAgg')
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from pathlib import Path
import slab
import numpy
import copy
import scipy
import pandas
pandas.set_option('display.max_rows', 1000, 'display.max_columns', 1000, 'display.width', 1000)

# ------ HRTF analysis ------ #

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
        vsis.append(vsi_across_bands(hrtf, bands, show=False, ear_idx=ear_idx))
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
        sp_str.append(spectral_strength_across_bands(hrtf, bands, show=False, ear=ear))
    mean_sp_str_across_bands = numpy.mean(sp_str, axis=0)
    if show:
        if not axis:
            fig, axis = plt.subplots()
        hrtf_plot.plot_spectral_strength_across_bands(mean_sp_str_across_bands, bands, axis=axis)
        err = scipy.stats.sem(sp_str, axis=0)
        axis.errorbar(axis.get_xticks(), numpy.mean(sp_str, axis=0), capsize=3,
                      yerr=err, fmt="o", c='k', elinewidth=0.5, markersize=3)
    return mean_sp_str_across_bands


# ------ SPCA ------ #
def hrtf_pca_space(hrtf_df, q=10, bandwidth=(4000, 16000)):
    global pca
    tf_data, hrtf_df = erb_binned_tf(hrtf_df, bandwidth)  # get binned DTFs for each subject HRTF
    tf_data = tf_data - tf_data.mean(axis=0)  # subtract mean transfer function
    pca = PCA(n_components=q)  # automated
    tf_pca = pca.fit_transform(tf_data)  # fit model

    hrtf_df = add_hrtf_pc_weights(hrtf_df, tf_pca) # add component weights to dataframe
    """
    sanity check:
    cc = []
    for i in range(10):
        cc.append(tf_pca[431, i] * pca.components_[i])  # matches tf_orig
    cc = numpy.sum(cc, axis=0)
    plt.figure()
    plt.plot(cc)    
    plt.plot(tf_data[431])
    # tf_data(588, 83) matches tf_pca(588, 10) * components
    
    
    c = []
    for i in range(10):
        c.append(hrtf_df.iloc[0]['pc weights'][0][0][i] * pca.components_[i])
    c = numpy.sum(c, axis=0)

    
    plt.figure()
    plt.plot(c)    
    fig, axes = plt.subplots(2,1)
    hrtf_df.iloc[0]['hrtf'][0].channel(0).tf(axis=axes[0])
    axes[0].set_xlim(4000, 16000)
    axes[0].set_ylim(-60, -10)
    axes[1].plot(c)
    
    
    tf_orig = PCA.inverse_transform(pca, X=tf_pca)
    tf_orig.reshape(tf_data.shape[0], tf_data.shape[1], tf_data.shape[2])  # matches tf_data
    # step by step PCA
    # cov_mtx = covariance_mtx(tf_data)
    # components = spca(cov_mtx, q)
    """
    return hrtf_df, pca

def erb_binned_tf(hrtf_df, bandwidth):
    hrtf_df['hrtf binned'] = ''
    for df_id, row in hrtf_df.iterrows():
        hrtf = hrtf_df.iloc[df_id]['hrtf']
        hrtf_binned = hrtf_processing.erb_filter_hrtf(hrtf, low_cutoff=bandwidth[0],
                                                      high_cutoff=bandwidth[1], return_bins=True)[0]
        hrtf_df.iloc[df_id]['hrtf binned'] = (hrtf_binned[:, :, 0], hrtf_binned[:, :, 1])
    tf_data = numpy.asarray(hrtf_df['hrtf binned'].tolist())
    tf_data = numpy.concatenate((tf_data[:, 0], tf_data[:, 1]))
    tf_data_c = tf_data.reshape(tf_data.shape[0] * tf_data.shape[1], tf_data.shape[2])
    # tf_data_c = tf_data_c.reshape(tf_data.shape[0], tf_data.shape[1], tf_data.shape[2])  # works reverse
    # tf_data_c = 20 * numpy.log10(tf_data_c)
    return tf_data_c, hrtf_df

def add_hrtf_pc_weights(hrtf_df, tf_pca):
    tf_pca = tf_pca.reshape(int(tf_pca.shape[0] / 7), 7, tf_pca.shape[1])
    tf_pca_left = tf_pca[:int(len(tf_pca) / 2)]
    tf_pca_right = tf_pca[int(len(tf_pca) / 2):]
    hrtf_df['pc weights'] = ''
    for df_id, row in hrtf_df.iterrows():
        hrtf_df.iloc[df_id]['pc weights'] = (tf_pca_left[df_id], tf_pca_right[df_id])
    return hrtf_df

def spca(cov_mtx, q=10):
    eig_vals, eig_vecs = numpy.linalg.eig(cov_mtx)  # eigenvalues and eigenvectors of covariance matrix
    eig_val_idx = numpy.argpartition(eig_vals, -q)[-q:]  # indices of q largest eigenvalues
    components = eig_vecs[:, eig_val_idx].T # corresponding eigenvectors = q basis functions / basis vectors
    return components

def covariance_mtx(dtf_list):
    cov_mtx = numpy.cov(dtf_list, rowvar=False)  # S(ij) = (1/n) * sum[D(ki),D(kj)] for i,j = 1,2,...,n frequency bins
    # # step by step takes much longer
    # n_observ = dtf_list.shape[0]
    # n_freqs = dtf_list.shape[1]
    # s = numpy.zeros((n_freqs, n_freqs))
    # for i in range(n_freqs):
    #     for j in range(n_freqs):
    #         s[i, j] = 1 / n_observ * numpy.sum(
    #         [(dtf_list[z, i] -  dtf_list[:, i].mean()) * (dtf_list[z, j] -  dtf_list[:, j].mean()) for z in range(n_observ)])
    return cov_mtx

def find_weights(basis_functions, dtf_list):
    # fit basis functions by finding weights that minimize MSE for each DTF
    weights = []
    for dtf in dtf_list:
        while mse > 100:
            w = numpy.ones((basis_functions.shape[1]))
            mse = numpy.mean(numpy.square((numpy.sum(numpy.sum((basis_vecs * w) + mean_dtf, axis=1), axis=1) / q) - dtf))
            # solve eigenvalue problem
        weights.append(w)

# ---- helper ----- #
def load_hrtf(subject_id, condition='Ears Free', processed=False):
    hrtf_df = hrtf_processing.get_hrtf_df(processed=processed)
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
