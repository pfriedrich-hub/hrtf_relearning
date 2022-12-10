import os
import slab
from pathlib import Path
import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy

# todo: fix problem of resolution downscaling with frequency for vsi:
#  for example kemar vsi with 0.1s looks better than for 0.05 seconds (higher frequencies have to
#  few samples here to be represented correctly in vsi)

def get_hrtfs(data_dir):
    subj_files = os.listdir(str(data_dir))
    subj_files = [fname for fname in subj_files if fname.endswith('.sofa')]  # remove sofa files from list
    subj_files.sort()
    hrtf_free = slab.HRTF(data_dir / subj_files[0])
    hrtf_mold = slab.HRTF(data_dir / subj_files[1])
    # todo test this again:
    # hrtf_free, hrtf_mold = hrtf_free.diffuse_field_equalization(), hrtf_mold.diffuse_field_equalization()
    return hrtf_free, hrtf_mold

# plot waterfall
def plot_tf(hrtf, sources, n_bins=None, kind=None, axis=None, ear=None, xlim=None, dfe=False):
    if dfe:
        hrtf = hrtf.diffuse_field_equalization()
    hrtf.plot_tf(sources, n_bins=n_bins, kind='waterfall', axis=axis, ear=ear, xlim=(4000, 16000))

def vsi_across_bands(hrtf, sources, n_bins, axis=None, dfe=False):
    if dfe:
        hrtf = hrtf.diffuse_field_equalization()
    # plot vsi across 1/2 octave frequency bands
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

    # plot
    if not axis:
        fig, axis = plt.subplots()
    axis.plot(vsi, c='k')
    # axis.errorbar(numpy.mean(vsi, axis=1), capsize=3, yerr=numpy.abs(numpy.diff(vsi, axis=0)),
    #                fmt="o", c='0.6', elinewidth=0.5, markersize=3)
    axis.set_xticks([0, 1, 2, 3, 4])
    labels = [item.get_text() for item in axis.get_xticklabels()]
    for idx, band in enumerate(bandwidths / 1000):
        labels[idx] = '%.1f - %.1f' % (band[0], band[1])
    axis.set_xticklabels(labels)
    axis.set_xlabel('Frequency bands (kHz)')
    axis.set_ylabel('VSI')


# hrtf_free.plot_tf(sources, n_bins=200, kind='image', ear='left', xlim=(4000, 16000))
def plot_correlation(hrtf_free, hrtf_mold, sources):
    # compare heatmap of hrtf free and with mold
    fig, axis = plt.subplots(2, 2, sharey=True)
    hrtf_free.plot_tf(sources, n_bins=96, kind='image', ear='left', xlim=(4000, 12000), axis=axis[0, 0])
    hrtf_mold.plot_tf(sources, n_bins=96, kind='image', ear='left', xlim=(4000, 12000), axis=axis[0, 1])
    fig.text(0.3, 0.9, 'Ear Free', ha='center')
    fig.text(0.7, 0.9, 'With Mold', ha='center')
    # plot hrtf autocorrelation free
    corr_mtx, cbar_1 = dtf_correlation(hrtf_free, hrtf_free, show=True, bandwidth=None,
                                         n_bins=96, axis=axis[1, 0])
    # plot hrtf correlation free vs mold
    cross_corr_mtx, cbar_2 = dtf_correlation(hrtf_free, hrtf_mold, show=True, bandwidth=None,
                                               n_bins=96, axis=axis[1, 1])
    fig.text(0.3, 0.5, 'Autocorrelation Ear Free', ha='center')
    fig.text(0.7, 0.5, 'Correlation Free vs. Mold', ha='center')
    cbar_1.remove()

def dtf_correlation(hrtf_1, hrtf_2, show=False, bandwidth=None, n_bins=96, axis=None, cbar=None):
    # get sources and dtfs
    sources = hrtf_1.cone_sources(0)
    dtf = hrtf_1.tfs_from_sources(sources, n_bins)
    dtf_2 = hrtf_2.tfs_from_sources(sources, n_bins)
    if bandwidth:  # cap dtf to bandwidth
        frequencies = numpy.linspace(0, hrtf_1[0].frequencies[-1], n_bins)
        dtf = dtf[numpy.logical_and(frequencies >= bandwidth[0], frequencies <= bandwidth[1])]
        dtf_2 = dtf_2[numpy.logical_and(frequencies >= bandwidth[0], frequencies <= bandwidth[1])]
    # calculate correlation coefficients
    n_sources = len(sources)
    corr_mtx = numpy.zeros((n_sources, n_sources))
    for i in range(n_sources):
        for j in range(n_sources):
            corr_mtx[i, j] = numpy.corrcoef(dtf[:, i], dtf_2[:, j])[1, 0]
    # plot correlation matrix
    if show:
        if axis is None:
            fig, axis = plt.subplots()
        else:
            fig = axis.figure
        levels = numpy.linspace(-1, 1, 11)
        contour = axis.contourf(hrtf_1.sources.vertical_polar[sources, 1],
                                hrtf_2.sources.vertical_polar[sources, 1], corr_mtx,
                                cmap='viridis', levels=levels)
        axis.set_xticks(numpy.arange(-37.5, 38, 12.5))
        axis.set_yticks(numpy.arange(-37.5, 38, 12.5))
        divider = make_axes_locatable(axis)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(contour, cax, orientation="vertical", ticks=numpy.linspace(-1, 1, 11),
                     label='Correlation Coefficient')
        axis.set_ylabel('Elevation (degrees)')
        axis.set_xlabel('Elevation (degrees)')
        plt.show()
    return corr_mtx, cbar

def vsi_dissimilarity(hrtf_1, hrtf_2, bandwidth):
    # get correlation matrices
    correlation_free_v_mold = dtf_correlation(hrtf_1, hrtf_2, bandwidth=bandwidth)
    autocorrelation_free = dtf_correlation(hrtf_1, hrtf_1, bandwidth=bandwidth)
    # VSI dissimilarity: euclidean distance between the matrices
    vsi_dissimilarity = numpy.linalg.norm(correlation_free_v_mold - autocorrelation_free)
    return vsi_dissimilarity

def average_hrtf(hrtf_list):
    dtf_list = []
    sources = hrtf_list[0].cone_sources(0)
    for hrtf in hrtf_list:
        dtf_left = hrtf.tfs_from_sources(sources, ear='left', n_bins=).T
        dtf_right = hrtf.tfs_from_sources(sources, ear='right').T
        dtfs = numpy.stack((dtf_left, dtf_right), axis=1)
        dtf_list.append(dtfs)
    dtf_list = numpy.asarray(dtf_list)
    mean_dtfs = numpy.mean(dtf_list, axis=0)
    return slab.HRTF(data=mean_dtfs, samplerate=97656, datatype='TF',
           sources=hrtf.sources.vertical_polar)

"""   
subject_id = 'cs'
condition = 'earmolds'
data_dir = Path.cwd() / 'data' / 'experiment' / 'bracket_1' / subject_id / condition
import datetime
date = datetime.datetime.now()

hrtf = slab.HRTF(str(data_dir / (subject_id + '_' + condition + date.strftime('_%d.%m'))) + '.sofa')

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
    
    
    
    
    
 # plot learning
    # plot_learning(data_dir, group_stats=False)

    if not group_stats:
        # compare free vs mold
        fig, axis = plt.subplots(2, 2)
        # hrtf_free, hrtf_mold = get_hrtfs(data_dir)
        src_free = hrtf_free.cone_sources(0, full_cone=True)
        src_mold = hrtf_mold.cone_sources(0, full_cone=True)
        # waterfall and vsi
        plot_vsi(hrtf_free, src_free, n_bins, axis=axis[0, 0])
        plot_vsi(hrtf_mold, src_mold, n_bins, axis=axis[0, 1])
        hrtf_free.plot_tf(src_free, n_bins=n_bins, kind='waterfall', axis=axis[1, 0])
        hrtf_mold.plot_tf(src_mold, n_bins=n_bins, kind='waterfall', axis=axis[1, 1])
        axis[0, 0].set_title('ears free')
        axis[0, 1].set_title('mold')

        # cross correlation
        # plot_hrtf_correlation(hrtf_free, hrtf_mold, src)"""