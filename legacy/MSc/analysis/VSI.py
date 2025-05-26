import numpy
import slab

"""
Note: in previous experiments VSI explained localization accuracy only when the analysis
was limited to frequency bands containing prominent spectral features (for example 6–12 kHz)
"""

hrtf = slab.HRTF.kemar()  # MIT Kemar
# to read from sofa file:
# hrtf = slab.HRTF('filename.sofa')

def vsi(hrtf, bandwidth=(4000, 16000), ear_idx=[0, 1], average=True):
    """
    :param hrtf: the HRTF of which the vertical spectral information is computed can be of datatype TF or IR
    :param bandwidth: the spectral band in which the vertical spectral information is computed
    :param ear_idx: which ear to use, 0 for the left, 1 for the right or a list [0, 1] for both ears
    :param average: whether to average left and right ear values
    :return: a quantification of vertical spectral information (Trapeau & Schönwiesner, 2015)
    """
    corr_mtx = hrtf_correlation(hrtf, hrtf, bandwidth, ear_idx, average=average)
    corr_mtx = mtx_remove_main_diag(corr_mtx)
    if not average and len(ear_idx)==2:
        vsi = 1 - numpy.mean(corr_mtx, axis=1)
    else:
        vsi = 1 - numpy.mean(corr_mtx)
    return vsi

def mtx_remove_main_diag(corr_mtx):
    """
    remove the main diagonal from the Corrleation matrix (see Trapeau & Schönwiesner, 2015)
    """
    mask = numpy.ones(corr_mtx.shape[-2:], dtype=bool)
    mask[numpy.diag_indices(corr_mtx.shape[-1])] = False
    if corr_mtx.shape[0] == 2: # in case of non averaged 2-ears matrix
        corr_mtx = corr_mtx[:, mask]  
    else:
        corr_mtx = corr_mtx[mask]
    return corr_mtx

def hrtf_correlation(hrtf_1, hrtf_2, bandwidth=(4000, 16000), ear_idx=[0, 1], average=True):
    """
    compute correlation matrix between 2 HRTFs.
    ear_idx: [0, 1] == both ears, [0] == left ear, [1] == right ear
    both HRTFs must have the same samplerate and number of bins
    and by default tfs across the vertical midline are used
    """
    # fetch band-limited TFs on the vertical midline
    freqs, _ = hrtf_1[0].tf(show=False)  # the frequencies of the HRTFs / HRIRs
    freq_idx = numpy.logical_and(freqs >= bandwidth[0], freqs <= bandwidth[1])  # select frequency band
    dtfs_1 = hrtf_1.tfs_from_sources(hrtf.cone_sources(0), n_bins=len(freqs), ear='both')[:, freq_idx] # retrieve dtfs
    dtfs_2 = hrtf_2.tfs_from_sources(hrtf.cone_sources(0), n_bins=len(freqs), ear='both')[:, freq_idx]
    corr_mtx = numpy.zeros((len(ear_idx), len(dtfs_1), len(dtfs_2)))  # matrix to hold correlation coefs
    for ear_id in ear_idx:     # compute correlation matrix
        for i, tf_i in enumerate(dtfs_1):
            for j, tf_j in enumerate(dtfs_2):
                corr_mtx[-ear_id, i, j] = numpy.corrcoef(tf_i[:, ear_id],
                                                        tf_j[:, ear_id])[1, 0]
    if average: # average left and right ear values
        corr_mtx = numpy.mean(corr_mtx, axis=0)
    return corr_mtx