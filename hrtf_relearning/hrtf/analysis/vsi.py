"""
vsi.py — Vertical Spectral Information (VSI) metrics.

Reference
---------
Trapeau & Schönwiesner (2016). Fast and persistent adaptation to new spectral
cues for sound localization suggests a many-to-one mapping mechanism.
J. Acoust. Soc. Am. 140(2), 879–890.

Functions
---------
vsi(hrtf, bandwidth)
    VSI for a single HRTF — measures how well elevations can be discriminated
    from the spectral shape of DTFs.

vsi_dissimilarity(hrtf_1, hrtf_2, bandwidth)
    VSI dissimilarity between two HRTFs — RMS distance between the
    cross-correlation matrix and the autocorrelation matrix.

Frequency band
--------------
The default bandwidth (5700–11300 Hz) corresponds to the octave band with the
highest VSI and the strongest correlation with vertical localization
performance (Trapeau & Schönwiesner 2016, Fig. 2 & 5).
"""

import numpy
import slab


def vsi(hrtf, bandwidth=(5700, 11300)):
    """
    Vertical Spectral Information index (Trapeau & Schönwiesner 2016).

    VSI = 1 − mean of all off-diagonal entries of the autocorrelation matrix,
    averaged over left and right ear.

    The autocorrelation matrix contains the Pearson correlation coefficients
    between every pair of DTFs at different elevations on the median plane,
    within the given frequency band.  A VSI of 0 means all DTFs are identical
    (no spectral information), while higher values indicate better elevation
    discriminability.

    Parameters
    ----------
    hrtf      : slab.HRTF
    bandwidth : (low_hz, high_hz)
        Frequency band for the correlation.  Default (5700, 11300) is the
        peak VSI band from Trapeau & Schönwiesner (2016).

    Returns
    -------
    float
    """
    freqs, _ = hrtf[0].tf(show=False)   # always works for both HRIR and TF filters
    freq_idx = numpy.logical_and(freqs >= bandwidth[0], freqs <= bandwidth[1])
    sources  = hrtf.cone_sources(0)
    n        = len(sources)
    n_bins   = len(freqs)

    ear_vsi = []
    for ear in ('left', 'right'):
        # tfs_from_sources → (n_sources, n_bins, 1); squeeze → (n_sources, n_bins)
        dtfs = hrtf.tfs_from_sources(sources, n_bins=n_bins, ear=ear).squeeze()[:, freq_idx]
        off_diag = [
            float(numpy.corrcoef(dtfs[i], dtfs[j])[0, 1])
            for i in range(n) for j in range(n) if i != j
        ]
        ear_vsi.append(1.0 - float(numpy.mean(off_diag)))

    return float(numpy.mean(ear_vsi))


def vsi_dissimilarity(hrtf_1, hrtf_2, bandwidth=(5700, 11300)):
    """
    VSI dissimilarity between two HRTFs (Trapeau & Schönwiesner 2016).

    Defined as the RMS distance between the cross-correlation matrix
    (hrtf_1 vs hrtf_2) and the autocorrelation matrix (hrtf_1 vs hrtf_1),
    averaged over left and right ear.

    A dissimilarity of 0 means the two HRTFs produce identical DTF
    correlation structures; larger values indicate that the spectral cues
    differ more.

    Parameters
    ----------
    hrtf_1, hrtf_2 : slab.HRTF
    bandwidth       : (low_hz, high_hz), default (5700, 11300)

    Returns
    -------
    float
    """
    freqs, _ = hrtf_1[0].tf(show=False)  # always works for both HRIR and TF filters
    freq_idx = numpy.logical_and(freqs >= bandwidth[0], freqs <= bandwidth[1])
    sources  = hrtf_1.cone_sources(0)
    n        = len(sources)
    n_bins   = len(freqs)

    ear_dissim = []
    for ear in ('left', 'right'):
        d1 = hrtf_1.tfs_from_sources(sources, n_bins=n_bins, ear=ear).squeeze()[:, freq_idx]
        d2 = hrtf_2.tfs_from_sources(sources, n_bins=n_bins, ear=ear).squeeze()[:, freq_idx]

        cross = numpy.array(
            [[numpy.corrcoef(d1[i], d2[j])[0, 1] for j in range(n)] for i in range(n)]
        )
        auto = numpy.array(
            [[numpy.corrcoef(d1[i], d1[j])[0, 1] for j in range(n)] for i in range(n)]
        )

        ear_dissim.append(float(numpy.sqrt(numpy.mean((cross - auto) ** 2))))

    return float(numpy.mean(ear_dissim))
