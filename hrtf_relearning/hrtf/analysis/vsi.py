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

Frequency band — TWO different results, do not conflate them
------------------------------------------------------------
Trapeau report two separate band analyses, and this module used to describe the
first one as if it were also the second:

* VSI is HIGHEST in the 5657–11314 Hz OCTAVE band, of five octave bands between
  4 and 16 kHz (:data:`PAPER_OCTAVE_BANDS`; their Fig. 2). This is a statement
  about where the DTF set is most elevation-discriminative — nothing about
  behaviour. It is the module default because it is the band the cue
  manipulations target.
* VSI CORRELATES WITH VERTICAL RMSE in the 5657–8000 Hz HALF-OCTAVE band, of
  four non-overlapping half-octave bands (:data:`PAPER_HALF_OCTAVE_BANDS`;
  their Fig. 5A): R = −0.53, Bonferroni-corrected p = 0.0126, ~25% of the
  behavioural variance. Different band, different band scheme.

Use :data:`BEHAVIOURAL_BAND` when replicating the behavioural correlation.

Spectral resolution
-------------------
Trapeau correlate the outputs of a bank of triangular band-pass filters spaced
0.0286 octaves apart (2% frequency steps, ~36 points per octave), not raw FFT
bins. Linearly spaced FFT bins put far more points — and therefore far more
weight — at the top of a band than at the bottom, which changes the correlation
coefficients. Pass ``resolution='filterbank'`` to match the paper;
``'fft'`` (the default) keeps the historical behaviour of this module.

Normalisation (``normalize=``) — read this before comparing ACROSS subjects
--------------------------------------------------------------------------
Trapeau correlates DTFs, i.e. transfer functions with the direction-independent
component removed. What that removal can be here is constrained by how these
HRTFs are made: only the az=0 vertical arc is measured, and the azimuths are
duplicates of it with spherical-head ILD/ITD imposed
(``record.processing.expand_azimuths_with_binaural_cues``). A diffuse-field
average over "all sources" is therefore just the arc mean plus a mean ILD term
— it carries no independent information.

``'none'`` (default)
    Correlate the raw transfer functions. Direction-INdependent spectrum is
    still present and is shared by every elevation, which inflates the
    correlations and depresses VSI by however much of that component the
    recording retained. Fine for WITHIN-subject comparisons (original vs
    modified share the same common component); NOT comparable across subjects.

``'arc_mean'``
    Subtract the mean dB spectrum across the correlated sources — Trapeau's DTF
    read literally on the measured arc. DEGENERATE and kept only as a control:
    mean-removed vectors sum to zero, so the pairwise covariances must sum to
    minus the total variance, mean(r) ≈ −1/(n−1) and VSI ≈ 1 + 1/(n−1) for every
    subject. Any spread it shows tracks grid density (n elevations), not cues.

``'detail'``
    Remove each direction's own cepstrally-smoothed envelope (``n_keep`` cosine
    coefficients, Kulkarni & Colburn 1998) and correlate the residual fine
    structure — the same envelope/detail split used by
    ``hrtf.modify.shift_spectral_detail``. Strips the broad common shape without
    imposing a zero-sum constraint, so between-subject spread survives. Use this
    for cross-subject work.

None of these is the paper's normalisation, and none can be: Trapeau divide each
transfer function by the grand RMS average over a spatially uniform sample of 48
directions, 22.5 deg apart in azimuth and elevation, INCLUDING THE REAR FIELD.
That estimate is independent of the 13 median-plane elevations they then
correlate, which is what keeps the correlations from being forced negative. A
frontal-arc-only recording has no such independent sample available.
"""

import numpy
import slab

from hrtf_relearning.hrtf.modify.shift_spectral_detail import smooth_magnitude

NORMALIZATIONS = ('none', 'arc_mean', 'detail')
DEFAULT_N_KEEP = 4

# Five octave bands between 4 and 16 kHz; VSI is highest in the middle one.
PAPER_OCTAVE_BANDS = ((4000, 8000), (4757, 9514), (5657, 11314),
                      (6727, 13454), (8000, 16000))
# Four non-overlapping half-octave bands; VSI correlated with vertical RMSE in
# the second one (R = -0.53, corrected p = 0.0126).
PAPER_HALF_OCTAVE_BANDS = ((4000, 5657), (5657, 8000),
                           (8000, 11314), (11314, 16000))
BEHAVIOURAL_BAND = (5657, 8000)
PAPER_BANDS = PAPER_OCTAVE_BANDS          # backwards-compatible alias

# Middlebrooks (1999) filter bank, used unchanged by Trapeau. NOT an auditory
# filterbank: no gammatones, no ERB spacing. It is a log-frequency resampling
# grid, chosen "to avoid over-representing higher frequencies and for ease in
# scaling DTFs in frequency" — 85 triangular filters from 3 to 16 kHz. For
# scale: the 0.057-octave 3-dB bandwidth is about a third of an ERB at 7 kHz
# (~0.15 octave), so these filters are far narrower than auditory filters.
OCTAVE_SPACING = 0.0286      # centre spacing [octaves] = 2% frequency steps
FILTER_SLOPE_DB = 105.0      # skirt slope [dB/octave] -> 0.057 oct 3-dB width
                             # (105 dB/oct x 0.0286 oct = 3 dB); "triangular"
                             # describes the shape on a dB vs log-f axis


def median_plane_sources(hrtf):
    """``cone_sources(0)`` with duplicate elevations dropped.

    Some recordings (494 sources instead of 475) carry the az=0 arc twice
    because the azimuth grid included 0 at both ends, so ``cone_sources(0)``
    returns each elevation twice. Duplicates correlate at r=1 with themselves,
    which depresses VSI, and they change n — which the ``arc_mean``
    normalisation is directly sensitive to. Always select elevations here.
    """
    sources = list(hrtf.cone_sources(0))
    elevations = hrtf.sources.vertical_polar[sources, 1]
    seen, unique = set(), []
    for source, elevation in zip(sources, numpy.round(elevations, 2)):
        if elevation not in seen:
            seen.add(elevation)
            unique.append(source)
    return unique


def filterbank_levels(freqs, tfs_db, bandwidth, octave_spacing=OCTAVE_SPACING,
                      slope_db=FILTER_SLOPE_DB):
    """Band levels from the Middlebrooks (1999) filter bank.

    Centre frequencies are spaced ``octave_spacing`` octaves apart (2% steps),
    so an octave band yields 36 points evenly spaced in LOG frequency instead of
    the top-heavy set of linear FFT bins. Each filter is triangular *on a dB vs
    log-frequency axis* — its magnitude falls linearly at ``slope_db`` dB per
    octave either side of the centre, giving a 3-dB bandwidth of
    ``2 * slope_db**-1 * 3`` octaves (0.057 at the paper's 105 dB/octave).

    Note this is a dB-domain triangle, not an amplitude-domain one: the skirts
    decay exponentially in amplitude and never reach exactly zero, so each
    filter integrates a good deal more than its 3-dB width. Filtering is done in
    the power domain, weights being the squared magnitude response.

    Returns (n_sources, n_bands) in dB.
    """
    freqs = numpy.asarray(freqs, dtype=float)
    low, high = float(bandwidth[0]), float(bandwidth[1])
    n_bands = int(round(numpy.log2(high / low) / octave_spacing))
    centers = low * 2.0 ** (numpy.arange(n_bands + 1) * octave_spacing)

    positive = freqs > 0
    log_freqs = numpy.full(freqs.shape, -numpy.inf)
    log_freqs[positive] = numpy.log2(freqs[positive])

    power = 10.0 ** (numpy.asarray(tfs_db, dtype=float) / 10.0)   # (n_src, n_bins)
    levels = numpy.empty((power.shape[0], len(centers)))
    for i, center in enumerate(centers):
        octaves = numpy.abs(log_freqs - numpy.log2(center))
        # |H| in dB is -slope*octaves; weights are |H|^2, i.e. power gain
        weights = 10.0 ** (-slope_db * octaves / 10.0)
        weights[~positive] = 0.0
        total = weights.sum()
        if total <= 0:      # centre outside the measured frequency range
            nearest = int(numpy.argmin(numpy.abs(freqs - center)))
            weights = numpy.zeros_like(freqs)
            weights[nearest] = total = 1.0
        levels[:, i] = power @ weights / total
    return 10.0 * numpy.log10(numpy.maximum(levels, numpy.finfo(float).tiny))


def dtf_matrix(hrtf, ear, sources=None, bandwidth=(5657, 11314),
               normalize='none', n_keep=DEFAULT_N_KEEP, resolution='fft',
               octave_spacing=OCTAVE_SPACING):
    """Spectra to correlate: (n_sources, n_points_in_band), dB.

    Normalisation is applied over the FULL spectrum and the band is selected
    afterwards, so the cepstral fit is not biased by a truncated slice.
    See the module docstring for what ``normalize`` and ``resolution`` mean.
    """
    if normalize not in NORMALIZATIONS:
        raise ValueError(f"normalize must be one of {NORMALIZATIONS}, got {normalize!r}")
    if resolution not in ('fft', 'filterbank'):
        raise ValueError(f"resolution must be 'fft' or 'filterbank', got {resolution!r}")
    freqs, _ = hrtf[0].tf(show=False)   # works for both HRIR and TF filters
    if sources is None:
        sources = median_plane_sources(hrtf)
    tfs = hrtf.tfs_from_sources(sources, n_bins=len(freqs), ear=ear).squeeze()  # (n_src, n_bins) dB

    if normalize == 'arc_mean':
        tfs = tfs - tfs.mean(axis=0, keepdims=True)
    elif normalize == 'detail':
        mag = 10.0 ** (tfs.T / 20.0)                      # (n_bins, n_src), linear
        envelope = smooth_magnitude(mag, n_keep=n_keep)
        tfs = (20.0 * numpy.log10(mag / envelope)).T      # back to (n_src, n_bins) dB

    if resolution == 'filterbank':
        return filterbank_levels(freqs, tfs, bandwidth, octave_spacing)
    freq_idx = numpy.logical_and(freqs >= bandwidth[0], freqs <= bandwidth[1])
    return tfs[:, freq_idx]


def vsi(hrtf, bandwidth=(5657, 11314), normalize='none', n_keep=DEFAULT_N_KEEP,
        resolution='fft'):
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
    normalize : 'none' | 'arc_mean' | 'detail'
        How the direction-independent component is removed. The default keeps
        historical behaviour and is only valid WITHIN a subject — see the module
        docstring before comparing values across subjects.
    n_keep : int
        Cosine coefficients kept for the envelope when ``normalize='detail'``.

    Returns
    -------
    float
    """
    sources = median_plane_sources(hrtf)
    n = len(sources)

    ear_vsi = []
    for ear in ('left', 'right'):
        dtfs = dtf_matrix(hrtf, ear, sources=sources, bandwidth=bandwidth,
                          normalize=normalize, n_keep=n_keep,
                          resolution=resolution)
        off_diag = [
            float(numpy.corrcoef(dtfs[i], dtfs[j])[0, 1])
            for i in range(n) for j in range(n) if i != j
        ]
        ear_vsi.append(1.0 - float(numpy.mean(off_diag)))

    return float(numpy.mean(ear_vsi))


def vsi_bands(hrtf, bands=PAPER_OCTAVE_BANDS, normalize='none',
              n_keep=DEFAULT_N_KEEP, resolution='fft'):
    """VSI in several bands at once: ``{band: vsi}``.

    Equivalent to calling :func:`vsi` per band, but the normalisation — the
    expensive part for ``'detail'``, a least-squares cosine fit per direction —
    is done once over the full spectrum and the bands are sliced out of the
    result. Use this for band sweeps; a per-band loop over :func:`vsi` re-fits
    the envelope every time and is ~n_bands slower for no difference in output.
    """
    freqs, _ = hrtf[0].tf(show=False)
    sources = median_plane_sources(hrtf)
    n = len(sources)
    # normalise once over the full spectrum, then slice/resample each band
    full = {ear: dtf_matrix(hrtf, ear, sources=sources,
                            bandwidth=(freqs[0], freqs[-1]),
                            normalize=normalize, n_keep=n_keep)
            for ear in ('left', 'right')}

    out = {}
    for band in bands:
        ear_vsi = []
        for ear in ('left', 'right'):
            if resolution == 'filterbank':
                data = filterbank_levels(freqs, full[ear], band)
            else:
                idx = numpy.logical_and(freqs >= band[0], freqs <= band[1])
                data = full[ear][:, idx]
            corr = numpy.corrcoef(data)
            # mean of the off-diagonal entries, diagonal is n ones
            ear_vsi.append(1.0 - (corr.sum() - n) / (n * (n - 1)))
        out[tuple(band)] = float(numpy.mean(ear_vsi))
    return out


def vsi_dissimilarity(hrtf_1, hrtf_2, bandwidth=(5700, 11300),
                      normalize='none', n_keep=DEFAULT_N_KEEP):
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
    normalize       : 'none' | 'arc_mean' | 'detail' — see the module docstring.
    n_keep          : envelope coefficients when ``normalize='detail'``

    Returns
    -------
    float
    """
    sources = median_plane_sources(hrtf_1)
    n = len(sources)

    ear_dissim = []
    for ear in ('left', 'right'):
        d1 = dtf_matrix(hrtf_1, ear, sources=sources, bandwidth=bandwidth,
                        normalize=normalize, n_keep=n_keep)
        d2 = dtf_matrix(hrtf_2, ear, sources=sources, bandwidth=bandwidth,
                        normalize=normalize, n_keep=n_keep)

        cross = numpy.array(
            [[numpy.corrcoef(d1[i], d2[j])[0, 1] for j in range(n)] for i in range(n)]
        )
        auto = numpy.array(
            [[numpy.corrcoef(d1[i], d1[j])[0, 1] for j in range(n)] for i in range(n)]
        )

        ear_dissim.append(float(numpy.sqrt(numpy.mean((cross - auto) ** 2))))

    return float(numpy.mean(ear_dissim))
