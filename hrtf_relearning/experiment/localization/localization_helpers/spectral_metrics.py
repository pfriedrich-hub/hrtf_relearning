"""
spectral_metrics.py

Cue depth, source variation and the cue:source ratio, on one shared
log-frequency axis so the two are directly comparable.

These are the primitives protocols/dev/stimulus_check.py works with, lifted
into an importable module because more than one caller now needs them and a
cell-by-cell protocol script must never be imported (its cells run at import).
The definitions are byte-identical to that file's, so numbers computed here are
comparable to the cue:source ratios quoted in stimulus_check cells 4 and 6.
stimulus_check.py still carries its own copies; migrate it when convenient.

Everything is measured as the rms per DCT coefficient of a set of 1/6-octave
smoothed log spectra, restricted to a ripple-density band:

    cue depth      = variation across ELEVATION of one ear's DTF
    ISD depth      = variation across elevation of the LEFT-RIGHT difference
    source variation = variation across TOKENS of the stimulus

`ripple_rms` takes the SD across the set before the band restriction, so each
measure is the elevation- (or token-) dependent part; the direction-independent
common transfer function and the broadband ILD drop out on their own.
"""
import numpy
import slab
from scipy.fftpack import dct

from hrtf_relearning.experiment.localization.localization_helpers.stimulus import (
    make_gapped_pinknoise, make_rippled_pinknoise, RMS_TILT, RMS_CUE)
from hrtf_relearning.utils import paths

FLO, FHI, NPTS = 3500.0, 16000.0, 192
LOGF = numpy.logspace(numpy.log10(FLO), numpy.log10(FHI), NPTS)
NOCT = numpy.log2(FHI / FLO)
BW = 2 ** (1 / 12)              # +/- 1/12 oct -> 1/6 octave window (~1 ERB at 8 kHz)
DENSITY = numpy.arange(NPTS) / (2 * NOCT)
CUE_BAND = (0.5, 2.0)
TILT_BAND = (0.2, 0.5)


def log_axis(flo=FLO, fhi=FHI, npts=NPTS):
    """(log-frequency points, ripple density per DCT coefficient) for an axis.

    Ripple density is defined per axis, so anything comparing a spectrum to a
    DCT band has to state which axis it is on. The default is the 3.5-16 kHz
    axis this codebase measures cue:source ratios on; the stimulus envelope
    itself lives on stimulus.SHAPE_FLO..SHAPE_FHI with SHAPE_N points, so
    figures that overlay envelopes on DTFs should pass that one.
    """
    logf = numpy.logspace(numpy.log10(flo), numpy.log10(fhi), npts)
    return logf, numpy.arange(npts) / (2 * numpy.log2(fhi / flo))


def log_spectrum(x, samplerate, logf=None):
    """1/6-octave-smoothed log power spectrum on `logf`, mean removed (shape only)."""
    logf = LOGF if logf is None else numpy.asarray(logf, dtype=float)
    x = numpy.asarray(x, dtype=float).squeeze()
    power = numpy.abs(numpy.fft.rfft(x)) ** 2
    freqs = numpy.fft.rfftfreq(len(x), 1 / samplerate)
    cumulative = numpy.concatenate([[0], numpy.cumsum(power)])
    lo = numpy.searchsorted(freqs, logf / BW)
    # at the bottom of a wide axis the 1/6-octave window can be narrower than
    # one FFT bin (500 Hz +/- 1/12 oct is 60 Hz; a 512-tap IR at 48828 Hz
    # resolves 95 Hz), which would leave an empty band and return -200 dB.
    # Take at least one bin. No-op on the default 3.5-16 kHz axis.
    hi = numpy.minimum(numpy.maximum(numpy.searchsorted(freqs, logf * BW), lo + 1),
                       len(freqs))
    band = (cumulative[hi] - cumulative[lo]) / numpy.maximum(hi - lo, 1)
    spectrum = 10 * numpy.log10(band + 1e-20)
    return spectrum - spectrum.mean()


def ripple_rms(spectra, band=CUE_BAND, density=None):
    """rms per DCT coefficient of a set of log spectra, within a ripple band.

    The SD is taken across the set first, so this is the across-set (elevation,
    or token) variation. `density` must match the axis the spectra are on --
    see `log_axis`.
    """
    density = DENSITY if density is None else numpy.asarray(density, dtype=float)
    coeffs = dct(numpy.asarray(spectra), type=2, norm='ortho', axis=1).std(0)
    sel = (density >= band[0]) & (density < band[1])
    return float(numpy.sqrt((coeffs[sel] ** 2).mean()))


def load_hrtf(sofa_name, subject_id=None):
    """slab.HRTF from SOFA_DIR/<subject_id>/<sofa_name>.sofa, with .name set."""
    subject_id = sofa_name.split('_')[0] if subject_id is None else subject_id
    hrtf = slab.HRTF(str(paths.SOFA_DIR / subject_id / f'{sofa_name}.sofa'))
    hrtf.name = sofa_name
    return hrtf


def column_indices(hrtf, azimuth=0.0, max_abs_elevation=35.0):
    """Indices of one azimuth column, sorted by elevation.

    The experiment negates azimuth relative to the SOFA convention (same
    convention as stimulus_check.cue_spectra). At azimuth 0 the two agree.
    """
    sources = hrtf.sources.vertical_polar
    sofa_az = numpy.mod(-azimuth, 360)
    az_grid = sources[:, 0]
    nearest = az_grid[numpy.argmin(numpy.abs(((az_grid - sofa_az + 180) % 360) - 180))]
    sel = ((numpy.abs(((az_grid - nearest + 180) % 360) - 180) < 1)
           & (numpy.abs(sources[:, 1]) <= max_abs_elevation))
    idx = numpy.flatnonzero(sel)
    return idx[numpy.argsort(sources[idx, 1])]


def dtf_spectra(hrtf, ear='left', azimuth=0.0, max_abs_elevation=35.0):
    """Log spectra of one DTF azimuth column across elevation -- the cue itself."""
    channel = 0 if ear == 'left' else 1
    idx = column_indices(hrtf, azimuth, max_abs_elevation)
    return numpy.array([log_spectrum(hrtf[i].data[:, channel], hrtf.samplerate)
                        for i in idx])


def cue_depth(hrtf, ear='left', band=CUE_BAND, **kwargs):
    """Elevation-dependent depth of one ear's spectral cue, dB rms."""
    return ripple_rms(dtf_spectra(hrtf, ear=ear, **kwargs), band)


def isd_depth(hrtf, band=CUE_BAND, **kwargs):
    """Elevation-dependent depth of the interaural spectral difference, dB rms.

    This is the cue that survives an unknown source spectrum: the source
    cancels in L-R at every frequency. Zero for a symmetric head; for real ears
    it is typically 55-80% of the monaural depth, because most of the
    directional variation is common to the two ears and cancels with it.
    """
    left = dtf_spectra(hrtf, ear='left', **kwargs)
    right = dtf_spectra(hrtf, ear='right', **kwargs)
    return ripple_rms(left - right, band)


def source_rms(n=120, band=CUE_BAND, kind='ripple', **stim_kwargs):
    """Across-token variation of the stimulus itself, dB rms in `band`."""
    spectra = []
    for _ in range(n):
        if kind == 'noise':
            sound = make_gapped_pinknoise()
        elif kind == 'ripple':
            sound = make_rippled_pinknoise(
                rms_tilt=stim_kwargs.get('rms_tilt', RMS_TILT),
                rms_cue=stim_kwargs.get('rms_cue', RMS_CUE))[0]
        else:
            raise ValueError(f"kind must be 'noise' or 'ripple', got {kind!r}")
        spectra.append(log_spectrum(sound.data[:, 0], sound.samplerate))
    return ripple_rms(numpy.array(spectra), band)


def calibrate_rms_cue(hrtf, ear='left', target_ratio=1.0, rms_tilt=3.0,
                      grid=numpy.arange(0.0, 6.01, 0.5), n=120, band=CUE_BAND,
                      verbose=True, **kwargs):
    """`rms_cue` whose in-band source variation gives cue:source = target_ratio.

    Run this ONCE for a study, not once per subject. The stimulus is held
    constant across participants: the comparisons that matter are within
    subject, so a listener who localizes a given ripple better than another is
    not a confound, and a constant stimulus keeps the method reportable in one
    line. What a constant cannot absorb is a listener at FLOOR, so choose it
    against the SHALLOWEST cue depth in the pool rather than a typical one --
    the same worst-case rule stimulus_check.py cell 6 uses to pick rms_tilt
    from the weaker ear. Cue depths differ enough for this to matter (AS 7.84,
    GS 13.79, FS 13.59 dB on this axis).

    Returns (rms_cue, report) where report holds the measured cue depth and the
    ratio at every grid point, so the choice is auditable rather than a bare
    number. Note the grid floor is not zero: with rms_cue=0 the below-band
    envelope already leaks a little into the cue band, which is the ceiling on
    how clean the separation can be.
    """
    depth = cue_depth(hrtf, ear=ear, band=band, **kwargs)
    ratios = []
    for value in grid:
        source = source_rms(n=n, band=band, rms_tilt=rms_tilt, rms_cue=float(value))
        ratios.append(depth / source if source > 0 else numpy.inf)
        if verbose:
            print(f'  rms_cue {value:4.1f} dB -> source {source:5.2f} dB, '
                  f'cue:source {ratios[-1]:5.2f}:1')
    ratios = numpy.array(ratios)
    best = int(numpy.argmin(numpy.abs(ratios - target_ratio)))
    report = {'cue_depth_db': depth, 'ear': ear, 'band': band,
              'rms_tilt': rms_tilt, 'target_ratio': target_ratio,
              'grid': [float(g) for g in grid], 'ratios': ratios.tolist(),
              'chosen_rms_cue': float(grid[best]),
              'chosen_ratio': float(ratios[best])}
    if verbose:
        print(f'  cue depth {depth:.2f} dB rms -> rms_cue {grid[best]:.1f} dB '
              f'({ratios[best]:.2f}:1, target {target_ratio:.2f}:1)')
    return float(grid[best]), report
