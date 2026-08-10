"""
stimulus.py

Single source of truth for the localization test stimulus, so the dome
(real loudspeaker) and AR/VR (pybinsim) conditions play the *same* stimulus and
only the transducer/level path differs.

`make_gapped_pinknoise` reproduces the original Localization_AR synthesis:
225 ms pinknoise (10 ms cosine edges) with four 25 ms gaps zeroed out at
25-50, 75-100, 125-150, 175-200 ms, each gap edge given a 5 ms raised-cosine
ramp. This yields five ~25 ms flat bursts drawn from one continuous noise
realization (independent per burst), 4x25 ms gaps -> 225 ms total.

`make_rippled_pinknoise` is the same burst train recoloured by a NEW random
spectral envelope on every trial. It exists because plain pinknoise does not
vary: its expected spectrum is identical every trial and only the realisation
noise differs, which at 1/6-octave resolution is 0.4 dB SD across trains (0.8
dB across the five bursts within one train) against a ~3.3 dB elevation cue in
a measured DTF. The sound at the eardrum therefore carries a near one-to-one
map from *absolute* spectrum to elevation, and a listener can solve the
localization task by template matching without ever separating source from
filter, so nothing distinguishes learning that generalises across sources from
learning tied to this one stimulus. Note what the fix buys and what it does
not: a moving source spectrum rules out a mapping keyed to the overall timbre
of the training stimulus, but NOT a lookup restricted to the preserved cue
band, since the notch is a shape-within-band feature that survives the
manipulation under either account. See docs/stimulus_spectral_variation.md,
"What the manipulation can and cannot decide".

Natural sources differ from one another mainly in broad spectral colouration,
which the auditory system must discount to read the finer pinna pattern, so
that is what the envelope imposes: it is band-limited to ripple densities below
RIPPLE_TILT_MAX (0.5 ripples/oct) and leaves the 0.5-2 ripples/oct band -- where
the notch shifts with elevation, cf. n_keep in hrtf.modify -- largely untouched.
With the defaults below every trial is strongly recoloured (6.4 dB SD across
trials) while the cue still stands ~2.8:1 above the source variation in the cue
band and is swamped (~0.37:1) below it.

The envelope is applied before gapping and is shared by all five bursts of a
trial (burst-to-burst r = 0.95 for the detrended log spectra, vs -0.01 for the
unmodified stimulus). That matters: integrating across bursts averages the
inherent realisation noise down but leaves the imposed colouration intact, so
the source spectrum is a stable property of the trial rather than per-burst
noise.

Absolute loudness is handled downstream, NOT here:
  - dome:   freefield speaker calibration (equalize=True)
  - AR/VR:  the matched pybinsim gain (loc_settings['gain'])
so callers pick `level` only to set the WAV/DAC reference; matching the two
transducer paths is a separate by-ear step (match_ar_dome_loudness.py).
"""
import numpy
import slab
from scipy.fftpack import idct

DURATION = 0.225          # s, whole burst train

# --- random spectral shape (make_rippled_pinknoise) --------------------------
SHAPE_FLO = 500.0         # Hz, shape is defined on a log axis over this range
SHAPE_FHI = 16000.0       # and held at the edge value outside it
SHAPE_N = 128             # points on that log axis
SHAPE_OCTAVES = numpy.log2(SHAPE_FHI / SHAPE_FLO)

RIPPLE_TILT_MAX = 0.5     # ripples/oct, upper edge of the "normalisable" band
RIPPLE_CUE_MAX = 2.0      # ripples/oct, upper edge of the pinna-notch band

# Defaults measured against FS_donor_AS (left ear, az -20), 400 realisations:
# 6.4 dB across-trial SD, cue:source 0.37:1 in the tilt band, 2.8:1 in the cue band.
#
# RMS_CUE is 0 by design. Variation inside the cue band is a cost, not a
# benefit -- it contaminates the very cue being read, and the work of defeating
# a fixed-template read-out is done entirely by the tilt band. It is kept as a
# parameter only as a diagnostic (deliberately contaminate the cue and watch
# performance fall). The residual 2.8:1 rather than infinity is leakage: a
# band-limited envelope cannot have a perfectly sharp edge at 0.5 ripples/oct,
# so raising RMS_TILT still erodes the cue ratio.
#
# Cue depth varies between ears and subjects -- re-check per subject with
# protocols/dev/stimulus_check.py, cell 6.
RMS_TILT = 8.0            # dB rms of the envelope below RIPPLE_TILT_MAX
RMS_CUE = 0.0             # dB rms of the envelope in the cue band (diagnostic)

_RNG = numpy.random.default_rng()


def _apply_gaps(stim):
    """Zero four 25 ms windows with 5 ms raised-cosine edges, in place."""
    n_silent = (numpy.arange(25, 221, 25).reshape(4, 2) * stim.samplerate / 1000).astype(int)
    ramp_len = int(.005 * stim.samplerate)
    half_len = int(ramp_len / 2)
    for start, end in n_silent:
        ramp_up = 0.5 * (1 - numpy.cos(numpy.linspace(0, numpy.pi, ramp_len)))
        ramp_down = 0.5 * (1 - numpy.cos(numpy.linspace(numpy.pi, 0, ramp_len)))
        ramp_up = ramp_up[:, numpy.newaxis]
        ramp_down = ramp_down[:, numpy.newaxis]
        # ramp the noise down into / up out of each gap, then silence the middle
        stim.data[start - half_len: start + half_len] *= (1 - ramp_up)
        stim.data[end - half_len: end + half_len] *= (1 - ramp_down)
        stim.data[start + half_len: end - half_len] = 0
    return stim


def _density(k):
    """Ripple density (ripples/octave) of DCT coefficient k on the shape axis."""
    return numpy.asarray(k) / (2 * SHAPE_OCTAVES)


def _band_indices(ripple_max=RIPPLE_CUE_MAX):
    k = numpy.arange(SHAPE_N)
    d = _density(k)
    return (k > 0) & (d < RIPPLE_TILT_MAX), (d >= RIPPLE_TILT_MAX) & (d < ripple_max)


def _scaled(sel, rms, rng):
    """Coefficients on `sel` scaled so their inverse DCT has rms `rms` dB."""
    c = numpy.zeros(SHAPE_N)
    c[sel] = rng.standard_normal(int(sel.sum()))
    s = idct(c, type=2, norm='ortho')
    return c * (rms / numpy.std(s - s.mean()))


def shape_coefficients(rms_tilt=RMS_TILT, rms_cue=RMS_CUE, rng=None,
                       flat_rms=None, ripple_max=RIPPLE_CUE_MAX):
    """Draw one random shape, return its DCT coefficients (length SHAPE_N).

    Each band is scaled so the shape it contributes has exactly the requested
    rms in dB over the log-frequency axis. Scaling coefficients scales the
    inverse DCT by the same factor, so the coefficients returned here fully
    determine the shape -- log them and the trial is reconstructible.

    Two modes:
      banded (default) -- rms_tilt below RIPPLE_TILT_MAX, rms_cue from there to
        ripple_max. With rms_cue=0 the elevation cue band is left alone.
      flat_rms         -- overrides both: one iid draw across 0 < density <
        ripple_max, scaled to flat_rms dB, i.e. variation at every spectral
        scale including the cue band. Note this is NOT the same as setting
        rms_tilt == rms_cue: the two bands hold 4 and ~15 coefficients, so
        normalising each to the same point-wise rms puts far more energy per
        coefficient in the tilt band.
    """
    rng = _RNG if rng is None else rng
    if flat_rms is not None:
        k = numpy.arange(SHAPE_N)
        sel = (k > 0) & (_density(k) < ripple_max)
        return _scaled(sel, flat_rms, rng)
    tilt_sel, cue_sel = _band_indices(ripple_max)
    coeffs = numpy.zeros(SHAPE_N)
    for sel, rms in ((tilt_sel, rms_tilt), (cue_sel, rms_cue)):
        if sel.any() and rms > 0:
            coeffs += _scaled(sel, rms, rng)
    return coeffs


def shape_from_coefficients(coeffs):
    """(frequencies, gain in dB) for a coefficient vector from `shape_coefficients`."""
    coeffs = numpy.asarray(coeffs, dtype=float)
    if coeffs.size < SHAPE_N:   # only the active coefficients were logged
        coeffs = numpy.pad(coeffs, (0, SHAPE_N - coeffs.size))
    shape = idct(coeffs, type=2, norm='ortho')
    freqs = numpy.logspace(numpy.log10(SHAPE_FLO), numpy.log10(SHAPE_FHI), SHAPE_N)
    return freqs, shape - shape.mean()


def n_active_coefficients(ripple_max=RIPPLE_CUE_MAX):
    """How many leading DCT coefficients are ever non-zero (the rest are 0)."""
    tilt_sel, cue_sel = _band_indices(ripple_max)
    return int(numpy.flatnonzero(tilt_sel | cue_sel).max()) + 1


def make_gapped_pinknoise(level=80):
    """225 ms gapped pinknoise (5x25 ms bursts, 4x25 ms gaps). Uses the slab
    default samplerate, so callers should set it before calling."""
    stim = slab.Sound.pinknoise(duration=DURATION, level=level).ramp(when='both', duration=0.01)
    return _apply_gaps(stim)


def make_rippled_pinknoise(level=80, rms_tilt=RMS_TILT, rms_cue=RMS_CUE,
                           seed=None, coeffs=None, flat_rms=None,
                           ripple_max=RIPPLE_CUE_MAX):
    """Gapped pinknoise x a fresh random smooth spectral shape.

    Same burst train as `make_gapped_pinknoise`, so timing, duration and
    bandwidth are unchanged; only the source spectrum moves from trial to
    trial. The shape is zero-phase and smooth (<= ripple_max ripples/oct,
    impulse response well under a millisecond), so it does not smear energy
    into the gaps -- it is applied before gapping in any case.

    Note the envelope spans the WHOLE frequency range (SHAPE_FLO..SHAPE_FHI),
    including every frequency that carries an elevation cue; what the default
    restricts is the spectral SCALE of the variation, not where in frequency it
    applies. A coarse envelope raises and lowers the level around a notch
    without filling it in or creating a new one.

    Parameters
    ----------
    rms_tilt, rms_cue : dB rms of the envelope below / within the cue band.
        rms_cue is 0 by default -- see the module docstring.
    flat_rms : dB rms, optional. Broadband-in-density variant: overrides
        rms_tilt/rms_cue and varies the envelope at every scale up to
        ripple_max, cue band included. Macpherson & Middlebrooks (2003) saw
        no degradation at 1 ripple/oct below ~20 dB peak-to-trough, so there is
        headroom here, but the cue ratio falls quickly -- measure before use.
    ripple_max : upper density limit of the envelope, ripples/oct.
    seed : int, optional. Reproducible shape for a given trial.
    coeffs : array, optional. Replay an exact shape logged earlier; overrides
        seed / rms.

    Returns
    -------
    stim : slab.Sound
    params : dict, everything needed to reconstruct this trial's spectrum.
    """
    rng = numpy.random.default_rng(seed) if seed is not None else _RNG
    if coeffs is None:
        coeffs = shape_coefficients(rms_tilt=rms_tilt, rms_cue=rms_cue, rng=rng,
                                    flat_rms=flat_rms, ripple_max=ripple_max)
    else:
        coeffs = numpy.asarray(coeffs, dtype=float)

    stim = slab.Sound.pinknoise(duration=DURATION, level=level)
    freqs, shape_db = shape_from_coefficients(coeffs)
    bins = numpy.fft.rfftfreq(stim.n_samples, 1 / stim.samplerate)
    gain = 10 ** (numpy.interp(numpy.clip(bins, SHAPE_FLO, SHAPE_FHI),
                               freqs, shape_db) / 20)
    data = numpy.fft.irfft(numpy.fft.rfft(stim.data[:, 0]) * gain, stim.n_samples)
    stim = slab.Sound(data[:, None], samplerate=stim.samplerate)
    stim.level = level
    stim = _apply_gaps(stim.ramp(when='both', duration=0.01))

    params = {'kind': 'ripple',
              'mode': 'flat' if flat_rms is not None else 'banded',
              'rms_tilt': float(rms_tilt), 'rms_cue': float(rms_cue),
              'flat_rms': None if flat_rms is None else float(flat_rms),
              'seed': None if seed is None else int(seed),
              'tilt_max': RIPPLE_TILT_MAX, 'ripple_max': float(ripple_max),
              'shape_flo': SHAPE_FLO, 'shape_fhi': SHAPE_FHI,
              'coeffs': [float(c) for c in coeffs[:n_active_coefficients(ripple_max)]]}
    return stim, params
