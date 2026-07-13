"""
modify.py — individualised-HRTF manipulations for the relearning experiment.

Two modes (see MODE in __main__):

- 'shift'  : the relearning manipulation. Split each HRTF magnitude spectrum
             into a coarse envelope and fine detail with a truncated cosine
             (Fourier) series on log-magnitude (Kulkarni & Colburn 1998,
             Nature 396:747), then translate the fine detail — the pinna
             peaks/notches carrying elevation cues — inside the Trapeau
             peak-VSI octave (5.7–11.3 kHz), by a constant amount on the
             ERB-number axis. The selection window shifts WITH the detail so the
             band is not cut short; the coarse envelope is left in place (and is
             the spectrum outside the shifted band), so every direction keeps a
             unique pattern simply displaced in ERB: a bijective, relearnable
             remap rather than a destroyed or conflicting cue. Magnitude-only
             (original phase kept); ITD/ILD are added downstream when lateral
             (az != 0) sources are synthesised.

- 'synth'  : legacy path. Cosine-series smoothing plus synthetic Gaussian
             spectral features, blended into the magnitude and resynthesised as
             minimum phase with onset-based ITD restoration.

Shared primitives (_smooth, minimum_phase_from_magnitude, restore_itd_from_onsets)
are used by 'synth'; 'shift' uses _smooth for the split and keeps original phase.
"""

import copy
import numpy
import matplotlib
import matplotlib.ticker
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from hrtf_relearning import PATH
from hrtf_relearning.utils import paths
hrtf_dir = paths.SOFA_DIR
import slab
from hrtf_relearning.hrtf.analysis.vsi import (vsi as _vsi, vsi_dissimilarity as _vsi_dissimilarity)

sub_id = 'AS'

SMOOTH = True
N_KEEP = 4
PLOT = 'image'   # 'image' (contour heatmap) | 'waterfall' | 'surface' (via plot_tf)
# fname = 'shift'   # output SOFA suffix: <sub_id>_<fname>.sofa

# ---------------------------------------------------------------------------
# Modification mode
# ---------------------------------------------------------------------------
# 'synth' : cepstral smoothing + synthetic spectral features (FEATURES below).
#           This is the original behaviour.
# 'shift' : cepstral envelope/detail split + frequency-shift of the subject's
#           OWN fine cues inside a band (SHIFT_* params below).  Ignores
#           SMOOTH / N_KEEP / FEATURES.
MODE = 'shift'
fname = MODE

# --- 'shift' mode parameters (only used when MODE == 'shift') --------------
# shift_detail selects the fine detail in SHIFT_BAND (Trapeau peak-VSI octave)
# and translates it along the ERB-number axis by SHIFT_ERB: each output frequency
# f samples the windowed detail at ERB(f) - SHIFT_ERB, so peaks/notches move a
# constant ERB step while the broad envelope stays put. The selection window
# moves with the detail, so the band is not cut short at its edges.
#   > 0  shifts cues UP   in frequency
#   < 0  shifts cues DOWN in frequency
#   = 0  is a rebuild no-op (analysis + resynthesis, cues unmoved)
# Equivalence above ~1 kHz: factor 1.3 ≈ 2.4 ERB, factor 1.4 ≈ 3.0 ERB.
# Set PLOT = 'waterfall' below to see the per-elevation spectra stacked.
SHIFT_BAND      = (5700, 11300)  # Trapeau et al. 2016 peak-VSI octave; None -> whole spectrum
SHIFT_ERB       = 2.5
SHIFT_ENV_NKEEP = 4      # Fourier coeffs kept for the envelope (M; Kulkarni & Colburn 1998)
SHIFT_SKIRT     = 0.25   # raised-cosine taper on the selection window [octaves]
SHIFT_EQ_RMS    = True   # match in-band detail energy per direction/ear

# ---------------------------------------------------------------------------
# Spectral feature list
# ---------------------------------------------------------------------------
# Each entry is one spectral feature (notch or peak) added to the HRTF.
# Parameters are linearly interpolated between two spatial anchor points
# X1 and X2 (azimuth, elevation) in degrees.
#
# Keys per feature
# ----------------
# freqs : (f_at_X1, f_at_X2)   centre frequency [Hz]
# width : (w_at_X1, w_at_X2)   Gaussian σ / bandwidth [Hz]
# depth : (d_at_X1, d_at_X2)   magnitude in dB
#     > 0  →  notch  (spectral attenuation below reference level)
#     < 0  →  peak   (spectral boost above reference level)
# X1, X2 : (azimuth, elevation) spatial anchor directions
# ---------------------------------------------------------------------------
FEATURES = [
    # {
    #     'freqs': (8000, 9000),  # centre freq at X1 and X2 [Hz]
    #     'width': (300,   300),   # Gaussian σ at X1 and X2 [Hz]
    #     'depth': (12.0,  12.0), # >0 = notch, <0 = peak [dB]
    #     'X1':    (0, 0),         # anchor 1 (az, az)
    #     'X2':    (-40, 40),      # anchor 2 (el, el)
    # },
    # # Add further features here, e.g.:
    # {
    #     'freqs': (11000, 10000),
    #     'width': (300, 300),
    #     'depth': (12, 12),   # negative → peak
    #     'X1': (0, 0),
    #     'X2': (-40, 40),
    # },
    # {
    #     'freqs': (15500, 16000),
    #     'width': (300, 300),
    #     'depth': (12, 12),  # negative → peak
    #     'X1': (0, 0),
    #     'X2': (-40, 40),
    # },
]

# FEATURES = []  # no features


# ---------------------------------------------------------------------------
# Shared helpers (same as modify.py)
# ---------------------------------------------------------------------------

def _smooth(mag, n_keep):
    """
    Smooth a one-sided magnitude spectrum via truncated cosine-series
    reconstruction of log-magnitude (Kulkarni & Colburn 1998).
    """
    mag = numpy.asarray(mag, dtype=float)
    if mag.ndim != 2:
        raise ValueError("mag must have shape (n_bins, n_channels)")
    n_bins, n_channels = mag.shape
    n_samples = 2 * (n_bins - 1)
    n_keep = int(n_keep)
    if n_keep < 1 or n_keep > n_bins:
        raise ValueError(f"n_keep must be between 1 and {n_bins}, got {n_keep}")

    log_mag = numpy.log(numpy.maximum(mag, numpy.finfo(float).tiny))
    k = numpy.arange(n_bins, dtype=float)[:, None]
    n = numpy.arange(n_bins, dtype=float)[None, :]
    basis = numpy.cos(2.0 * numpy.pi * k * n / float(n_samples))

    log_mag_smooth = numpy.empty_like(log_mag)
    for ch in range(n_channels):
        coeffs, _, _, _ = numpy.linalg.lstsq(basis, log_mag[:, ch], rcond=None)
        coeffs[n_keep:] = 0.0
        log_mag_smooth[:, ch] = basis @ coeffs

    return numpy.exp(log_mag_smooth)


def minimum_phase_from_magnitude(mag):
    """Real-cepstrum minimum-phase reconstruction from a one-sided magnitude spectrum."""
    mag = numpy.asarray(mag, dtype=float)
    n_bins, n_channels = mag.shape
    n_samples = 2 * (n_bins - 1)
    tiny = numpy.finfo(float).tiny
    spec_min = numpy.empty((n_bins, n_channels), dtype=complex)

    for ch in range(n_channels):
        mag_ch = numpy.maximum(mag[:, ch], tiny)
        log_mag_half = numpy.log(mag_ch)
        log_mag_full = numpy.concatenate((log_mag_half, log_mag_half[-2:0:-1]))
        cep = numpy.fft.ifft(log_mag_full).real
        cep_min = numpy.zeros_like(cep)
        cep_min[0] = cep[0]
        cep_min[1:n_samples // 2] = 2.0 * cep[1:n_samples // 2]
        cep_min[n_samples // 2] = cep[n_samples // 2]
        spec_min[:, ch] = numpy.exp(numpy.fft.fft(cep_min))[:n_bins]

    return spec_min


def find_ir_onsets(ir, threshold_db=15.0):
    ir = numpy.asarray(ir, dtype=float)
    n_samples, n_channels = ir.shape
    onsets = numpy.zeros(n_channels, dtype=int)
    for ch in range(n_channels):
        x = numpy.abs(ir[:, ch])
        peak_idx = int(numpy.argmax(x))
        peak_val = float(x[peak_idx])
        if peak_val <= 0:
            continue
        threshold = peak_val / (10.0 ** (float(threshold_db) / 20.0))
        above = numpy.where(x[:peak_idx + 1] >= threshold)[0]
        onsets[ch] = int(above[0]) if len(above) else 0
    return onsets


def restore_itd_from_onsets(ir_original, ir_processed, threshold_db=15.0):
    ir_original  = numpy.asarray(ir_original,  dtype=float)
    ir_processed = numpy.asarray(ir_processed, dtype=float)
    n_samples, n_channels = ir_original.shape
    out = numpy.zeros_like(ir_processed)
    on_orig = find_ir_onsets(ir_original,  threshold_db=threshold_db)
    on_proc = find_ir_onsets(ir_processed, threshold_db=threshold_db)
    for ch in range(n_channels):
        delta = int(on_orig[ch] - on_proc[ch])
        if delta > 0:
            out[:, ch] = numpy.concatenate(
                (numpy.zeros(delta), ir_processed[:-delta, ch]))
        elif delta < 0:
            d = -delta
            out[:, ch] = numpy.concatenate(
                (ir_processed[d:, ch], numpy.zeros(d)))
        else:
            out[:, ch] = ir_processed[:, ch]
    return out


def linear_notch_position(azimuth, elevation, X1, X2, Y):
    X = numpy.column_stack((X1, X2))
    model = LinearRegression()
    model.fit(X, numpy.array(Y))
    return float(model.predict(numpy.array((azimuth, elevation)).reshape(1, -1))[0])


def linear_notch_width(azimuth, elevation, X1, X2, Y):
    X = numpy.column_stack((X1, X2))
    model = LinearRegression()
    model.fit(X, numpy.array(Y))
    sigma = float(model.predict(numpy.array((azimuth, elevation)).reshape(1, -1))[0])
    return float(max(sigma, numpy.finfo(float).eps))


def linear_scaling_factor(azimuth, elevation, X1, X2, Y):
    X = numpy.column_stack((X1, X2))
    model = LinearRegression()
    model.fit(X, numpy.array(Y))
    return float(model.predict(numpy.array((azimuth, elevation)).reshape(1, -1))[0])


def _compute_feature_params(azimuth, elevation, feature):
    """
    Return interpolated spectral-feature parameters for a given source direction.

    Parameters
    ----------
    azimuth, elevation : float
    feature : dict with keys freqs, width, depth, X1, X2

    Returns
    -------
    dict with keys mu, sigma, depth_db (signed: >0 notch, <0 peak).
    """
    X1, X2 = feature['X1'], feature['X2']
    return {
        "mu":       linear_notch_position(azimuth, elevation, X1=X1, X2=X2, Y=feature['freqs']),
        "sigma":    linear_notch_width(   azimuth, elevation, X1=X1, X2=X2, Y=feature['width']),
        "depth_db": linear_scaling_factor(azimuth, elevation, X1=X1, X2=X2, Y=feature['depth']),
    }


# ---------------------------------------------------------------------------
# Core processing — replacement / blending approach
# ---------------------------------------------------------------------------

def smooth_and_replace_hrtf(
        hrtf,
        n_keep=12,
        smooth=True,
        features=None,
        ref_n_keep=4,
        onset_threshold_db=15.0,
):
    """
    Apply spectral smoothing and direction-dependent spectral features to a
    slab.HRTF object, using a **linear magnitude blend** rather than a
    multiplicative dB gain.

    For each feature the Gaussian serves as a crossfade weight between the
    original (smoothed) magnitude and a *target* magnitude:

        w(f)       = exp(-0.5 * ((f - mu) / sigma)^2)
        target(f)  = mag_ref(f) * 10^(-depth_db / 20)
        mag_out(f) = (1 - w(f)) * mag_in(f)  +  w(f) * target(f)

    where mag_ref is a heavily smoothed version of the spectrum (controlled by
    ref_n_keep).  Using the smooth reference as the target anchor means the
    replacement defines what the spectral *trend* should be at those bins,
    rather than compounding with any existing fine structure.

    Parameters
    ----------
    hrtf : slab.HRTF
    n_keep : int
        Cosine coefficients for the main smoothing step.
    smooth : bool
        Apply main smoothing before feature insertion.
    features : list of dict
        Same format as in modify.py (freqs, width, depth, X1, X2).
        depth > 0 → notch, depth < 0 → peak.
        Pass [] or None to skip.
    ref_n_keep : int, default 4
        Cosine coefficients for the heavily-smoothed reference spectrum used
        as the blend target anchor.  Fewer coefficients → smoother reference,
        so the target level reflects the broad spectral trend rather than any
        fine peaks or dips already present.
    onset_threshold_db : float, default 15.0
        Threshold for onset-based ITD restoration.

    Returns
    -------
    slab.HRTF  (deep copy, processed)
    """
    if features is None:
        features = []

    out = copy.deepcopy(hrtf)

    for filt, source in zip(out, out.sources.vertical_polar):
        azimuth, elevation = float(source[0]), float(source[1])

        ir_original = numpy.asarray(filt.data, dtype=float)
        if ir_original.ndim != 2 or ir_original.shape[1] != 2:
            raise ValueError("Each HRIR must have shape (n_samples, 2)")

        n_samples, _ = ir_original.shape
        fs = filt.samplerate
        freqs = numpy.fft.rfftfreq(n_samples, d=1.0 / fs)

        spec_original = numpy.fft.rfft(ir_original, axis=0)
        mag_original  = numpy.abs(spec_original)

        # 1) Optional main smoothing
        mag_processed = _smooth(mag_original, n_keep=n_keep) if smooth else mag_original.copy()

        # 2) Spectral replacement — blend mag_processed toward a clean target
        if features:
            # Compute a heavily-smoothed reference that captures only the
            # broad spectral trend (used as target anchor for replacements)
            mag_ref = _smooth(mag_processed, n_keep=ref_n_keep)

            for feat in features:
                params = _compute_feature_params(azimuth, elevation, feat)
                mu, sigma, depth_db = params["mu"], params["sigma"], params["depth_db"]

                # Gaussian blending weight: 1 at centre, 0 in the tails
                w = numpy.exp(-0.5 * ((freqs - mu) / sigma) ** 2)

                # Target magnitude: reference level shifted by depth_db
                # depth_db > 0 → pull below reference (notch)
                # depth_db < 0 → push above reference (peak)
                target = mag_ref * (10.0 ** (-depth_db / 20.0))

                # Leave DC and Nyquist untouched
                w[0]  = 0.0
                w[-1] = 0.0

                # Linear blend in the magnitude domain
                mag_processed = (
                    (1.0 - w[:, None]) * mag_processed
                    + w[:, None] * target
                )

            # Clip against extreme values after all features applied
            mag_processed = numpy.clip(
                mag_processed,
                10.0 ** (-80.0 / 20.0),
                10.0 ** ( 80.0 / 20.0),
            )

        # 3) Minimum-phase reconstruction
        spec_processed = minimum_phase_from_magnitude(mag_processed)

        # 4) Back to time domain
        ir_processed = numpy.fft.irfft(spec_processed, n=n_samples, axis=0)

        # 5) Restore original ITD
        ir_processed = restore_itd_from_onsets(
            ir_original, ir_processed, threshold_db=onset_threshold_db,
        )

        filt.data = ir_processed

    return out


# ---------------------------------------------------------------------------
# Band-limited frequency shift of fine spectral detail
# ---------------------------------------------------------------------------
# Ported from hrtf_course.manipulations.shift_band (2026-06-17).  Reuses this
# module's native primitives (_smooth, minimum_phase_from_magnitude,
# restore_itd_from_onsets) rather than the vendored copies in hrtf_course.
#
# Where smooth_and_replace_hrtf *adds synthetic* spectral features, shift_band
# warps the subject's *own* fine spectral structure (the high-quefrency detail
# that carries vertical-localisation cues) up or down in frequency within a
# chosen band, leaving the broad spectral envelope in place.  Intended as a
# fallback manipulation for the main experiment when fully synthetic cues prove
# unlearnable: the shifted cues are spectrally realistic but remapped.


def octave_band(center_hz, fraction=1.0):
    """Return (low, high) Hz for an octave (or fractional-octave) band.

    >>> octave_band(8000)            # one octave centred on 8 kHz
    (5656.85..., 11313.70...)
    >>> octave_band(8000, 1 / 3)     # third-octave
    (7127.18..., 8979.69...)
    """
    if center_hz <= 0:
        raise ValueError(f"center_hz must be positive, got {center_hz}")
    if fraction <= 0:
        raise ValueError(f"fraction must be positive, got {fraction}")
    factor = 2.0 ** (fraction / 2.0)
    return float(center_hz / factor), float(center_hz * factor)


def erb_bandwidth(center_hz):
    """Glasberg & Moore (1990) equivalent rectangular bandwidth, in Hz.

    Useful for choosing band widths comparable to a single auditory filter.
    """
    f_kHz = center_hz / 1000.0
    return 24.7 * (4.37 * f_kHz + 1.0)


def hz_to_erb(f):
    """ERB-number (Glasberg & Moore 1990) for a frequency in Hz."""
    return 21.4 * numpy.log10(4.37 * numpy.asarray(f, dtype=float) / 1000.0 + 1.0)


def erb_to_hz(e):
    """Inverse of :func:`hz_to_erb`: frequency in Hz for an ERB-number."""
    return (10.0 ** (numpy.asarray(e, dtype=float) / 21.4) - 1.0) * 1000.0 / 4.37


def band_window(freqs, low_hz, high_hz, skirt_octaves=0.25):
    """Smooth in-band window on a log-frequency axis.

    The window is 1 inside ``[low_hz, high_hz]`` and tapers to 0 over a skirt of
    ``skirt_octaves`` on each side via a raised cosine in log frequency.  DC and
    Nyquist are forced to 0 so the manipulation stays phase-coherent and cannot
    blow up the band edges during minimum-phase reconstruction.
    """
    if not (0 < low_hz < high_hz):
        raise ValueError(f"need 0 < low_hz < high_hz, got {low_hz}, {high_hz}")
    if skirt_octaves < 0:
        raise ValueError(f"skirt_octaves must be >= 0, got {skirt_octaves}")

    f = numpy.asarray(freqs, dtype=float)
    w = numpy.zeros_like(f)

    pos = f > 0
    if not numpy.any(pos):
        return w

    f_pos = f[pos]
    log_f = numpy.log2(f_pos)
    log_lo = numpy.log2(low_hz)
    log_hi = numpy.log2(high_hz)
    skirt = float(skirt_octaves)

    if skirt == 0:
        w_pos = ((log_f >= log_lo) & (log_f <= log_hi)).astype(float)
    else:
        w_pos = numpy.zeros_like(f_pos)

        ramp_up = (log_f >= log_lo - skirt) & (log_f < log_lo)
        x = (log_f[ramp_up] - (log_lo - skirt)) / skirt
        w_pos[ramp_up] = 0.5 * (1 - numpy.cos(numpy.pi * x))

        flat = (log_f >= log_lo) & (log_f <= log_hi)
        w_pos[flat] = 1.0

        ramp_dn = (log_f > log_hi) & (log_f <= log_hi + skirt)
        x = (log_f[ramp_dn] - log_hi) / skirt
        w_pos[ramp_dn] = 0.5 * (1 + numpy.cos(numpy.pi * x))

    w[pos] = w_pos
    w[0] = 0.0
    if f.size and f[-1] > 0:
        w[-1] = 0.0
    return w


def shift_detail(
        hrtf,
        shift_erb,
        band=(5700.0, 11300.0),
        envelope_n_keep=4,
        skirt_octaves=0.25,
        equalize_rms=True,
):
    """Select the fine spectral detail inside ``band`` (the Trapeau peak-VSI
    octave by default) and translate it along the ERB-number axis by
    ``shift_erb`` ERB, with the coarse envelope held in place.

    The selection window is applied to the SOURCE detail and shifts WITH it, so
    the band is never cut short at its (moving) edges — the whole windowed band
    is relocated intact.  Outside the shifted band the spectrum is the smooth
    envelope.  Pass ``band=None`` to shift the whole-spectrum detail (no
    selection).

    Cepstral split (Kulkarni & Colburn 1998, Nature 396:747):

    1. ``envelope = _smooth(|H|, n_keep=envelope_n_keep)`` — the coarse spectral
       shape (first ``envelope_n_keep`` cosine coefficients of log-magnitude).
    2. ``detail   = log|H| - envelope`` — the fine structure: the pinna peaks
       and notches that carry the vertical-localisation cue.
    3. Select the detail in ``band`` with a raised-cosine window (skirt
       ``skirt_octaves``), then translate it along the ERB-number axis: each
       output frequency ``f`` samples the windowed detail at the frequency whose
       ERB number is ``ERB(f) - shift_erb``.  Because the window is on the
       source it moves with the detail — the band is not cut short.  ``> 0`` up,
       ``< 0`` down; a constant ERB step preserves notch spacing on the auditory
       scale.
    4. Optionally rescale the shifted detail so its in-band energy matches the
       source per direction/ear (keeps spectral contrast / in-notch power
       constant).
    5. Recombine ``new_log_mag = envelope + detail_shifted`` (smooth envelope
       outside the shifted band).  Magnitude-only: keep the ORIGINAL phase
       (onset / ITD structure untouched; interaural cues are imposed upstream
       when the measured frontal arc is expanded across azimuth,
       record/processing.py::expand_azimuths_with_binaural_cues).

    Parameters
    ----------
    hrtf : slab.HRTF
        Input HRTF.  Not modified (a deep copy is returned).
    shift_erb : float
        Displacement of the detail along the ERB-number axis, in ERB.
        ``> 0`` shifts cues up in frequency, ``< 0`` down, ``== 0`` is a
        rebuild no-op.  A constant ERB step ≈ a constant frequency-scale factor
        above ~1 kHz: factor 1.3 ≈ 2.4 ERB, factor 1.4 ≈ 3.0 ERB.
    band : (low_hz, high_hz) or None, default (5700, 11300)
        Band whose detail is selected and shifted (default = the Trapeau et al.
        2016 peak-VSI octave).  ``None`` shifts the whole-spectrum detail.
    envelope_n_keep : int, default 4
        Cosine (Fourier) coefficients retained for the envelope (M in Kulkarni
        & Colburn 1998).  Lower → more of the spectrum counts as shiftable
        detail; higher → only the sharpest peaks/notches move.
    skirt_octaves : float, default 0.25
        Raised-cosine taper on the selection window, in octaves (ignored when
        ``band`` is None).
    equalize_rms : bool, default True
        Rescale the shifted detail so its in-band energy matches the source per
        direction/ear (removes the overall spectral-contrast / in-notch-power
        change as a confound; cf. Zonooz et al. 2019).

    Returns
    -------
    slab.HRTF  (deep copy, processed)
    """
    if envelope_n_keep < 1:
        raise ValueError(f"envelope_n_keep must be >= 1, got {envelope_n_keep}")

    out = copy.deepcopy(hrtf)

    for filt in out:
        ir_original = numpy.asarray(filt.data, dtype=float)
        if ir_original.ndim != 2 or ir_original.shape[1] != 2:
            raise ValueError("Each HRIR must have shape (n_samples, 2)")

        n_samples, _ = ir_original.shape
        fs = filt.samplerate
        freqs = numpy.fft.rfftfreq(n_samples, d=1.0 / fs)

        spec_original = numpy.fft.rfft(ir_original, axis=0)
        mag_in = numpy.abs(spec_original)

        eps = numpy.finfo(float).tiny
        log_mag_db = 20.0 * numpy.log10(numpy.maximum(mag_in, eps))

        # 1) coarse envelope via the shared cepstral smoother
        envelope_mag = _smooth(mag_in, n_keep=int(envelope_n_keep))
        envelope_db = 20.0 * numpy.log10(numpy.maximum(envelope_mag, eps))

        # 2) fine detail (high-quefrency residual)
        detail_db = log_mag_db - envelope_db

        # 3) select the cue-bearing detail in `band` (window on the SOURCE), then
        #    translate it along the ERB axis by `shift_erb`. Because the window is
        #    applied before the shift it moves WITH the detail, so the band is not
        #    cut short at its (moving) edges. band=None -> shift the whole detail.
        if band is None:
            detail_src = detail_db
        else:
            w = band_window(freqs, band[0], band[1], skirt_octaves=skirt_octaves)
            detail_src = w[:, None] * detail_db
        src_freqs = erb_to_hz(hz_to_erb(freqs) - shift_erb)
        detail_shifted = numpy.empty_like(detail_src)
        for ch in range(detail_src.shape[1]):
            detail_shifted[:, ch] = numpy.interp(
                src_freqs, freqs, detail_src[:, ch],
                left=detail_src[0, ch], right=detail_src[-1, ch],
            )

        # 4) match the shifted detail's in-band energy to the source per
        #    direction/ear (keep spectral contrast / in-notch power constant;
        #    cf. Zonooz et al. 2019)
        if equalize_rms:
            for ch in range(detail_src.shape[1]):
                e_in = float(numpy.sum(detail_src[:, ch] ** 2))
                e_out = float(numpy.sum(detail_shifted[:, ch] ** 2))
                if e_out > 0:
                    detail_shifted[:, ch] *= numpy.sqrt(e_in / e_out)

        # 5) recombine; envelope untouched (smooth outside the shifted band)
        new_log_mag = envelope_db + detail_shifted
        mag_out = 10.0 ** (new_log_mag / 20.0)

        # magnitude-only edit: keep the ORIGINAL phase (onset / ITD structure
        # untouched here; interaural cues imposed upstream in the azimuth
        # expansion, record/processing.py::expand_azimuths_with_binaural_cues).
        spec_processed = mag_out * numpy.exp(1j * numpy.angle(spec_original))
        ir_processed = numpy.fft.irfft(spec_processed, n=n_samples, axis=0)

        filt.data = ir_processed

    return out


def _build_tf_image(hrtf, sourceidx, ear, n_bins, xlim, floor_db=-25):
    """
    Build the image array used by plot_tf's 'image' mode, using
    tfs_from_sources to obtain the raw dB data.

    Returns
    -------
    freqs      : 1-D array, frequency axis trimmed to xlim[1]
    elevations : 1-D array, elevation for each source
    img        : 2-D array (n_freq_bins, n_sources), clipped at floor_db
    """
    chan = {'left': 0, 'right': 1}[ear]
    n_b = n_bins if n_bins is not None else hrtf[sourceidx[0]].n_taps
    # tfs_from_sources returns (n_sources, n_bins, 1) — squeeze to (n_sources, n_bins)
    tfs = hrtf.tfs_from_sources(sourceidx, n_bins=n_b, ear=ear)
    img = numpy.clip(tfs.squeeze(-1).T, floor_db, None)   # (n_bins, n_sources)
    freqs, _ = hrtf[sourceidx[0]].tf(channels=chan, n_bins=n_bins, show=False)
    elevations = hrtf.sources.vertical_polar[sourceidx, 1]
    mask = freqs <= xlim[1]
    return freqs[mask], elevations, img[mask, :]


def plot(hrtf, hrtf_modified, kind='image', ear='left', n_bins=None, xlim=(1000, 18000),
         vsi_orig=None, vsi_mod=None, vsi_dis=None, vsi_bw=None):
    sources = hrtf.cone_sources(0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    if kind == 'image':
        # Build raw dB image data for both HRTFs via tfs_from_sources
        freqs, elevations, img_orig = _build_tf_image(hrtf,         sources, ear, n_bins, xlim)
        _,     _,          img_mod  = _build_tf_image(hrtf_modified, sources, ear, n_bins, xlim)

        # Joint colorbar limits across both images
        vmin   = float(min(img_orig.min(), img_mod.min()))
        vmax   = float(max(img_orig.max(), img_mod.max()))
        levels = numpy.linspace(vmin, vmax, 21)

        ct = None
        for ax, img, title, vsi_val in zip(
                axes,
                [img_orig, img_mod],
                ['original', 'modified'],
                [vsi_orig,  vsi_mod]):
            ct = ax.contourf(freqs, elevations, img.T, cmap='hot', levels=levels)
            ax.set_title(title)
            # VSI value as a second line below the x-axis label
            xlabel = 'Frequency [kHz]'
            if vsi_val is not None:
                xlabel += f'\nVSI = {vsi_val:.3f}'
            ax.set(xlabel=xlabel, ylabel='Elevation [°]', xlim=xlim)
            ax.xaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, pos: str(int(x / 1000))))
            ax.autoscale(tight=True)
            ax.tick_params('both', length=2, pad=2)

        # Single shared colorbar to the right of both subplots
        # fig.colorbar(ct, ax=list(axes), location='right', shrink=1, pad=0.02)

        cbar_ticks = numpy.arange(vmin, vmax, 6)
        cax_pos = list(axes[-1].get_position().bounds)  # (x0, y0, width, height)
        cax_pos[2] = cax_pos[2] * 0.06  # cbar width in fractions of axis width
        cax_pos[0] = 0.92
        cbar_axis = fig.add_axes(cax_pos)
        cbar = fig.colorbar(ct, cbar_axis, orientation='vertical', ticks=cbar_ticks)

        # VSI dissimilarity as a footer below both plots
        if vsi_dis is not None:
            bw_str = (f'{vsi_bw[0]/1000:.1f}–{vsi_bw[1]/1000:.1f} kHz'
                      if vsi_bw is not None else '')
            fig.text(0.5, 0.0,
                     f'VSI dissimilarity = {vsi_dis:.3f}   ({bw_str}, Trapeau et al. 2016)',
                     ha='center', va='bottom', fontsize=9)
            # plt.tight_layout(rect=[0, 0.07, 1, 1])
        else:
            plt.tight_layout()

    else:
        # waterfall / surface: fall back to plot_tf (no shared scale needed)
        hrtf.plot_tf(         sources, kind=kind, axis=axes[0], ear=ear, xlim=xlim, show=False)
        hrtf_modified.plot_tf(sources, kind=kind, axis=axes[1], ear=ear, xlim=xlim, show=False)
        axes[0].set_title('original')
        axes[1].set_title('modified')
        plt.tight_layout()

    plt.show(block=False)
    plt.pause(0.1)
    return fig


def plot_split_qc(hrtf, envelope_n_keep, ear='right', xlim=(2000, 18000), band=None):
    """QC for the coarse/fine split (constraint c).

    For each median-plane elevation, overlay the full log-magnitude (thin grey)
    and the truncated-cosine envelope (thick red) that shift_band holds fixed.
    The envelope should be smooth AND roughly elevation-invariant: if the red
    curves still track elevation, the split is freezing a cue that also carries
    elevation information (a cue conflict), not cleanly separating macro shape
    from the fine structure being shifted.  Curves are mean-removed and stacked.
    """
    chan = {'left': 0, 'right': 1}[ear]
    sources = hrtf.cone_sources(0)
    elevations = hrtf.sources.vertical_polar[sources, 1]
    order = numpy.argsort(elevations)
    sources_sorted = numpy.array(sources)[order]

    fig, ax = plt.subplots(figsize=(6, 8))
    offset = 25.0  # dB between stacked elevations
    for row, idx in enumerate(sources_sorted):
        ir = numpy.asarray(hrtf[idx].data, dtype=float)
        n = ir.shape[0]
        freqs = numpy.fft.rfftfreq(n, d=1.0 / hrtf[idx].samplerate)
        mag = numpy.abs(numpy.fft.rfft(ir, axis=0))
        env = _smooth(mag, n_keep=envelope_n_keep)
        eps = numpy.finfo(float).tiny
        full_db = 20.0 * numpy.log10(numpy.maximum(mag[:, chan], eps))
        env_db = 20.0 * numpy.log10(numpy.maximum(env[:, chan], eps))
        m = (freqs >= xlim[0]) & (freqs <= xlim[1])
        y0 = row * offset
        ax.plot(freqs[m], full_db[m] - full_db[m].mean() + y0, lw=0.6, color='0.6')
        ax.plot(freqs[m], env_db[m] - env_db[m].mean() + y0, lw=1.8, color='C3')
        ax.text(xlim[1], y0, f'{elevations[order][row]:.0f}°', va='center', fontsize=7)

    if band is not None:
        ax.axvspan(band[0], band[1], color='C0', alpha=0.08, lw=0)
    ax.set(xlabel='Frequency [Hz]', ylabel='elevation (stacked, mean-removed dB)',
           xlim=xlim, title=f'split QC: envelope (M={envelope_n_keep}) vs full — {ear} ear')
    ax.set_yticks([])
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
    return fig


if __name__ == '__main__':
    if MODE not in ('synth', 'shift'):
        raise ValueError(f"MODE must be 'synth' or 'shift', got {MODE!r}")

    hrtf = slab.HRTF(hrtf_dir / sub_id / str(sub_id + '.sofa'))

    if MODE == 'shift':
        print(f"shift_detail: band={SHIFT_BAND} Hz, shift={SHIFT_ERB} ERB")
        hrtf_modified = shift_detail(
            hrtf,
            shift_erb=SHIFT_ERB,
            band=SHIFT_BAND,
            envelope_n_keep=SHIFT_ENV_NKEEP,
            skirt_octaves=SHIFT_SKIRT,
            equalize_rms=SHIFT_EQ_RMS,
        )
    else:  # 'synth'
        hrtf_modified = smooth_and_replace_hrtf(
            hrtf,
            n_keep=N_KEEP,
            smooth=SMOOTH,
            features=FEATURES,
            ref_n_keep=4,
            onset_threshold_db=15.0,
        )

    # VSI metrics in the 5.7–11.3 kHz band (peak VSI band, Trapeau et al. 2016)
    VSI_BW = (5700, 11300)
    vsi_orig = _vsi(hrtf,          bandwidth=VSI_BW)
    vsi_mod  = _vsi(hrtf_modified, bandwidth=VSI_BW)
    vsi_dis  = _vsi_dissimilarity(hrtf, hrtf_modified, bandwidth=VSI_BW)

    fig = plot(hrtf, hrtf_modified, PLOT, ear='right',
               vsi_orig=vsi_orig, vsi_mod=vsi_mod, vsi_dis=vsi_dis, vsi_bw=VSI_BW)
    qc_fig = (plot_split_qc(hrtf, SHIFT_ENV_NKEEP, ear='right', band=SHIFT_BAND)
              if MODE == 'shift' else None)
    input('press enter to save')
    fig.savefig(paths.subject_plot_dir(sub_id) / str(sub_id + f'_{fname}.png'),
                bbox_inches='tight')
    if qc_fig is not None:
        qc_fig.savefig(paths.subject_plot_dir(sub_id) / str(sub_id + f'_{fname}_split_qc.png'),
                       bbox_inches='tight')
    (hrtf_dir / sub_id).mkdir(parents=True, exist_ok=True)
    hrtf_modified.write_sofa(hrtf_dir / sub_id / str(sub_id + f'_{fname}.sofa'))
