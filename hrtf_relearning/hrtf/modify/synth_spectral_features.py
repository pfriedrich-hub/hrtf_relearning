"""
synth_spectral_features.py — replace a subject's elevation cues with synthetic ones.

The counterpart to shift_spectral_detail.py. Where that manipulation moves the
subject's OWN pinna cues to new frequencies, this one erases them and writes in
cues we designed:

  1. SMOOTH the magnitude spectrum with a truncated cosine series on
     log-magnitude (Kulkarni & Colburn 1998, Nature 396:747), keeping N_KEEP
     coefficients. High N_KEEP keeps some native fine structure; low N_KEEP
     wipes it, leaving only the broad shape.
  2. ADD synthetic Gaussian notches/peaks whose centre frequency, width and
     depth are interpolated linearly across direction, so each elevation gets a
     different but systematic pattern — a cue with a known, exactly specified
     mapping.
  3. RESYNTHESISE as minimum phase, then put the original ITD back from the
     measured onsets.

Each feature is a crossfade toward a target rather than a multiplicative dB gain::

    w(f)       = exp(-0.5 * ((f - mu) / sigma) ** 2)
    target(f)  = mag_ref(f) * 10 ** (-depth_db / 20)
    mag_out(f) = (1 - w(f)) * mag_in(f) + w(f) * target(f)

``mag_ref`` is a heavily smoothed reference (REF_N_KEEP), so a feature defines
what the spectral TREND should be at those bins instead of compounding with fine
structure that is already there.

NOTE on phase: unlike shift_spectral_detail this path rebuilds the phase
(minimum-phase reconstruction), so the measured phase is NOT preserved. The ITD
is restored afterwards from the original onsets — check the result if the input
HRTF carries binaural cues you care about.

Run this file directly to apply it to one subject's SOFA: set SUB_ID and the
FEATURES list below, run, eyeball the before/after image, then press enter to
write <SUB_ID>_synth.sofa next to the native one.
"""

import copy
import numpy
from sklearn.linear_model import LinearRegression

from hrtf_relearning.hrtf.modify.shift_spectral_detail import smooth_magnitude

# ---------------------------------------------------------------------------
# Parameters — edit these, then run this file
# ---------------------------------------------------------------------------
SUB_ID     = 'CO'            # subject with a measured <id>.sofa in data/hrtf/sofa/<id>/
OUT_SUFFIX = 'synth'         # writes <SUB_ID>_<OUT_SUFFIX>.sofa

SMOOTH     = True            # smooth before inserting features (erases native cues)
N_KEEP     = 4               # cosine coeffs kept by the main smoothing (M)
REF_N_KEEP = 4               # coeffs for the reference the features are anchored to
ONSET_DB   = 15.0            # threshold for onset detection / ITD restoration

PLOT_KIND  = 'image'         # 'image' (before/after heatmap) | 'waterfall' | 'surface'
EAR        = 'right'         # ear shown in the QC plot
VSI_BW     = (5700, 11300)   # band for the VSI / VSI-dissimilarity readout

# ---------------------------------------------------------------------------
# Synthetic features
# ---------------------------------------------------------------------------
# One entry per feature. Parameters are interpolated linearly between two
# spatial anchor points X1 and X2 (azimuth, elevation) in degrees, so a feature
# sweeps in frequency/width/depth across the field.
#
#   freqs : (f_at_X1, f_at_X2)   centre frequency [Hz]
#   width : (w_at_X1, w_at_X2)   Gaussian sigma [Hz]
#   depth : (d_at_X1, d_at_X2)   magnitude [dB]; > 0 = notch, < 0 = peak
#   X1, X2 : (azimuth, elevation) anchor directions [deg]
FEATURES = [
    {
        'freqs': (6000, 9000),   # centre freq at X1 and X2 [Hz]
        'width': (300,   300),   # Gaussian sigma at X1 and X2 [Hz]
        'depth': (12.0,  12.0),  # >0 = notch, <0 = peak [dB]
        'X1':    (0, 0),         # anchor 1 (az, el)
        'X2':    (-40, 40),      # anchor 2 (az, el)
    },
    {
        'freqs': (11000, 10000),  # centre freq at X1 and X2 [Hz]
        'width': (300, 300),  # Gaussian sigma at X1 and X2 [Hz]
        'depth': (12.0, 12.0),  # >0 = notch, <0 = peak [dB]
        'X1': (0, 0),  # anchor 1 (az, el)
        'X2': (-40, 40),  # anchor 2 (az, el)
    },
    # add further features here, e.g.
    # {'freqs': (11000, 10000), 'width': (300, 300), 'depth': (12, 12),
    #  'X1': (0, 0), 'X2': (-40, 40)},
]


# ---------------------------------------------------------------------------
# Direction interpolation of feature parameters
# ---------------------------------------------------------------------------

def _interpolate(azimuth, elevation, X1, X2, Y):
    """Value of a feature parameter at (az, el), linear in the two anchors."""
    model = LinearRegression()
    model.fit(numpy.column_stack((X1, X2)), numpy.array(Y))
    return float(model.predict(numpy.array((azimuth, elevation)).reshape(1, -1))[0])


def feature_params(azimuth, elevation, feature):
    """Interpolated (mu, sigma, depth_db) for one feature at one direction.

    ``depth_db`` keeps its sign: > 0 is a notch, < 0 a peak.
    """
    X1, X2 = feature['X1'], feature['X2']
    sigma = _interpolate(azimuth, elevation, X1, X2, feature['width'])
    return {
        'mu':       _interpolate(azimuth, elevation, X1, X2, feature['freqs']),
        'sigma':    max(sigma, float(numpy.finfo(float).eps)),
        'depth_db': _interpolate(azimuth, elevation, X1, X2, feature['depth']),
    }


# ---------------------------------------------------------------------------
# Minimum phase and ITD restoration
# ---------------------------------------------------------------------------

def minimum_phase_from_magnitude(mag):
    """Real-cepstrum minimum-phase reconstruction from a one-sided magnitude."""
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
    """First sample within ``threshold_db`` of the peak, per channel."""
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
    """Shift each channel so its onset lands where the measured one did.

    The minimum-phase rebuild discards the measured delay, so without this the
    ITD would collapse.
    """
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


# ---------------------------------------------------------------------------
# The manipulation
# ---------------------------------------------------------------------------

def synth_spectral_features(hrtf, features=None, n_keep=4, smooth=True,
                            ref_n_keep=4, onset_threshold_db=15.0):
    """Smooth away the native fine structure and write in synthetic features.

    Parameters
    ----------
    hrtf : slab.HRTF
        Input HRTF. Not modified (a deep copy is returned).
    features : list of dict or None
        Feature definitions, format as in FEATURES above. ``None`` or ``[]``
        applies the smoothing only — useful as a cue-free control.
    n_keep : int, default 4
        Cosine coefficients kept by the main smoothing step.
    smooth : bool, default True
        Smooth before inserting features. ``False`` writes the features on top
        of the native fine structure instead of replacing it.
    ref_n_keep : int, default 4
        Coefficients for the heavily-smoothed reference the features are
        anchored to. Fewer -> the target level follows the broad trend rather
        than any peaks or dips already present.
    onset_threshold_db : float, default 15.0
        Threshold for the onset-based ITD restoration.

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

        n_samples = ir_original.shape[0]
        freqs = numpy.fft.rfftfreq(n_samples, d=1.0 / filt.samplerate)

        spec_original = numpy.fft.rfft(ir_original, axis=0)
        mag_original  = numpy.abs(spec_original)

        # 1) smoothing — this is what erases the subject's own cues
        mag_processed = (smooth_magnitude(mag_original, n_keep=n_keep) if smooth
                         else mag_original.copy())

        # 2) synthetic features, blended toward a clean target
        if features:
            mag_ref = smooth_magnitude(mag_processed, n_keep=ref_n_keep)

            for feat in features:
                p = feature_params(azimuth, elevation, feat)
                mu, sigma, depth_db = p['mu'], p['sigma'], p['depth_db']

                # Gaussian blending weight: 1 at centre, 0 in the tails
                w = numpy.exp(-0.5 * ((freqs - mu) / sigma) ** 2)

                # target: reference level shifted by depth_db
                # depth_db > 0 -> pull below reference (notch)
                # depth_db < 0 -> push above reference (peak)
                target = mag_ref * (10.0 ** (-depth_db / 20.0))

                # leave DC and Nyquist untouched
                w[0] = 0.0
                w[-1] = 0.0

                mag_processed = ((1.0 - w[:, None]) * mag_processed
                                 + w[:, None] * target)

            mag_processed = numpy.clip(mag_processed,
                                       10.0 ** (-80.0 / 20.0),
                                       10.0 ** (80.0 / 20.0))

        # 3) minimum-phase resynthesis, then put the measured ITD back
        spec_processed = minimum_phase_from_magnitude(mag_processed)
        ir_processed = numpy.fft.irfft(spec_processed, n=n_samples, axis=0)
        filt.data = restore_itd_from_onsets(ir_original, ir_processed,
                                            threshold_db=onset_threshold_db)

    return out


def describe(features, n_keep, smooth):
    """Print what the current configuration will do."""
    lines = [f"synth_spectral_features: smooth={smooth} (M={n_keep}), "
             f"{len(features)} synthetic feature(s)"]
    for i, feat in enumerate(features):
        kind = 'notch' if feat['depth'][0] > 0 else 'peak'
        lines.append(
            f"  [{i}] {kind}: {feat['freqs'][0]:.0f}->{feat['freqs'][1]:.0f} Hz, "
            f"sigma {feat['width'][0]:.0f}->{feat['width'][1]:.0f} Hz, "
            f"{abs(feat['depth'][0]):.1f}->{abs(feat['depth'][1]):.1f} dB "
            f"between {feat['X1']} and {feat['X2']} deg")
    if not features:
        lines.append("  no features — smoothing only (cue-free control)")
    text = "\n".join(lines)
    print(text)
    return text


# ---------------------------------------------------------------------------
# Run on one subject
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import slab
    from hrtf_relearning.utils import paths
    from hrtf_relearning.hrtf.modify.plot_compare import plot
    from hrtf_relearning.hrtf.analysis.vsi import (
        vsi as _vsi, vsi_dissimilarity as _vsi_dissimilarity,
    )

    sofa_dir = paths.SOFA_DIR / SUB_ID
    native_path = sofa_dir / f'{SUB_ID}.sofa'
    if not native_path.exists():
        raise FileNotFoundError(
            f'no SOFA at {native_path} — check SUB_ID in the config block')

    hrtf = slab.HRTF(str(native_path))
    hrtf.name = SUB_ID
    print(f'loaded {native_path.name}')

    describe(FEATURES, N_KEEP, SMOOTH)
    hrtf_synth = synth_spectral_features(
        hrtf,
        features=FEATURES,
        n_keep=N_KEEP,
        smooth=SMOOTH,
        ref_n_keep=REF_N_KEEP,
        onset_threshold_db=ONSET_DB,
    )

    vsi_o = _vsi(hrtf,       bandwidth=VSI_BW)
    vsi_m = _vsi(hrtf_synth, bandwidth=VSI_BW)
    vsi_d = _vsi_dissimilarity(hrtf, hrtf_synth, bandwidth=VSI_BW)
    print(f'VSI  native={vsi_o:.3f}  synth={vsi_m:.3f}  dissimilarity={vsi_d:.3f}')

    fig = plot(hrtf, hrtf_synth, PLOT_KIND, ear=EAR,
               vsi_orig=vsi_o, vsi_mod=vsi_m, vsi_dis=vsi_d, vsi_bw=VSI_BW)

    stem = f'{SUB_ID}_{OUT_SUFFIX}'
    print(f'about to write {stem}.sofa')

    input('press enter to save (ctrl-c to discard)')
    plot_dir = paths.subject_plot_dir(SUB_ID)
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / f'{stem}.png', bbox_inches='tight')
    sofa_dir.mkdir(parents=True, exist_ok=True)
    out_path = sofa_dir / f'{stem}.sofa'
    hrtf_synth.write_sofa(str(out_path))
    print(f'wrote {out_path}')
