import copy
import numpy
import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from hrtf_relearning import PATH
hrtf_dir = PATH / 'data' /'hrtf'/'sofa'
import slab

sub_id = 'VD'

SMOOTH = True
N_KEEP = 8 # all subjects reported complete externalization of the virtual sound image

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
#     > 0  →  notch  (spectral attenuation)
#     < 0  →  peak   (spectral boost)
# X1, X2 : (azimuth, elevation) spatial anchor directions
# ---------------------------------------------------------------------------
FEATURES = [
    {
        'freqs': (6000, 8500),  # centre freq at X1 and X2 [Hz]
        'width': (400,   400),   # Gaussian σ at X1 and X2 [Hz]
        'depth': (12.0,  12.0), # >0 = notch, <0 = peak [dB]
        'X1':    (0, 0),         # anchor 1 (az, az)
        'X2':    (-40, 40),      # anchor 2 (el, el)
    },
    # Add further features here, e.g.:
    {
        'freqs': (9500, 11000),
        'width': (300, 300),
        'depth': (12, 12),   # negative → peak
        'X1': (0, 0),
        'X2': (40, -40),
    },
]

def _smooth(mag, n_keep):
    """
    Smooth a one-sided HRTF magnitude spectrum by truncating the cosine-series
    representation of its log-magnitude spectrum.

    This implements the smoothing idea described in Kulkarni & Colburn (1998):
    the log-magnitude spectrum is expressed as a cosine / Fourier series and
    reconstructed using only the first M coefficients. Exact reconstruction
    corresponds to the full set of coefficients; stronger smoothing uses fewer
    coefficients.

    Parameters
    ----------
    mag : numpy.ndarray
        One-sided magnitude spectrum with shape (n_bins, n_channels), typically
        from numpy.fft.rfft(...). Must be strictly non-negative.
    n_keep : int
        Number of cosine coefficients to retain, counting from the DC term C(0).
        Thus:
            - n_keep = full number of coefficients -> exact reconstruction
            - n_keep = 1 -> flat log-magnitude at its average value
        This interpretation matches the paper's description that the most extreme
        smoothing retains only the average spectral shape. :contentReference[oaicite:3]{index=3}

    Returns
    -------
    numpy.ndarray
        Smoothed one-sided magnitude spectrum with the same shape as `mag`.

    Notes
    -----
    The paper expresses the smoothing in terms of
        log|H[k]| = sum_n C(n) cos(2*pi*n*k/N)
    and reconstructs using a truncated series. Here this is implemented directly
    by solving for cosine-series coefficients on the one-sided frequency grid
    k = 0 ... N/2, where N is the original HRIR length. :contentReference[oaicite:4]{index=4}
    """
    mag = numpy.asarray(mag, dtype=float)
    if mag.ndim != 2:
        raise ValueError("mag must have shape (n_bins, n_channels)")

    n_bins, n_channels = mag.shape
    if n_bins < 2:
        raise ValueError("mag must contain at least DC and one more frequency bin")

    n_samples = 2 * (n_bins - 1)  # because n_bins = N/2 + 1 for rfft of length N
    max_coeffs = n_bins
    n_keep = int(n_keep)

    if n_keep < 1 or n_keep > max_coeffs:
        raise ValueError(f"n_keep must be between 1 and {max_coeffs}, got {n_keep}")

    log_mag = numpy.log(numpy.maximum(mag, numpy.finfo(float).tiny))

    # Cosine basis matching the paper's formula:
    # log|H[k]| = sum_n C(n) cos(2*pi*n*k/N), k=0..N/2
    k = numpy.arange(n_bins, dtype=float)[:, None]  # (n_bins, 1)
    n = numpy.arange(n_bins, dtype=float)[None, :]  # (1, n_bins)
    basis = numpy.cos(2.0 * numpy.pi * k * n / float(n_samples))  # (n_bins, n_bins)

    log_mag_smooth = numpy.empty_like(log_mag)

    for ch in range(n_channels):
        coeffs, _, _, _ = numpy.linalg.lstsq(basis, log_mag[:, ch], rcond=None)
        coeffs[n_keep:] = 0.0
        log_mag_smooth[:, ch] = basis @ coeffs

    mag_smooth = numpy.exp(log_mag_smooth)
    return mag_smooth


def _compute_feature_params(azimuth, elevation, feature):
    """
    Return interpolated spectral-feature parameters for a given source direction.

    Parameters
    ----------
    azimuth, elevation : float
        Source direction in degrees.
    feature : dict
        Feature specification with keys:
            freqs : (f_at_X1, f_at_X2)  centre frequency [Hz]
            width : (w_at_X1, w_at_X2)  Gaussian σ [Hz]
            depth : (d_at_X1, d_at_X2)  signed depth [dB]
                positive → notch (attenuation)
                negative → peak  (boost)
            X1, X2 : (azimuth, elevation)  spatial anchor directions

    Returns
    -------
    dict with keys ``mu``, ``sigma``, ``depth_db`` (signed).
    """
    X1 = feature['X1']
    X2 = feature['X2']

    mu = linear_notch_position(azimuth, elevation, X1=X1, X2=X2, Y=feature['freqs'])
    sigma = linear_notch_width(azimuth, elevation, X1=X1, X2=X2, Y=feature['width'])
    # preserve sign: positive = notch, negative = peak
    depth_db = linear_scaling_factor(azimuth, elevation, X1=X1, X2=X2, Y=feature['depth'])

    return {
        "mu":       float(mu),
        "sigma":    float(max(sigma, numpy.finfo(float).eps)),
        "depth_db": float(depth_db),
    }

def minimum_phase_from_magnitude(mag):
    """
    Reconstruct a one-sided minimum-phase HRTF spectrum from a one-sided
    magnitude spectrum.

    Parameters
    ----------
    mag : numpy.ndarray
        One-sided magnitude spectrum with shape (n_bins, n_channels),
        corresponding to an rFFT of an even-length HRIR.

    Returns
    -------
    numpy.ndarray
        One-sided complex minimum-phase spectrum with shape
        (n_bins, n_channels).

    Notes
    -----
    The paper states that modified HRTFs were implemented as
    "minimum-phase filters, augmented with a frequency-independent time delay"
    consistent with the original overall ITD.

    This function performs the minimum-phase part only. ITD restoration is done
    later in `restore_itd_from_onsets(...)`.

    Method
    ------
    A standard real-cepstrum minimum-phase reconstruction is used:
      1. build a full even-symmetric log-magnitude spectrum
      2. take the IFFT to obtain the real cepstrum
      3. zero the anti-causal cepstral part and double the causal part
      4. exponentiate the resulting complex log spectrum
    """
    mag = numpy.asarray(mag, dtype=float)
    if mag.ndim != 2:
        raise ValueError("mag must have shape (n_bins, n_channels)")

    n_bins, n_channels = mag.shape
    if n_bins < 2:
        raise ValueError("mag must contain at least DC and one more frequency bin")

    n_samples = 2 * (n_bins - 1)
    tiny = numpy.finfo(float).tiny
    spec_min = numpy.empty((n_bins, n_channels), dtype=complex)

    for ch in range(n_channels):
        mag_ch = numpy.maximum(mag[:, ch], tiny)
        log_mag_half = numpy.log(mag_ch)

        # Build full, even-symmetric log-magnitude spectrum of length N.
        # For even N:
        # [0 ... N/2, N/2-1 ... 1]
        log_mag_full = numpy.concatenate(
            (log_mag_half, log_mag_half[-2:0:-1]),
            axis=0,
        )

        cep = numpy.fft.ifft(log_mag_full).real

        cep_min = numpy.zeros_like(cep)
        cep_min[0] = cep[0]

        # Keep causal cepstrum and double positive quefrencies
        cep_min[1:n_samples // 2] = 2.0 * cep[1:n_samples // 2]

        # Nyquist term stays unchanged for even-length transforms
        cep_min[n_samples // 2] = cep[n_samples // 2]

        log_spec_min = numpy.fft.fft(cep_min)
        spec_full_min = numpy.exp(log_spec_min)

        spec_min[:, ch] = spec_full_min[:n_bins]

    return spec_min


def find_ir_onsets(ir, threshold_db=15.0):
    """
    Estimate per-channel HRIR onset samples from a time-domain impulse response.

    Parameters
    ----------
    ir : numpy.ndarray
        Time-domain HRIR with shape (n_samples, n_channels).
    threshold_db : float, default=15.0
        Threshold below the channel peak, in dB. The onset is defined as the
        first sample up to the peak whose absolute amplitude exceeds
        peak / 10^(threshold_db / 20).

    Returns
    -------
    numpy.ndarray
        Integer onset sample for each channel, shape (n_channels,).

    Notes
    -----
    This is a lightweight NumPy replacement for pyfar onset detection because
    the slab Filter objects stored in the HRTF are not directly compatible with
    pyfar's find_impulse_response_start function.

    The paper preserves overall ITD by supplementing the modified minimum-phase
    HRTFs with a frequency-independent delay. Here that delay is estimated from
    binaural onset differences in the original and processed HRIRs.
    """
    ir = numpy.asarray(ir, dtype=float)
    if ir.ndim != 2:
        raise ValueError("ir must have shape (n_samples, n_channels)")

    n_samples, n_channels = ir.shape
    onsets = numpy.zeros(n_channels, dtype=int)

    for ch in range(n_channels):
        x = numpy.abs(ir[:, ch])
        peak_idx = int(numpy.argmax(x))
        peak_val = float(x[peak_idx])

        if peak_val <= 0:
            onsets[ch] = 0
            continue

        threshold = peak_val / (10.0 ** (float(threshold_db) / 20.0))
        above = numpy.where(x[:peak_idx + 1] >= threshold)[0]

        if len(above) == 0:
            onsets[ch] = 0
        else:
            onsets[ch] = int(above[0])

    return onsets

def restore_itd_from_onsets(ir_original, ir_processed, threshold_db=15.0):
    """
    Restore original per-ear onset positions by shifting each processed ear
    independently so that its onset matches the corresponding original ear onset.

    Parameters
    ----------
    ir_original : numpy.ndarray
        Original HRIR, shape (n_samples, 2).
    ir_processed : numpy.ndarray
        Processed HRIR, shape (n_samples, 2).
    threshold_db : float, default=15.0
        Threshold used in `find_ir_onsets(...)`.

    Returns
    -------
    numpy.ndarray
        Processed HRIR with restored per-ear onsets, shape (n_samples, 2).

    Notes
    -----
    This restores both the common delay and the ITD, because both ears are
    aligned to their original onset positions independently.
    """
    ir_original = numpy.asarray(ir_original, dtype=float)
    ir_processed = numpy.asarray(ir_processed, dtype=float)

    if ir_original.shape != ir_processed.shape:
        raise ValueError("ir_original and ir_processed must have the same shape")
    if ir_original.ndim != 2 or ir_original.shape[1] != 2:
        raise ValueError("Both HRIRs must have shape (n_samples, 2)")

    n_samples, n_channels = ir_original.shape
    out = numpy.zeros_like(ir_processed)

    on_orig = find_ir_onsets(ir_original, threshold_db=threshold_db)
    on_proc = find_ir_onsets(ir_processed, threshold_db=threshold_db)

    for ch in range(n_channels):
        delta = int(on_orig[ch] - on_proc[ch])

        if delta > 0:
            # delay channel
            if delta >= n_samples:
                out[:, ch] = 0.0
            else:
                out[:, ch] = numpy.concatenate(
                    (numpy.zeros(delta, dtype=ir_processed.dtype), ir_processed[:-delta, ch]),
                    axis=0,
                )
        elif delta < 0:
            # advance channel
            d = -delta
            if d >= n_samples:
                out[:, ch] = 0.0
            else:
                out[:, ch] = numpy.concatenate(
                    (ir_processed[d:, ch], numpy.zeros(d, dtype=ir_processed.dtype)),
                    axis=0,
                )
        else:
            out[:, ch] = ir_processed[:, ch]

    return out


def linear_notch_position(azimuth, elevation, X1, X2, Y):
    X = numpy.column_stack((X1, X2))
    model = LinearRegression()
    model.fit(X, numpy.array(Y))
    mu = model.predict(numpy.array((azimuth, elevation)).reshape(1, -1))[0]
    return float(mu)

def linear_notch_width(azimuth, elevation, X1, X2, Y):
    X = numpy.column_stack((X1, X2))
    model = LinearRegression()
    model.fit(X, numpy.array(Y))
    sigma = model.predict(numpy.array((azimuth, elevation)).reshape(1, -1))[0]
    return float(max(sigma, numpy.finfo(float).eps))

def linear_scaling_factor(azimuth, elevation, X1, X2, Y):
    X = numpy.column_stack((X1, X2))
    model = LinearRegression()
    model.fit(X, numpy.array(Y))
    scaling = model.predict(numpy.array((azimuth, elevation)).reshape(1, -1))[0]
    return float(scaling)

def smooth_and_notch_hrtf(
        hrtf,
        n_keep=12,
        smooth=True,
        features=None,
        onset_threshold_db=15.0,
):
    """
    Apply paper-style HRTF spectral smoothing, optional artificial spectral
    features (notches and/or peaks), minimum-phase reconstruction, and
    onset-based ITD restoration to a slab.HRTF object.

    Parameters
    ----------
    hrtf : slab.HRTF
        Input HRTF object. Each element is assumed to contain a binaural HRIR
        in `filt.data` with shape (n_samples, 2).
    n_keep : int
        Number of cosine coefficients retained in the log-magnitude smoothing
        step. This is the smoothing parameter analogous to the paper's M,
        interpreted here as the number of retained coefficients including C(0).
    smooth : bool, default=True
        If True, apply Kulkarni & Colburn (1998) log-magnitude smoothing.
    features : list of dict, optional
        Spectral features to add. Each dict must contain:
            freqs : (f_at_X1, f_at_X2)  Gaussian centre frequency [Hz]
            width : (w_at_X1, w_at_X2)  Gaussian σ [Hz]
            depth : (d_at_X1, d_at_X2)  signed depth [dB]
                positive → notch (attenuation)
                negative → peak  (boost)
            X1, X2 : (azimuth, elevation)  spatial anchor directions
        Parameters are linearly interpolated between X1 and X2 for every
        source direction. Multiple features are accumulated in the dB domain
        before converting to a linear gain, so they interact additively in dB.
        Pass an empty list (or None) to skip this step.
    onset_threshold_db : float, default=15.0
        Threshold for onset detection when restoring ITD.

    Returns
    -------
    slab.HRTF
        Deep-copied and processed HRTF object.

    Processing steps
    ----------------
    For each HRIR:
      1. Transform to one-sided frequency domain via rFFT
      2. Smooth the one-sided magnitude spectrum using truncated cosine-series
         reconstruction of log-magnitude, following Kulkarni & Colburn (1998)
      3. Accumulate all spectral features in the dB domain and apply as a
         single linear gain vector
      4. Reconstruct a minimum-phase spectrum from the final magnitude
      5. Transform back to the time domain via irFFT
      6. Restore the original onset-based ITD by a pure right-ear shift

    Notes
    -----
    Smoothing follows the paper-style approach: log-magnitude is expressed as
    a truncated cosine/Fourier series and the result is implemented as a
    minimum-phase filter with the original ITD restored as a frequency-
    independent delay.
    """
    if features is None:
        features = []

    out = copy.deepcopy(hrtf)

    for filt, source in zip(out, out.sources.vertical_polar):
        azimuth, elevation = float(source[0]), float(source[1])

        ir_original = numpy.asarray(filt.data, dtype=float)
        if ir_original.ndim != 2 or ir_original.shape[1] != 2:
            raise ValueError(
                "Each HRIR must have shape (n_samples, 2) for binaural processing"
            )

        n_samples, n_channels = ir_original.shape
        fs = filt.samplerate

        freqs = numpy.fft.rfftfreq(n_samples, d=1.0 / fs)
        spec_original = numpy.fft.rfft(ir_original, axis=0)
        mag_original = numpy.abs(spec_original)

        # 1) Paper-style smoothing in log-magnitude / cosine-series domain
        if smooth:
            mag_processed = _smooth(mag_original, n_keep=n_keep)
        else:
            mag_processed = mag_original

        # 2) Accumulate spectral features in the dB domain, then apply once
        if features:
            combined_db = numpy.zeros(len(freqs))
            for feat in features:
                params = _compute_feature_params(azimuth, elevation, feat)
                # positive depth_db → attenuation (notch); negative → boost (peak)
                gaussian = numpy.exp(
                    -0.5 * ((freqs - params["mu"]) / params["sigma"]) ** 2
                )
                combined_db += -params["depth_db"] * gaussian

            # leave DC and Nyquist untouched
            combined_db[0]  = 0.0
            combined_db[-1] = 0.0

            combined_lin = 10.0 ** (combined_db / 20.0)
            # guard against extreme attenuation / boost
            combined_lin = numpy.clip(
                combined_lin,
                10.0 ** (-80.0 / 20.0),
                10.0 ** ( 80.0 / 20.0),
            )
            mag_processed = mag_processed * combined_lin[:, None]

        # 3) Enforce minimum phase once, after smoothing + features
        spec_processed = minimum_phase_from_magnitude(mag_processed)

        # 4) Back to time domain
        ir_processed = numpy.fft.irfft(spec_processed, n=n_samples, axis=0)

        # 5) Restore original ITD from HRIR onsets
        ir_processed = restore_itd_from_onsets(
            ir_original,
            ir_processed,
            threshold_db=onset_threshold_db,
        )

        filt.data = ir_processed

    return out

def plot(hrtf, hrtf_modified, kind='image', ear='left', xlim=(1000, 18000), clip_db=-25, n_levels=20):
    """
    Plot original and modified HRTF transfer functions side by side with a shared colorbar.

    Uses tfs_from_sources to extract raw dB data from both HRTFs, computes the joint
    min/max across both panels (after clipping at clip_db), and passes shared contour
    levels to both contourf calls so the color scale is identical.

    Parameters
    ----------
    hrtf, hrtf_modified : slab.HRTF
    kind : str  (currently only 'image' is implemented here)
    ear : 'left' | 'right'
    xlim : (f_low, f_high) in Hz  frequency display range
    clip_db : float  lower dB clip (matches slab's internal -25 dB clip)
    n_levels : int  number of contour levels
    """
    sources = hrtf.cone_sources(0)
    elevations = hrtf.sources.vertical_polar[sources, 1]

    # Frequency axis from the first filter's tf method
    freqs = hrtf[0].tf(show=False)[0]
    freq_mask = (freqs >= xlim[0]) & (freqs <= xlim[1])
    freqs_plot = freqs[freq_mask]

    # Raw TF data: (n_sources, n_bins, 1) → (n_sources, n_bins)
    tf_orig = hrtf.tfs_from_sources(sources, n_bins=None, ear=ear)[:, :, 0]
    tf_mod  = hrtf_modified.tfs_from_sources(sources, n_bins=None, ear=ear)[:, :, 0]

    # Frequency-band selection and dB clip (matching slab's own clipping in plot_tf)
    tf_orig = numpy.clip(tf_orig[:, freq_mask], clip_db, None)
    tf_mod  = numpy.clip(tf_mod[:, freq_mask],  clip_db, None)

    # Joint colour range
    vmin = min(tf_orig.min(), tf_mod.min())
    vmax = max(tf_orig.max(), tf_mod.max())
    levels = numpy.linspace(vmin, vmax, n_levels)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    c0 = ax[0].contourf(freqs_plot, elevations, tf_orig, cmap='hot', levels=levels)
    c1 = ax[1].contourf(freqs_plot, elevations, tf_mod,  cmap='hot', levels=levels)
    ax[0].set_title('original')
    ax[1].set_title('modified')
    for a in ax:
        a.set_xlabel('Frequency (Hz)')
        a.set_ylabel('Elevation (°)')

    # Single shared colorbar
    fig.colorbar(c0, ax=ax.tolist(), orientation='vertical', label='dB')

    plt.show(block=False)
    plt.pause(0.1)
    return fig

if __name__ == '__main__':
    # modify
    hrtf = slab.HRTF(hrtf_dir / str(sub_id + '.sofa'))
    hrtf_modified = smooth_and_notch_hrtf(
        hrtf,
        n_keep=N_KEEP,
        smooth=SMOOTH,
        features=FEATURES,
        onset_threshold_db=15.0,
    )

    # plot
    fig = plot(hrtf, hrtf_modified, 'image', ear='right')
    input('press enter to save')
    fig.savefig(PATH / 'data' / 'results' / 'plot' / sub_id / str(sub_id + '_modified.png'))
    hrtf_modified.write_sofa(hrtf_dir / str(sub_id + '_notch.sofa'))
    import logging
    logging.info(f"Modified HRTF written to {hrtf_dir / str(sub_id + '_notch.sofa')}")