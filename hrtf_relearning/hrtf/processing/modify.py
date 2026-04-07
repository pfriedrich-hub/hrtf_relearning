import copy
import numpy
import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from hrtf_relearning import PATH
hrtf_dir = PATH / 'data' /'hrtf'/'sofa'
import slab

sub_id = 'PC'

SMOOTH = True
N_KEEP = 12
NOTCH = True
# notch parameters
notch_freqs = (6000, 12000)   # notch center frequency for azimuth-driven (X1) and elevation-driven (X2) variation [Hz]
notch_width = (300, 300)      # notch bandwidth (Gaussian σ) for X1 and X2 [Hz]
notch_depth = (12.0, 12.0)    # notch attenuation for X1 and X2 [dB]

# X1 and X2 define two spatial anchor directions,
# and the notch parameters are linearly interpolated between them in azimuth–elevation space.
notch_X1 = (0, 0)     # azimuth reference direction (azimuth-driven variation)
notch_X2 = (-60, 60)  # elevation reference direction (elevation-driven variation)

# notch_X1 = (0, -60)  # this is what chatgpt thinks is correct for elevation dependent notch
# notch_X2 = (0, 60)  # doesn't work

"""
Artificial spectral notch parameters.

A direction-dependent spectral notch is added to the HRTF magnitude response.
The notch characteristics are defined at two spatial anchor points (X1 and X2)
and are linearly interpolated for all other directions.

Coordinates
-----------
X1, X2 : (azimuth, elevation) in degrees
    Spatial anchor points that define where the notch parameters are specified.

Parameter Mapping
-----------------
notch_freqs : (f1, f2) in Hz
    Center frequency of the notch at X1 and X2. The notch frequency shifts
    linearly between these two values as the source direction moves between
    the anchor points.

notch_width : (w1, w2) in Hz
    Standard deviation / bandwidth of the Gaussian notch. Larger values
    produce broader spectral attenuation. Width is interpolated between
    X1 and X2 across the spatial field.

notch_depth : (d1, d2) in dB
    Depth of the attenuation at the notch center frequency. Higher values
    correspond to stronger attenuation. Depth varies linearly between
    the anchor points.

Spatial Behaviour
-----------------
For a given source direction (azimuth, elevation), the notch parameters
(center frequency, width, and depth) are obtained by linear interpolation
between X1 and X2. This creates a smooth spatial gradient so that the notch
changes gradually across azimuth and elevation rather than abruptly.

Typical use
-----------
This mechanism can simulate direction-dependent spectral cues similar to
pinna-induced spectral notches, where notch frequency and depth vary
systematically with source direction.
"""

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


def _add_notch(
        azimuth,
        elevation,
        notch_freqs=(6000.0, 12000.0),
        notch_width=(300.0, 300.0),
        notch_depth=(12.0, 12.0),
        X1=(0, 0),
        X2=(-60, 60),
):
    """
    Return artificial spectral-notch parameters for a given source direction.
    """
    mu = linear_notch_position(
        azimuth,
        elevation,
        X1=X1,
        X2=X2,
        Y=notch_freqs,
    )
    sigma = linear_notch_width(
        azimuth,
        elevation,
        X1=X1,
        X2=X2,
        Y=notch_width,
    )
    depth_db = abs(
        linear_scaling_factor(
            azimuth,
            elevation,
            X1=X1,
            X2=X2,
            Y=notch_depth,
        )
    )

    return {
        "mu": float(mu),
        "sigma": float(max(sigma, numpy.finfo(float).eps)),
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
        add_notch=True,
        onset_threshold_db=15.0,
        notch_freqs=(6000.0, 12000.0),
        notch_width=(300.0, 300.0),
        notch_depth=(12.0, 12.0),
        notch_X1=(0, 0),
        notch_X2=(-60, 60),
):
    """
    Apply paper-style HRTF spectral smoothing, optional artificial spectral
    notch insertion, minimum-phase reconstruction, and onset-based ITD
    restoration to a slab.HRTF object.

    Parameters
    ----------
    hrtf : slab.HRTF
        Input HRTF object. Each element is assumed to contain a binaural HRIR
        in `filt.data` with shape (n_samples, 2).
    n_keep : int
        Number of cosine coefficients retained in the log-magnitude smoothing
        step. This is the smoothing parameter analogous to the paper's M,
        interpreted here as the number of retained coefficients including C(0).

    add_notch : bool, default=True
        If True, apply the artificial direction-dependent notch after smoothing
        and before minimum-phase reconstruction.
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
      3. Optionally apply an artificial Gaussian notch in magnitude
      4. Reconstruct a minimum-phase spectrum from the final magnitude
      5. Transform back to the time domain via irFFT
      6. Restore the original onset-based ITD by a pure right-ear shift

    Notes
    -----
    This is "strict paper-style" for the smoothing stage:
      - smoothing is performed on log-magnitude using a truncated Fourier /
        cosine series, not directly in the time domain
      - the resulting magnitude is implemented as a minimum-phase filter
      - overall ITD is restored as a frequency-independent delay.

    The artificial notch is your additional manipulation and is not part of the
    original paper.
    """
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

        if smooth:
            # 1) Paper-style smoothing in log-magnitude / cosine-series domain
            mag_processed = _smooth(mag_original, n_keep=n_keep)
        else:
            mag_processed = mag_original

        # 2) Optional artificial spectral notch in the magnitude domain
        if add_notch:
            notch = _add_notch(
                azimuth,
                elevation,
                notch_freqs=notch_freqs,
                notch_width=notch_width,
                notch_depth=notch_depth,
                X1=notch_X1,
                X2=notch_X2,
            )
            notch_db = -notch["depth_db"] * numpy.exp(
                -0.5 * ((freqs - notch["mu"]) / notch["sigma"]) ** 2
            )
            notch_lin = 10.0 ** (notch_db / 20.0)

            # leave DC and Nyquist untouched
            notch_lin[0] = 1.0
            notch_lin[-1] = 1.0

            # guard against pathological attenuation
            notch_lin = numpy.clip(notch_lin, 10.0 ** (-80.0 / 20.0), 1.0)
            mag_processed = mag_processed * notch_lin[:, None]


        # 3) Enforce minimum phase once, after smoothing + notch
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

def plot(hrtf, hrtf_modified, kind='image', ear='left'):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    hrtf.plot_tf(hrtf.cone_sources(0), kind=kind, axis=ax[0], ear=ear)
    hrtf_modified.plot_tf(hrtf.cone_sources(0), kind=kind, axis=ax[1], ear=ear)
    ax[0].set_title('original')
    ax[1].set_title('modified')
    plt.show(block=False)
    plt.pause(0.1)  # give Qt time to draw
    return fig

if __name__ == '__main__':
    # modify
    hrtf = slab.HRTF(hrtf_dir / str(sub_id + '.sofa'))
    hrtf_modified = smooth_and_notch_hrtf(
        hrtf,
        n_keep=N_KEEP,
        smooth=SMOOTH,
        add_notch=NOTCH,
        onset_threshold_db=15.0,
        notch_freqs=notch_freqs,
        notch_width=notch_width,
        notch_depth=notch_depth,
        notch_X1=notch_X1,
        notch_X2=notch_X2,
    )

    # plot
    fig = plot(hrtf, hrtf_modified, 'image', ear='right')
    input('press enter to save')
    fig.savefig(PATH / 'data' / 'results' / 'plot' / sub_id / str(sub_id + '_modified.png'))
    hrtf_modified.write_sofa(hrtf_dir / str(sub_id + '_notch.sofa'))
    import logging
    logging.info(f"Modified HRTF written to {hrtf_dir / str(sub_id + '_notch.sofa')}")