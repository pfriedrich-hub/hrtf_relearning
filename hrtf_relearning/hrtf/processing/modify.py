"""
modify_replace.py — HRTF modification with spectral *replacement* rather than multiplicative gain.

Difference from modify.py
-------------------------
modify.py treats the Gaussian as a **dB gain** applied on top of the existing spectrum:

    gain_db(f) = -depth_db * exp(-0.5 * ((f-mu)/sigma)^2)
    mag_out(f) = mag_in(f) * 10^(gain_db(f)/20)

This compounds multiplicatively with whatever spectral shape is already present at
those frequencies.

This script treats the Gaussian as a **blending weight** that crossfades the original
magnitude toward a target level:

    w(f)       = exp(-0.5 * ((f-mu)/sigma)^2)          # 0 at edges, 1 at centre
    target(f)  = mag_ref(f) * 10^(-depth_db/20)         # >0 depth -> notch, <0 -> peak
    mag_out(f) = (1 - w(f)) * mag_in(f) + w(f) * target(f)

where mag_ref is the smoothed (trend) version of the spectrum, used so that the target
represents a clean level independent of fine spectral structure at those bins.

At the Gaussian centre (w = 1):   mag_out = target   (original is fully replaced)
At the Gaussian tails (w → 0):    mag_out = mag_in   (original is fully preserved)
In between:                        linear crossfade between the two

The two approaches agree at both extremes (w = 0 and w = 1) but differ in between:
the replacement/blend is linear in the magnitude domain, whereas the gain approach is
linear in dB (i.e. exponential in magnitude).  The blend approach results in a
"softer" notch shape in the transition zone and replaces the actual spectral power in
those bins with the feature's target power rather than attenuating whatever happens
to be there.
"""

import copy
import numpy
import matplotlib
import matplotlib.ticker
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from hrtf_relearning import PATH
hrtf_dir = PATH / 'data' / 'hrtf' / 'sofa'
import slab
from hrtf_relearning.hrtf.analysis.vsi import vsi as _vsi, vsi_dissimilarity as _vsi_dissimilarity

sub_id = 'NKa'

SMOOTH = True
N_KEEP = 4
PLOT = 'image'
fname = 's_4_notch'
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
    {
        'freqs': (6000, 8000),  # centre freq at X1 and X2 [Hz]
        'width': (300,   300),   # Gaussian σ at X1 and X2 [Hz]
        'depth': (12.0,  12.0), # >0 = notch, <0 = peak [dB]
        'X1':    (0, 0),         # anchor 1 (az, az)
        'X2':    (-40, 40),      # anchor 2 (el, el)
    },
    # Add further features here, e.g.:
    {
        'freqs': (9000, 11000),
        'width': (300, 300),
        'depth': (12, 12),   # negative → peak
        'X1': (0, 0),
        'X2': (40, -40),
    },
    {
        'freqs': (16000, 17000),
        'width': (300, 300),
        'depth': (12, 12),  # negative → peak
        'X1': (0, 0),
        'X2': (-40, 40),
    },
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


if __name__ == '__main__':
    hrtf = slab.HRTF(hrtf_dir / str(sub_id + '.sofa'))
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
    input('press enter to save')
    fig.savefig(PATH / 'data' / 'results' / 'plot' / sub_id / str(sub_id + '_modified.png'),
                bbox_inches='tight')
    hrtf_modified.write_sofa(hrtf_dir / str(sub_id + f'_{fname}.sofa'))
