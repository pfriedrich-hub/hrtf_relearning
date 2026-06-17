"""
diagnostics.py — Did the MESM speaker calibration actually work?

Quantifies and visualises what `calibrate_mesm` achieved, comparing the raw
(PASS A) recordings against the verified (PASS B) recordings made with the
calibration applied. Two things the calibration is supposed to buy you:

1. Level matching  — all speakers equally loud (broadband).
2. Spectral matching/flattening — the speakers share the same magnitude
   response shape (low inter-speaker dispersion) and each is reasonably flat.

The headline number is the inter-speaker spectral dispersion: the standard
deviation across speakers at each frequency, RMS-averaged over the usable band.
A good calibration collapses this curve.

Usage
-----
Called automatically at the end of `calibrate_mesm.main()`. Can also be run on
saved data::

    from .diagnostics import plot_calibration_result, calibration_metrics
    metrics = calibration_metrics(raw, verified, fs=FS)
    plot_calibration_result(raw, verified, calibration, fs=FS)
"""
from __future__ import annotations

import logging

import numpy as np
import pyfar

# usable analysis band (Hz) for dispersion / flatness metrics
BAND_LO = 100.0
BAND_HI = 16_000.0
SMOOTH_FRACTION = 6   # 1/6-octave magnitude smoothing for readable curves


# ---------------------------------------------------------------------------
# Magnitude spectra (interpolated onto a shared log-frequency grid)
# ---------------------------------------------------------------------------
# Recordings from different passes have different lengths (applying the FIR
# lengthens the verified pass), so raw FFT bins don't line up. Interpolating
# every magnitude spectrum onto one fixed log-spaced grid makes all downstream
# comparisons valid regardless of recording length.

def _freq_grid(fs: int, n: int = 512):
    return np.logspace(np.log10(20.0), np.log10(fs / 2), n)


def _mag_on_grid(recording, fs: int, grid, smooth: int = SMOOTH_FRACTION):
    """Magnitude (dB) of one recording, smoothed and interpolated onto `grid`."""
    data = np.asarray(recording.data).squeeze()
    sig = pyfar.Signal(data, fs)
    if smooth:
        sig, _ = pyfar.dsp.smooth_fractional_octave(sig, smooth)
    freqs = sig.frequencies
    mag = 20 * np.log10(np.abs(sig.freq.squeeze()) + 1e-12)
    return np.interp(grid, freqs, mag)


def _band_mask(freqs, lo=BAND_LO, hi=BAND_HI):
    return (freqs >= lo) & (freqs <= hi)


def _spectra(recordings: dict, fs: int, grid=None):
    """(grid, {idx: mag_db}) — all on a shared log-frequency grid."""
    if grid is None:
        grid = _freq_grid(fs)
    return grid, {idx: _mag_on_grid(rec, fs, grid) for idx, rec in recordings.items()}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calibration_metrics(raw: dict, verified: dict, fs: int, log: bool = True) -> dict:
    """
    Quantify the improvement from raw → calibrated.

    Returns a dict with, for both passes:
      - dispersion_db   : RMS over band of the across-speaker std of (shape) spectra
      - level_spread_db : max-min of broadband level across speakers
      - flatness_db     : mean over speakers of each speaker's in-band std (own flatness)
    """
    grid = _freq_grid(fs)

    def _summarise(recordings):
        _, spectra = _spectra(recordings, fs, grid)
        m = _band_mask(grid)
        mags = np.stack([mag[m] for mag in spectra.values()])  # (n_spk, n_freq)

        # shape-only: remove each speaker's in-band mean so level doesn't leak in
        shape = mags - mags.mean(axis=1, keepdims=True)
        dispersion = float(np.sqrt(np.mean(np.std(shape, axis=0) ** 2)))

        levels = np.array([rec.level for rec in recordings.values()])
        level_spread = float(levels.max() - levels.min())

        flatness = float(np.mean(np.std(shape, axis=1)))
        return dispersion, level_spread, flatness

    d0, l0, f0 = _summarise(raw)
    d1, l1, f1 = _summarise(verified)
    metrics = {
        "dispersion_db":   {"raw": d0, "calibrated": d1},
        "level_spread_db": {"raw": l0, "calibrated": l1},
        "flatness_db":     {"raw": f0, "calibrated": f1},
    }
    if log:
        logging.info(
            "\nCalibration result (%g–%g Hz band)\n"
            "  metric                       raw     ->  calibrated\n"
            "  inter-speaker dispersion   %6.2f dB ->  %6.2f dB\n"
            "  broadband level spread     %6.2f dB ->  %6.2f dB\n"
            "  per-speaker flatness (std) %6.2f dB ->  %6.2f dB",
            BAND_LO, BAND_HI, d0, d1, l0, l1, f0, f1,
        )
    return metrics


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_calibration_result(raw: dict, verified: dict, calibration: dict, fs: int,
                            show: bool = True):
    """
    Six-panel figure summarising the calibration outcome.

    A/B  overlaid speaker spectra, raw vs calibrated (same y-scale)
    C    inter-speaker dispersion (std across speakers) vs frequency
    D    broadband level per speaker, raw vs verified
    E    applied inverse-filter magnitude responses
    """
    import matplotlib.pyplot as plt

    grid = _freq_grid(fs)
    freqs, raw_spec = _spectra(raw, fs, grid)
    _, cal_spec = _spectra(verified, fs, grid)
    m = _band_mask(freqs, 20, fs / 2)

    idxs = sorted(raw_spec)
    metrics = calibration_metrics(raw, verified, fs, log=False)

    fig = plt.figure(figsize=(15, 9))
    fig.suptitle("MESM speaker calibration — outcome", fontsize=13)
    gs = fig.add_gridspec(3, 2)

    # mean-normalise spectra for the overlays so shape comparison is fair
    bm = _band_mask(freqs)

    def _norm(spec):
        return {i: mag - mag[bm].mean() for i, mag in spec.items()}

    raw_n, cal_n = _norm(raw_spec), _norm(cal_spec)

    ax_a = fig.add_subplot(gs[0, 0])
    for i in idxs:
        ax_a.semilogx(freqs[m], raw_n[i][m], lw=0.8, label=f"spk {i}")
    ax_a.set_title(f"Raw spectra  (dispersion {metrics['dispersion_db']['raw']:.2f} dB)")
    ax_a.set_ylabel("rel. magnitude (dB)")
    ax_a.set_ylim(-25, 15); ax_a.grid(True, which="both", alpha=0.3)

    ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
    for i in idxs:
        ax_b.semilogx(freqs[m], cal_n[i][m], lw=0.8)
    ax_b.set_title(f"Calibrated spectra  (dispersion {metrics['dispersion_db']['calibrated']:.2f} dB)")
    ax_b.grid(True, which="both", alpha=0.3)

    # C — dispersion vs frequency
    ax_c = fig.add_subplot(gs[1, :])
    raw_std = np.std(np.stack([raw_n[i] for i in idxs]), axis=0)
    cal_std = np.std(np.stack([cal_n[i] for i in idxs]), axis=0)
    ax_c.semilogx(freqs[m], raw_std[m], lw=1.2, color="C3", label="raw")
    ax_c.semilogx(freqs[m], cal_std[m], lw=1.2, color="C2", label="calibrated")
    ax_c.axvspan(BAND_LO, BAND_HI, color="grey", alpha=0.08, label="metric band")
    ax_c.set_title("Inter-speaker dispersion — std across speakers (lower = better matched)")
    ax_c.set_ylabel("std across speakers (dB)")
    ax_c.set_xlabel("Frequency (Hz)")
    ax_c.grid(True, which="both", alpha=0.3); ax_c.legend(ncol=3, fontsize=8)

    # D — broadband level matching
    ax_d = fig.add_subplot(gs[2, 0])
    width = 0.4
    raw_lvl = [raw[i].level for i in idxs]
    cal_lvl = [verified[i].level for i in idxs]
    x = np.arange(len(idxs))
    ax_d.bar(x - width / 2, raw_lvl, width, color="C3", label="raw")
    ax_d.bar(x + width / 2, cal_lvl, width, color="C2", label="calibrated")
    ax_d.axhline(np.mean(cal_lvl), color="k", lw=0.8, ls="--")
    ax_d.set_xticks(x); ax_d.set_xticklabels([f"{i}" for i in idxs])
    ax_d.set_title(
        f"Broadband level  (spread {metrics['level_spread_db']['raw']:.1f} → "
        f"{metrics['level_spread_db']['calibrated']:.1f} dB)")
    ax_d.set_xlabel("speaker"); ax_d.set_ylabel("level (dB)")
    ax_d.set_ylim(min(raw_lvl + cal_lvl) - 3, max(raw_lvl + cal_lvl) + 3)
    ax_d.legend(fontsize=8)

    # E — applied inverse filters
    ax_e = fig.add_subplot(gs[2, 1])
    for i in idxs:
        filt = calibration[i]["filter"]
        h = np.asarray(filt.data).squeeze()
        H = pyfar.Signal(h, fs)
        fr = H.frequencies
        mag = 20 * np.log10(np.abs(H.freq.squeeze()) + 1e-12)
        fm = _band_mask(fr, 20, fs / 2)
        ax_e.semilogx(fr[fm], mag[fm], lw=0.8, label=f"spk {i}")
    ax_e.set_title("Applied inverse filters")
    ax_e.set_xlabel("Frequency (Hz)"); ax_e.set_ylabel("gain (dB)")
    ax_e.grid(True, which="both", alpha=0.3)

    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, fontsize=7)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if show:
        plt.show()
    return fig
