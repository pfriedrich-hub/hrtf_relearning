"""Fit the spherical-head radius acoustically, from measured lateral ITDs.

Why
---
`head_radius` feeds `spherical_head`, which sets every off-midline ITD in the
azimuth expansion (and the low-frequency magnitude anchors in
`lowfreq_extrapolate`). It is an EFFECTIVE radius -- the radius of the sphere
whose ITDs match the listener's -- not an anatomical measurement. Half the
ear-to-ear distance underestimates it, because the acoustic path around a real
head is longer than around a sphere of that radius: the head is elongated
front-to-back and the ears sit behind and below the widest point. Half the
inion-nasion depth overestimates it for the mirror-image reason. Standard adult
value is 0.0875 m; Kuhn (1977) had to raise it to 0.093 m to match measured
low-frequency ITDs.

Nothing in a subject's own recording can check the value, because only the
az = 0 arc is measured and every azimuth cue is synthesised. This module closes
that loop: record a few LATERAL directions, measure the ITD, and solve for the
radius that reproduces it.

Two things this module has to get right, both of which it got WRONG before
2026-08-19 and both of which are validated against KEMAR below.

1. AZIMUTH SIGN. The dome speaker table uses the compass convention, positive
   azimuth to the RIGHT. pyfar `top_elev` (and SOFA/AES69, and therefore every
   SOFA this project writes) uses the mathematical convention, positive azimuth
   to the LEFT. Feeding dome azimuths straight into `spherical_head` produces a
   MIRRORED model: measured and model ITDs then have opposite signs, the fit
   has nothing to latch onto and runs to the bound. Convert at the boundary
   with `dome_to_sofa_azimuth` -- never by negating the ITD, which hides the
   error somewhere it will be re-derived wrongly later.

2. WINDOWING. A dome recording is 0.2 s of direct sound plus room. The
   interaural phase of direct + reverberant field is not the phase of a delay,
   so a narrow-band phase slope over it is meaningless -- on the KEMAR data it
   read +1244 us where the truth was +411. Cross-correlation survives it (it
   locks to the direct-path peak) but the phase estimator does not. Deconvolve
   to an IR and window to the direct sound (`window_direct`) before using the
   phase estimator. `fit_from_sofa` is safe as-is: SOFA IRs are already short.

How to get the measurements
---------------------------
`record_head_radius` does the whole thing: record (or load) the horizontal row
with the in-ear mics already in place, deconvolve, window, measure, convert the
azimuth convention, fit, and save. `Recordings.record_dome` filters the dome
down to the vertical arc with `azimuth=(-1, 1)`; widening that to lateral
speakers at el 0 is all the extra measurement needs.

Validation (KEMAR, 2026-08-19)
------------------------------
Six lateral directions on the dome, mics in KEMAR's ears, against the published
MIT KEMAR SOFA measured with the identical estimator and fitter:

    estimator                     this rig     published MIT KEMAR
    phase slope, windowed IR      0.0722 m     0.0754 m     (resid 27 / 29 us)
    cross-correlation             0.0824 m     0.0853 m     (resid 25 /  8 us)

2-3 mm agreement on a known head, with both estimators, and the per-direction
ITDs match published KEMAR to ~20 us. The offset BETWEEN the two estimators
(~10 mm) is a property of the estimator, not an error: the low-frequency phase
ITD of a real head exceeds its broadband/onset ITD (Kuhn 1977), so the sphere
that matches the phase slope is smaller. `estimator='phase'` is the default
because `expand_azimuths_with_binaural_cues` imposes ITD via the same phase
measure -- the fitted radius then reproduces the ITD the pipeline will actually
synthesise. Read `xcorr` as the cross-check, not as a competing answer.

Caveat the fit reports for you
------------------------------
`residual_us` is the RMS mismatch between the fitted sphere and the
measurements. A real head is not a sphere and the ears are not at +-90 deg, so
a few tens of microseconds is normal; a large residual, a residual that grows
systematically with azimuth, or `at_bound`, means a single sphere does not
describe this listener and the fitted number is a compromise rather than a
measurement.
"""
import json
import logging
from pathlib import Path

import numpy
import pyfar
from scipy.optimize import minimize_scalar

from hrtf_relearning.hrtf.processing.spherical_head import spherical_head

ITD_BAND_HZ = (200.0, 1500.0)
DEFAULT_DISTANCE_M = 1.4
RADIUS_BOUNDS_M = (0.055, 0.115)
WINDOW_SAMPLES = 512
FALLBACK_RADIUS_M = 0.0875


# ---------------------------------------------------------------------
# conventions
# ---------------------------------------------------------------------

def dome_to_sofa_azimuth(azimuth_deg):
    """Dome/compass azimuth (+ = right) -> SOFA & pyfar azimuth (+ = left).

    The dome speaker table and every recording key written by
    `Recordings.record_dome` use the compass convention. `spherical_head`,
    pyfar `top_elev` and the SOFA files this project writes use the opposite
    one. Skipping this conversion mirrors the model and the fit collapses onto
    `RADIUS_BOUNDS_M` with a ~1000 us residual.
    """
    return -numpy.asarray(azimuth_deg, dtype=float)


sofa_to_dome_azimuth = dome_to_sofa_azimuth  # the map is its own inverse


# ---------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------

def interaural_delay_us(spectrum, freqs, band=ITD_BAND_HZ):
    """Interaural delay of one binaural spectrum, in microseconds.

    Slope of the unwrapped interaural phase difference over `band`. Positive
    means the right ear is later. `spectrum` is (2, n_freq), left ear first.
    Same estimator and sign convention as
    `processing._interaural_delay_s` and `expand_azimuths_with_binaural_cues`
    step 3a, so a fit made here is consistent with what the pipeline imposes.

    Requires a signal whose interaural transfer function IS a delay: an
    anechoic or windowed impulse response. See `window_direct`.
    """
    ipd = numpy.unwrap(numpy.angle(spectrum[1] * numpy.conj(spectrum[0])))
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if mask.sum() < 3:
        raise ValueError(f"band {band} covers only {mask.sum()} bins - too few to fit a slope")
    return -numpy.polyfit(2 * numpy.pi * freqs[mask], ipd[mask], 1)[0] * 1e6


def crosscorrelation_delay_us(time_data, fs, band=None):
    """Interaural delay from the cross-correlation peak, in microseconds.

    Positive means the right ear is later -- same convention as
    `interaural_delay_us`. Parabolic interpolation around the peak gives
    sub-sample resolution (~2 us at 48.8 kHz).

    Unlike the phase slope this locks onto the DIRECT-path peak, so it survives
    an unwindowed room recording: on the KEMAR row it returned the same radius
    from the raw sweeps (0.0832 m) as from windowed IRs (0.0824 m), where the
    phase estimator was off by a factor of three. Use it for a quick check
    without deconvolution, and as the cross-check on the phase fit.
    """
    x = numpy.asarray(time_data, dtype=float)
    if x.shape[0] != 2:
        raise ValueError(f"expected (2, n_samples), got {x.shape}")
    n = 1
    while n < 2 * x.shape[-1]:
        n *= 2
    left = numpy.fft.rfft(x[0], n)
    right = numpy.fft.rfft(x[1], n)
    cross = right * numpy.conj(left)
    if band is not None:
        freqs = numpy.fft.rfftfreq(n, 1 / fs)
        cross = cross * ((freqs >= band[0]) & (freqs <= band[1]))
    cc = numpy.fft.irfft(cross, n)
    cc = numpy.concatenate([cc[-n // 2:], cc[:n // 2]])
    lags = numpy.arange(-n // 2, n // 2)
    k = int(numpy.argmax(numpy.abs(cc)))
    if 0 < k < len(cc) - 1:
        y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
        denominator = y0 - 2 * y1 + y2
        shift = 0.5 * (y0 - y2) / denominator if denominator != 0 else 0.0
    else:
        shift = 0.0
    return float((lags[k] + shift) / fs * 1e6)


ESTIMATORS = ("phase", "xcorr")


def _measure(time_data, fs, estimator="phase", band=ITD_BAND_HZ):
    """Dispatch to one estimator. `time_data` is (2, n_samples), left first."""
    x = numpy.asarray(time_data, dtype=float)
    if x.shape[0] != 2:
        raise ValueError(f"expected (2, n_samples), got {x.shape}")
    if estimator == "xcorr":
        return crosscorrelation_delay_us(x, fs)
    if estimator == "phase":
        return interaural_delay_us(numpy.fft.rfft(x, axis=-1),
                                   numpy.fft.rfftfreq(x.shape[-1], 1 / fs), band)
    raise ValueError(f"estimator must be one of {ESTIMATORS}, got {estimator!r}")


def window_direct(time_data, n_samples=WINDOW_SAMPLES, pre_samples=32,
                  fade_samples=64, onset_threshold_db=20.0):
    """Window a binaural IR down to its direct sound, keeping the ITD intact.

    ONE common window for both ears, anchored on the earlier of the two onsets.
    Windowing the ears independently would slide them onto a common onset and
    delete exactly the quantity being measured.

    The onset is the first sample of the two-ear envelope within
    `onset_threshold_db` of its peak, backed off by `pre_samples`.
    """
    x = numpy.asarray(time_data, dtype=float)
    if x.shape[0] != 2:
        raise ValueError(f"expected (2, n_samples), got {x.shape}")
    envelope = numpy.abs(x).max(axis=0)
    threshold = envelope.max() * 10 ** (-onset_threshold_db / 20)
    onset = int(numpy.argmax(envelope > threshold))
    start = max(0, onset - pre_samples)
    segment = x[:, start:start + n_samples]
    if segment.shape[-1] < n_samples:
        segment = numpy.pad(segment, ((0, 0), (0, n_samples - segment.shape[-1])))
    window = numpy.ones(n_samples)
    rise = min(8, n_samples // 8)
    window[:rise] = numpy.hanning(2 * rise)[:rise]
    fade = min(fade_samples, n_samples // 2)
    window[-fade:] = numpy.hanning(2 * fade)[fade:]
    return segment * window


def itd_from_binaural(time_data, fs, reference=None, band=ITD_BAND_HZ,
                      estimator="phase", window=False):
    """ITD in us from a binaural time signal, shape (2, n_samples).

    `reference`, if given, is a binaural recording of the SAME speaker with the
    mics co-located; dividing by it removes any mic/speaker phase mismatch.

    WINDOWING. With `estimator='phase'` the signal must be an anechoic or
    windowed impulse response -- pass `window=True` for a deconvolved IR, or
    window it yourself. Handing this function a RAW dome sweep with the phase
    estimator is the 2026-08-19 bug: 200 ms of room makes the low-frequency
    interaural phase meaningless and the reported ITD was 3x the truth.
    `estimator='xcorr'` is robust to this and needs no window.
    """
    x = numpy.asarray(time_data, dtype=float)
    if x.shape[0] != 2:
        raise ValueError(f"expected (2, n_samples), got {x.shape}")
    if reference is not None:
        spectrum = numpy.fft.rfft(x, axis=-1)
        ref = numpy.fft.rfft(numpy.asarray(reference, dtype=float), axis=-1)
        if ref.shape != spectrum.shape:
            raise ValueError("reference must have the same shape as the measurement")
        x = numpy.fft.irfft(spectrum / (ref + 1e-30), n=x.shape[-1], axis=-1)
    if window:
        x = window_direct(x)
    return _measure(x, fs, estimator, band)


# ---------------------------------------------------------------------
# model + fit
# ---------------------------------------------------------------------

def model_itd_us(azimuths_deg, head_radius, elevations_deg=0.0,
                 distance_m=DEFAULT_DISTANCE_M, n_samples=512, fs=48828,
                 band=ITD_BAND_HZ, estimator="phase"):
    """ITD of the spherical-head model at the given directions, in us.

    `azimuths_deg` are SOFA/pyfar azimuths (+ = LEFT). Pass dome azimuths
    through `dome_to_sofa_azimuth` first.

    Measured with the same estimator as the data, so the fit is not absorbing a
    difference between two ways of defining ITD -- a phase-slope model fitted
    against cross-correlation measurements biases the radius by ~5 mm.
    """
    azimuths_deg = numpy.atleast_1d(numpy.asarray(azimuths_deg, dtype=float))
    elevations_deg = numpy.broadcast_to(
        numpy.asarray(elevations_deg, dtype=float), azimuths_deg.shape)
    coordinates = pyfar.Coordinates(
        azimuths_deg.tolist(), elevations_deg.tolist(),
        [float(distance_m)] * azimuths_deg.size,
        domain="sph", convention="top_elev", unit="deg")
    head = pyfar.Coordinates(0, [head_radius, -head_radius], 0)
    shtf = spherical_head(coordinates, head=head, n_samples=n_samples, sampling_rate=fs)
    if estimator == "phase":
        return numpy.array([interaural_delay_us(shtf.freq[i], shtf.frequencies, band)
                            for i in range(azimuths_deg.size)])
    return numpy.array([_measure(shtf.time[i], fs, estimator, band)
                        for i in range(azimuths_deg.size)])


def fit_head_radius(azimuths_deg, itds_us, elevations_deg=0.0,
                    distance_m=DEFAULT_DISTANCE_M, n_samples=512, fs=48828,
                    band=ITD_BAND_HZ, bounds=RADIUS_BOUNDS_M, estimator="phase"):
    """Solve for the head radius whose model ITDs best match the measurements.

    `azimuths_deg` are SOFA/pyfar azimuths (+ = LEFT); see
    `dome_to_sofa_azimuth`. `estimator` must be the one the measurements were
    made with.

    Returns a dict with `head_radius` (m), `residual_us` (RMS), `azimuths`,
    `measured_us`, `model_us`, `residuals_us`, `estimator` and `at_bound`
    (True if the solution ran into `bounds`, which means the data are not
    describable by a sphere in the plausible range -- most often because the
    azimuth convention was not converted).
    """
    azimuths_deg = numpy.asarray(azimuths_deg, dtype=float)
    itds_us = numpy.asarray(itds_us, dtype=float)
    if azimuths_deg.shape != itds_us.shape:
        raise ValueError("azimuths_deg and itds_us must have the same shape")
    if numpy.all(numpy.abs(azimuths_deg) < 5):
        raise ValueError("all directions are within 5 deg of the midline - "
                         "ITD is ~0 there and carries no information about the radius")

    lateral = numpy.abs(azimuths_deg) >= 5
    if lateral.sum() > 1 and numpy.corrcoef(azimuths_deg[lateral], itds_us[lateral])[0, 1] < 0:
        logging.warning(
            "fit_head_radius: ITD DECREASES with azimuth. In the SOFA/pyfar "
            "convention (+ azimuth = LEFT) a positive azimuth must give a "
            "POSITIVE ITD (right ear later). These look like dome azimuths - "
            "pass them through dome_to_sofa_azimuth() first.")

    def cost(radius):
        model = model_itd_us(azimuths_deg, radius, elevations_deg,
                             distance_m, n_samples, fs, band, estimator)
        return float(numpy.mean((model - itds_us) ** 2))

    result = minimize_scalar(cost, bounds=bounds, method="bounded",
                             options={"xatol": 1e-5})
    radius = float(result.x)
    model = model_itd_us(azimuths_deg, radius, elevations_deg,
                         distance_m, n_samples, fs, band, estimator)
    residuals = model - itds_us
    at_bound = bool(min(abs(radius - bounds[0]), abs(radius - bounds[1])) < 1e-4)
    if at_bound:
        logging.warning("fit_head_radius: solution sits on the bound (%.4f m) - "
                        "a sphere does not describe these ITDs.", radius)
    return {
        "head_radius": radius,
        "residual_us": float(numpy.sqrt(numpy.mean(residuals ** 2))),
        "azimuths": azimuths_deg,
        "measured_us": itds_us,
        "model_us": model,
        "residuals_us": residuals,
        "at_bound": at_bound,
        "band_hz": tuple(band),
        "distance_m": float(distance_m),
        "estimator": estimator,
    }


def fit_from_sofa(sofa_file, elevation_tol_deg=1.0, min_abs_azimuth=5.0,
                  band=ITD_BAND_HZ, bounds=RADIUS_BOUNDS_M, estimator="phase"):
    """Recover the head radius an existing SOFA was built with.

    Reads the horizontal row, measures its ITDs and fits. For SOFAs from this
    pipeline the azimuth ITDs ARE the model, so the residual should be ~0 and
    the fit tells you exactly which radius produced the file -- useful when the
    build parameters were not recorded.

    No azimuth conversion here: SOFA azimuths are already in the SOFA
    convention. No windowing either: SOFA IRs are already short.
    """
    import h5py
    sofa_file = Path(sofa_file)
    with h5py.File(sofa_file, "r") as sofa:
        ir = numpy.array(sofa["Data.IR"])
        pos = numpy.array(sofa["SourcePosition"])
        fs = float(numpy.array(sofa["Data.SamplingRate"]).ravel()[0])

    azimuth = (pos[:, 0] + 180) % 360 - 180
    keep = (numpy.abs(pos[:, 1]) < elevation_tol_deg) & (numpy.abs(azimuth) >= min_abs_azimuth)
    if not keep.any():
        raise ValueError(f"{sofa_file.name}: no horizontal off-midline directions found")
    order = numpy.argsort(azimuth[keep])
    idx = numpy.where(keep)[0][order]

    itds = numpy.array([_measure(ir[i], fs, estimator, band) for i in idx])
    return fit_head_radius(azimuth[idx], itds, elevations_deg=pos[idx, 1],
                           distance_m=float(numpy.mean(pos[idx, 2])),
                           n_samples=ir.shape[-1], fs=fs, band=band, bounds=bounds,
                           estimator=estimator)


def report(fit, label=""):
    """One-line-per-direction summary of a fit."""
    print(f"{label}head_radius = {fit['head_radius']:.4f} m   "
          f"residual {fit['residual_us']:.2f} us RMS   [{fit.get('estimator', 'phase')}]"
          f"{'   [AT BOUND]' if fit['at_bound'] else ''}")
    print(f"{'':4}{'az (sofa)':>11}{'measured':>12}{'model':>10}{'diff':>9}")
    for az, meas, mod, res in zip(fit["azimuths"], fit["measured_us"],
                                  fit["model_us"], fit["residuals_us"]):
        print(f"{'':4}{az:+11.1f}{meas:+12.1f}{mod:+10.1f}{res:+9.1f}")


# ---------------------------------------------------------------------
# protocol wrapper
# ---------------------------------------------------------------------

def record_head_radius(subject_id, azimuth_range=(-60, 60), elevation=(-1, 1),
                       n_recordings=10, hp_freq=120, fs=48828,
                       estimator="phase", band=ITD_BAND_HZ,
                       distance_m=DEFAULT_DISTANCE_M, base_dir=None,
                       overwrite=False, show=True, save=None):
    """Measure the horizontal row and fit this listener's effective head radius.

    Records lateral speakers at el 0 with the in-ear mics already in place,
    deconvolves, windows to the direct sound, converts the azimuth convention
    and fits. Re-running loads the stored sweeps instead of re-recording.

    Everything lands INSIDE the subject's own recording folder -- `azimuth.npz`,
    `azimuth_params.txt` and `head_radius_fit.json` next to `recordings.npz`.
    The earlier version wrote a whole sibling `rec/<id>_azimuth/` folder, which
    doubled the number of entries in `rec/` for no benefit.

    Returns the fit dict from `fit_head_radius`, with `fit['cross_check']`
    holding the same fit made with the other estimator. If the two disagree by
    much more than ~10 mm, something is wrong with the measurement rather than
    with the choice of estimator.
    """
    from hrtf_relearning.hrtf.record.recordings import Recordings
    from hrtf_relearning.hrtf.record import processing
    from hrtf_relearning.utils import paths

    base_dir = Path(base_dir) if base_dir is not None else paths.REC_DIR
    subject_dir = base_dir / subject_id
    npz_file = subject_dir / "azimuth.npz"
    legacy = base_dir / f"{subject_id}_azimuth"

    if npz_file.exists() and not overwrite:
        logging.info(f"Loading azimuth sweeps from {npz_file}")
        recordings = Recordings.load(subject_dir, filename="azimuth.npz")
    elif (legacy / "recordings.npz").exists() and not overwrite:
        logging.warning(
            f"Using the legacy sibling folder {legacy}. Move its recordings.npz "
            f"to {npz_file} (and params.txt to azimuth_params.txt) to keep "
            f"everything for one subject in one place.")
        recordings = Recordings.load(legacy)
    else:
        logging.info(f"Recording the horizontal row for '{subject_id}'")
        recordings = Recordings.record_dome(
            id=f"{subject_id}_azimuth", azimuth=azimuth_range, elevation=elevation,
            n_directions=1, n_recordings=n_recordings, hp_freq=hp_freq, fs=fs,
            equalize_dome=False, key=True)
        subject_dir.mkdir(parents=True, exist_ok=True)
        recordings.to_npz(subject_dir, overwrite=True, filename="azimuth.npz")

    irs = processing.compute_ir(recordings, inversion_range_hz=(hp_freq, 20e3),
                                onset_threshold_db=10, align_interaural=False)

    dome_azimuths, itds, cross = [], [], []
    other = "xcorr" if estimator == "phase" else "phase"
    for key in irs.data:
        _index, azimuth, _elevation = key.split("_")
        if abs(float(azimuth)) < 5:      # ITD ~0 at the midline, no information there
            continue
        windowed = window_direct(numpy.asarray(irs.data[key].time))
        dome_azimuths.append(float(azimuth))
        itds.append(_measure(windowed, fs, estimator, band))
        cross.append(_measure(windowed, fs, other, band))

    order = numpy.argsort(dome_azimuths)
    dome_azimuths = numpy.asarray(dome_azimuths)[order]
    sofa_azimuths = dome_to_sofa_azimuth(dome_azimuths)
    itds = numpy.asarray(itds)[order]
    cross = numpy.asarray(cross)[order]

    fit = fit_head_radius(sofa_azimuths, itds, distance_m=distance_m, fs=fs,
                          band=band, estimator=estimator)
    fit["cross_check"] = fit_head_radius(sofa_azimuths, cross, distance_m=distance_m,
                                         fs=fs, band=band, estimator=other)
    fit["dome_azimuths"] = dome_azimuths

    if show:
        report(fit, label=f"{subject_id}: ")
        report(fit["cross_check"], label=f"{subject_id} cross-check: ")

    out = subject_dir / "head_radius_fit.json"
    out.write_text(json.dumps(_jsonable(fit), indent=2))
    logging.info(f"Wrote {out}")

    # `save` also files the fit with the subject's results, so the radius a SOFA
    # was built with travels with the behavioural data. Accepts a Subject, or an
    # id string, or True to reuse subject_id.
    if save is not None and save is not False:
        save_id = (getattr(save, "id", None) or getattr(save, "subject_id", None)
                   or (subject_id if save is True else str(save)))
        try:
            results = paths.subject_dir(save_id)
            results.mkdir(parents=True, exist_ok=True)
            copy = results / "head_radius_fit.json"
            copy.write_text(json.dumps(_jsonable(fit), indent=2))
            logging.info(f"Wrote {copy}")
        except Exception as error:      # never lose a recording over bookkeeping
            logging.warning(f"Could not file the fit with subject '{save_id}': {error}")

    return fit


def usable_radius(fit, fallback=FALLBACK_RADIUS_M, max_residual_us=150.0,
                  max_cross_check_m=0.025):
    """The fitted radius if the fit is trustworthy, else `fallback`, loudly.

    Use this rather than `fit['head_radius']` when the value feeds straight into
    a build, so a bad measurement cannot silently set every synthesised ITD in
    the SOFA. Rejects a solution that hit the bounds, one whose residual is far
    above the ~30 us a real head gives, and one the other ITD estimator
    disagrees with by more than `max_cross_check_m`.

    For scale: KEMAR returns 0.0722 m with a 27 us residual and a 10 mm
    cross-check gap. The pre-2026-08-19 sign bug returned 0.055 m AT BOUND with
    a 1104 us residual.
    """
    reasons = []
    if fit.get("at_bound"):
        reasons.append(f"solution sits on the bound ({fit['head_radius']:.4f} m)")
    if fit.get("residual_us", 0) > max_residual_us:
        reasons.append(f"residual {fit['residual_us']:.0f} us > {max_residual_us:.0f}")
    cross = fit.get("cross_check")
    if cross is not None:
        gap = abs(cross["head_radius"] - fit["head_radius"])
        if gap > max_cross_check_m:
            reasons.append(f"the two ITD estimators disagree by {gap*1000:.0f} mm")
    if reasons:
        logging.error(
            "head-radius fit is NOT usable (%s) -- falling back to %.4f m. "
            "Record that the radius was not measured for this subject.",
            "; ".join(reasons), fallback)
        return fallback
    logging.info("head-radius fit accepted: %.4f m (residual %.0f us)",
                 fit["head_radius"], fit["residual_us"])
    return fit["head_radius"]


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


# %% recover the radius an existing SOFA was built with
if __name__ == "__main__":
    from hrtf_relearning.utils import paths

    logging.basicConfig(level=logging.INFO)
    for subject_id in ("NW",):
        report(fit_from_sofa(paths.SOFA_DIR / subject_id / f"{subject_id}.sofa"),
               label=f"{subject_id}: ")
