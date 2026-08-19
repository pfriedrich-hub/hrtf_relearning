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

How to get the measurements
---------------------------
`Recordings.record_dome` filters the dome down to the vertical arc with
`azimuth=(-1, 1)`. Widen that to pick up lateral speakers at el 0 (or rotate
the head in azimuth, the same trick record_dome already uses for elevation) and
record with the in-ear mics already in place. The excitation cancels in the
interaural ratio, so the ITD can be read straight off the recorded sweeps --
no deconvolution needed. Pass a same-speaker reference recording if you want
the mic/speaker phase mismatch divided out as well.

Validate the method on KEMAR first: its off-axis ITDs are published, so a fit
against KEMAR tests the procedure before it is applied to a participant.

Caveat the fit reports for you
------------------------------
`residual_us` is the RMS mismatch between the fitted sphere and the
measurements. A real head is not a sphere and the ears are not at +-90 deg, so
a few tens of microseconds is normal; a large residual, or a residual that
grows systematically with azimuth, means a single sphere does not describe this
listener and the fitted number is a compromise rather than a measurement.
"""
import logging
from pathlib import Path

import numpy
import pyfar
from scipy.optimize import minimize_scalar

from hrtf_relearning.hrtf.processing.spherical_head import spherical_head

ITD_BAND_HZ = (200.0, 1500.0)
DEFAULT_DISTANCE_M = 1.4
RADIUS_BOUNDS_M = (0.055, 0.115)


def interaural_delay_us(spectrum, freqs, band=ITD_BAND_HZ):
    """Interaural delay of one binaural spectrum, in microseconds.

    Slope of the unwrapped interaural phase difference over `band`. Positive
    means the right ear is later. `spectrum` is (2, n_freq), left ear first.
    Same estimator and sign convention as
    `processing._interaural_delay_s` and `expand_azimuths_with_binaural_cues`
    step 3a, so a fit made here is consistent with what the pipeline imposes.
    """
    ipd = numpy.unwrap(numpy.angle(spectrum[1] * numpy.conj(spectrum[0])))
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if mask.sum() < 3:
        raise ValueError(f"band {band} covers only {mask.sum()} bins - too few to fit a slope")
    return -numpy.polyfit(2 * numpy.pi * freqs[mask], ipd[mask], 1)[0] * 1e6


def itd_from_binaural(time_data, fs, reference=None, band=ITD_BAND_HZ):
    """ITD in us from a binaural time signal, shape (2, n_samples).

    Works on raw recorded sweeps as well as on impulse responses: the
    excitation is common to both ears and cancels in the interaural ratio.
    `reference`, if given, is a binaural recording of the SAME speaker with the
    mics co-located; dividing by it removes any mic/speaker phase mismatch.
    """
    time_data = numpy.asarray(time_data)
    if time_data.shape[0] != 2:
        raise ValueError(f"expected (2, n_samples), got {time_data.shape}")
    spectrum = numpy.fft.rfft(time_data, axis=-1)
    freqs = numpy.fft.rfftfreq(time_data.shape[-1], 1 / fs)
    if reference is not None:
        ref = numpy.fft.rfft(numpy.asarray(reference), axis=-1)
        if ref.shape != spectrum.shape:
            raise ValueError("reference must have the same shape as the measurement")
        spectrum = spectrum / (ref + 1e-30)
    return interaural_delay_us(spectrum, freqs, band)


def model_itd_us(azimuths_deg, head_radius, elevations_deg=0.0,
                 distance_m=DEFAULT_DISTANCE_M, n_samples=512, fs=48828,
                 band=ITD_BAND_HZ):
    """ITD of the spherical-head model at the given directions, in us.

    Measured with the same phase-slope estimator as the data, so the fit is not
    absorbing a difference between two ways of defining ITD.
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
    return numpy.array([interaural_delay_us(shtf.freq[i], shtf.frequencies, band)
                        for i in range(azimuths_deg.size)])


def fit_head_radius(azimuths_deg, itds_us, elevations_deg=0.0,
                    distance_m=DEFAULT_DISTANCE_M, n_samples=512, fs=48828,
                    band=ITD_BAND_HZ, bounds=RADIUS_BOUNDS_M):
    """Solve for the head radius whose model ITDs best match the measurements.

    Returns a dict with `head_radius` (m), `residual_us` (RMS), `azimuths`,
    `measured_us`, `model_us`, `residuals_us` and `at_bound` (True if the
    solution ran into `bounds`, which means the data are not describable by a
    sphere in the plausible range).
    """
    azimuths_deg = numpy.asarray(azimuths_deg, dtype=float)
    itds_us = numpy.asarray(itds_us, dtype=float)
    if azimuths_deg.shape != itds_us.shape:
        raise ValueError("azimuths_deg and itds_us must have the same shape")
    if numpy.all(numpy.abs(azimuths_deg) < 5):
        raise ValueError("all directions are within 5 deg of the midline - "
                         "ITD is ~0 there and carries no information about the radius")

    def cost(radius):
        model = model_itd_us(azimuths_deg, radius, elevations_deg,
                             distance_m, n_samples, fs, band)
        return float(numpy.mean((model - itds_us) ** 2))

    result = minimize_scalar(cost, bounds=bounds, method="bounded",
                             options={"xatol": 1e-5})
    radius = float(result.x)
    model = model_itd_us(azimuths_deg, radius, elevations_deg,
                         distance_m, n_samples, fs, band)
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
    }


def fit_from_sofa(sofa_file, elevation_tol_deg=1.0, min_abs_azimuth=5.0,
                  band=ITD_BAND_HZ, bounds=RADIUS_BOUNDS_M):
    """Recover the head radius an existing SOFA was built with.

    Reads the horizontal row, measures its ITDs and fits. For SOFAs from this
    pipeline the azimuth ITDs ARE the model, so the residual should be ~0 and
    the fit tells you exactly which radius produced the file -- useful when the
    build parameters were not recorded.
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

    freqs = numpy.fft.rfftfreq(ir.shape[-1], 1 / fs)
    itds = numpy.array([interaural_delay_us(numpy.fft.rfft(ir[i], axis=-1), freqs, band)
                        for i in idx])
    return fit_head_radius(azimuth[idx], itds, elevations_deg=pos[idx, 1],
                           distance_m=float(numpy.mean(pos[idx, 2])),
                           n_samples=ir.shape[-1], fs=fs, band=band, bounds=bounds)


def report(fit, label=""):
    """One-line-per-direction summary of a fit."""
    print(f"{label}head_radius = {fit['head_radius']:.4f} m   "
          f"residual {fit['residual_us']:.2f} us RMS"
          f"{'   [AT BOUND]' if fit['at_bound'] else ''}")
    print(f"{'':4}{'az':>8}{'measured':>12}{'model':>10}{'diff':>9}")
    for az, meas, mod, res in zip(fit["azimuths"], fit["measured_us"],
                                  fit["model_us"], fit["residuals_us"]):
        print(f"{'':4}{az:+8.1f}{meas:+12.1f}{mod:+10.1f}{res:+9.1f}")


# %% recover the radius an existing SOFA was built with
if __name__ == "__main__":
    from hrtf_relearning.utils import paths

    logging.basicConfig(level=logging.INFO)
    for subject_id in ("NW",):
        report(fit_from_sofa(paths.SOFA_DIR / subject_id / f"{subject_id}.sofa"),
               label=f"{subject_id}: ")

# %% fit a subject from lateral recordings
# from hrtf_relearning.hrtf.record.recordings import Recordings
# recordings = Recordings.load(paths.REC_DIR / SUBJECT_ID)     # lateral speakers, el 0
# azimuths, itds = [], []
# for key, repeats in recordings.data.items():
#     _spk, azimuth, elevation = key.split("_")
#     if abs(float(elevation)) > 1 or abs(float(azimuth)) < 5:
#         continue
#     averaged = numpy.mean([r.data.T for r in repeats], axis=0)
#     azimuths.append(float(azimuth))
#     itds.append(itd_from_binaural(averaged, recordings.params["fs"]))
# report(fit_head_radius(azimuths, itds), label=f"{SUBJECT_ID}: ")
