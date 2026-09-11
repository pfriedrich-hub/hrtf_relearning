"""Where the listener's ears actually sat, fitted from the midline sweeps.

The dome is a sphere: every speaker is the same distance from its centre. That
is only true for the LISTENER if the listener's ears are AT the centre, which
is what the chair height sets. If the chair is wrong the listener sits off the
centre by `d_vert`, and then the speaker at elevation E is at

    r(E) = distance - d_vert * sin(E) - d_fwd * (cos(E) - 1)

so the time of flight varies linearly with sin(E). That is a large, easily
measured signal -- 1 cm of vertical offset moves the arrival time at +-37.5 deg
by 35 us, and the estimator below resolves ~1 cm -- and it is measured from the
sweeps that are being recorded anyway, so it costs nothing.

WHY IT MATTERS. A chair-height error is not visible in any behavioural block:
the dome localization test zeroes the head tracker on the centre speaker and
labels its targets relative to the same centre, so a bodily offset cancels;
and in the AR condition the source is rendered relative to the tracker's zero
AND the response is read in that same frame, so a tracker offset cancels there
too. The one place it does NOT cancel is HERE, in the recording: a listener
sitting `d_vert` too low records the acoustics of a source at elevation E+delta
into the filter LABELLED E. Every later AR rendering then reproduces that
mislabelling, and the listener points delta degrees off, on every trial, for
the rest of the experiment -- a constant elevation bias that no amount of
re-calibration during testing can remove, because the error is baked into the
SOFA.

So this is a recording-time gate, not a test-time one. Run it before the
pipeline spends twenty minutes on a subject whose geometry was wrong.

Measured on the first five subjects to be checked this way (200-1500 Hz band):
GM 4.3, AS 5.2, LS 4.0, NR 2.5, FP 3.9 cm -- i.e. the ears sit a few cm ABOVE
the dome's geometric centre by construction of the chair and headrest, with a
spread of about +-1.5 cm. `EXPECTED_VERTICAL_M` below is that rig constant, not
a target to aim at. The absolute value is only good to ~1.5 cm (it moves a
little with the analysis band); the spread between subjects is what the
tolerance is set from, and it is comfortably tight enough to catch the error
this exists to catch, which is a chair left at someone else's setting.
"""

from pathlib import Path
import json
import logging

import numpy
import scipy.optimize

SPEED_OF_SOUND = 343.0

#: Vertical ear offset this rig produces when the chair IS set correctly, in
#: metres, positive = ears above the dome's geometric centre. Measured, not
#: chosen -- see the module docstring.
EXPECTED_VERTICAL_M = 0.040

#: How far from `EXPECTED_VERTICAL_M` a subject may sit before this is called a
#: setup error. 3 cm is ~1.2 deg of source-direction error at 1.4 m and about
#: twice the observed between-subject spread.
TOLERANCE_M = 0.030

#: Band for the arrival-time estimate. Low enough to be below the pinna's own
#: direction-dependent delays and above the room's modal region; the same band
#: `fit_head_radius` and `zero_frontal_interaural` use for ITD.
DELAY_BAND_HZ = (200.0, 1500.0)


def _relative_delay(signal, reference, samplerate, band):
    """Sub-sample delay of `signal` re `reference`, in microseconds.

    Band-limited cross-correlation with a parabolic fit on the peak. Both are
    recordings of the SAME excitation sweep, so the peak is unambiguous and no
    deconvolution is needed.
    """
    n = len(signal)
    frequencies = numpy.fft.rfftfreq(n, 1 / samplerate)
    weight = ((frequencies >= band[0]) & (frequencies <= band[1])).astype(float)
    spectrum = (numpy.fft.rfft(signal) * numpy.conj(numpy.fft.rfft(reference))
                * weight)
    correlation = numpy.fft.irfft(spectrum, n=n)
    correlation = numpy.concatenate([correlation[-n // 2:], correlation[:n // 2]])
    peak = int(numpy.argmax(numpy.abs(correlation)))
    if 0 < peak < len(correlation) - 1:
        left, centre, right = numpy.abs(correlation[peak - 1:peak + 2])
        shift = 0.5 * (left - right) / (left - 2 * centre + right + 1e-30)
    else:
        shift = 0.0
    return ((peak + shift) - n // 2) / samplerate * 1e6


def fit_head_position(recordings, distance=1.4, band=DELAY_BAND_HZ):
    """Fit the listener's ear position relative to the dome centre.

    Parameters
    ----------
    recordings : Recordings
        The subject's raw midline sweeps, as returned by
        `Recordings.record_dome` / `Recordings.load`. Only az ~ 0 directions
        are used.
    distance : float
        Dome radius in metres. Only scales the reported angle, not the fit.
    band : (float, float)
        Analysis band in Hz.

    Returns
    -------
    dict with `vertical_m` (positive = ears above the dome centre, i.e. chair
    too high), `forward_m` (positive = ears forward of centre), `residual_us`,
    the per-direction `elevations` and `delays_us`, and `angular_error_deg`,
    the direction error the vertical offset implies at `distance`.

    The two ears are averaged before the delay is measured: on the midline they
    are symmetric about the median plane, so their mean cancels any LATERAL
    offset of the head and leaves the vertical and fore/aft terms the model
    actually contains.
    """
    elevations, traces, samplerate = [], [], None
    for key, repeats in recordings.items():
        _, azimuth, elevation = recordings.parse_key(key)
        if abs(azimuth) > 1.0:
            continue
        data = numpy.asarray([numpy.asarray(rep.data) for rep in repeats])
        samplerate = samplerate or int(repeats[0].samplerate)
        elevations.append(elevation)
        traces.append(data.mean(axis=0).mean(axis=-1))   # over repeats, then ears
    if len(elevations) < 4:
        raise ValueError(
            f"need at least 4 midline directions to fit a head position, got "
            f"{len(elevations)}")

    elevations = numpy.asarray(elevations, dtype=float)
    traces = numpy.asarray(traces)
    order = numpy.argsort(elevations)
    elevations, traces = elevations[order], traces[order]

    reference = traces[int(numpy.argmin(numpy.abs(elevations)))]
    delays = numpy.asarray([_relative_delay(trace, reference, samplerate, band)
                            for trace in traces])

    radians = numpy.radians(elevations)

    def residual(params):
        offset, vertical, forward = params
        model = (offset
                 - vertical / SPEED_OF_SOUND * numpy.sin(radians) * 1e6
                 - forward / SPEED_OF_SOUND * (numpy.cos(radians) - 1) * 1e6)
        return model - delays

    solution = scipy.optimize.least_squares(residual, [0.0, 0.0, 0.0])
    _, vertical, forward = solution.x
    return {
        "vertical_m": float(vertical),
        "forward_m": float(forward),
        "angular_error_deg": float(numpy.degrees(numpy.arctan2(vertical, distance))),
        "residual_us": float(numpy.sqrt(numpy.mean(solution.fun ** 2))),
        "elevations": elevations.tolist(),
        "delays_us": delays.tolist(),
        "band_hz": list(band),
        "distance_m": float(distance),
    }


def check_head_position(fit, expected=EXPECTED_VERTICAL_M, tolerance=TOLERANCE_M,
                        subject_id=None):
    """True if the listener sat where they should have, else False, loudly.

    Does NOT raise: the recording already exists by the time this can run, and
    throwing it away automatically is worse than telling the experimenter what
    happened. The point is that the message appears BEFORE the pipeline builds
    a SOFA nobody will question later.
    """
    deviation = fit["vertical_m"] - expected
    if abs(deviation) <= tolerance:
        logging.info(
            "head position OK: ears %.1f cm above the dome centre "
            "(expected %.1f +- %.1f), residual %.0f us",
            fit["vertical_m"] * 100, expected * 100, tolerance * 100,
            fit["residual_us"])
        return True
    logging.error(
        "CHAIR HEIGHT IS WRONG for %s -- the ears sat %.1f cm above the dome "
        "centre, %+.1f cm off the %.1f cm this rig gives when the chair is "
        "right. That is %+.1f deg of source-direction error, and it will be "
        "BAKED INTO THE SOFA as a constant elevation bias on every AR trial "
        "this subject ever runs (it cancels in the dome test, so you will not "
        "see it there). Fix the chair and re-record with overwrite_rec=True.",
        subject_id or "this subject", fit["vertical_m"] * 100, deviation * 100,
        expected * 100, fit["angular_error_deg"] - numpy.degrees(
            numpy.arctan2(expected, fit["distance_m"])))
    return False


def save_head_position(fit, directory, filename="head_position_fit.json"):
    """Write the fit next to the recordings it was made from."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    with path.open("w") as handle:
        json.dump(fit, handle, indent=2)
    return path
