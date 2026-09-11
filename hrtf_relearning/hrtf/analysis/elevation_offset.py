"""Is the SOFA's elevation LABELLING right? Measured, not argued.

A constant elevation bias in the AR condition that is absent on the dome has
exactly two homes: the SOFA (its filters are labelled with the wrong elevation,
so rendering "0 deg" delivers the acoustics of some other direction) or the
headphone chain (the HP inverse filter leaves a residual in the 5-13 kHz band
that the elevation percept reads). Nothing in the behavioural data separates
them -- both produce a constant offset that survives re-calibration, room
changes and headphone re-seats -- and nothing in the recordings on disk does
either: a rigid relabelling of one listener's DTF surface is not resolvable
against between-listener pinna variability (tried on n=5; the shift estimates
scatter by +-7 deg, which is larger than the effect).

`acoustic_test` in HRIR_Recording.py separates them in one measurement, because
it captures BOTH legs through the SAME in-ear mics in the SAME session: the
dome leg (real speakers, known elevations) and the HP+HRIR leg (the rendering).
That makes it a within-subject, within-session comparison with none of the
variability that defeats the cross-subject version. This module turns its two
output dicts into a number.

    hp_rec, dome_rec, level_match = acoustic_test(...)
    report = elevation_offset(hp_rec, dome_rec)
    print(format_offset(report))

If `offset_deg` comes back near zero, the SOFA's labels are right and a
behavioural bias is the headphone chain (or the listener). If it comes back at
the size of the behavioural bias and with the same sign, the SOFA is
mislabelled and the subject has to be re-recorded -- no amount of testing-side
calibration can fix it, because every AR trial replays the mislabelled filter.

NOTE ON SAMPLING. `acoustic_test` defaults to every third midline source, i.e.
12.5 deg spacing, because the dome leg can only play from physical speakers.
That is coarse for estimating an offset of under ~10 deg: the fit interpolates
between them and the peak gets broad. If this is the question being asked,
run acoustic_test over every source it can reach and expect the confidence
interval below to tighten accordingly.
"""

import logging
import re

import numpy

#: The elevation cue lives here. Same band `donor_selection` and the notch work
#: use; below it the torso and head shadow dominate, above it the mics and the
#: HP inverse filter are both unreliable.
PINNA_BAND_HZ = (5000.0, 13000.0)

#: Torso/shoulder reflections. Useful as a CONTROL: they are generated below
#: the pinna and do not move when the head pitches, so a real SOFA mislabelling
#: caused by head pitch shows an offset in the pinna band and none here, while
#: a whole-body displacement shows one in both.
TORSO_BAND_HZ = (700.0, 2500.0)


def _parse_elevation(key):
    """Elevation out of an `acoustic_test` key, which is `str(vertical_polar row)`."""
    numbers = re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", str(key))
    if len(numbers) < 2:
        raise ValueError(f"cannot read an elevation from key {key!r}")
    return float(numbers[1])


def _surface(recordings, band, smooth_octaves=1 / 8):
    """(elevations, ears x frequency spectra) from one leg of the test.

    Each direction is level-normalised inside `band` and the across-direction
    mean is removed, so what is compared is the direction-DEPENDENT shape --
    the same quantity a DTF holds. A constant gain difference between the two
    legs, which `acoustic_test` measures separately and which is exactly what
    `match_level` exists for, therefore cannot influence this at all.
    """
    elevations, spectra = [], []
    samplerate = None
    for key, sound in recordings.items():
        data = numpy.asarray(sound.data, dtype=float)
        samplerate = samplerate or float(sound.samplerate)
        n = len(data)
        frequencies = numpy.fft.rfftfreq(n, 1.0 / samplerate)
        magnitude = 20 * numpy.log10(numpy.abs(numpy.fft.rfft(data, axis=0)) + 1e-12)
        elevations.append(_parse_elevation(key))
        spectra.append(magnitude.T)                      # (ear, frequency)
    elevations = numpy.asarray(elevations)
    spectra = numpy.asarray(spectra)                     # (direction, ear, frequency)

    smoothed = numpy.empty_like(spectra)
    for i, centre in enumerate(frequencies):
        selection = ((frequencies >= centre * 2 ** -smooth_octaves)
                     & (frequencies <= centre * 2 ** smooth_octaves))
        smoothed[..., i] = (spectra[..., selection].mean(axis=-1) if selection.any()
                            else spectra[..., i])

    order = numpy.argsort(elevations)
    elevations, smoothed = elevations[order], smoothed[order]
    selection = (frequencies >= band[0]) & (frequencies <= band[1])
    surface = smoothed[..., selection]
    surface = surface - surface.mean(axis=-1, keepdims=True)      # per-direction level
    surface = surface - surface.mean(axis=0, keepdims=True)       # across-direction mean
    return elevations, surface


def _correlation(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    return float(a @ b / (numpy.linalg.norm(a) * numpy.linalg.norm(b) + 1e-30))


def elevation_offset(hp_recordings, dome_recordings, band=PINNA_BAND_HZ,
                     max_offset_deg=20.0, step_deg=0.25):
    """How far the HP+HRIR rendering sits from the dome, in degrees of elevation.

    Returns a dict with `offset_deg` (positive = the rendering delivers the
    acoustics of a HIGHER elevation than the label says, which makes the
    listener point UP and produces a POSITIVE behavioural intercept),
    `peak_correlation`, the full `profile` of correlation against offset, and
    `half_width_deg`, the half-width of the peak at 0.05 below its maximum --
    read that as the resolution this particular measurement supports, not as a
    confidence interval in the statistical sense.
    """
    dome_elevations, dome_surface = _surface(dome_recordings, band)
    hp_elevations, hp_surface = _surface(hp_recordings, band)

    spacing = float(numpy.median(numpy.diff(dome_elevations)))
    if spacing > 8.0:
        logging.warning(
            "acoustic_test sampled elevation every %.1f deg; an offset smaller "
            "than that is being read off an interpolation. Re-run over more "
            "sources if the number matters.", spacing)

    offsets = numpy.arange(-max_offset_deg, max_offset_deg + step_deg / 2, step_deg)
    profile = []
    for offset in offsets:
        # `offset` is a property of the RENDERING: at offset d the HP leg's
        # filter labelled e holds the acoustics of e + d. To line that up with
        # the dome's content at e, sample the HP surface at e - d. Getting this
        # backwards inverts the reported sign, which is the one thing this
        # number must not do -- the sign is what says which way the listener
        # points. Validated by injecting a known shift; see the module tests.
        shifted = numpy.stack(
            [[numpy.interp(dome_elevations - offset, hp_elevations, hp_surface[:, ear, j])
              for j in range(hp_surface.shape[2])]
             for ear in range(hp_surface.shape[1])], axis=0)          # (ear, freq, dir)
        profile.append(numpy.mean([_correlation(shifted[ear].T, dome_surface[:, ear, :])
                                   for ear in range(hp_surface.shape[1])]))
    profile = numpy.asarray(profile)
    peak = int(numpy.argmax(profile))
    above = numpy.flatnonzero(profile >= profile[peak] - 0.05)
    return {
        "offset_deg": float(offsets[peak]),
        "peak_correlation": float(profile[peak]),
        "half_width_deg": float((offsets[above[-1]] - offsets[above[0]]) / 2),
        "profile": {"offsets_deg": offsets.tolist(),
                    "correlation": profile.tolist()},
        "band_hz": list(band),
        "elevation_spacing_deg": spacing,
        "n_directions": int(len(dome_elevations)),
    }


def format_offset(report, control=None):
    """One-screen summary. `control` = the same call made in TORSO_BAND_HZ."""
    lines = [
        f"HP+HRIR vs dome, {report['band_hz'][0]/1000:.0f}-"
        f"{report['band_hz'][1]/1000:.0f} kHz, {report['n_directions']} directions "
        f"at {report['elevation_spacing_deg']:.1f} deg spacing",
        f"  elevation offset  {report['offset_deg']:+.1f} deg "
        f"(+-{report['half_width_deg']:.1f}), peak r = {report['peak_correlation']:.2f}",
    ]
    if control is not None:
        lines.append(f"  torso-band control {control['offset_deg']:+.1f} deg, "
                     f"peak r = {control['peak_correlation']:.2f}")
        lines.append(
            "  -> pinna offset with no torso offset = the HEAD was pitched during "
            "the recording; both offset together = the whole listener was displaced")
    if report["peak_correlation"] < 0.4:
        lines.append("  WARNING: the two legs barely correlate -- check the level "
                     "match, the HP filter and that both legs used the same "
                     "equalize_dome before believing the offset")
    return "\n".join(lines)
