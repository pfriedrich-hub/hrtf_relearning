"""Nearest pyBinSim filter for a head-relative source direction.

All AR/VR scripts used to pick the filter with

    numpy.argmin(numpy.linalg.norm(rel_target - sources, axis=1))

on (az, el, r) with az in [0, 360). That distance does not wrap at 360 deg:
a relative azimuth just below 360 (target 0-2 deg to the RIGHT of the nose)
is ~360 deg from the 0 deg filter, so it always snapped to the last grid
azimuth (355.75 deg on the current 4.2 deg grid) -- rendering the source one
grid step (4.25 deg) to the right whenever the listener was on target.
Found 2026-10-21 from SK's report; in her training holds 34% of samples were
affected. Mirror-image error on the left side does not exist (0 deg is the
first grid azimuth, so small positive azimuths already map to 0 correctly).
"""
import numpy


def nearest_filter_idx(sources, az, el):
    """Index into `sources` (N, 3; vertical-polar, az in [0, 360), HRTF
    convention) of the filter nearest to (az, el), with az wrap-safe."""
    sources = numpy.asarray(sources, dtype=float)
    d_az = (float(az) - sources[:, 0] + 180) % 360 - 180
    d_el = float(el) - sources[:, 1]
    return int(numpy.argmin(numpy.hypot(d_az, d_el)))
