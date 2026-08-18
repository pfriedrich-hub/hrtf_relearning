"""Distance law of the training game: how far the head is from the target,
turned into an inter-pulse interval.

Shared by both training conditions so they stay acoustically comparable.
`distance_to_interval` returns MILLISECONDS: `Training_Dome` writes that
straight to the `interval` tag of the pulse circuit, `Training_AR` divides by
1000 for the pyBinSim pulse worker.

Two mappings:

'ar'
    log1p ramp from `min_pulse_interval` at the edge of the target window to
    `max_pulse_interval` at the far corner of the target area, and exactly 0
    inside the window -- both games read 0 as "play continuously", which is the
    same signal as the scoring criterion. Normalised by `target_size` and by the
    extent of the target area, so changing the window or switching between a
    midline and a whole-dome/hemifield area rescales the ramp instead of
    silently changing the difficulty.

'legacy'
    the mapping of the original dome script: `max_pulse_interval` scaled by
    (log((d - 2) / max_distance + 0.05) + 3) / 3. Kept so the feel of the
    earlier dome sessions can be reproduced. Clamped at 0 -- the original took
    the log of a negative number inside 2 degrees and wrote the resulting
    negative interval to the processors.

`settings` is the game settings dict of either condition: it must carry
`target_size` and the target area, as `az_range`/`ele_range` or
`azimuth_range`/`elevation_range`. `pulse_map`, `max_pulse_interval` and
`min_pulse_interval` are optional and fall back to `PULSE_DEFAULTS`.
"""
import numpy

PULSE_DEFAULTS = {
    'ar':     dict(max_pulse_interval=350, min_pulse_interval=75),   # ms
    'legacy': dict(max_pulse_interval=500, min_pulse_interval=0),    # ms
}

# steepness of the 'ar' log1p ramp
STEEPNESS = 5


def target_area_extent(settings):
    """Distance in degrees from straight ahead to the far corner of the target area."""
    az_range = settings.get('az_range', settings.get('azimuth_range'))
    ele_range = settings.get('ele_range', settings.get('elevation_range'))
    if az_range is None or ele_range is None:
        raise KeyError("settings needs 'az_range'/'ele_range' (or "
                       "'azimuth_range'/'elevation_range') to scale the pulse interval")
    return float(numpy.linalg.norm(numpy.subtract([0, 0], [az_range[0], ele_range[0]])))


def distance_to_interval(distance, settings):
    """Inter-pulse interval in MILLISECONDS for a head-to-target distance in degrees."""
    mapping = settings.get('pulse_map', 'ar')
    if mapping not in PULSE_DEFAULTS:
        raise ValueError(f"pulse_map must be one of {list(PULSE_DEFAULTS)}, got {mapping!r}")
    defaults = PULSE_DEFAULTS[mapping]
    max_interval = settings.get('max_pulse_interval') or defaults['max_pulse_interval']
    min_interval = settings.get('min_pulse_interval')
    if min_interval is None:
        min_interval = defaults['min_pulse_interval']
    max_distance = target_area_extent(settings)

    if mapping == 'legacy':
        interval_scale = (distance - 2 + 1e-9) / max_distance
        interval = max_interval * (numpy.log(max(interval_scale + 0.05, 1e-9)) + 3) / 3
        return float(max(0.0, interval))

    if distance <= settings['target_size']:
        return 0.0
    norm_dist = (distance - settings['target_size']) / (max_distance - settings['target_size'])
    norm_dist = float(numpy.clip(norm_dist, 0, 1))
    scale = numpy.log1p(STEEPNESS * norm_dist) / numpy.log1p(STEEPNESS)
    return float(min_interval + (max_interval - min_interval) * scale)
