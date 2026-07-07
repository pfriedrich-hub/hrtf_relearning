"""
stimulus.py

Single source of truth for the localization test stimulus, so the dome
(real loudspeaker) and AR/VR (pybinsim) conditions play the *same* stimulus and
only the transducer/level path differs.

`make_gapped_pinknoise` reproduces the original Localization_AR synthesis:
225 ms pinknoise (10 ms cosine edges) with four 25 ms gaps zeroed out at
25-50, 75-100, 125-150, 175-200 ms, each gap edge given a 5 ms raised-cosine
ramp. This yields five ~25 ms flat bursts drawn from one continuous noise
realization (independent per burst), 4x25 ms gaps -> 225 ms total.

Absolute loudness is handled downstream, NOT here:
  - dome:   freefield speaker calibration (equalize=True)
  - AR/VR:  the matched pybinsim gain (loc_settings['gain'])
so callers pick `level` only to set the WAV/DAC reference; matching the two
transducer paths is a separate by-ear step (match_ar_dome_loudness.py).
"""
import numpy
import slab


def make_gapped_pinknoise(level=80):
    """225 ms gapped pinknoise (5x25 ms bursts, 4x25 ms gaps). Uses the slab
    default samplerate, so callers should set it before calling."""
    stim = slab.Sound.pinknoise(duration=0.225, level=level).ramp(when='both', duration=0.01)
    n_silent = (numpy.arange(25, 221, 25).reshape(4, 2) * stim.samplerate / 1000).astype(int)
    ramp_len = int(.005 * stim.samplerate)
    half_len = int(ramp_len / 2)
    for start, end in n_silent:
        ramp_up = 0.5 * (1 - numpy.cos(numpy.linspace(0, numpy.pi, ramp_len)))
        ramp_down = 0.5 * (1 - numpy.cos(numpy.linspace(numpy.pi, 0, ramp_len)))
        ramp_up = ramp_up[:, numpy.newaxis]
        ramp_down = ramp_down[:, numpy.newaxis]
        # ramp the noise down into / up out of each gap, then silence the middle
        stim.data[start - half_len: start + half_len] *= (1 - ramp_up)
        stim.data[end - half_len: end + half_len] *= (1 - ramp_down)
        stim.data[start + half_len: end - half_len] = 0
    return stim
