"""
ar_level_match.py

OBJECTIVE loudness match between the DOME localization test (real loudspeakers
via freefield) and the AR localization test (pybinsim over headphones), measured
through the in-ear mics.

This is the measurement `match_ar_dome_loudness.py` has a TODO for. Unlike
`hp_vs_dome_check.py` and `HRIR_Recording.acoustic_test`, which measure the
freefield `play_and_record_headphones` path, this one measures the REAL AR
chain: the binsim filter set (reverb, DRR, hp_filter), the WAV RMS, the runtime
`/pyBinSimLoudness` gain and the OS master fader. A number from here is
therefore a drop-in for `loc_settings['gain']`; a number from those two is not.

How the capture works
---------------------
pybinsim plays asynchronously over OSC and the TDT records on its own trigger,
so the two cannot be sample-aligned. They do not need to be: the probe is a
LONG stationary pinknoise (default 4 s) and the capture is a SHORTER silent
`play_and_record` (default 1.5 s) started inside it. Whatever the latency, the
captured window sits in steady state. The window is then trimmed to its steady
portion by a sliding-window envelope before the level is taken, so a late start
or an early end costs accuracy rather than correctness.

The dome leg needs none of that -- the TDT plays and records on one trigger --
so it captures the same probe directly.

Routing prerequisite
--------------------
The headphones must be on the SOUNDCARD (as in the real AR test) while the
in-ear mics are on the TDT inputs. If the rig cannot do both at once, this
measurement cannot run as written; `measure_ar_dome_level` checks for that
rather than returning a confident wrong number -- see the noise-floor gate.

NOT VALIDATED ON HARDWARE. Written against the OSC/worker API in
`Localization_AR.py` and the freefield calls used elsewhere in the package, but
never executed on the rig. Run `probe_binsim_capture()` first.
"""

import contextlib
import logging
import multiprocessing as mp
import time

import numpy
import slab
import freefield

from hrtf_relearning.utils import paths

DEFAULT_BAND = (200.0, 16000.0)   # same convention as ild_band in record/processing.py

#: Level the AR test writes its stimulus WAV at -- `Localization.make_stim()`
#: ends with `stim.level = 80`. The probe must use the same value or the
#: measured gain will not transfer to the real test.
AR_WAV_LEVEL_DB = 80.0

#: Below this SNR the mics are almost certainly not hearing the headphones,
#: i.e. the routing is wrong. Measured against a gain=0 capture.
MIN_SNR_DB = 15.0


# ---------------------------------------------------------------------------
# level helpers
# ---------------------------------------------------------------------------

def band_level_db(x, samplerate, band=DEFAULT_BAND):
    """Parseval-normalised band power of a 1-D signal, in dB.

    Mean-square of the band-limited signal, so the value is a physical level
    and does not scale with the transform length -- the dome and binsim
    captures can differ in length after steady-state trimming.
    """
    x = numpy.asarray(x, dtype=float)
    n = len(x)
    spec = numpy.abs(numpy.fft.rfft(x)) ** 2
    freqs = numpy.fft.rfftfreq(n, 1.0 / samplerate)
    sel = (freqs >= band[0]) & (freqs <= band[1])
    if not sel.any():
        raise ValueError(f'band {band} selects no frequency bins at fs={samplerate}')
    return float(10 * numpy.log10(2.0 * spec[sel].sum() / n ** 2 + 1e-30))


def steady_slice(x, samplerate, win_s=0.05, rel_threshold=0.5):
    """Longest run of `x` whose short-time RMS stays above `rel_threshold` of peak.

    Returns (slice, fraction_of_input). Used to trim a capture that may start
    before, or end after, the pybinsim probe -- the level is then taken over
    the part that actually carries the probe.
    """
    x = numpy.asarray(x, dtype=float)
    hop = max(1, int(win_s * samplerate))
    n_win = len(x) // hop
    if n_win < 3:
        return slice(0, len(x)), 1.0
    env = numpy.sqrt((x[:n_win * hop].reshape(n_win, hop) ** 2).mean(axis=1))
    if env.max() <= 0:                      # nothing arrived at all
        return slice(0, len(x)), 0.0
    above = env >= rel_threshold * env.max()

    best_len = best_start = 0
    run_start = None
    for i, flag in enumerate(numpy.append(above, False)):
        if flag and run_start is None:
            run_start = i
        elif not flag and run_start is not None:
            if i - run_start > best_len:
                best_len, best_start = i - run_start, run_start
            run_start = None
    if best_len == 0:
        return slice(0, len(x)), 1.0
    return (slice(best_start * hop, (best_start + best_len) * hop),
            best_len / n_win)


# ---------------------------------------------------------------------------
# pybinsim session
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def binsim_session(subject, hrir_settings, loc_settings, os_volume=50):
    """Build the binsim files, start the worker and OSC clients, tear down after.

    Yields (loc, osc_filter, osc_play). Mirrors what `Localization.run()` does
    around a test, minus the motion sensor. `os_volume` is pinned because
    pybinsim SPL scales with the OS fader -- the match is only valid if the
    real test runs at the same value.
    """
    from hrtf_relearning.experiment.localization.Localization_AR import Localization
    from hrtf_relearning.experiment.misc.system_volume import set_windows_volume

    set_windows_volume(os_volume)
    loc = Localization(subject, hrir_settings, loc_settings)

    osc_filter = loc._make_osc_client(port=10000)   # /pyBinSim_ds_Filter
    osc_play = loc._make_osc_client(port=10003)     # /pyBinSimLoudness, /pyBinSimFile
    worker = mp.Process(target=loc._binsim_stream, args=(loc.hrir_name,))
    worker.start()
    time.sleep(1.5)   # let the stream come up
    try:
        yield loc, osc_filter, osc_play
    finally:
        try:
            osc_play.send_message('/pyBinSimLoudness', 0)
            time.sleep(0.1)
        except Exception:
            pass
        worker.terminate()
        worker.join(timeout=3)
        if worker.is_alive():
            worker.kill()
            worker.join()


def _filter_index(loc, source):
    """Index of the binsim filter for `source` = (az, el), mirroring play_stimulus."""
    rel_az = (-float(source[0]) + 360) % 360
    rel = numpy.array((rel_az, float(source[1]), loc.hrir_sources[0, 2]))
    return int(numpy.argmin(numpy.linalg.norm(rel - loc.hrir_sources, axis=1)))


def _capture_silent(duration, samplerate, speaker=(0, 0)):
    """Record the in-ear mics for `duration` without playing anything audible.

    freefield has no record-only entry point, so this plays digital silence
    from a dome speaker on the same trigger that arms the recording buffer.
    Requires the dome initialised with default='play_birec'.
    """
    silence = slab.Sound.silence(duration=duration, samplerate=samplerate)
    spk = freefield.pick_speakers(speaker)[0]
    return freefield.play_and_record(
        spk, silence, compensate_delay=False, compensate_attenuation=False,
        equalize=False, recording_samplerate=samplerate)


# ---------------------------------------------------------------------------
# the measurement
# ---------------------------------------------------------------------------

def probe_binsim_capture(loc, osc_filter, osc_play, gain=0.2, source=(0.0, 0.0),
                         probe_duration=4.0, capture_duration=1.5, settle=0.75,
                         band=DEFAULT_BAND, samplerate=None):
    """Routing check: is anything from pybinsim reaching the in-ear mics?

    Run this BEFORE `measure_ar_dome_level` on a rig where the headphone/mic
    routing has not been verified. Returns a dict with the signal and
    noise-floor band levels and their difference; an SNR below `MIN_SNR_DB`
    means the mics are not hearing the headphones (headphones still on the TDT
    output, wrong soundcard device, gain 0, OS muted).
    """
    fs = int(samplerate or loc.samplerate)
    probe = _write_probe(loc, probe_duration, fs)
    idx = _filter_index(loc, source)

    signal = _play_and_capture(loc, osc_filter, osc_play, idx, probe, gain,
                               capture_duration, settle, fs)
    floor = _capture_silent(capture_duration, fs)

    out = {'band_hz': list(band), 'gain': float(gain), 'source': list(source)}
    for ear, name in enumerate(('left', 'right')):
        sig_db = band_level_db(numpy.asarray(signal.data)[:, ear], fs, band)
        flr_db = band_level_db(numpy.asarray(floor.data)[:, ear], fs, band)
        out[name] = {'signal_db': sig_db, 'floor_db': flr_db, 'snr_db': sig_db - flr_db}
        logging.info('%-5s  signal %7.2f dB   floor %7.2f dB   SNR %6.2f dB',
                     name, sig_db, flr_db, sig_db - flr_db)
    worst = min(out['left']['snr_db'], out['right']['snr_db'])
    out['ok'] = bool(worst >= MIN_SNR_DB)
    if not out['ok']:
        logging.error(
            'pybinsim is not reaching the in-ear mics (worst SNR %.1f dB < %.1f). '
            'Check that the headphones are on the SOUNDCARD, the OS volume is up '
            'and unmuted, and the binsim worker is running.', worst, MIN_SNR_DB)
    return out


def _write_probe(loc, duration, fs, level_db=AR_WAV_LEVEL_DB):
    """Long stationary pinknoise at the AR test's own WAV level."""
    probe = slab.Sound.pinknoise(duration=duration, samplerate=fs)
    probe = probe.ramp(when='both', duration=0.05)
    probe.level = level_db
    path = loc.sound_path / 'ar_level_probe.wav'
    probe.write(path)
    return path


def _play_and_capture(loc, osc_filter, osc_play, filter_idx, probe_path, gain,
                      capture_duration, settle, fs):
    """Start the long probe, wait `settle`, capture a shorter window inside it."""
    src = loc.hrir_sources[filter_idx]
    osc_filter.send_message('/pyBinSim_ds_Filter',
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             float(src[0]), float(src[1]), 0, 0, 0, 0])
    time.sleep(0.1)
    osc_play.send_message('/pyBinSimLoudness', float(gain))
    osc_play.send_message('/pyBinSimFile', str(probe_path))
    time.sleep(settle)
    try:
        return _capture_silent(capture_duration, fs)
    finally:
        osc_play.send_message('/pyBinSimLoudness', 0)


def measure_ar_dome_level(loc, osc_filter, osc_play, *,
                          sources=((0.0, 0.0),),
                          gain=None,
                          band=DEFAULT_BAND,
                          probe_duration=4.0,
                          capture_duration=1.5,
                          settle=0.75,
                          dome_level_db=None,
                          samplerate=None,
                          check_routing=True,
                          prompt=input):
    """Measure dome-minus-pybinsim level per source and per ear.

    Both legs are measured with the SAME stationary pinknoise probe, so the
    result is a pure chain difference and carries no stimulus-shape assumption.

    Parameters
    ----------
    loc, osc_filter, osc_play
        A live `Localization` and its OSC clients -- from `binsim_session`, or
        the ones `match_ar_dome_loudness.py` already sets up. The dome must be
        initialised with ``default='play_birec'`` (both in-ear mics).
    sources
        (az, el) pairs to measure. One frontal source is enough to derive the
        gain; more gives a direction-resolved check that no elevation is
        rendered at the wrong level.
    gain
        pybinsim gain to measure at. Defaults to ``loc.settings['gain']``.
    dome_level_db
        Level to play the dome probe at. Default None reads the dome test's own
        stimulus level from ``LocalizationDome.make_stim()``, so the comparison
        is against what a participant actually gets on the dome.
    check_routing
        Run `probe_binsim_capture` first and refuse to report a number if the
        mics are not hearing the headphones.
    prompt
        Called once before each leg with the instruction text. Default `input`
        blocks for Enter; pass `print` (or a no-op) to run unattended if the
        headphones can stay in place for both legs.

    Returns
    -------
    dict with, per source and ear, the dome and binsim band levels and their
    difference, plus:

    ``gain_suggested``
        ``gain * 10 ** (offset_db_mean / 20)`` -- **this is the number to put in
        `loc_settings['gain']`.** pybinsim applies the loudness value as a
        linear multiplier, so scaling it by the measured ratio closes the gap.
    ``warnings``
        Non-empty if the suggested gain exceeds 1.0 (headroom -- see
        REFERENCE_LEVEL in write_filters), if the steady fraction of any
        capture was low (probe/capture timing), or if routing was not checked.
    """
    from hrtf_relearning.experiment.localization.Localization_dome import LocalizationDome

    fs = int(samplerate or loc.samplerate)
    gain = float(loc.settings['gain'] if gain is None else gain)
    warnings = []

    routing = None
    if not check_routing:
        warnings.append('routing not checked')

    if dome_level_db is None:
        dome_level_db = float(numpy.mean(LocalizationDome.make_stim().level))

    dome_probe = slab.Sound.pinknoise(duration=capture_duration, samplerate=fs)
    dome_probe = dome_probe.ramp(when='both', duration=0.05)
    dome_probe.level = dome_level_db

    binsim_probe = _write_probe(loc, probe_duration, fs)
    indices = {src: _filter_index(loc, src) for src in sources}

    # --- phase 1: every source through pybinsim, headphones ON --------------
    # Both legs are grouped so the headphones are put on and taken off once,
    # not once per source: re-seating them between sources would add the
    # re-seat spread (~1 dB rms, see DT990_hp_repeats) to the comparison.
    prompt('Headphones ON (fed by the soundcard), mics in. Enter to measure pybinsim...')
    if check_routing:
        routing = probe_binsim_capture(loc, osc_filter, osc_play, gain=gain,
                                       source=sources[0], probe_duration=probe_duration,
                                       capture_duration=capture_duration,
                                       settle=settle, band=band, samplerate=fs)
        if not routing['ok']:
            raise RuntimeError(
                'pybinsim output is not reaching the in-ear mics -- see the log. '
                'Fix the routing, or pass check_routing=False to measure anyway.')
    binsim_rec = {}
    for src in sources:
        binsim_rec[src] = _play_and_capture(loc, osc_filter, osc_play, indices[src],
                                            binsim_probe, gain, capture_duration,
                                            settle, fs)

    # --- phase 2: every source from the dome, headphones OFF ----------------
    prompt('Headphones OFF, mics still in. Enter to measure the dome...')
    dome_rec = {}
    for src in sources:
        spk = freefield.pick_speakers(src)[0]
        dome_rec[src] = freefield.play_and_record(
            spk, dome_probe, compensate_delay=True, compensate_attenuation=False,
            equalize=True, recording_samplerate=fs)

    per_source = {}
    for src in sources:
        key = f'{float(src[0]):.1f}_{float(src[1]):.1f}'
        entry = {'filter_source': loc.hrir_sources[indices[src]].tolist(),
                 'dome_db': [], 'binsim_db': [], 'offset_db': [],
                 'steady_fraction': []}
        for ear in (0, 1):
            b = numpy.asarray(binsim_rec[src].data)[:, ear]
            sl, frac = steady_slice(b, fs)
            b_db = band_level_db(b[sl], fs, band)
            d_db = band_level_db(numpy.asarray(dome_rec[src].data)[:, ear], fs, band)
            entry['binsim_db'].append(b_db)
            entry['dome_db'].append(d_db)
            entry['offset_db'].append(d_db - b_db)
            entry['steady_fraction'].append(float(frac))
            if frac < 0.5:
                warnings.append(
                    f'{key} ear {ear}: only {frac:.0%} of the capture was steady '
                    '-- increase probe_duration or reduce capture_duration')
        per_source[key] = entry
        logging.info('%-12s dome %7.2f/%7.2f  binsim %7.2f/%7.2f  offset %+6.2f/%+6.2f dB',
                     key, *entry['dome_db'], *entry['binsim_db'], *entry['offset_db'])

    per_ear = numpy.array([per_source[k]['offset_db'] for k in per_source])
    median_per_ear = numpy.median(per_ear, axis=0)
    offset_mean = float(numpy.median(per_ear))
    gain_suggested = gain * 10 ** (offset_mean / 20.0)

    if gain_suggested > 1.0:
        warnings.append(
            f'suggested gain {gain_suggested:.3f} > 1.0 -- pybinsim would need more '
            'headroom than it has; raise the OS volume or the filter REFERENCE_LEVEL '
            'instead of the gain')

    result = {
        'band_hz': list(band),
        'sources': [list(s) for s in sources],
        'per_source': per_source,
        'offset_db_median_per_ear': median_per_ear.tolist(),
        'offset_db_mean': offset_mean,
        'offset_db_spread': float(per_ear.max() - per_ear.min()),
        'gain_used': gain,
        'gain_suggested': float(gain_suggested),
        'levels': {'binsim_wav_db': AR_WAV_LEVEL_DB, 'dome_probe_db': float(dome_level_db)},
        'hrir_name': loc.hrir_name,
        'routing_check': routing,
        'path': 'pybinsim (reverb, hp_filter, WAV RMS, gain, OS fader) vs dome',
        'warnings': warnings,
    }

    logging.info('median per ear: L %+.2f  R %+.2f dB  |  overall %+.2f dB '
                 '(spread %.2f dB)', median_per_ear[0], median_per_ear[1],
                 offset_mean, result['offset_db_spread'])
    logging.info("-> loc_settings['gain'] = %.4f  (was %.4f). Valid only at the OS "
                 'volume this was measured at.', gain_suggested, gain)
    for w in warnings:
        logging.warning('%s', w)
    return result
