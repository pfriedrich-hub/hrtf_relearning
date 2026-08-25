"""
HRIR_Recording.py

First-session pipeline: record an individual HRIR, calibrate headphones, then
localization-test dome vs. virtual (pybinsim) rendering.

Run cell by cell (# %%) in an IDE/console -- do NOT run this top-to-bottom as
a plain script. Nothing here loops or blocks on input; rerun any cell as
needed (e.g. redo the dome localization, rerun VR localization on a
different headphone profile).

Steps:
    1. Record (or load) HRIR
    2. Calibrate headphones       (mics still in)
    3. Acoustic sanity check      (dome speaker vs HRIR rendering, spectrum comparison)  [optional]
    4. Dome localization          (real speakers, vertical midline)
    4b. Dome training             (only if step 4 is at floor)                 [optional]
    5. Virtual localization       (pybinsim, same locations, independent randomisation)
"""

# %% imports and config ------------------------------------------------------
import matplotlib
from hrtf_relearning.utils.mpl_backend import use_interactive
use_interactive()
from matplotlib import pyplot as plt
import numpy
import copy
import logging
import freefield
import slab
import hrtf_relearning as hr
from hrtf_relearning.experiment.localization.Localization_AR import Localization
from hrtf_relearning.experiment.localization.Localization_dome import LocalizationDome
from hrtf_relearning.experiment.training.Training_Dome import TrainingDome
from hrtf_relearning.hrtf.record.record_hrir import record_hrir, record_reference
from hrtf_relearning.hrtf.record.recordings import Recordings
from hrtf_relearning.hrtf.record.fit_head_radius import (
    record_head_radius, usable_radius, fit_from_sofa, FALLBACK_RADIUS_M)
from hrtf_relearning.hrtf.record.calibration.calibrate_headphones import calibrate_headphones
from hrtf_relearning.utils import paths
import json

SUBJECT_ID   = 'NR'          # edit per participant
REFERENCE_ID = 'ref_20.08'   # fresh id -> step 0b records it; reused id -> loaded
EQUALIZE_DOME = False        # subject AND reference; they must match. See step 0b.
HEAD_RADIUS = 0.075          # fallback
N_DIRECTIONS = 3              # directions for the HRIR recording
N_RECORDINGS = 10
FS           = 48828
HP_FREQ      = 120
N_REC_HP     = 3
SHOW         = True

# step 0, acoustic head radius
AZ_RANGE     = (-60, 60)      # lateral speakers to sweep, degrees
AZ_ELEVATION = (-1, 1)        # horizontal row only
N_REC_AZ     = 10

ROOT = hr.PATH
slab.set_default_samplerate(FS)
freefield.set_logger('info')
subject = hr.Subject(SUBJECT_ID)

# %% step 0: acoustic head radius ----------------------------------------------
# Mics already in the ears. Records the horizontal row, fits the sphere whose ITDs match this listener
logging.info('--- Step 0: acoustic head radius ---')
az_fit = record_head_radius(
    SUBJECT_ID, azimuth_range=AZ_RANGE, elevation=AZ_ELEVATION,
    n_recordings=N_REC_AZ, hp_freq=HP_FREQ, fs=FS, show=SHOW, save=subject)
# usable_radius returns the fitted value, or falls back to 0.0875 with a loud
# error if the fit hit a bound / has a huge residual / the two ITD estimators
# disagree. Read the printed table anyway: KEMAR gives 0.0722 m, residual 27 us.
HEAD_RADIUS = usable_radius(az_fit)


# %% step 1: record / load HRIR ------------------------------------------------
logging.info('--- Step 1: HRIR recording ---')
hrir = record_hrir(
    subject_id     = SUBJECT_ID,
    reference_id   = REFERENCE_ID,
    n_directions   = N_DIRECTIONS,
    n_recordings   = N_RECORDINGS,
    fs             = FS,
    hp_freq        = HP_FREQ,
    equalize_dome  = EQUALIZE_DOME,
    head_radius    = HEAD_RADIUS,
    show           = SHOW,
    overwrite_rec  = False,
    overwrite_hrir = False,
)

# %% step 2: headphone calibration ---------------------------------------------
logging.info('--- Step 2: HP calibration ---')
# hp_filter = calibrate_headphones(SUBJECT_ID, 'MYSPHERE', N_REC_HP, SHOW, True)
hp_filter = calibrate_headphones(SUBJECT_ID, 'DT990', N_REC_HP, SHOW, False, overwrite=True)

# %% stimulus for every localization test in this file -------------------------
STIM = 'ripple'            # -> 'ripple' once the depth is settled
STIM_SETTINGS = {'rms_tilt': 3}        # empty = inherit localization_helpers.stimulus defaults

# %% step 4: dome localization ---------------------------------------------------
logging.info('--- Step 4: Dome localization ---')
dome_loc = LocalizationDome(subject, {'targets_per_speaker': 3, 'min_distance': 15,
    'stim': STIM, 'stim_settings': STIM_SETTINGS})
dome_loc.run()

# %% step 4b: OPTIONAL dome training -- only if step 4 is at floor ---------------
training = TrainingDome(subject, region='midline')
training.run(n_games=1)


# %% step 5b: virtual localization -- DT990 ---------------------------------------
logging.info('--- Step 5: HP localization (DT990) ---')
ar_loc_settings = {'kind': 'standard', 'azimuth_range': (-1, 1), 'elevation_range': (-35, 35),
    'targets_per_speaker': 2, 'min_distance': 15, 'gain': .2, 'stim': STIM,
    'stim_settings': STIM_SETTINGS}
dt990_hrir_settings = dict(name=SUBJECT_ID, subject_id=SUBJECT_ID, ear=None, mirror=False,
    reverb=True, drr=20, hp_filter=True, hp='DT990', convolution='cpu', storage='cpu')
ar_loc = Localization(subject, dt990_hrir_settings, ar_loc_settings)
ar_loc.run()

# %% helper: acoustic_test (run this cell before the Step 3 cell below) ---------
def acoustic_test(hrir, hp_filter, subject_id, hp_id, *, equalize_dome,
                  show=True, match_level=True, level_band=(200.0, 16000.0),
                  dome_level_db=85.0, hp_level_db=65.0, binsim=None):
    """
    Compare real loudspeaker recordings against HRIR headphone renderings.

    Plays a log-chirp from every third vertical-midline speaker and records
    binaurally via the in-ear mics -- once from the dome (remove headphones)
    and once via HP+HRIR (put headphones on). Overlays spectra per source.

    equalize_dome : bool, REQUIRED, keyword-only
        Dome equalization for the loudspeaker leg. **Must be the same value the
        HRIR and its reference were recorded with.** It is required rather than
        defaulted because it was hardcoded ``equalize=True`` until 2026-08-24,
        which silently compared a dome-equalized recording against an
        unequalized HRIR: measured on ref_20.08 vs ref_19.08, the dome EQ is
        worth +8.9 dB at 2 kHz falling to -3.1 dB at 16 kHz, i.e. a ~9 dB
        tilt across 2-16 kHz that shows up as a spurious low-frequency excess
        in the HP+HRIR trace and looks like an HRTF error.

    match_level : bool, default True
        Shift the HP+HRIR curve down onto the dome curve by the measured
        `level_match['offset_db']` before overlaying, so the panels compare
        spectral SHAPE. Only the HP curve moves, by exactly the number this
        function returns, so what you see is what that number does. The dome
        curve keeps its absolute values, and any vertical gap left inside the
        plotted window is real: it means the rendering has a spectral tilt
        relative to the dome, not merely a gain difference. Figure only -- the
        returned recordings are untouched.

    level_band : (float, float), default (200, 16000)
        Band for the level match, in Hz. Same convention as the `ild_band` used
        throughout `record/processing.py`. The chirp runs 200 Hz-18 kHz, so this
        covers it without the roll-off at either end.

    dome_level_db, hp_level_db : float
        The playback levels actually used for the two legs. Recorded in the
        returned dict so the offset can be interpreted; change them here if you
        change the levels below.

    binsim : dict or None
        When given, ALSO measure the real AR chain -- pybinsim with reverb,
        hp_filter, the WAV RMS, `loc_settings['gain']` and the OS fader -- against
        the dome, and put the result in `level_match['binsim']`. That measurement
        is the one whose number transfers to `loc_settings['gain']`; the freefield
        HP number above does not (see the caveat). Requires the headphones to be
        fed by the SOUNDCARD while the in-ear mics stay on the TDT; it prompts for
        headphones on, then off, and refuses to report a number if the mics are
        not hearing the headphones.

        Keys: `subject`, `hrir_settings`, `loc_settings` (as for the AR test),
        optional `os_volume` (default 50) and `sources` (default: the same
        directions this function just measured). Example::

            binsim=dict(subject=subject, hrir_settings=dt990_hrir_settings,
                        loc_settings=ar_loc_settings)

        Costs ~30 s of setup because the binsim worker has to come up, so it is
        off by default.

    Returns
    -------
    hp_recordings, dome_recordings : dict
        Raw recordings keyed by source position, unmodified.
    level_match : dict
        Loudness relation between the two legs, per direction and per ear, as
        dome-minus-HP band levels over `level_band`. `offset_db_mean` is the
        single number: **multiply the HP rendering by `hp_gain_factor` to make
        it as loud as the dome**, at the playback levels recorded in
        `playback_levels`.

        CAVEAT, read before using this for the AR test: this measures the
        **freefield `play_and_record_headphones`** path, NOT pybinsim. The AR
        localization runs through pybinsim, whose SPL is a function of the WAV
        RMS, the runtime `/pyBinSimLoudness` gain (`loc_settings['gain']`) and
        the OS master fader -- none of which this measurement touches. So this
        number is not a drop-in for `loc_settings['gain']`. What it IS good for:
        it makes the HRIR-rendering half of that chain objective instead of
        by-ear, and it is a direction-resolved check that no single elevation is
        rendered at the wrong level. See `match_ar_dome_loudness.py` for the
        pybinsim leg, which still has to be done by ear on a fixed OS volume.
    """
    fs = hrir.samplerate

    def _band_level_db(sound, channel):
        """Band power over `level_band`, in dB.

        Parseval-normalised, i.e. the mean-square of the band-limited signal,
        not a raw FFT-bin sum -- so it is a physical level and does not scale
        with the transform length. (Both legs are captured with the same 1 s
        chirp at the same `recording_samplerate` anyway, so this only matters
        if someone changes one of them.)
        """
        x = numpy.asarray(sound.data)[:, channel].astype(float)
        n = len(x)
        spec = numpy.abs(numpy.fft.rfft(x)) ** 2
        freqs = numpy.fft.rfftfreq(n, 1.0 / sound.samplerate)
        band = (freqs >= level_band[0]) & (freqs <= level_band[1])
        if not band.any():
            raise ValueError(f'level_band {level_band} selects no frequency bins')
        return float(10 * numpy.log10(2.0 * spec[band].sum() / n ** 2 + 1e-30))

    signal = slab.Sound.chirp(
        duration=1.0, level=70, samplerate=fs,
        kind='logarithmic', from_frequency=200, to_frequency=18000,
    )
    signal = signal.ramp(when='both', duration=0.01)

    src_idx = hrir.cone_sources(0)[::3]
    src_idx.sort()

    # --- headphones ---
    if freefield.PROCESSORS.mode != 'bi_play_rec':
        freefield.initialize('headphones', default='bi_play_rec')

    hp_signal = hp_filter.apply(signal)
    input('Put on headphones and press Enter to continue...')
    hp_recordings = {}
    for src in hrir.sources.vertical_polar[src_idx]:
        idx = hrir.get_source_idx(src[0], src[1])[0]
        filtered = hrir.apply(idx, hp_signal)
        filtered.level = 65
        hp_recordings[str(src)] = freefield.play_and_record_headphones(
            speaker='both', sound=filtered, compensate_delay=True, distance=0,
            compensate_attenuation=False, equalize=False, recording_samplerate=fs,
        )

    # --- dome speakers ---
    freefield.initialize('dome', default='play_birec')
    spk_signal = copy.deepcopy(signal)
    spk_signal.level = 85
    input('Remove headphones and press Enter to continue...')
    dome_recordings = {}
    for src in hrir.sources.vertical_polar[src_idx]:
        speaker = freefield.pick_speakers((src[0], src[1]))
        dome_recordings[str(src)] = freefield.play_and_record(
            speaker, spk_signal, compensate_delay=True,
            compensate_attenuation=False, equalize=equalize_dome,
            recording_samplerate=fs,
        )

    # --- level relation between the two legs -----------------------------
    dome_db, hp_db, offset_db = {}, {}, {}
    for src in dome_recordings:
        dome_db[src] = [_band_level_db(dome_recordings[src], c) for c in (0, 1)]
        hp_db[src] = [_band_level_db(hp_recordings[src], c) for c in (0, 1)]
        offset_db[src] = [d - h for d, h in zip(dome_db[src], hp_db[src])]

    per_ear = numpy.array([offset_db[src] for src in offset_db])   # (n_src, 2)
    median_per_ear = numpy.median(per_ear, axis=0)
    mean_offset = float(numpy.median(per_ear))

    level_match = {
        'band_hz': [float(level_band[0]), float(level_band[1])],
        'dome_level_db': dome_db,
        'hp_level_db': hp_db,
        'offset_db': offset_db,                       # dome - HP, per direction, [L, R]
        'offset_db_median_per_ear': median_per_ear.tolist(),
        'offset_db_mean': mean_offset,                # median over directions AND ears
        'offset_db_spread': float(per_ear.max() - per_ear.min()),
        'hp_gain_factor': float(10 ** (mean_offset / 20.0)),
        'playback_levels': {'dome_db': float(dome_level_db), 'hp_db': float(hp_level_db)},
        'path': 'freefield play_and_record_headphones -- NOT pybinsim',
    }

    logging.info('--- acoustic_test level match (%g-%g Hz) ---',
                 level_band[0], level_band[1])
    logging.info('%-28s %8s %8s', 'source', 'L dB', 'R dB')
    for src in offset_db:
        logging.info('%-28s %8.2f %8.2f', src, offset_db[src][0], offset_db[src][1])
    logging.info('median per ear: L %.2f  R %.2f  |  overall %.2f dB '
                 '(spread across directions/ears %.2f dB)',
                 median_per_ear[0], median_per_ear[1], mean_offset,
                 level_match['offset_db_spread'])
    logging.info('-> multiply the HP rendering by %.3f to match the dome '
                 '(dome %g dB vs HP %g dB playback). pybinsim is NOT in this path.',
                 level_match['hp_gain_factor'], dome_level_db, hp_level_db)

    # --- optional: the real pybinsim chain, not the freefield HP path ------
    if binsim:
        # imported lazily: it pulls in Localization_AR, pythonosc and pybinsim,
        # none of which step 3 needs when binsim is off.
        from hrtf_relearning.experiment.localization.localization_helpers.ar_level_match \
            import binsim_session, measure_ar_dome_level

        binsim_sources = binsim.get('sources') or [
            (float(src[0]), float(src[1])) for src in hrir.sources.vertical_polar[src_idx]]
        with binsim_session(binsim['subject'], binsim['hrir_settings'],
                            binsim['loc_settings'],
                            os_volume=binsim.get('os_volume', 50)) as (loc, osc_f, osc_p):
            level_match['binsim'] = measure_ar_dome_level(
                loc, osc_f, osc_p, sources=binsim_sources, band=level_band)

    if show:
        fmin, fmax = 2e3, 18.2e3
        ticks = 2 ** numpy.arange(numpy.log2(fmin), numpy.log2(fmax), 1)

        def _shift_last_line(ax, db):
            """Move the line just added to ax by `db` decibels."""
            line = ax.lines[-1]
            line.set_ydata(numpy.asarray(line.get_ydata()) + db)

        fig, axes = plt.subplots(
            nrows=len(src_idx), ncols=2, figsize=(12, 3 * len(src_idx)), layout='tight'
        )
        eq_note = 'dome EQ on' if equalize_dome else 'dome EQ off'
        lvl_note = (f', HP +{mean_offset:.1f} dB to match dome over '
                    f'{level_band[0]:.0f}-{level_band[1]:.0f} Hz') if match_level else ''
        fig.suptitle(f'{subject_id} — acoustic test ({hp_id}, {eq_note}{lvl_note})')
        for row, (dome_item, hp_item) in enumerate(
            zip(dome_recordings.items(), hp_recordings.items())
        ):
            for col in range(2):
                ax = axes[row, col]
                dome_item[1].channel(col).spectrum(axis=ax)
                hp_item[1].channel(col).spectrum(axis=ax)
                if match_level:
                    # only the HP curve moves, by exactly the returned number
                    _shift_last_line(ax, offset_db[hp_item[0]][col])
                ax.set_title(f'{dome_item[0]}° — {"L" if col == 0 else "R"}')
                ax.set_xlim(fmin, fmax)
                ax.set_xticks(ticks)
                ax.set_xticklabels(
                    [f"{int(t/1000)}k" if t >= 1000 else str(int(t)) for t in ticks]
                )
                if match_level:
                    ax.set_ylabel('Power [dB/Hz], HP level-matched')
                    ax.relim()
                    ax.autoscale_view(scalex=False)
                ax.legend(['Dome', 'HP+HRIR'])

        save_dir = paths.subject_acoustic_dir(subject_id)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / f'acoustic_test_{hp_id}.svg')
        plt.show()
    return hp_recordings, dome_recordings, level_match

# %% step 3: acoustic sanity check (optional) -----------------------------------
# equalize_dome MUST match what the HRIR/reference were recorded with (EQUALIZE_DOME).
logging.info('--- Step 3: Acoustic test ---')
# Add binsim=dict(subject=subject, hrir_settings=dt990_hrir_settings,
# loc_settings=ar_loc_settings) to ALSO measure the real AR chain and get a gain
# you can paste into loc_settings['gain'] -- see the docstring. Headphones must
# be on the soundcard for that leg. Off here because it costs ~30 s of setup.
hp_rec, dome_rec, level_match = acoustic_test(
    hrir, hp_filter, subject_id=SUBJECT_ID, hp_id='DT990',
    equalize_dome=EQUALIZE_DOME, show=SHOW)
# dome-minus-HP band level. NOT a pybinsim gain -- see the docstring caveat.
print(f"HP needs {level_match['offset_db_mean']:+.2f} dB "
      f"(x{level_match['hp_gain_factor']:.3f}) to match the dome; "
      f"spread {level_match['offset_db_spread']:.2f} dB across directions/ears")
if 'binsim' in level_match:
    b = level_match['binsim']
    print(f"AR chain: loc_settings['gain'] = {b['gain_suggested']:.4f} "
          f"(was {b['gain_used']:.4f}), spread {b['offset_db_spread']:.2f} dB")

# %% step 5a: virtual localization -- MYSPHERE (optional) ------------------------
logging.info('--- Step 5: HP localization (MYSPHERE) ---')
ar_loc_settings = {'kind': 'standard', 'azimuth_range': (-1, 1), 'elevation_range': (-35, 35),
    'targets_per_speaker': 2, 'min_distance': 15, 'gain': .07, 'stim': 'noise'}
mysphere_hrir_settings = dict(name=SUBJECT_ID, subject_id=SUBJECT_ID, ear=None, mirror=False,
    reverb=True, drr=20, hp_filter=True, hp='MYSPHERE', convolution='cpu', storage='cpu')
ar_loc = Localization(subject, mysphere_hrir_settings, ar_loc_settings)
ar_loc.run()

# %% step 0b: reference (ONCE per reference id -- skip if REFERENCE_ID exists) ---
# Mics on the STAND at the listening position, no listener in the chair. Two
# minutes. Refuses to overwrite an existing id, so it is safe to leave this cell
# in place and just run it when REFERENCE_ID is new -- rerunning it with an old
# id raises rather than silently replacing the sweeps.
#
# You can also skip this cell entirely: record_hrir() below records the
# reference itself if REFERENCE_ID does not exist yet, with the same
# EQUALIZE_DOME as the subject. The only reason to do it here is ordering --
# record_hrir does the SUBJECT first, so you would have to move the mics from
# the ears to the stand in the middle of the call.
logging.info('--- Step 0b: reference ---')
reference_rec = record_reference(REFERENCE_ID, n_recordings=N_RECORDINGS, fs=FS,
                                 hp_freq=HP_FREQ, equalize_dome=EQUALIZE_DOME)