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

SUBJECT_ID   = 'kemar_test_24.08'          # edit per participant
REFERENCE_ID = 'ref_20.08'   # fresh id -> step 0b records it; reused id -> loaded
EQUALIZE_DOME = False        # subject AND reference; they must match. See step 0b.
HEAD_RADIUS = 0.075          # fallback
N_DIRECTIONS = 1              # directions for the HRIR recording
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
    overwrite_rec  = True,
    overwrite_hrir = True,
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

# # %% step 3: acoustic sanity check (optional) -----------------------------------
# logging.info('--- Step 3: Acoustic test ---')
hp_rec, dome_rec = acoustic_test(hrir, hp_filter, subject_id=SUBJECT_ID, hp_id='DT990', show=SHOW)

# %% helper: acoustic_test (define before running the Step 3 cell) -----------
def acoustic_test(hrir, hp_filter, subject_id, hp_id, show=True):
    """
    Compare real loudspeaker recordings against HRIR headphone renderings.

    Plays a log-chirp from every third vertical-midline speaker and records
    binaurally via the in-ear mics -- once from the dome (remove headphones)
    and once via HP+HRIR (put headphones on). Overlays spectra per source.
    """
    fs = hrir.samplerate
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
            compensate_attenuation=False, equalize=True, recording_samplerate=fs,
        )

    if show:
        fmin, fmax = 2e3, 18.2e3
        ticks = 2 ** numpy.arange(numpy.log2(fmin), numpy.log2(fmax), 1)
        fig, axes = plt.subplots(
            nrows=len(src_idx), ncols=2, figsize=(12, 3 * len(src_idx)), layout='tight'
        )
        fig.suptitle(f'{subject_id} — acoustic test ({hp_id})')
        for row, (dome_item, hp_item) in enumerate(
            zip(dome_recordings.items(), hp_recordings.items())
        ):
            for col in range(2):
                ax = axes[row, col]
                dome_item[1].channel(col).spectrum(axis=ax)
                hp_item[1].channel(col).spectrum(axis=ax)
                ax.set_title(f'{dome_item[0]}° — {"L" if col == 0 else "R"}')
                ax.set_xlim(fmin, fmax)
                ax.set_xticks(ticks)
                ax.set_xticklabels(
                    [f"{int(t/1000)}k" if t >= 1000 else str(int(t)) for t in ticks]
                )
                ax.legend(['Dome', 'HP+HRIR'])

        save_dir = paths.subject_acoustic_dir(subject_id)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / f'acoustic_test_{hp_id}.svg')
        plt.show()
    return hp_recordings, dome_recordings

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