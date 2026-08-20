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
from hrtf_relearning.hrtf.record.fit_head_radius import record_head_radius
from hrtf_relearning.hrtf.record.calibration.calibrate_headphones import calibrate_headphones
from hrtf_relearning.utils import paths
import json

SUBJECT_ID   = 'Kemar_reseated_2'          # edit per participant
REFERENCE_ID = 'ref_20.08'   # fresh id -> step 0b records it; reused id -> loaded
EQUALIZE_DOME = False        # subject AND reference; they must match. See step 0b.

# The dome EQ mismatch (project_dome_eq_mismatch): every reference recorded
# before 2026-08-20 -- ref_03.04, kemar_reference, ref_19.08, ref_19.08_swapped
# -- has equalize_dome=True while every subject has False. record_dome passes
# this to freefield.play_and_record(equalize=...), which pre-filters the emitted
# sweep per speaker; processing.equalize() then divides subject by reference PER
# SPEAKER, so the dome EQ does not cancel and leaves HRTF / E_k. Each midline
# elevation is a different speaker (20..26), so the residual is
# elevation-dependent -- roughly 1.1 dB rms at 200-1000 Hz, 1.5 dB 1-4 kHz,
# 2.5 dB 4-16 kHz.
#   To stop inheriting it: give REFERENCE_ID a FRESH name and run step 0b once.
#   Everything recorded from then on is internally consistent, at the cost of
#   about two minutes on the first session of the day. Subjects already on disk
#   keep the old reference and the old residual; that is a separate decision.

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
# Mics already in the ears. Records the horizontal row, fits the sphere whose
# ITDs match this listener, and returns the radius in metres -- also stored as
# subject.head_radius (<ID>.pkl + .json). Re-running loads instead of
# re-recording. If the fit is not usable it warns and returns the 0.0875 m
# fallback; details in fit_head_radius.py. KEMAR: 0.0722 m.
logging.info('--- Step 0: acoustic head radius ---')
HEAD_RADIUS = record_head_radius(
    SUBJECT_ID, azimuth_range=AZ_RANGE, elevation=AZ_ELEVATION,
    n_recordings=N_REC_AZ, hp_freq=HP_FREQ, fs=FS, show=SHOW, save=subject)

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
# The test stimulus varies its source spectrum on every trial, so that
# localization cannot be solved by matching an absolute spectrum to a stored
# template -- see learning_transfer.STIM for the full reason. Set in one place
# here so the dome reference and the virtual test are never run on different
# stimuli, which is the comparison this session exists to make.
#   !! Depth not yet settled -- see learning_transfer.STIM_SETTINGS. Verify with
#      protocols/dev/stimulus_check.py (cells 11-13 dome, 7-9 AR) before use.
STIM = 'ripple'            # -> 'ripple' once the depth is settled
STIM_SETTINGS = {'rms_tilt': 3}        # empty = inherit localization_helpers.stimulus defaults

# %% step 4: dome localization ---------------------------------------------------
# Real speakers, vertical midline. Each run gets a fresh timestamped filename
# (see LocalizationDome.__init__), so repeats are stored as separate sequences
# rather than overwriting one another -- rerun this cell to redo it.
logging.info('--- Step 4: Dome localization ---')
dome_loc = LocalizationDome(subject, {'targets_per_speaker': 3, 'min_distance': 15,
    'stim': STIM, 'stim_settings': STIM_SETTINGS})
dome_loc.run()

# %% step 4b: OPTIONAL dome training -- only if step 4 is at floor ---------------
# NOT part of the standard session. Run it when a participant cannot do the task
# on REAL SPEAKERS -- repeated step-4 runs at chance, elevation gain near zero,
# responses that ignore elevation entirely. That is a participant who has not
# understood the task or is not attending to the spectral cue, and it is worth
# separating from a participant whose HRIR is bad: the dome is the ceiling
# condition, so if they cannot localize HERE, nothing measured over headphones
# afterwards means anything.
#
# The coin game on the vertical midline, same speakers step 4 tests. Targets are
# weighted by the per-sector error of their last matching localization run
# (find_last_matching_sequence), so it trains where they are actually wrong --
# which is why this must run AFTER at least one step-4 block, not before.
#
# One or two games (~90 s each) is the intended dose; this is a task-comprehension
# rescue, not adaptation training, and every game played is exposure that a naive
# baseline no longer has. RECORD IN THE SUBJECT NOTES that training was given,
# and re-run step 4 afterwards rather than reusing the pre-training block.
#
# Processor modes take care of themselves: TrainingDome switches the RX8s to the
# pulse circuit and LocalizationDome.run() switches them back to 'play_rec', each
# guarding on freefield.PROCESSORS.mode. So step 4 can simply be re-run.
#
# Esc at the "play again" prompt ends the session.
training = TrainingDome(subject, region='midline')
training.run(n_games=1)

# ...then re-run the step 4 cell above to see whether it took.

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
# hp_rec, dome_rec = acoustic_test(hrir, hp_filter, subject_id=SUBJECT_ID, hp_id='DT990', show=SHOW)

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
# logging.info('--- Step 5: HP localization (MYSPHERE) ---')
# ar_loc_settings = {'kind': 'standard', 'azimuth_range': (-1, 1), 'elevation_range': (-35, 35),
#     'targets_per_speaker': 2, 'min_distance': 15, 'gain': .07, 'stim': 'noise'}
# mysphere_hrir_settings = dict(name=SUBJECT_ID, subject_id=SUBJECT_ID, ear=None, mirror=False,
#     reverb=True, drr=20, hp_filter=True, hp='MYSPHERE', convolution='cpu', storage='cpu')
# ar_loc = Localization(subject, mysphere_hrir_settings, ar_loc_settings)
# ar_loc.run()