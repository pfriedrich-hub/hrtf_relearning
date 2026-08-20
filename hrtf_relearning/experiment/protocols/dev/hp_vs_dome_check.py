"""
hp_vs_dome_check.py

Dome loudspeakers vs. headphone HRIR rendering, on the freefield headphone
path: an objective level match (step 1) and a 2AFC discrimination test
(step 2).

Salvaged 2026-08-20 from hrtf/record/test_hrir_recording.py, which was removed
as dead code (audit finding H, docs/hrir_recording_audit.md): its entry point
could not run (it called record_hrir(overwrite=...), a parameter that does not
exist) and its acoustic_test is live in protocols/HRIR_Recording.py step 3.
These two functions were the only parts of that file with no home elsewhere,
so they live here now. NEITHER HAS BEEN RUN SINCE THE SALVAGE, and the
behavioural half was a sketch to begin with -- treat both as a starting point,
not a validated protocol. The original todos are kept and marked.

NOT the same thing as
localization/localization_helpers/match_ar_dome_loudness.py, which matches AR
(pybinsim) rendering to the dome BY EAR. This file measures the dome-vs-
headphone difference OBJECTIVELY, through the in-ear mics, on the freefield
play_and_record_headphones path. The two rendering paths have different gain
structures, so a number measured here does not transfer to pybinsim.

Prerequisites
    - an HRIR and a headphone filter for this subject (HRIR_Recording.py
      steps 1-2)
    - step 1 needs the in-ear mics IN and the headphones put on/taken off
      between cells
    - step 2 is a listening test: mics out, headphones on

Run cell by cell (# %%) in an IDE/console -- do NOT run this top-to-bottom as
a plain script. Each rig action (put headphones on, take them off) is a cell
boundary rather than an input() prompt; the only prompts left are the
participant's own responses in step 2.
"""

# %% imports and config ------------------------------------------------------
import copy
import json
import logging

import numpy

import freefield
import slab

from hrtf_relearning.hrtf.record.calibration.calibrate_headphones import load_hp_filter
from hrtf_relearning.utils import paths

SUBJECT_ID = 'Kemar_reseated_2'      # edit per participant
HP_ID = 'DT990'                      # headphone model, must match the filter on disk
FS = 48828
slab.set_default_samplerate(FS)

# Levels. The two source functions disagreed, and the disagreement is worth
# keeping: acoustic_test used 65 dB (headphones) / 85 dB (dome) with the note
# "these levels work, use them for behavioral testing", while behavioral_test
# itself used 70 / 80 and a "todo adjust level". The acoustic_test pair is
# carried over here as the default; step 1 is what tells you whether it is
# right for THIS subject and headphone.
LEVEL_HP = 65
LEVEL_SPK = 85

N_TRIALS = 50                        # 2AFC trials in step 2
SEED = None                          # set an int to reproduce a sequence

logging.basicConfig(level=logging.INFO)


# %% step 0: load HRIR and headphone filter ----------------------------------
hrir = slab.HRTF(str(paths.SOFA_DIR / f'{SUBJECT_ID}.sofa'))
hp_filter = load_hp_filter(
    paths.REC_DIR / SUBJECT_ID / f'{HP_ID}_equalization.npz', 'slab')

# vertical-midline sources of this HRIR, used by both steps
midline_idx = sorted(hrir.cone_sources(0))
midline_elevations = hrir.sources.vertical_polar[midline_idx][:, 1]
logging.info(f'{len(midline_idx)} midline sources, '
             f'elevations {midline_elevations.min():.0f}..{midline_elevations.max():.0f} deg')


# %% step 1a: dome level, mics in ears, NO headphones ------------------------
# Objective dome-vs-headphone level match through the in-ear mics. Frontal
# speaker only; the dome EQ is applied (equalize=True) so this is the level a
# participant actually gets on the dome.
freefield.initialize('dome', default='play_birec')

probe = slab.Sound.chirp(duration=1.0, level=LEVEL_SPK, samplerate=FS,
                         kind='logarithmic', from_frequency=200, to_frequency=18000)
probe = probe.ramp(when='both', duration=0.01)

speaker = freefield.pick_speakers((0, 0))
dome_rec = freefield.play_and_record(
    speaker, probe, compensate_delay=True, compensate_attenuation=False,
    equalize=True, recording_samplerate=FS)
dome_level = dome_rec.level
logging.info(f'dome level at (0, 0): {numpy.round(dome_level, 2)} dB')


# %% step 1b: headphone level, mics still in, headphones ON ------------------
# Same probe through the headphones, equalized with this subject's HP filter
# (freefield.load_equalization reads the pickle that calibrate_headphones ->
# ff_equalization(save_freefield=True) wrote).
freefield.initialize('headphones', default='bi_play_rec')
freefield.load_equalization(freefield.DIR / 'data' / f'calibration_{HP_ID}.pkl')

hp_probe = slab.Sound.chirp(duration=1.0, level=LEVEL_HP, samplerate=FS,
                            kind='logarithmic', from_frequency=200, to_frequency=18000)
hp_probe = hp_probe.ramp(when='both', duration=0.01)

hp_rec = freefield.play_and_record_headphones(
    speaker='both', sound=hp_probe, compensate_delay=True, distance=0,
    compensate_attenuation=False, equalize=True, recording_samplerate=FS)
hp_level = hp_rec.level

level_difference = numpy.atleast_1d(dome_level) - numpy.atleast_1d(hp_level)
logging.info(f'dome - headphones: {numpy.round(level_difference, 2)} dB per channel')
# Original note from test_hrir_recording.equalize_loudness, measured with
# LEVEL_HP == LEVEL_SPK: "diff is about 19 on both channels". Here the probe
# levels already differ by LEVEL_SPK - LEVEL_HP, so a difference near
# (LEVEL_SPK - LEVEL_HP) - 19 means the two paths are matched at these
# settings. Feed the residual back into LEVEL_HP and rerun 1b to close it.


# %% step 2: 2AFC discrimination, mics OUT, headphones on --------------------
# "Can the participant tell a real loudspeaker from its HRIR rendering?"
# Randomly interleaved dome / headphone presentations at random midline
# elevations; the experimenter types what the participant reports.
#
# todo (original) settle the test signal -- pinknoise is a placeholder, and a
#      fixed-spectrum signal is exactly what makes template matching possible;
#      consider the 'ripple' stimulus used everywhere else (see
#      localization_helpers/stimulus.py).
# todo (original) collect responses through the freefield button box rather
#      than the console, so the participant is not cued by typing.
def run_discrimination(hrir, hp_filter, n_trials=N_TRIALS, seed=SEED):
    """Interleaved dome/headphone 2AFC on the vertical midline.

    Returns
    -------
    trials : dict
        'source' (0 dome, 1 headphones), 'elevation', 'response' per trial.
    """
    rng = numpy.random.default_rng(seed)
    signal = slab.Sound.pinknoise(duration=0.5, samplerate=hrir.samplerate)

    sources = rng.integers(0, 2, n_trials)
    elevations = hrir.sources.vertical_polar[sorted(hrir.cone_sources(0))][:, 1]
    ele_idx = rng.integers(0, len(elevations), n_trials)

    spk_signal = copy.deepcopy(signal)
    spk_signal.level = LEVEL_SPK
    hp_signal = hp_filter.apply(signal)

    if freefield.PROCESSORS.mode != 'play_birec':
        freefield.initialize('dome', default='play_birec')

    responses = []
    for source, idx in zip(sources, ele_idx):
        elevation = elevations[idx]
        if source == 0:                       # real loudspeaker
            speaker = freefield.pick_speakers((0, elevation))
            freefield.set_signal_and_speaker(spk_signal, speaker)
            freefield.play()
        else:                                 # headphone rendering
            src_idx = hrir.get_source_idx(0, elevation)[0]
            filtered = hrir.apply(src_idx, hp_signal)
            filtered.level = LEVEL_HP
            filtered.play()
        response = input('response (0 = speaker, 1 = headphones): ')
        responses.append(int(response) if response.strip() in ('0', '1') else None)

    return {'source': sources.tolist(),
            'elevation': elevations[ele_idx].tolist(),
            'response': responses}


trials = run_discrimination(hrir, hp_filter)

scored = [(s, r) for s, r in zip(trials['source'], trials['response']) if r is not None]
correct = sum(s == r for s, r in scored)
logging.info(f'{correct}/{len(scored)} correct '
             f'({100 * correct / max(len(scored), 1):.0f} %; chance = 50 %)')


# %% step 3: save ------------------------------------------------------------
out_dir = paths.subject_acoustic_dir(SUBJECT_ID)
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / f'hp_vs_dome_{HP_ID}.json'
out_file.write_text(json.dumps(
    {'subject_id': SUBJECT_ID,
     'hp_id': HP_ID,
     'level_hp': LEVEL_HP,
     'level_spk': LEVEL_SPK,
     'dome_level': numpy.atleast_1d(dome_level).tolist(),
     'hp_level': numpy.atleast_1d(hp_level).tolist(),
     'level_difference': level_difference.tolist(),
     'trials': trials},
    indent=2))
logging.info(f'saved {out_file}')
