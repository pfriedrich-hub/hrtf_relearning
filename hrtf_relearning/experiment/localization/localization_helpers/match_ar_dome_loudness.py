"""
match_ar_dome_loudness.py

By-ear loudness matching between the DOME localization test (real loudspeakers,
freefield) and the AR/VR localization test (pybinsim over headphones).

Why this exists
---------------
The dome test plays its stim through freefield at a calibrated `level` (dB SPL
via speaker equalization). The AR/VR test does NOT use that path: it writes the
stim to `localization.wav` and plays it through pybinsim, where the physical
output level is set by the runtime `/pyBinSimLoudness` gain (currently
`loc_settings['gain'] = 0.2`). Nothing links that gain to the dome's SPL, so the
two tests are not loudness-matched. KEMAR recording through pybinsim isn't
available, so we match BY EAR: play the dome reference, play the pybinsim path,
nudge `GAIN`, repeat until they sound equally loud, then paste GAIN back into
`loc_settings['gain']` for the AR/VR tests.

How to use  (run cell-by-cell, # %%, in an IDE/console)
-------------------------------------------------------
1. Run the SETUP cell once (builds binsim files, starts the binsim worker +
   OSC clients, initialises the dome).
2. DOME cell: rerun to hear the dome reference (remove headphones).
3. PYBINSIM cell: edit `GAIN`, rerun to hear the pybinsim path (headphones on).
   Alternate 2 <-> 3 until they sound equally loud.
4. RESULT cell: prints the GAIN to set in loc_settings['gain'].
5. TEARDOWN cell when done.

Stimulus note
-------------
The dome and AR/VR tests use *different* stimuli (dome: 5x25 ms pinknoise burst
train @ level 85; AR: 225 ms gapped pinknoise @ level 80). Loudness-matching two
different stimuli by ear is inherently fuzzy. Set MATCH_STIM = 'native' to hear
each test's real stim (ecologically valid), or 'dome' to play the identical
dome-style burst train through BOTH paths for a cleaner apples-to-apples match.
"""

# TO DO, next time KEMAR is in the chair: replace this by-ear match with an
# objective one -- KEMAR in the rig, in-ear mics, bi-rec RCX, measure the
# dome and the headphone path and take the difference. Also check stimulus
# generation across conditions while the manikin is set up.
#
# Do not write it from scratch first: `equalize_loudness()` in
# _to_delete/dead_code_20260819/record/test_hrir_recording.py already does
# the objective dome-vs-headphone match through the in-ear mics, via
# freefield.play_and_record_headphones, and carries a measured result
# ('diff is about 19 on both channels'). Salvage that before it is deleted.

# %% SETUP -------------------------------------------------------------------
import multiprocessing as mp
import time
import numpy
import slab
import freefield

import hrtf_relearning as hr
from hrtf_relearning.experiment.misc.system_volume import set_windows_volume
from hrtf_relearning.experiment.localization.Localization_AR import Localization
from hrtf_relearning.experiment.localization.Localization_dome import LocalizationDome

OS_VOLUME = 50   # Windows master volume (%) the loudness match is made at

# --- what to compare against what ---
SUBJECT_ID = 'JS'
HRIR_NAME  = 'JS_synth'     # SOFA basename used by the AR/VR test
HP         = 'DT990'
MATCH_STIM = 'native'       # 'native' (each test's own stim) or 'dome' (same stim both paths)
REF_SOURCE = (0., 0.)       # az, el of the frontal reference location

_hrir_settings = dict(name=HRIR_NAME, subject_id=SUBJECT_ID, ear=None, mirror=False,
                      reverb=True, drr=20, hp_filter=True, hp=HP,
                      convolution='cpu', storage='cpu')
_loc_settings = {'kind': 'sectors', 'azimuth_range': (-1, 1), 'elevation_range': (-35, 35),
                 'targets_per_sector': 3, 'min_distance': 15, 'gain': 0.2,
                 'sector_size': (7, 14), 'replace': False, 'stim': 'noise'}

set_windows_volume(OS_VOLUME)   # pin OS volume; keep identical in the real tests

subject = hr.Subject(SUBJECT_ID)

# Builds binsim files + gives us the AR test's stim generation and OSC/worker helpers
loc = Localization(subject, _hrir_settings, _loc_settings)

# start the pybinsim worker + OSC clients (mirrors Localization.run)
osc_filter = loc._make_osc_client(port=10000)   # /pyBinSim_ds_Filter
osc_play   = loc._make_osc_client(port=10003)   # /pyBinSimLoudness, /pyBinSimFile
binsim_worker = mp.Process(target=loc._binsim_stream, args=(loc.hrir_name,))
binsim_worker.start()
time.sleep(1.5)   # let the stream come up

# frontal filter index for the pybinsim path (mirror az like play_stimulus does)
_rel_az = (-REF_SOURCE[0] + 360) % 360
_rel = numpy.array((_rel_az, REF_SOURCE[1], loc.hrir_sources[0, 2]))
FILTER_IDX = int(numpy.argmin(numpy.linalg.norm(_rel - loc.hrir_sources, axis=1)))
print(f"pybinsim reference filter -> {loc.hrir_sources[FILTER_IDX]}")

# dome for the loudspeaker reference
if freefield.PROCESSORS is None or freefield.PROCESSORS.mode != 'play_rec':
    freefield.initialize('dome', default='play_rec', sensor_tracking=False)


def _make_pybinsim_stim():
    """The stim that the pybinsim path will play, written to localization.wav.
    Dome and AR now share one synthesis (make_gapped_pinknoise), so 'native' and
    'dome' produce the same stimulus; kept only so the WAV stays at the AR test's
    own level (80) and the derived GAIN transfers straight into the real test."""
    if MATCH_STIM == 'dome':
        stim = LocalizationDome.make_stim(level=80)
    else:
        stim = loc.make_stim()                        # AR test's real 225 ms gapped noise @ level 80
    stim.write(loc.sound_path / 'localization.wav')
    return stim


def _make_dome_stim():
    """The stim played from the real loudspeakers -- default (level=None)
    preserves the dome's loudness so this match matches the real dome test."""
    return LocalizationDome.make_stim()


def play_dome():
    """Play the dome reference stim from REF_SOURCE (headphones off)."""
    dome_stim = _make_dome_stim()
    spk = freefield.pick_speakers(REF_SOURCE)[0]
    freefield.set_signal_and_speaker(signal=dome_stim, speaker=spk.index, equalize=True)
    freefield.play()
    freefield.wait_to_finish_playing()
    print(f"  DOME     (level {numpy.mean(dome_stim.level):.0f}, equalized) from {REF_SOURCE}")


def play_binsim(gain):
    """Play the pybinsim path at `gain` from the frontal filter (headphones on)."""
    stim = _make_pybinsim_stim()
    osc_filter.send_message('/pyBinSim_ds_Filter',
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             float(loc.hrir_sources[FILTER_IDX][0]),
                             float(loc.hrir_sources[FILTER_IDX][1]), 0, 0, 0, 0])
    time.sleep(0.1)
    osc_play.send_message('/pyBinSimLoudness', gain)
    osc_play.send_message('/pyBinSimFile', str(loc.sound_path / 'localization.wav'))
    time.sleep(float(stim.duration) + 0.2)
    osc_play.send_message('/pyBinSimLoudness', 0)
    print(f"  PYBINSIM @ gain = {gain}")


print("Setup done. MATCH_STIM =", MATCH_STIM)


# %% MATCH LOOP -- interactively enter gains until the two sound equally loud --
# Commands at the prompt:
#   <number>  set gain and play pybinsim   (e.g. 0.25)
#   d         (re)play the dome reference
#   b         (re)play pybinsim at current gain
#   <Enter>   A/B: play dome then pybinsim
#   q         quit the loop (keeps current gain)
GAIN = 0.20   # starting gain

play_dome()
play_binsim(GAIN)
while True:
    cmd = input(f"[gain={GAIN}] number / d / b / Enter=A-B / q > ").strip().lower()
    if cmd == 'q':
        break
    elif cmd == 'd':
        play_dome()
    elif cmd == 'b':
        play_binsim(GAIN)
    elif cmd == '':
        play_dome()
        play_binsim(GAIN)
    else:
        try:
            GAIN = float(cmd)
        except ValueError:
            print("  ? enter a number, or d / b / Enter / q")
            continue
        play_binsim(GAIN)

print(f"\nMatched gain for AR/VR:  loc_settings['gain'] = {GAIN}\n"
      f"(subject={SUBJECT_ID}, hrir={HRIR_NAME}, hp={HP}, stim={MATCH_STIM}, os_volume={OS_VOLUME})")


# %% TEARDOWN ----------------------------------------------------------------
try:
    osc_play.send_message('/pyBinSimLoudness', 0)
    time.sleep(0.1)
except Exception:
    pass
binsim_worker.terminate()
binsim_worker.join(timeout=3)
if binsim_worker.is_alive():
    binsim_worker.kill()
    binsim_worker.join()
print("Torn down.")
