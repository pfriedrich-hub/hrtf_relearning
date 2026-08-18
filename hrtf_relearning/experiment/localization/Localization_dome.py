"""
Dome loudspeaker localization test.

Mirrors Localization_AR.Localization in API and conventions:
  - Subject for data persistence
  - make_sequence (std_targets) for trial sequences
  - meta-motion sensor for head-pose response
  - Enter key to advance trials
"""
import matplotlib
from hrtf_relearning.utils.mpl_backend import use_interactive
use_interactive()
import numpy
import slab
import freefield
import datetime
import logging
from pynput import keyboard

import hrtf_relearning
from hrtf_relearning.experiment.localization.localization_helpers.make_sequence import make_sequence
from hrtf_relearning.experiment.localization.localization_helpers.stimulus import (
    make_gapped_pinknoise, make_rippled_pinknoise, RMS_TILT, RMS_CUE, RIPPLE_CUE_MAX)
from hrtf_relearning.experiment.misc import meta_motion
from hrtf_relearning.experiment.analysis.localization.localization_analysis import (
    plot_localization, plot_elevation_response,
)
from hrtf_relearning.utils import paths

ROOT = hrtf_relearning.PATH

# TDT playback rate. The stimulus helpers generate at slab's DEFAULT samplerate,
# and slab's own default is 8000 Hz -- so a dome block run from a script that
# never called slab.set_default_samplerate plays a stimulus synthesised at 8 kHz
# out of a 48828 Hz converter: 6.1x upshift, 37 ms instead of 225 ms, nothing
# below ~500 Hz. It is audible immediately (thin, no low end) but easy to
# mistake for a change in the stimulus itself. Localization_AR sets the rate
# from the HRIR in its __init__; the dome has no HRIR to take it from, so it is
# pinned here.
SAMPLERATE = 48828


class LocalizationDome:
    """
    Localization test using real dome loudspeakers.

    Plays pink noise bursts from the vertical midline speakers (az ≈ 0)
    that match the HRIR recording locations. Head orientation at response
    time is captured via the meta-motion sensor, consistent with
    Localization_AR.Localization.

    Parameters
    ----------
    subject : Subject
    hrir_settings : dict
        Must contain 'name' (str, SOFA basename) and optionally 'hp' (str,
        headphone model). Used only for sequence metadata — no file is loaded.
    loc_settings : dict, optional
        Sequence and stimulus parameters. Keys: 'targets_per_speaker' (int),
        'min_distance' (float), 'gain' (float), 'stim' ('noise' | 'ripple'),
        'stim_settings' (dict, envelope parameters when stim='ripple').

    Notes
    -----
    The 'ripple' option is here to test the random source envelope in the
    FREE FIELD, with the listener's own ears and no processing chain: no
    headphone equalization, no binaural synthesis, no reverberation model, no
    head-tracked convolution. If the envelope costs elevation gain here, it is
    the envelope and nothing else. Read the result as an UPPER BOUND on
    localizability: own ears give deeper, sharper spectral cues than the
    cepstrally smoothed and donor-modified transfer functions used in the
    learning experiment, so an envelope that leaves dome performance intact can
    still be too deep for a modified HRTF. Confirm in AR before committing a
    value to the learning-transfer protocol.
    """

    def __init__(self, subject, loc_settings=None):
        self.subject = subject
        date = datetime.datetime.now().strftime('%d.%m_%H-%M')
        self.filename = f"{subject.id}_{date}_dome"

        if loc_settings is None:
            loc_settings = {
                'targets_per_speaker': 3,
                'min_distance': 15,
            }
        self.stim_type = loc_settings.get('stim', 'noise')
        self.stim_settings = loc_settings.get('stim_settings', {}) or {}
        # see SAMPLERATE -- must happen before any stimulus is synthesised
        slab.set_default_samplerate(loc_settings.get('samplerate', SAMPLERATE))

        # Vertical midline speaker positions (hardcoded to match dome layout)
        midline = numpy.array([[  0. , -37.5],
                               [  0. , -25. ],
                               [  0. , -12.5],
                               [  0. ,   0. ],
                               [  0. ,  12.5],
                               [  0. ,  25. ],
                               [  0. ,  37.5]])
        self.sequence = make_sequence({'kind': 'standard', **loc_settings}, midline)
        self.sequence.name = self.filename
        self.sequence.label = 'dome'
        # 'noise' | 'ripple', matching Localization_AR. Dome runs recorded before
        # the ripple option was added carry the old label 'pinknoise_burst';
        # treat that as 'noise'.
        self.sequence.stim = self.stim_type
        self.sequence.stim_settings = self.stim_settings
        # per-trial source-spectrum recipe, appended by play_trial(). Empty for
        # 'noise'; for 'ripple' it holds that trial's DCT coefficients, so the
        # source spectrum of every trial is exactly reconstructible.
        self.sequence.stim_params = []
        self.target = None

    def write(self):
        self.subject.localization[self.filename] = self.sequence
        self.subject.write()

    def run(self):
        if freefield.PROCESSORS.mode != 'play_rec':
            freefield.initialize('dome', default='play_rec', sensor_tracking=False)
        self.motion_sensor = self._init_sensor()

        try:
            for self.target in self.sequence:
                self.wait_for_enter('Look at the center and press Enter...')
                self.motion_sensor.calibrate()
                self.play_trial()

            self.subject.last_sequence = self.sequence
            self.write()
            logging.info('Dome localization complete.')
            plot_dir = paths.subject_plot_dir(self.subject.id)
            plot_elevation_response(self.sequence, filepath=plot_dir)
            plot_localization(self.sequence, report_stats=['elevation'], filepath=plot_dir)
        finally:
            self.motion_sensor.halt()

    def play_trial(self):
        stim, params = self.make_stim(stim=self.stim_type,
                                      stim_settings=self.stim_settings,
                                      return_params=True)
        self.sequence.stim_params.append(params)
        speaker = freefield.pick_speakers((float(self.target[0]), float(self.target[1])))[0]
        freefield.set_signal_and_speaker(signal=stim, speaker=speaker.index, equalize=True)
        freefield.play()
        freefield.wait_to_finish_playing()
        self.wait_for_enter()
        response = self.motion_sensor.get_pose()
        progress = self.sequence.this_n / len(self.sequence.conditions) * 100
        logging.info(f'{progress:.1f}% | Target: {self.target} | Response: {response}')
        self.sequence.add_response(numpy.array((response, self.target)))
        self.write()

    @staticmethod
    def _legacy_burst_train_level(level=85):
        """Overall RMS of the pre-unification dome burst-train (5x25 ms bursts
        each ramped 10 ms, 4x25 ms gaps). Used only to preserve the dome's
        loudness after switching to the AR synthesis, so the current AR<->dome
        by-ear match (gain 0.07) stays valid until re-calibrated with KEMAR."""
        noise = slab.Sound.pinknoise(duration=0.025, level=level).ramp(when='both', duration=0.01)
        silence = slab.Sound.silence(duration=0.025)
        stim = slab.Sound.sequence(noise, silence, noise, silence, noise,
                                   silence, noise, silence, noise)
        return float(numpy.mean(stim.ramp(when='both', duration=0.01).level))

    @staticmethod
    def make_stim(level=None, stim='noise', stim_settings=None, return_params=False):
        """225 ms gapped pinknoise, synthesis IDENTICAL to the AR/VR condition
        (localization_helpers.stimulus). Only the overall loudness is
        dome-specific: `level=None` (default) preserves the loudness of the old
        dome burst-train so the existing gain match holds; pass an explicit dB
        value once you re-calibrate with KEMAR recordings.

        stim='ripple' recolours the train with a fresh random envelope on every
        call, exactly as in the AR condition. The level is set AFTER filtering,
        so both conditions play at the same overall rms -- note that equal rms
        is not equal loudness for a recoloured stimulus, so expect ripple tokens
        to differ slightly in loudness from each other. That is intrinsic to the
        manipulation and is the same in AR.

        `return_params=False` keeps the original signature (returns a Sound), so
        existing callers such as match_ar_dome_loudness are unaffected.
        """
        stim_settings = stim_settings or {}
        # Guard against synthesising at slab's 8000 Hz module default -- see
        # SAMPLERATE. 8000 is never a rate this experiment uses, so reaching
        # here with it set means nobody configured slab in this session.
        if slab.get_default_samplerate() != SAMPLERATE:
            logging.warning('slab default samplerate was %d, forcing %d for the dome stimulus',
                            slab.get_default_samplerate(), SAMPLERATE)
            slab.set_default_samplerate(SAMPLERATE)
        if stim == 'noise':
            sound, params = make_gapped_pinknoise(), {'kind': 'noise'}
        elif stim == 'ripple':
            sound, params = make_rippled_pinknoise(
                rms_tilt=stim_settings.get('rms_tilt', RMS_TILT),
                rms_cue=stim_settings.get('rms_cue', RMS_CUE),
                flat_rms=stim_settings.get('flat_rms', None),
                ripple_max=stim_settings.get('ripple_max', RIPPLE_CUE_MAX))
        else:
            raise ValueError('stim must be "noise" or "ripple".')
        sound.level = LocalizationDome._legacy_burst_train_level() if level is None else level
        return (sound, params) if return_params else sound

    @staticmethod
    def _init_sensor():
        device = meta_motion.get_device()
        state = meta_motion.State(device)
        return meta_motion.Sensor(state)

    @staticmethod
    def wait_for_enter(msg=None):
        if msg:
            print(msg)
        def on_press(key):
            if key == keyboard.Key.enter:
                listener.stop()
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()
