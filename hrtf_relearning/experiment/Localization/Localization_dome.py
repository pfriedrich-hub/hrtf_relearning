"""
Dome loudspeaker localization test.

Mirrors Localization_AR.Localization in API and conventions:
  - Subject for data persistence
  - make_sequence (std_targets) for trial sequences
  - meta-motion sensor for head-pose response
  - Enter key to advance trials
"""
import matplotlib
matplotlib.use('TkAgg')
import numpy
import slab
import freefield
import datetime
import logging
from pynput import keyboard

import hrtf_relearning
from hrtf_relearning.experiment.misc.localization_helpers.make_sequence import make_sequence
from hrtf_relearning.experiment.misc.training_helpers import meta_motion

ROOT = hrtf_relearning.PATH


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
    hrir : slab.HRTF
    repetitions : int
        How many times each speaker location is presented.
    min_distance : float
        Minimum angular distance (°) between successive targets.
    """

    # todo init class with all settings, so we can call it from test_hrir_recording.py
    def __init__(self, subject, hrir, repetitions=3, min_distance=15):
        self.subject = subject
        date = datetime.datetime.now().strftime('%d.%m_%H-%M')
        self.filename = f"{subject.id}_{date}_{hrir.name}_dome"

        slab.set_default_samplerate(hrir.samplerate)
        self.hrir_sources = hrir.sources.vertical_polar  # (az, el, r)

        # Pre-filter to vertical midline (az ≈ 0) — same locations as HRIR recording
        sources = hrir.sources.vertical_polar
        midline = sources[numpy.isclose(sources[:, 0], 0, atol=1.0), :2]  # (az, el) # hardcode instead
        # hardcode instead
        midline = numpy.array([[  0. , -37.5],
       [  0. , -25. ],
       [  0. , -12.5],
       [  0. ,   0. ],
       [  0. ,  12.5],
       [  0. ,  25. ],
       [  0. ,  37.5]])
        settings = {
            'kind': 'standard',
            'targets_per_speaker': repetitions,
            'min_distance': min_distance,
            'gain': 1,
        }
        self.sequence = make_sequence(settings, midline)
        self.sequence.name = self.filename
        self.sequence.hrir = hrir.name
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
        finally:
            self.motion_sensor.halt()

    def play_trial(self):
        stim = self.make_stim()
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
    def make_stim(level=85):
        noise = slab.Sound.pinknoise(duration=0.025, level=level).ramp(when='both', duration=0.01)
        silence = slab.Sound.silence(duration=0.025)
        stim = slab.Sound.sequence(noise, silence, noise, silence, noise,
                                   silence, noise, silence, noise)
        return stim.ramp(when='both', duration=0.01)

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
