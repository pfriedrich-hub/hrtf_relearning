from analysis.localization import *  # also set mpl backend
import multiprocessing as mp
import datetime
import time
from pathlib import Path
from pythonosc import udp_client
from experiment.misc.training_helpers import meta_motion
from experiment.misc.make_sequence import *
from hrtf.processing.hrtf2binsim import hrtf2binsim
from experiment.Subject import Subject
from pynput import keyboard
date = datetime.datetime.now()
date = f'{date.strftime("%d")}.{date.strftime("%m")}_{date.strftime("%H")}.{date.strftime("%M")}'
logging.getLogger().setLevel('INFO')
data_dir = Path.cwd() / 'data'

# --- Load Subject ----
id = 'PF'
subject = Subject(id)

# --- HRTF settings ----
# hrir='KU100'
# hrir ='single_notch'
# hrir = 'pf_just_itd'
hrir = 'pf_high_res_itd'

# ---- specify ear for unilateral testing, None defaults to binaural testing
ear = None
# ear = 'left'
reverb = True

# --- load and process HRIR
hrir = hrtf2binsim(hrir, ear, reverb, overwrite=False)
hrir_dir = Path.cwd() / 'data' / 'hrtf' / 'binsim' / hrir.name

class Localization:
    """
    Localization test:
        Test localization at uniformly random positions within sectors
    """
    def __init__(self, subject, hrir):
        # make trial sequence and write to subject
        # self.settings = {'azimuth_range': (-35, 0), 'elevation_range': (-35, 35), 'sector_size': (7, 14),
        #                  'targets_per_sector': 3, 'min_distance': 15, 'gain': .5}
        self.settings = {'azimuth_range': (-35, 35), 'elevation_range': (-50, 50), 'sector_size': (14, 20),
                         'targets_per_sector': 3, 'min_distance': 35, 'gain': .5}  # azimuth test
        # self.settings = {'azimuth_range': (-40, 40), 'elevation_range': (-30, 30), 'sector_size': (20, 20),
        #                  'targets_per_sector': 3, 'min_distance': 20, 'gain': .8}  # ff HRTF
        self.subject = subject
        self.filename = subject.id + f'_{hrir.name}' + '_loc_' + date

        # metadata
        slab.set_default_samplerate(hrir.samplerate)
        self.hrir_sources = hrir.sources.vertical_polar
        self.sound_path = data_dir / 'hrtf' / 'binsim' / hrir.name / 'sounds'
        self.target = None

        # init pybinsim
        self.osc_client_1 = self._make_osc_client(port=10000)
        self.osc_client_2 = self._make_osc_client(port=10003)
        self.binsim_worker = mp.Process(target=self._binsim_stream, args=(hrir.name,))
        self.binsim_worker.start()

        # init motion sensor
        self.motion_sensor = self.init_sensor()
        time.sleep(.2)

    def write(self):
        self.subject.localization[self.filename] = self.sequence
        self.subject.write()

    def run(self):
        self.sequence = make_sequence_from_sources(self.settings, self.hrir)
        # self.sequence = make_sequence(self.settings)
        self.sequence.name = self.filename
        self.write()
        self.play_sound('beep')
        for self.target in self.sequence:
            self.wait_for_button('Look at the Center and press Enter')
            self.motion_sensor.calibrate()
            self.play_trial()  # generate and play stim, get pose response
            self.write()  # write to file
        self.sequence.response_errors = target_p(self.sequence, show=False)
        self.write()
        logging.info('Finished.')
        return

    def play_trial(self):
        # generate stimulus
        self.stim = self.make_stim()  # generate a new stim each trial
        self.stim.write(self.sound_path / 'localization.wav')
        # play stim
        self.play_stimulus()
        time.sleep(self.stim.duration)
        # get response
        self.wait_for_button()
        response = self.motion_sensor.get_pose()
        progress = self.sequence.this_n / len(self.sequence.conditions) * 100
        logging.info(f'{progress:.1f}% | Target: {self.target} | Response: {response}')
        time.sleep(.25)
        self.subject.localization[self.filename].add_response(numpy.array((response, self.target)))

    def play_stimulus(self):
        pose = self.motion_sensor.get_pose()
        relative_coords = self.target - pose  # mimic freefield setup
        # find the closest filter idx and send to pybinsim
        relative_coords[0] = (-relative_coords[0] + 360) % 360  # mirror and convert to HRTF convention [0 < az < 360]
        rel_target = numpy.array((relative_coords[0], relative_coords[1], self.hrir_sources[0, 2]))
        filter_idx = numpy.argmin(numpy.linalg.norm(rel_target - self.hrir_sources, axis=1))
        rel_hrtf_coords = self.hrir_sources[filter_idx]
        self.osc_client_1.send_message('/pyBinSim_ds_Filter', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                        float(rel_hrtf_coords[0]), float(rel_hrtf_coords[1]), 0,
                                                        0, 0, 0])
        logging.debug(f'Set filter for {self.hrir_sources[filter_idx]}')
        # play
        self.osc_client_2.send_message('/pyBinSimLoudness', self.settings['gain'])
        self.osc_client_2.send_message('/pyBinSimFile', str(self.sound_path / 'localization.wav'))
        time.sleep(.5)
        self.osc_client_2.send_message('/pyBinSimLoudness', 0)

    def play_sound(self, kind):
        logging.info(f'Playing {kind} sound')
        name = f'{kind}.wav'
        duration = slab.Sound(self.sound_path / name).duration
        self.osc_client_2.send_message('/pyBinSimLoudness', self.settings['gain'])
        self.osc_client_2.send_message('/pyBinSimFile', str(self.sound_path / name))
        time.sleep(duration)
        self.osc_client_2.send_message('/pyBinSimLoudness', 0)

    @staticmethod
    def _make_osc_client(port, ip='127.0.0.1'):
        return udp_client.SimpleUDPClient(ip, port)

    @staticmethod
    def _binsim_stream(hrir_name):
        import pybinsim
        pybinsim.logger.setLevel(logging.ERROR)
        binsim = pybinsim.BinSim(data_dir / 'hrtf' / 'wav' / hrir_name / f'{hrir_name}_test_settings.txt')
        binsim.stream_start()  # run binsim loop

    @staticmethod
    def init_sensor():
        # init motion sensor
        device = meta_motion.get_device()  # Ensure this function initializes the hardware correctly
        state = meta_motion.State(device)
        return meta_motion.Sensor(state)

    @staticmethod
    def make_stim():
        stim = slab.Sound.pinknoise(duration=0.225, level=90).ramp(when='both', duration=0.01)
        n_silent = (numpy.arange(25,221,25).reshape(4,2) * stim.samplerate / 1000).astype(int)
        ramp_len = int(.005 * stim.samplerate)
        half_len = int(ramp_len / 2)
        for start, end in n_silent:
            ramp_up = 0.5 * (1 - numpy.cos(numpy.linspace(0, numpy.pi, ramp_len)))
            ramp_down = 0.5 * (1 - numpy.cos(numpy.linspace(numpy.pi, 0, ramp_len)))
            ramp_up = ramp_up[:, numpy.newaxis]
            ramp_down = ramp_down[:, numpy.newaxis]
            # Apply ramps at the edges of the silent region
            stim.data[start - half_len: start + half_len] *= (1 - ramp_up)
            stim.data[end - half_len: end + half_len] *= (1 - ramp_down)
            # Silence the center
            stim.data[start + half_len: end - half_len] = 0
        # noise = slab.Sound.pinknoise(duration=0.025, level=90)
        # noise = noise.ramp(when='both', duration=0.01)
        # silence = slab.Sound.silence(duration=0.025)
        # stim = slab.Sound.sequence(noise, silence, noise, silence, noise,
        #                            silence, noise, silence, noise)
        stim.ramp('both', 0.01)
        return stim

    @staticmethod
    def wait_for_button(msg=None):
        if msg: print(msg)
        def on_press(key):
            if key == keyboard.Key.enter:
                listener.stop()  # stop listening once Enter is pressed
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()  # block until listener.stop() is called

if __name__ == "__main__":
    loc_test = Localization(subject, hrir)
    loc_test.run()
    sequence = subject.localization[loc_test.filename]
    plot_localization(sequence, report_stats=['elevation', 'azimuth'],
                      filepath=data_dir / 'results' / 'plot' / subject.id)

