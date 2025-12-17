from hrtf_relearning.analysis.localization import *  # also set mpl backend
import multiprocessing as mp
import hrtf_relearning
import datetime
import time
from pathlib import Path
from pythonosc import udp_client
from hrtf_relearning.experiment.misc.training_helpers import meta_motion
from hrtf_relearning.experiment.misc.make_sequence import *
from hrtf_relearning.hrtf.binsim.hrtf2binsim import hrtf2binsim
from hrtf_relearning.experiment.Subject import Subject
from pynput import keyboard
date = datetime.datetime.now()
date = f'{date.strftime("%d")}.{date.strftime("%m")}_{date.strftime("%H")}.{date.strftime("%M")}'
logging.getLogger().setLevel('INFO')
ROOT = Path(hrtf_relearning.__file__).resolve().parent


# --- settings ----
SUBJECT_ID = "test"
HRIR_NAME = "KU100"  # 'KU100', 'kemar', etc.
EAR = None

# --- load and process HRIR
hrir = hrtf2binsim(HRIR_NAME, EAR, reverb=True, hp_filter=True,
                   convolution='cpu', storage='cpu', overwrite=False)
slab.set_default_samplerate(hrir.samplerate)
HRIR_DIR = Path.cwd() / "data" / "hrtf" / "binsim" / hrir.name
subject = Subject(SUBJECT_ID)

class Localization:
    """
    Localization test:
        Test localization at uniformly random positions within sectors
    """
    def __init__(self, subject, hrir):
        # make trial sequence and write to subject

        self.settings = {'kind': 'sectors',
                         'azimuth_range': (-35, 35), 'elevation_range': (-35, 35),
                         'sector_size': (14, 14),
                         'targets_per_sector': 3, 'replace': False, 'min_distance': 15,
                         'gain': .2}
        # alternative setting: play 3 times from each source in the hrir (works well for dome recorded hrirs)
        #self.settings = {'kind': 'standard', 'azimuth_range': (-45, 45), 'elevation_range': (-45, 45),
         #                 'targets_per_speaker': 2, 'min_distance': 30, 'gain': .4}
        self.subject = subject
        self.filename = subject.id + f'_{hrir.name}' + '_loc_' + date
        # metadata
        slab.set_default_samplerate(hrir.samplerate)
        self.hrir_sources = hrir.sources.vertical_polar
        self.sound_path = ROOT / 'data' / 'hrtf' / 'binsim' / hrir.name / 'sounds'
        self.target = None

        # make sequence
        self.sequence = make_sequence(self.settings, self.hrir_sources)
        # self.sequence = make_sequence(self.settings)
        self.sequence.name = self.filename

    def write(self):
        self.subject.localization[self.filename] = self.sequence
        self.subject.write()

    def run(self):
        # init pybinsim
        self.osc_client_1 = self._make_osc_client(port=10000)
        self.osc_client_2 = self._make_osc_client(port=10003)
        self.binsim_worker = mp.Process(target=self._binsim_stream, args=(hrir.name,))
        self.binsim_worker.start()

        # init motion sensor
        self.motion_sensor = self.init_sensor()
        time.sleep(.2)

        self.play_sound('beep')
        for self.target in self.sequence:
            self.wait_for_button('Look at the Center and press Enter')
            self.motion_sensor.calibrate()
            self.play_trial()  # generate and play stim, get pose response and write to file
        self.subject.last_sequence = self.sequence
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
        self.sequence.add_response(numpy.array((response, self.target)))
        self.write()  # write to file

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
        binsim = pybinsim.BinSim(ROOT / 'data'  / 'hrtf' / 'binsim' / hrir_name / f'{hrir_name}_test_settings.txt')
        binsim.stream_start()  # run binsim loop

    @staticmethod
    def init_sensor():
        # init motion sensor
        device = meta_motion.get_device()  # Ensure this function initializes the hardware correctly
        state = meta_motion.State(device)
        return meta_motion.Sensor(state)



    @staticmethod
    def make_stim():
        # stim = slab.Sound.pinknoise(duration=0.5, level=90).ramp(when='both', duration=0.01)

        stim = slab.Sound.pinknoise(duration=0.225, level=80).ramp(when='both', duration=0.01)
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
        # stim.ramp('both', 0.01)
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
    plot_localization(sequence, report_stats=['elevation'], filepath=ROOT / 'data'  / 'results' / 'plot' / subject.id)
    plot_elevation_response(sequence, filepath=ROOT / 'data'  / 'results' / 'plot' / subject.id)

