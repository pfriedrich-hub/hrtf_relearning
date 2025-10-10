import multiprocessing as mp
import datetime
import time
from pathlib import Path
import logging
from pythonosc import udp_client
from experiment.misc import meta_motion
from analysis.localization import *
from experiment.misc.make_sequence import *
from hrtf.processing.hrtf2binsim import hrtf2binsim
from experiment.Subject import Subject
date = datetime.datetime.now()
date = f'{date.strftime("%d")}.{date.strftime("%m")}_{date.strftime("%H")}.{date.strftime("%M")}'
logging.getLogger().setLevel('INFO')
data_dir = Path.cwd() / 'data'

# --- Load Subject ----
id = 'PF'
subject = Subject(id)

# --- HRTF settings ----

# --- select sofa file
sofa_name ='KU100'
# sofa_name ='single_notch'
# sofa_name ='kemar'

# ---- specify ear for unilateral testing, None defaults to binaural testing
# ear = None
ear = 'left'

# --- load and process HRIR
hrir = hrtf2binsim(sofa_name, ear, overwrite=False)
slab.set_default_samplerate(hrir.samplerate)
hrir_dir = Path.cwd() / 'data' / 'hrtf' / 'wav' / hrir.name

class Localization:
    """
    Localization test:
        Test localization at uniformly random positions within sectors
    """
    def __init__(self, subject, hrir):
        # make trial sequence and write to subject
        self.settings = {'azimuth_range': (-35, 35), 'elevation_range': (-35, 35), 'sector_size': (14, 14),
                         'targets_per_sector': 3, 'min_distance': 30, 'gain': .5}
        # self.settings = {'azimuth_range': (-35, 35), 'elevation_range': (-14, 14), 'sector_size': (14, 14),
        #                  'targets_per_sector': 3, 'min_distance': 10, 'gain': .5}
        self.subject = subject
        self.filename = subject.id + f'_{hrir.name}' + '_loc_' + date

        # metadata
        slab.set_default_samplerate(hrir.samplerate)
        self.hrir_sources = hrir.sources.vertical_polar
        self.stim_path = data_dir / 'hrtf' / 'wav' / hrir.name / 'sounds' / 'localization.wav'
        self.target = None
        # self.stim = self.make_stim()  # use the same stimulus across trials

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
        self.sequence = make_sequence_from_sources(self.settings, self.hrir_sources)
        self.sequence.name = self.filename
        self.write()
        # for self.target in self.subject.localization[self.filename]:
        for self.target in self.sequence:
            input('Look at the Center and press Enter')  # calibrate sensor
            self.motion_sensor.calibrate()
            self.play_trial()  # generate and play stim, get pose response
            self.write()  # write to file
        logging.info('Finished.')
        return

    def play_trial(self):
        # generate stimulus
        self.stim = self.make_stim()  # generate a new stim each trial
        self.stim.write(self.stim_path)
        # play stim
        self.play_sound()
        time.sleep(self.stim.duration)
        # get response
        input('Aim at Sound and press Enter to confirm\n')
        response = self.motion_sensor.get_pose()
        progress = self.sequence.this_n / len(self.sequence.conditions) * 100
        logging.info(f'{progress:.1f}% | Target: {self.target} | Response: {response}')
        # logging.debug(f'got response: {response}')
        time.sleep(.25)
        self.subject.localization[self.filename].add_response(numpy.array((response, self.target)))

    def play_sound(self):
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
        self.osc_client_2.send_message('/pyBinSimFile', str(self.stim_path))
        time.sleep(.5)
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

    def init_sensor(self):
        # init motion sensor
        device = meta_motion.get_device()  # Ensure this function initializes the hardware correctly
        state = meta_motion.State(device)
        return meta_motion.Sensor(state)

    @staticmethod
    def make_stim():
        noise = slab.Precomputed(lambda: slab.Sound.pinknoise(duration=0.025, level=90)
                                 .ramp(when='both', duration=0.01), n=5)
        silence = slab.Sound.silence(duration=0.025)
        stim = slab.Sound.sequence(noise[0], silence, noise[1], silence, noise[2],
                                   silence, noise[3], silence, noise[4])

        # noise = slab.Sound.pinknoise(duration=0.025, level=90)
        # noise = noise.ramp(when='both', duration=0.01)
        # silence = slab.Sound.silence(duration=0.025)
        # stim = slab.Sound.sequence(noise, silence, noise, silence, noise,
        #                            silence, noise, silence, noise)
        # stim.ramp('both', 0.01)
        return stim

if __name__ == "__main__":
    loc_test = Localization(subject, hrir)
    loc_test.run()
    sequence = subject.localization[loc_test.filename]
    plot_localization(sequence, report_stats=['elevation', 'azimuth'],
                      filepath=data_dir / 'results' / 'plot' / subject.id)

    #todo add target p
