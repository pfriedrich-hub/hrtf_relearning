import datetime
import time
from pathlib import Path
from hrtf_relearning.experiment.misc.training_helpers import meta_motion
from analysis.localization import *
from hrtf_relearning.experiment.misc.localization_helpers.make_sequence import *
from hrtf_relearning.hrtf.binsim.hrtf2binsim import hrtf2binsim
from hrtf_relearning.experiment.Subject import Subject
date = datetime.datetime.now()
date = f'{date.strftime("%d")}.{date.strftime("%m")}_{date.strftime("%H")}.{date.strftime("%M")}'
logging.getLogger().setLevel('INFO')
data_dir = Path.cwd() / 'data'

"""
A test version of the localization using slab instead of pybinsim.
"""
# --- Load Subject ----
id = 'PF'
subject = Subject(id)

# --- HRTF settings ---- #

# --- select sofa file
sofa_name ='KU100'
# sofa_name ='single_notch'
# sofa_name ='kemar'

# ---- specify ear for unilateral testing, None defaults to binaural testing
ear = None
# ear = 'left'

# --- load and process HRIR
hrir = hrtf2binsim(sofa_name, ear, overwrite=False)
slab.set_default_samplerate(hrir.samplerate)
hrir_dir = Path.cwd() / 'data' / 'hrtf' / 'binsim' / hrir.name

class Localization:
    """
    Localization test:
        Test localization at uniformly random positions within sectors
    """
    def __init__(self, subject, hrir):
        # make trial sequence and write to subject
        self.settings = {'azimuth_range': (-35, 35), 'elevation_range': (-14, 14), 'sector_size': (14, 14),
                         'targets_per_sector': 3, 'min_distance': 10, 'gain': .5}
        # self.settings = {'azimuth_range': (-1, 0), 'elevation_range': (-1, 0), 'sector_size': (1, 1),
        #                  'targets_per_sector': 15, 'min_distance': 0, 'gain': .5}
        self.subject = subject
        self.filename = subject.id + f'_{hrir.name}' + date

        # metadata
        slab.set_default_samplerate(hrir.samplerate)
        self.hrir = hrir
        self.hrir_sources = hrir.sources.vertical_polar
        self.target = None

        # init motion sensor
        self.motion_sensor = self.init_sensor()
        time.sleep(.2)

    def write(self):
        self.subject.localization[self.filename] = self.sequence
        self.subject.write()

    def run(self):
        self.sequence = make_sequence_from_sources(self.settings, self.hrir_sources)
        # self.sequence = make_sequence(self.settings)
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
        # play stim
        self.play_sound()
        time.sleep(self.stim.duration)
        # get response
        input('Aim at Sound and press Enter to confirm\n')
        response = self.motion_sensor.get_pose()
        progress = self.sequence.this_n / len(self.sequence.conditions) * 100
        logging.info(f'{progress}% | Target: {self.target} | Response: {response}')
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
        # --- use slab to filter and play the sound --- #
        self.hrir[filter_idx].apply(self.stim).play()
        time.sleep(self.stim.duration)

    def init_sensor(self):
        # init motion sensor
        device = meta_motion.get_device()  # Ensure this function initializes the hardware correctly
        state = meta_motion.State(device)
        return meta_motion.Sensor(state)

    @staticmethod
    def make_stim():
        noise = slab.Sound.pinknoise(duration=0.05, level=90)
        noise = noise.ramp(when='both', duration=0.01)
        silence = slab.Sound.silence(duration=0.025)
        stim = slab.Sound.sequence(noise, silence, noise, silence, noise,
                                   silence, noise, silence, noise)
        stim.ramp('both', 0.01)
        return stim

if __name__ == "__main__":
    loc_test = Localization(subject, hrir)
    loc_test.run()
    sequence = subject.localization[loc_test.filename]
    plot_localization(sequence, report_stats=['elevation', 'azimuth'],
                      filepath=data_dir / 'results' / 'plot' / subject.id)

