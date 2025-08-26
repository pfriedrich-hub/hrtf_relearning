import argparse
import multiprocessing as mp
import pybinsim
import datetime
import time
from pythonosc import udp_client
from experiment.misc import meta_motion
from analysis.localization_analysis import *
from experiment.misc.make_sequence import *
from hrtf.processing.binsim.hrir2wav import *
from experiment.Subject import Subject
date = datetime.datetime.now()
date = f'{date.strftime("%d")}.{date.strftime("%m")}.{date.strftime("%H")}:{date.strftime("%M")}'
logging.getLogger().setLevel('INFO')
pybinsim.logger.setLevel(logging.WARNING)
data_dir = Path.cwd() / 'data'

subject = 'PF'

# hrtf_name ='KU100_HRIR_L2702'
hrtf_name ='single_notch'
subject_id = f'{subject}_{hrtf_name}'

class Localization:
    """
    Localization test:
        Test localization at uniformly random positions within sectors
    """
    def __init__(self, subject_id, hrtf_name):
        # make trial sequence and write to subject
        azimuth_range = (-30, 30)
        elevation_range = (-30, 30)
        sector_size = (10, 10)
        targets_per_sector = 3
        min_distance = 30
        self.gain = .5
        self.subject = Subject(subject_id)
        self.filename = subject_id + '_loc_' + date
        self.subject.localization[self.filename] = make_sequence(azimuth_range, elevation_range, sector_size, # (azimuth_size, elevation_size)
            targets_per_sector, min_distance)
        self.subject.write()

        # metadata
        hrtf = slab.HRTF(data_dir / 'hrtf' / 'sofa' / f'{hrtf_name}.sofa')
        slab.set_default_samplerate(hrtf.samplerate)
        self.hrtf_sources = hrtf.sources.vertical_polar
        self.stim_path = data_dir / 'hrtf' / 'wav' / hrtf_name / 'sounds' / 'localization.wav'
        self.target = None

        # init pybinsim
        self.osc_client_1 = self._init_osc_client(port=10000)
        self.osc_client_2 = self._init_osc_client(port=10003)
        mp.Process(target=self.binsim_stream, args=()).start()

        # init motion sensor
        self.motion_sensor = self.init_sensor()
        time.sleep(.2)

    def run(self):
        sequence = self.subject.localization[self.filename]
        # for self.target in self.subject.localization[self.filename]:
        for self.target in sequence:
            progress = sequence.this_n / len(sequence.conditions) * 100
            logging.info(f'{progress}% | Target: {self.target}')
            # calibrate sensor
            input('Look at the Center and press Enter')
            self.motion_sensor.calibrate()
            # generate and play stim, get pose response
            self.play_trial()
            # write to file
            self.subject.write()
        return

    def play_trial(self):
        # generate stimulus
        noise = slab.Sound.pinknoise(duration=0.025, level=90)
        noise = noise.ramp(when='both', duration=0.01)
        silence = slab.Sound.silence(duration=0.025)
        stim = slab.Sound.sequence(noise, silence, noise, silence, noise,
                                   silence, noise, silence, noise)
        stim.ramp('both', 0.01)
        stim.write(self.stim_path)
        # play stim
        self.play_sound()
        time.sleep(stim.duration)
        # get response
        input('Aim at Sound and press Enter to confirm\n')
        response = self.motion_sensor.get_pose()
        time.sleep(.25)
        self.subject.localization[self.filename].add_response(numpy.array((response, self.target)))

    def play_sound(self):
        # set filter
        pose = self.motion_sensor.get_pose()
        # set distance for play_session
        relative_coords = self.target - pose
        # find the closest filter idx and send to pybinsim
        relative_coords[0] = (-relative_coords[0] + 360) % 360  # mirror and convert to HRTF convetion [0 < az < 360]
        rel_target = numpy.array((relative_coords[0], relative_coords[1], self.hrtf_sources[0, 2]))
        filter_idx = numpy.argmin(numpy.linalg.norm(rel_target - self.hrtf_sources, axis=1))
        rel_hrtf_coords = self.hrtf_sources[filter_idx]
        self.osc_client_1.send_message('/pyBinSim_ds_Filter', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                        float(rel_hrtf_coords[0]), float(rel_hrtf_coords[1]), 0,
                                                        0, 0, 0])
        logging.debug(f'set filter for {self.hrtf_sources[filter_idx]}')
        # play
        self.osc_client_2.send_message('/pyBinSimLoudness', self.gain)
        self.osc_client_2.send_message('/pyBinSimFile', str(self.stim_path))
        time.sleep(.5)
        self.osc_client_2.send_message('/pyBinSimLoudness', 0)

    @staticmethod
    def _init_osc_client(port):
        host = '127.0.0.1'
        mode = 'client'
        ip = '127.0.0.1'
        parser = argparse.ArgumentParser()
        parser.add_argument("--host", default=host)
        parser.add_argument("--mode", default=mode)
        parser.add_argument("--ip", default=ip, help="The ip of the OSC server")
        parser.add_argument("--port", type=int, default=port, help="The port the OSC server is listening on")
        args = parser.parse_args()
        return udp_client.SimpleUDPClient(args.ip, args.port)

    def binsim_stream(self):
        binsim = pybinsim.BinSim(data_dir / 'hrtf' / 'wav' / hrtf_name / f'{hrtf_name}_test_settings.txt')
        binsim.stream_start()  # run binsim loop

    def init_sensor(self):
        # init motion sensor
        device = meta_motion.get_device()  # Ensure this function initializes the hardware correctly
        state = meta_motion.State(device)
        return meta_motion.Sensor(state)

if __name__ == "__main__":
    hrir2wav(hrtf_name)
    loc_test = Localization(subject_id, hrtf_name)
    loc_test.run()
    sequence = Subject(subject_id).localization[loc_test.filename]
    plot_localization(sequence)
