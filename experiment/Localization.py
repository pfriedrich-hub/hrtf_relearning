import argparse
import logging
import time
# from experiment.misc.localization_analysis import *
from pythonosc import udp_client
from experiment.misc import meta_motion
from experiment.misc.make_sequence import *
from hrtf.processing.hrtf2wav import *
logging.getLogger().setLevel('INFO')
import pybinsim
import datetime
date = datetime.datetime.now()
date = f'{date.strftime("%d")}_{date.strftime("%m")}'
data_dir = Path.cwd() / 'data'
from experiment.Subject import Subject


subject_id = 'test'
hrtf_name ='KU100_HRIR_L2702'
slab.set_default_samplerate(slab.HRTF(data_dir / 'hrtf' / 'sofa' / f'{hrtf_name}.sofa').samplerate)

class Localization:
    def __init__(self, subject_id, hrtf_name):
        # make trial sequence and write to subject
        self.subject = Subject(subject_id)
        self.filename = subject_id + '_loc_' + date
        self.subject.localization[self.filename], *_ = make_sequence(
            azimuth_range=(-30, 30),
            elevation_range=(-30, 30),
            sector_size=(10, 10),  # (azimuth_size, elevation_size)
            min_sector_distance=10,
            points_per_sector=1)
        self.subject.write()

        # metadata
        self.hrtf_sources = slab.HRTF(data_dir / 'hrtf' / 'sofa' / f'{hrtf_name}.sofa').sources.vertical_polar
        self.stim_path = data_dir / 'hrtf' / 'wav' / hrtf_name / 'sounds' / 'noise_burst.wav'
        self.target = None

        # init pybinsim
        self.osc_client = self._init_osc_client()
        self.binsim = self.init_pybinsim()

        # init motion sensor
        self.motion_sensor = self.init_sensor()
        time.sleep(.2)

    def run(self):
        # enable audio
        self.binsim.config.configurationDict['loudnessFactor'] = 0.5
        for self.target in self.subject.localization[self.filename]:
            logging.info(f'Target: {self.target}')
            # calibrate sensor
            input('Look at the Center and press Enter\n')
            self.motion_sensor.calibrate()
            # generate and play stim, get pose response
            self.play_trial()
            logging.info(f'{self.target}')
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
        self.play_sound(self.stim_path, self.target)
        time.sleep(stim.duration)

        # get response
        input('Aim at Sound and press Enter to confirm.')
        response = self.motion_sensor.get_pose()
        self.play_sound(data_dir / 'hrtf' / 'wav' / hrtf_name / 'sounds' / 'beep.wav', self.target)
        time.sleep(.25)
        self.subject.localization[self.filename].add_response(numpy.array((response, self.target)))

    def play_sound(self, path, target):
        # set filter
        pose = self.motion_sensor.get_pose()
        # set distance for play_session
        relative_coords = target - pose
        # find the closest filter idx and send to pybinsim
        relative_coords[0] = (-relative_coords[0] + 360) % 360  # mirror and convert to HRTF convetion [0 < az < 360]
        rel_target = numpy.array((relative_coords[0], relative_coords[1], self.hrtf_sources[0, 2]))
        filter_idx = numpy.argmin(numpy.linalg.norm(rel_target - self.hrtf_sources, axis=1))
        self.osc_client.send_message('/pyBinSim', [0, int(filter_idx), 0, 0, 0, 0, 0])
        logging.info(f'set filter for {self.hrtf_sources[filter_idx]}')
        # time.sleep(.1)
        # play
        self.osc_client.send_message('/pyBinSimFile', str(path))
        logging.info(f'Playing {path.stem}')

    @staticmethod
    def _init_osc_client():
        host = '127.0.0.1'
        mode = 'client'
        ip = '127.0.0.1'
        port = 10000
        parser = argparse.ArgumentParser()
        parser.add_argument("--host", default=host)
        parser.add_argument("--mode", default=mode)
        parser.add_argument("--ip", default=ip, help="The ip of the OSC server")
        parser.add_argument("--port", type=int, default=port, help="The port the OSC server is listening on")
        args = parser.parse_args()
        return udp_client.SimpleUDPClient(args.ip, args.port)

    def init_pybinsim(self):
        binsim = pybinsim.BinSim(data_dir / 'hrtf' / 'wav' / hrtf_name / f'{hrtf_name}_settings.txt')
        pybinsim.logger.setLevel(logging.INFO)
        binsim.soundHandler.loopSound = False
        binsim.config.configurationDict['loudnessFactor'] = 0
        self.osc_client.send_message('/pyBinSimFile', str(self.stim_path))
        binsim.stream_start()
        return binsim

    def init_sensor(self):
        # init motion sensor
        device = meta_motion.get_device()  # Ensure this function initializes the hardware correctly
        state = meta_motion.State(device)
        return meta_motion.Sensor(state)


if __name__ == "__main__":
    make_wav(hrtf_name)
    loc_test = Localization(subject_id, hrtf_name)
    loc_test.run()

    sequence = Subject(subject_id).localization[loc_test.filename]
    # localization_accuracy(sequence)



    #
    # @staticmethod
    # def _make_sequence(az_range, ele_range, min_dist, n_unique_targets, n_repteitions):
    #     """
    #     Create a sequence of n_trials target locations
    #     with more than min_dist angular distance between successive targets
    #
    #     randomly select [n_trials / 3] unique target locations
    #     create a sequence of n_trials target locations so that each unique location appears 3 times in the sequence
    #     with min_dist angular distance between successive targets
    #     """
    #     logging.info('Setting up trial sequence.')
    #     azimuth_range = numpy.arange(az_range[0], az_range[1] + 1)  # Example: from -90° to 90° with 1° steps
    #     elevation_range = numpy.arange(ele_range[0], ele_range[1] + 1)
    #     target_azimuths = numpy.random.choice(azimuth_range, n_unique_targets, replace=False)
    #     target_elevations = numpy.random.choice(elevation_range, n_unique_targets, replace=False)
    #     targets = numpy.column_stack((target_azimuths, target_elevations))
    #     targets = numpy.repeat(targets, n_repteitions, axis=0)
    #     targets = numpy.random.permutation(targets).tolist()
    #     sequence = [targets.pop(0)]
    #     while targets:
    #         for tar in targets:
    #             if numpy.linalg.norm(numpy.subtract(tar, sequence[-1])) >= min_dist:
    #                 sequence.append(tar)
    #                 targets.remove(tar)
    #                 break  # Restart checking from the beginning
    #     return slab.Trialsequence(sequence)
    #

