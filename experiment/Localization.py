import argparse
import logging
import multiprocessing as mp
from experiment.misc.localization_analysis import *
from pythonosc import udp_client
from experiment.misc import meta_motion
from experiment.misc.make_sequence import *
from hrtf.processing.hrtf2wav import *
logging.getLogger().setLevel('INFO')
import pybinsim
pybinsim.logger.setLevel(logging.INFO)
import datetime
date = datetime.datetime.now()
date = f'{date.strftime("%d")}_{date.strftime("%m")}'
data_dir = Path.cwd() / 'data'
from experiment.Subject import Subject

subject_id = 'test'
hrtf_name ='KU100_HRIR_L2702'

class Localization:
    def __init__(self, subject_id, hrtf_name):
        # make trial sequence and write to subject
        azimuth_range = (-40, 40)
        elevation_range = (-40, 40)
        sector_size = (20, 20)
        targets_per_sector = 3
        min_distance = 20
        self.loudness = .5
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
        # enable audio
        for self.target in self.subject.localization[self.filename]:
            logging.info(f'Target: {self.target}')
            # calibrate sensor
            input('Look at the Center and press Enter\n')
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
        input('Aim at Sound and press Enter to confirm.')
        response = self.motion_sensor.get_pose()
        # todo check if beep is neccessary
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
        self.osc_client_2.send_message('/pyBinSimLoudness', 0.5 * self.loudness)
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
    make_wav(hrtf_name)
    loc_test = Localization(subject_id, hrtf_name)
    loc_test.run()

    sequence = Subject(subject_id).localization[loc_test.filename]
    plot_localization(sequence)
