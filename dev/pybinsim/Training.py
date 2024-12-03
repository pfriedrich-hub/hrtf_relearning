from platform import processor

import matplotlib
matplotlib.use('TkAgg')
from dev.pybinsim.Pulse_Stream import Pulse_Stream
from matplotlib import pyplot as plt
import freefield
from pathlib import Path
import argparse
from pythonosc import udp_client
import time
import numpy
import slab
data_dir = Path.cwd() / 'data'
fs = 44100
slab.set_default_samplerate(fs)

# todo adjust pulse train to make it intuitive, get a good HRIR from jakab

class Training:
    def __init__(self, filename='kemar', target_size=5, target_time=.5, game_time=180, trial_time=30,
                 az_range=(-30, 30), ele_range=(-30, 30)):

        # general parameters
        self.filename = filename
        self.hrtf = slab.HRTF(data_dir / 'hrtf' / 'sofa' / str(filename + '.sofa'))
        self.target_size = target_size
        self.target_time = target_time
        self.game_time = game_time
        self.trial_time = trial_time
        self.az_range = az_range
        self.ele_range = ele_range
        self.target = None
        self.scores = []

        # init pybinsim streamer and osc messenger
        self.osc_client = self._make_osc_client()
        self.pulse_stream = Pulse_Stream(self.filename)
        self.filter_idx = 0  # filter index in the initial osc message

        # load sounds
        self.sounds = {
                        'coins': slab.Sound.read(data_dir / 'sounds' / 'coins.wav'),
                        'coin': slab.Sound.read(data_dir / 'sounds' / 'coin.wav'),
                        'buzzer': slab.Sound.read(data_dir / 'sounds' / 'buzzer.wav'),
                        }
        for sound, key in zip(self.sounds.values(), self.sounds.keys()):
            self.sounds[key] = sound.resample(fs)
            self.sounds[key].level = 75


    def __repr__(self):
        return f'{type(self)} Sessions played: {len(self.scores)} Scores: {repr(self.scores)}'

    def run(self):
        # init sensor
        freefield.initialize(setup='dome', default=None, device=None, sensor_tracking=True)
        freefield.SENSOR.set_fusion_mode('NDOF')
        # start background threads
        # self.pulse_stream.start()
        self.pulse_stream.audio_stream.start()
        self.pulse_stream.set_interval.start()
        while True:
            self.training_session()
            self._wait_for_button('Press Enter to play again.')

    def training_session(self):
        self.game_over, self.score = False, 0        # reset countdown and score
        self.game_start = time.time()
        while not self.game_over:  # loop over trials until end time has passed
            trial_prep = time.time()  # time between trials
            self.set_target()  # get next target
            self._wait_for_button('Press Enter to start.')
            freefield.calibrate_sensor(led_feedback=False, button_control=False)
            self.game_start += time.time() - trial_prep  # count time only while playing
            self.play_trial()
            self.scores.append(self.score)
            print(f'Run {len(self.scores)}: {self.score} points')

    def play_trial(self):
        self.trial_start = time.time()  # get trial start time
        time_on_target = 0
        count_down = False  # condition for counting time on target
        # within trial loop: continuously update headpose and monitor time
        while True:
            self.get_headpose()  # read headpose from sensor
            self.set_filter() # set the HRIR to pybinsim
            self.set_pulse_train() # convert distance to pulse interval and pass to pulse_stream object
            # check for hits
            if self.distance < self.target_size:
                if not count_down:  # start counting time as longs as pose matches target
                    time_on_target, count_down = time.time(), True
            else:
                time_on_target, count_down = time.time(), False  # reset timer if pose no longer matches target
            # end trial if goal conditions are met
            if time.time() > time_on_target + self.target_time:
                if time.time() - self.trial_start <= 3:
                    points = 2
                    self.end_trial('coins')
                else:
                    points = 1
                    self.end_trial('coin')
                self.score += points
                print('Score! %i' % points)
                break
            # end trial after 10 seconds
            if time.time() > self.trial_start + self.trial_time:
                self.end_trial('buzzer')
                # print('Time out')
                break
            # end training sequence if game time is up
            if time.time() > self.game_start + self.game_time:
                self.game_over = True
                self.end_trial('buzzer')
                break
            else:
                continue

    def set_target(self, min_dist=45):
        target = (numpy.random.randint(self.az_range[0], self.az_range[1]),
                  numpy.random.randint(self.ele_range[0], self.ele_range[1]))
        if self.target:  # check if target is at least min_dist away from previous target
            diff = numpy.diff((target, self.target), axis=0)[0]
            euclidean_dist = numpy.sqrt(diff[0] ** 2 + diff[1] ** 2)
            while euclidean_dist < min_dist:
                target = (numpy.random.randint(self.az_range[0], self.az_range[1]),
                          numpy.random.randint(self.ele_range[0], self.ele_range[1]))
                diff = numpy.diff((target, self.target), axis=0)[0]
                euclidean_dist = numpy.sqrt(diff[0] ** 2 + diff[1] ** 2)
        print('\n TARGET| azimuth: %.1f, elevation %.1f' % (target[0], target[1]))
        self.target = target

    def end_trial(self, sound='buzzer'):
        self.pulse_stream.halt()  # end binsim thread
        self.sounds[sound].play()  # play end sound
        time.sleep(self.sounds[sound].duration)

    def get_headpose(self):
        self.headpose = freefield.get_head_pose()

    def set_filter(self):
        # get sound source coordinates relative to head pose
        rel_coords = self.target - self.headpose
        # convert coordinates to HRTF convention (=physics convention)
        if rel_coords[0] > 0:
            rel_coords[0] = 360 - rel_coords[0]
        elif rel_coords[0] < 0:
            rel_coords[0] *= -1
        # find idx of the nearest filter in the hrtf
        filter_coords = self.hrtf._get_coordinates((rel_coords[0], rel_coords[1],
                        self.hrtf.sources.vertical_polar[0,2]), 'spherical').cartesian
        distances = numpy.sqrt(((filter_coords - self.hrtf.sources.cartesian) ** 2).sum(axis=1))
        next_idx = int(numpy.argmin(distances))
        if next_idx != self.filter_idx:
            # change filter
            filter_msg = [0, next_idx, 0, 0, 0, 0, 0]
            self.osc_client.send_message('/pyBinSim', filter_msg)
            self.filter_idx = next_idx

    def set_pulse_train(self):
        dst = self.headpose - self.target
        self.distance = numpy.linalg.norm(dst)
        print('distance: azimuth %.1f, elevation %.1f, total %.2f'
              % (dst[0], dst[1], self.distance), end="\r", flush=True)
        # maximal pulse interval in ms
        max_interval = 500
        # max displacement from center
        max_distance = numpy.linalg.norm(numpy.min((self.az_range,self.ele_range), axis=1) - numpy.array((0,0)))
        # scale distance with maximal distance
        interval_scale = (self.distance - 2 + 1e-9) / max_distance
        # scale interval logarithmically with distance
        interval = max_interval * (numpy.log(interval_scale + 0.05) + 3) / 3  # log scaling
        self.pulse_stream.update_interval(interval)

    @staticmethod
    def _make_osc_client():
        # Create OSC client
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
        osc_client = udp_client.SimpleUDPClient(args.ip, args.port)
        return osc_client

    @staticmethod
    def _stop_game():
        freefield.halt()

    @staticmethod
    def _make_sequence(targets, n_reps, min_dist=45):
        # create n_reps elements sequence with more than min_dist angular distance between successive targets
        def euclidean(D2array):
            diff = numpy.diff(D2array, axis=0)
            return numpy.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
        n_targets = targets.shape[0]
        n_trials = n_reps * n_targets
        sequence = numpy.zeros((n_trials, 2))
        while True:
            for s in range(n_reps):
                dist = numpy.zeros(n_targets)
                while any(dist < min_dist):
                    seq = targets[numpy.random.choice(n_targets, n_targets, replace=False), :]
                    dist = euclidean(seq)
                sequence[s*n_targets:s*n_targets+n_targets] = seq
            if all(euclidean(sequence) >= 35):
                return sequence

    @staticmethod
    def _wait_for_button(*msg, button=''):
        response = None
        while response != button:
            if msg: response = input(msg)
            else: response = input('Waiting for button.')
        return response

if __name__ == "__main__":
    self = Training('kemar')
    self.run()