import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import freefield
from pathlib import Path
import argparse
from pythonosc import udp_client
import time
import numpy
import slab
from dev.pybinsim.binsim import *
import threading

sound_dir = Path.cwd() / 'data' / 'sounds'
fs = 44100
slab.set_default_samplerate(fs)

class Training:
    def __init__(self, filename='kemar', target_size=5, target_time=.5, game_time=180, trial_time=30,
                 az_range=(-30, 30), ele_range=(-30, 30)):
        # general parameters
        self.filename = filename
        self.target_size = target_size
        self.target_time = target_time
        self.game_time = game_time
        self.trial_time = trial_time
        self.az_range = az_range
        self.ele_range = ele_range
        self.target = None
        self.scores = []
        # load sounds
        self.sounds = {
                        'coins': slab.Sound.read(sound_dir / 'coins.wav'),
                        'coin': slab.Sound.read(sound_dir / 'coin.wav'),
                        'buzzer': slab.Sound.read(sound_dir / 'buzzer.wav'),
                        # 'pulses': [slab.Sound.read(filename) for filename in
                        #           sorted(list((sound_dir / 'pinknoise_pulses').glob('*wav')))]
                        'noise': slab.Sound.read(sound_dir / 'pinknoise_44100.wav'),
                        'silence': slab.Sound.read(sound_dir / 'silence_44100.wav'),
                        }

        for sound, key in zip(self.sounds.values(), self.sounds.keys()):
            self.sounds[key] = sound.resample(fs)
        self.sounds['coins'].level, self.sounds['coin'].level, self.sounds['buzzer'].level = 75, 75, 75
        # pybinsim
        self.osc_client = self._make_osc_client()
        self.binsim = threading.Thread(target=binsim_start, args=(self.filename,))

    def __repr__(self):
        return f'{type(self)} Sessions played: {len(self.scores)} Scores: {repr(self.scores)}'

    def run(self):
        # freefield.initialize(setup=None, sensor_tracking=True)
        while True:
            self.training_session()
            self.wait_for_button('Press button to play again.')

    def stop(self):
        freefield.halt()

    def training_session(self):

        self.game_over, self.score = False, 0        # reset countdown and score
        self.game_start = time.time()

        while not self.game_over:  # loop over trials until end time has passed
            trial_prep = time.time()  # time between trials
            self.set_target()  # get next target
            self._wait_for_button('Press Enter to start.')
            # freefield.calibrate_sensor(led_feedback=self.led_feedback, button_control=self.button_control)
            self.game_start += time.time() - trial_prep  # count time only while playing
            self.play_trial()
            self.scores.append(self.score)
            print(f'Run {len(self.scores)}: {self.score} points')

    def play_trial(self):
        self.binsim.start()  # start pulse train
        self.trial_start = time.time()  # get trial start time
        count_down = False  # condition for counting time on target
        # within trial loop: continuously update headpose and monitor time
        while True:
            self.update_headpose()  # read headpose from sensor
            self.set_pulse_train() # get distance and set pulse train params to binsim

            if self.distance < self.target_size:
                if not count_down:  # start counting time as longs as pose matches target
                    time_on_target, count_down = time.time(), True
            else:
                time_on_target, count_down = time.time(), False  # reset timer if pose no longer matches target
            # end trial if goal conditions are met
            if time.time() > time_on_target + self.target_time:
                if time.time() - self.trial_start <= 3:
                    points = 2
                    self.sounds['coins'].play
                else:
                    points = 1
                    self.end_sound('coin')
                self.score += points
                print('Score! %i' % points)
                break
            # end trial after 10 seconds
            if time.time() > self.trial_start + self.trial_time:
                self.end_sound('buzzer')
                # print('Time out')
                break
            # end training sequence if game time is up
            if time.time() > self.game_start + self.game_time:
                self.game_over = True
                self.end_sound('buzzer')
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

    def end_sound(self, sound='buzzer'):
        self.binsim.join()  # end binsim thread
        self.sounds[sound].play()  # play end sound

    def update_headpose(self):
        self.pose = freefield.get_head_pose()

    def set_pulse_train(self):
        dst = self.pose - self.target
        self.distance = numpy.sqrt(numpy.sum(numpy.square(dst)))  # faster than reading from DSP
        # OR
        self.distance = numpy.linalg.norm(self.pose - self.target)

        print('distance: azimuth %.1f, elevation %.1f, total %.2f'
              % (dst[0], dst[1], self.distance), end="\r", flush=True)



        # # I how is idx related to the pulse duration? - linear
        # idx = numpy.arange(0, len(self.sounds['pulses']))
        # n = 0.025 # intercept
        # m = 0.01 # increment from numpy.linspace in make_pulses.py
        # duration = m * idx + n
        # # II how is distance related to pulse duration? - nonlinear
        # # from hrtf training :
        # ele_dist = numpy.abs(target[1] - pose[1])
        # # total distance of head pose from target
        # total_distance = numpy.sqrt(az_dist ** 2 + ele_dist ** 2)
        # # distance of current head pose from target window
        # window_distance = total_distance - goal_attr['target_size']
        # # scale ISI with total distance; use scale factor for pulse interval duration
        # interval_scale = (total_distance - 2 + 1e-9) / pulse_attr['max_distance']
        # interval = pulse_attr['max_pulse_interval'] * (numpy.log(interval_scale + 0.05) + 3) / 3  # log scaling
        # # III how is distance related to index?
        # i = int(dist * m + n)
        # # i controls pulse speed should depend on distance
        # sound_msg = str(self.sounds['pulses'][i])


        self.osc_client.send_message('/pyBinSimFile', sound_msg)

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
    def _wait_for_button(*msg):
        response = None
        if msg: response = input(msg)
        else: input('Waiting for button.')
        return response

    @ staticmethod
    def _make_osc_client():
        # Create OSC client
        ip = '127.0.0.1'
        port = 10000
        parser = argparse.ArgumentParser()
        parser.add_argument("--ip", default=ip, help="The ip of the OSC server")
        parser.add_argument("--port", type=int, default=port, help="The port the OSC server is listening on")
        args = parser.parse_args()
        osc_client = udp_client.SimpleUDPClient(args.ip, args.port)
        return osc_client

    # def wait_for_button(self):
    #     if self.processor == 'RP2':  # calibrate (wait for button)
    #         print('Press button to start sensor calibration')
    #     elif self.processor == 'RM1':
    #         input('Press button to start sensor calibration')

# if __name__ == "__main__":
    # training = Training('kemar')
    # training.run()