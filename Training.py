import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import freefield
from pathlib import Path
import time
import numpy
import slab
data_path = Path.cwd() / 'data'
fs = 24414  # RP 2 limitation

class Training():
    def __init__(self, processor='RP2', target_size=5, target_time=.5, game_time=180, trial_time=30,
                 az_range=(-30, 30), ele_range=(-30, 30)):
        self.processor = processor
        if self.processor == 'RP2':
            self.zbus = True
            self.connection = 'zBus'
            self.led_feedback = False
            self.button_control = True
        elif self.processor == 'RM1':
            self.zbus = False
            self.connection = 'USB'
            self.led_feedback = False
            self.button_control = False
        self.target_size = target_size
        self.target_time = target_time
        self.game_time = game_time
        self.trial_time = trial_time
        self.az_range = az_range
        self.ele_range = ele_range
        self.target = None
        self.scores = []
        self.sounds = {'coins': slab.Sound(data=data_path / 'sounds' / 'coins.wav'),
                  'coin': slab.Sound(data=data_path / 'sounds' / 'coin.wav'),
                  'buzzer': slab.Sound(data_path / 'sounds' / 'buzzer.wav')}
        for sound, key in zip(self.sounds.values(), self.sounds.keys()):
            self.sounds[key] = sound.resample(fs)

        self.sounds['coins'].level, self.sounds['coin'].level, self.sounds['buzzer'].level = 80, 80, 85

    def __repr__(self):
        return f'{type(self)} Sessions played: {len(self.scores)} Scores: {repr(self.scores)}'

    def run(self):
        freefield.set_logger('debug')
        freefield.initialize(setup='dome', zbus=self.zbus, connection=self.connection, sensor_tracking=True,
            device=[self.processor, self.processor, data_path / 'rcx' / f've_training_{self.processor}.rcx'])

        while True:
            self.training_session()
            print('Press button to play again.')
            self.wait_for_button()

    def stop(self):
        freefield.halt()

    def training_session(self):
        self.game_over, self.score = False, 0        # reset game parameters
        self.game_start = time.time()
        # while not self.game_over:  # loop over trials until end time has passed
        trial_prep = time.time()  # time between trials
        self.set_target()  # get next target
        self.wait_for_button()
        freefield.calibrate_sensor(led_feedback=self.led_feedback, button_control=self.button_control)
        self.game_start += time.time() - trial_prep  # count time only while playing
        self.play_trial()
        self.scores.append(self.score)
        print(f'Run {len(self.scores)}: {self.score} points')

    def play_trial(self):
        freefield.play(1, self.processor)  # start pulse train
        self.trial_start = time.time()  # get trial start time
        count_down = False  # condition for counting time on target
        # within trial loop: continuously update headpose and monitor time
        while True:
            self.update_headpose()  # read headpose from sensor and send to dsp
            # distance = freefield.read('distance', self.processor))  # get headpose - target distance
            dst = self.pose - self.target
            distance = numpy.sqrt(numpy.sum(numpy.square(dst)))  # faster than reading from DSP
            print('distance: azimuth %.1f, elevation %.1f, total %.2f'
                  % (dst[0], dst[1], distance), end="\r", flush=True)
            if distance < self.target_size:
                if not count_down:  # start counting down time as longs as pose matches target
                    time_on_target, count_down = time.time(), True
            else:
                time_on_target, count_down = time.time(), False  # reset timer if pose no longer matches target
            # end trial if goal conditions are met
            if time.time() > time_on_target + self.target_time:
                if time.time() - self.trial_start <= 3:
                    points = 2
                    self.play_end_sound('coins')
                else:
                    points = 1
                    self.play_end_sound('coin')
                self.score += points
                print('Score! %i' % points)
                break
            # end trial after 10 seconds
            if time.time() > self.trial_start + self.trial_time:
                self.play_end_sound('buzzer')
                # print('Time out')
                break
            # end training sequence if game time is up
            if time.time() > self.game_start + self.game_time:
                self.game_over = True
                self.play_end_sound('buzzer')
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
        freefield.write('target_az', target[0], self.processor)
        freefield.write('target_ele', target[1], self.processor)
        print('\n TARGET| azimuth: %.1f, elevation %.1f' % (target[0], target[1]))
        self.target = target

    def play_end_sound(self, sound='buzzer'):
        goal_sound = self.sounds[sound]
        freefield.write('n_goal', goal_sound.n_samples, self.processor)
        freefield.write('goal', goal_sound, self.processor)
        freefield.play(2, self.processor)  # stop pulse train and play buzzer sound
        freefield.wait_to_finish_playing(self.processor, 'goal_play')

    def update_headpose(self):
        # get headpose from sensor and write to processor
        self.pose = freefield.get_head_pose()
        freefield.write('head_az', self.pose[0], self.processor)
        freefield.write('head_ele', self.pose[1], self.processor)

    def wait_for_button(self):
        if self.processor == 'RP2':  # calibrate (wait for button)
            print('Press button to start sensor calibration')
        elif self.processor == 'RM1':
            input('Press button to start sensor calibration')

# if __name__ == "__main__":
#
#     training = Training('RM1')
#     training.run()