import slab
import numpy
from pathlib import Path
from collections import namedtuple
import freefield
from RM1 import connect_RM1, array2RM1

class training:
    def __init__(self):
        settings = namedtuple('settings',
                                   ['target_size', 'target_time','trial_time', 'game_time'])
        self.settings = settings(.5, .5, 90, 10)

        sound_path = Path.cwd()/'data'/'sounds'
        sounds = dict(coins=slab.Sound(data=sound_path/'coins.wav'),
                    coin=slab.Sound(data=sound_path/'coin.wav'),
                    buzzer=slab.Sound(sound_path/'buzzer.wav'))
        sounds['coins'].level, sounds['coin'].level, sounds['buzzer'].level = 70, 70, 75
        self.sounds = sounds

        variables = namedtuple('vars',
                            ['current_target', 'previous_target', 'score',
                                        'game_time', 'trial_time', 'game_over'])
        self.variables = variables(None, None, 0, 0, 0)
        self.scores = []

    def init_setup(self):
        self.dsp = connect_RM1(rcx_path=Path.cwd()/'data'/'rcx'/'ve_training.rcx')
        freefield.calibrate_sensor(led_feedback=False, button_control=False)

    def run_session(self):
        # set target sound location
        self.variables.current_target = self.get_target()
        self.dsp.SetTagVal('target_az', self.variables.current_target[0])
        self.dsp.SetTagVal('target_ele', self.variables.current_target[1])
        self.score, self.game_time, self.trial_time = 0, 0, 0  # reset trial parameters
        while not end:  # loop over trials
            play_trial()  # play trial
            self.variables.current_target = (numpy.random.randint(-60, 60), numpy.random.randint(-60, 60))

    def run_trial(self):
        count_down = False  # condition for counting time on target
        dsp.SoftTrg(1)  # start pulse train
        trial_start = time.time()
        while True:  # start within trial loop
            update_headpose()  # read headpose from sensor and send to dsp
            distance = dsp.GetTagVal('distance')  # get headpose - target distance
            if distance < settings['target_size']:
                if not count_down:  # start counting down time as longs as pose matches target
                    start_time, count_down = time.time(), True
            else:
                start_time, count_down = time.time(), False  # reset timer if pose no longer matches target
            if time.time() > start_time + settings['target_time']:  # end trial if goal conditions are met
                dsp.SoftTrg(1)  # stop pulse train
                if time.time() - trial_start <= 3:
                    points = 2
                    array2RM1(sounds['coins'], dsp)
                else:
                    points = 1
                    array2RM1(sounds['coin'], dsp)
                dsp.SoftTrg(2)
                variables['score'] += points
                print('Score! %i' % points)
                break
            if time.time() > trial_start + settings['trial_time']:  # end trial after 10 seconds
                dsp.SoftTrg(1)  # stop pulse train
                break
            if time.time() > game_start + prep_time + goal_attr['game_time']:  # end training sequence if time is up
                end = True
                array2RM1(sounds['buzzer'], dsp)
                dsp.SoftTrg(2)
                print('Final score: %i points' % score)
                break
            else:
                continue
            while freefield.read('goal_playback', processor='RX81', n_samples=1):
                time.sleep(0.1)

    def update_headpose(self):
        # read out headpose from sensor
        pose = freefield.get_head_pose()
        # send headpose to dsp
        dsp.SetTagVal('head_az', pose[0])
        dsp.SetTagVal('head_ele', pose[1])

    def set_target(self):
        if not self.variables.current_target:
            self.variables.current_target = (numpy.random.randint(-60, 60), numpy.random.randint(-60, 60))
        else:
            euclidean_dist = 0
            while euclidean_dist < 45:
                next_target = (numpy.random.randint(-60, 60), numpy.random.randint(-60, 60))
                diff = numpy.diff((self.variables.current_target[1:], next_target[1:]), axis=0)
                euclidean_dist = numpy.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)