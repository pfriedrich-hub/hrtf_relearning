import slab
import time
import numpy
from pathlib import Path
from collections import namedtuple
import freefield
from RM1 import connect_RM1, array2RM1, wait_for_button

class training:
    def __init__(self):
        # training game settings
        settings = namedtuple('settings',
                                   ['target_size', 'target_time','trial_time_limit', 'session_time_limit'])
        self.settings = settings(.5, .5, 10, 90)

        # set initial training variables
        variables = namedtuple('vars',
                            ['current_target', 'previous_target', 'session_score', 'scores'
                                        'session_time', 'trial_time', 'game_over'])
        self.variables = variables(None, None, 0, [], 0, 0)

        # load game sounds
        sound_path = Path.cwd()/'data'/'sounds'
        sounds = dict(coins=slab.Sound(data=sound_path/'coins.wav'),
                    coin=slab.Sound(data=sound_path/'coin.wav'),
                    buzzer=slab.Sound(sound_path/'buzzer.wav'))
        sounds['coins'].level, sounds['coin'].level, sounds['buzzer'].level = 70, 70, 75
        self.sounds = sounds

    def init_setup(self):
        """ initialize setup """
        self.dsp = connect_RM1(rcx_path=Path.cwd()/'data'/'rcx'/'ve_training.rcx')
        freefield.calibrate_sensor(led_feedback=False, button_control=False)
        self.initialized = True

    def run_session(self):
        """ run a training session """
        global session_start_time
        # set target sound location
        self.set_target()
        self.variables.session_score, self.variables.game_time, self.variables.trial_time = 0, 0, 0  # reset training session parameters
        while not self.variables.game_over:  # loop over trials
            session_start_time = time.time()
            self.play_trial()  # play trial
            self.set_target()

    def run_trial(self):
        """ run a trial in a session """
        count_down = False  # condition for counting time on target
        # freefield.calibrate_sensor() or wait_for_button()
        self.dsp.SoftTrg(1)  # start pulse train
        trial_start_time = time.time()
        while True:  # start intra-trial loop
            self.update_headpose()  # read headpose from sensor and send to dsp
            distance = self.dsp.GetTagVal('distance')  # get headpose - target distance
            self.variables.trial_time = time.time() - trial_start_time
            self.variables.session_time = time.time() - session_start_time
            # test score conditions
            if distance < self.settings.target_size:
                if not count_down:  # start counting time as longs as pose matches target
                    start_time, count_down = time.time(), True
            else:
                start_time, count_down = time.time(), False  # reset timer if pose no longer matches target
            if time.time() > start_time + self.settings.target_time:  # end trial if score conditions are met
                self.dsp.SoftTrg(1)  # stop pulse train
                if self.variables.trial_time <= 3:
                    self.variables['score'] += 2
                    array2RM1(self.sounds['coins'], self.dsp)
                else:
                    self.variables['score'] += 1
                    array2RM1(self.sounds['coin'], self.dsp)
                self.dsp.SoftTrg(2)
                break

            # no score conditions
            if self.variables.trial_time > self.settings.trial_time_limit:  # end trial after 10 seconds
                self.dsp.SoftTrg(2)  # stop pulse train
                break
            if self.variables.session_time > self.settings.session_time_limit:  # end training sequence if time is up
                self.variables.game_over = True
                self.varray2RM1(self.sounds['buzzer'], self.dsp)
                self.dsp.SoftTrg(3)
                print('Final score: %i Points' % self.score)
                break
            else:
                continue
            while self.dsp.GetTagVal('running'):  # wait for goal sound playback
                time.sleep(0.1)

    def update_headpose(self):
        """ read out headpose from sensor and send to dsp """
        pose = freefield.get_head_pose()
        self.dsp.SetTagVal('head_az', pose[0])
        self.dsp.SetTagVal('head_ele', pose[1])

    def set_target(self):
        """ set a new target location """
        if not self.variables.current_target:
            self.variables.current_target = (numpy.random.randint(-60, 60), numpy.random.randint(-60, 60))
        else:
            euclidean_dist = 0
            while euclidean_dist < 45:
                next_target = (numpy.random.randint(-60, 60), numpy.random.randint(-60, 60))
                diff = numpy.diff((self.variables.current_target[1:], next_target[1:]), axis=0)
                euclidean_dist = numpy.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
            self.variables.current_target = next_target
        self.dsp.SetTagVal('target_az', self.variables.current_target[0])
        self.dsp.SetTagVal('target_ele', self.variables.current_target[1])