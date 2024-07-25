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
        self.variables.current_target = (numpy.random.randint(-60, 60), numpy.random.randint(-60, 60))
        self.dsp.SetTagVal('target_az', self.variables.current_target[0])
        self.dsp.SetTagVal('target_ele', self.variables.current_target[1])
        self.score, self.game_time, self.trial_time = 0, 0, 0  # reset trial parameters
        while not end:  # loop over trials
            play_trial()  # play trial

    def run_trial(self):
