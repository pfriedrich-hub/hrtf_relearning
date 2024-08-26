import freefield
import slab
import numpy
import datetime
date = datetime.datetime.now()
from matplotlib import pyplot as plt
from pathlib import Path
from Subject import Subject
from old.MSc.analysis.localization_analysis import localization_accuracy
fs = 25000
slab.set_default_samplerate(fs)
data_dir = Path.cwd() / 'data'

condition = 'test_condition'
subject = Subject(id='test_id')

class Localization():
    def __init__(self, subject, n_reps=3,  az_range=(-52.5, 52.5), n_az=7,
                                        ele_range=(-37.5, 37.5), n_ele=7, processor='RP2'):
        # Settings
        az_grid = numpy.linspace(az_range[0], az_range[1], n_az)
        ele_grid = numpy.linspace(ele_range[0], ele_range[1], n_ele)
        self.targets = numpy.array(numpy.meshgrid(az_grid, ele_grid)).T.reshape(-1, 2)
        self.n_reps = n_reps
        self.trial_sequence = Localization._make_sequence(self.targets, self.n_reps, min_dist=35)

        # Processors
        self.processor = processor
        if self.processor == 'RP2':
            self.proc_settings = {'zbus': True, 'connection': 'zBus',
                                  'led_feedback': False, 'button_control': True}
        elif self.processor == 'RM1':
            self.proc_settings = {'zbus': False, 'connection': 'USB',
                                  'led_feedback': False, 'button_control': False}

    # def __repr__(self):
    #     return f'{type(self)} Sessions played: {len(self.scores)} Scores: {repr(self.scores)}'

    def run(self):
        freefield.set_logger('debug')
        freefield.initialize(setup='dome', zbus=self.proc_settings['zBus'], connection=self.proc_settings['connection'],
                             sensor_tracking=True, device=[self.processor, self.processor,
                                data_dir / 'rcx' / f've_training_{self.processor}.rcx'])

        self.wait_for_button('Press button to start localization test.')
        for target in self.trial_sequence:
            self.trial_sequence.add_response(self.localization_trial(target))

        subject.localization_data.append(self.trial_sequence)

    def localization_trial(self, target):
        self.wait_for_button('Press button to start next trial')
        freefield.calibrate_sensor(led_feedback=self.led_feedback, button_control=self.button_control)
        self.set_target(target)  # set next target location
        freefield.play_sound(sound=self.make_stim())  # play stimuli
        self.wait_for_button()  # wait for response
        pose = freefield.get_head_pose()  # get pose
        print('Response| azimuth %.1f |elevation %.1f' % (pose[0], pose[1]))
        freefield.play_sound(sound=slab.Sound.tone(frequency=1000, duration=0.25, level=70))  # play confirmation sound
        return numpy.array((pose, target))

    @staticmethod
    def _make_sequence(targets, n_reps, min_dist=35):
        """
        Create a sequence of n_reps elements
        with more than min_dist angular distance between successive targets
        """
        def euclidean(D2array):
            diff = numpy.diff(D2array, axis=0)
            return numpy.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
        n_targets = targets.shape[0]
        n_trials = n_reps * n_targets
        sequence = numpy.zeros((n_trials, 2))
        print('Setting target sequence...')
        while True:
            for s in range(n_reps):
                dist = numpy.zeros(n_targets)
                while any(dist < min_dist):
                    seq = targets[numpy.random.choice(n_targets, size=n_targets, replace=False), :]
                    dist = euclidean(seq)
                sequence[s*n_targets:s*n_targets+n_targets] = seq
                if all(euclidean(sequence) >= min_dist):
                    return slab.Trialsequence(sequence)

    def set_target(self, target):
        freefield.write('target_az', target[0], self.processor)
        freefield.write('target_ele', target[1], self.processor)
        print('\n Target| azimuth %.1f |elevation %.1f' % (target[0], target[1]))
        self.target = target

    def play_sound(self, sound):
        freefield.write('n_sound', sound.n_samples, self.processor)
        freefield.write('sound', sound, self.processor)
        freefield.play(1, self.processor)  # stop pulse train and play buzzer sound
        freefield.wait_to_finish_playing(self.processor, 'playback')

    def make_stim(self):
        noise = slab.Sound.pinknoise(duration=0.025, level=90).ramp(when='both', duration=0.01)
        silence = slab.Sound.silence(duration=0.025)
        return slab.Sound.sequence(noise, silence, noise, silence, noise, silence, noise, silence, noise)

    def wait_for_button(self, *msg):
        response = None
        if self.processor == 'RP2':
            if msg: print(msg)
            else: print('Waiting for button.')
            while not response:
                response = freefield.wait_for_button('RP2', 'response')
        elif self.processor == 'RM1':
            if msg: response = input()
            else: input('Waiting for button.')
        return response

    def stop(self):
        freefield.halt()

# if __name__ == "__main__":
#     subject = Participants(id='test')
#     localization = Localization('RM1', subject)
#     localization.run()
#     subject.write()