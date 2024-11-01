import freefield
import slab
import datetime
date = datetime.datetime.now()
date = f'{date.strftime("%d")}_{date.strftime("%m")}'
from pathlib import Path
from Subject import Subject
from dev.localization_analysis import *

fs = 25000
slab.set_default_samplerate(fs)
data_dir = Path.cwd() / 'data'

subject_id = 'Paul'
condition = 'AachenHRTF'

class Localization:
    def __init__(self, n_reps=3, az_range=(-52.5, 52.5), n_az=7, ele_range=(-37.5, 37.5), n_ele=7, processor='RM1'):
        # Settings
        az_grid = numpy.linspace(az_range[0], az_range[1], n_az)
        ele_grid = numpy.linspace(ele_range[0], ele_range[1], n_ele)
        self.targets = numpy.array(numpy.meshgrid(az_grid, ele_grid)).T.reshape(-1, 2)
        self.n_reps = n_reps

        # Processors
        self.processor = processor
        if self.processor == 'RP2':
            self.proc_settings = {'zBus': True, 'connection': 'zBus',
                                  'led_feedback': False, 'button_control': True}
        elif self.processor == 'RM1':
            self.proc_settings = {'zBus': False, 'connection': 'USB',
                                  'led_feedback': False, 'button_control': False}

    # def __repr__(self):
    #     return f'{type(self)} Sessions played: {len(self.scores)} Scores: {repr(self.scores)}'

    def run(self):
        freefield.set_logger('warning')
        freefield.initialize(setup='dome', zbus=self.proc_settings['zBus'], connection=self.proc_settings['connection'],
                             sensor_tracking=True, device=[self.processor, self.processor,
                                data_dir / 'rcx' / f'localization_{self.processor}.rcx'])
        self.trial_sequence = Localization._make_sequence(self.targets, self.n_reps, min_dist=35)
        self.wait_for_button('Press button to start localization test.')
        for trial_idx, trial in enumerate(self.trial_sequence):
            target = trial[:2]
            self.trial_sequence[trial_idx, 2:] = self.localization_trial(target)
            self.wait_for_button('Press button to continue')
        return self.trial_sequence

    def localization_trial(self, target):
        freefield.calibrate_sensor(led_feedback=self.proc_settings['led_feedback'],
                                   button_control=self.proc_settings['button_control'])
        self.set_target(target)  # set next target location
        self.play_sound(sound=Localization._make_stim())  # play stimuli
        self.wait_for_button()  # wait for response
        pose = freefield.get_head_pose()  # get pose
        print('Response| azimuth %.1f |elevation %.1f' % (pose[0], pose[1]))
        self.play_sound(sound=slab.Sound.tone(frequency=1000, duration=0.25, level=70))  # play confirmation sound
        return pose

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
                    sequence = numpy.c_[sequence, numpy.zeros(n_trials), numpy.zeros(n_trials)]
                    return sequence

    @staticmethod
    def _make_stim():
        noise = slab.Sound.pinknoise(duration=0.025, level=90).ramp(when='both', duration=0.01)
        silence = slab.Sound.silence(duration=0.025)
        return slab.Sound.sequence(noise, silence, noise, silence, noise, silence, noise, silence, noise)

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

    def wait_for_button(self, *msg):
        response = None
        if self.processor == 'RP2':
            if msg: print(msg)
            else: print('Waiting for button.')
            while not response:
                response = freefield.wait_for_button('RP2', 'response')
        elif self.processor == 'RM1':
            if msg: response = input(msg)
            else: input('Waiting for button.')
        return response

    def stop(self):
        freefield.halt()

if __name__ == "__main__":
    subject = Subject(id=subject_id)
    localization = Localization(n_reps=3, processor='RM1')
    localization_data = localization.run()
    subject.localization_data.append({'data': localization_data, 'condition': condition, 'date': date})
    subject.write()
    localization_accuracy(data=localization_data)
