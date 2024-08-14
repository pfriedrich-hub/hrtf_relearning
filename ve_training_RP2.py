import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import freefield
from pathlib import Path
import time
import numpy
import slab
data_path = Path.cwd() / 'data'

def training():
    global settings, sounds, variables
    # init RP2 processor
    freefield.initialize(setup='dome', device=['RP2', 'RP2', data_path / 'rcx' / 've_training_RP2.rcx'],
                         sensor_tracking=True)

    # game settings, variables and sounds
    settings = {'target_size': .5, 'target_time': .5,
                 'game_time': 90, 'trial_time': 10,
                'az_range': [-60, 60], 'ele_range': [-60, 60]}
    variables = {'target': None, 'score': 0, 'end': False}
    sounds = {'coins': slab.Sound(data=data_path / 'sounds' / 'coins.wav'),
              'coin': slab.Sound(data=data_path / 'sounds' / 'coin.wav'),
              'buzzer': slab.Sound(data_path / 'sounds' / 'buzzer.wav')}
    sounds['coins'].level, sounds['coin'].level, sounds['buzzer'].level = 70, 70, 75

    # start training
    while True:
        training_session()
        print('Press button to play again.')
        freefield.wait_for_button('RP2')
    freefield.halt()

def training_session():
    variables['end'], variables['score'], variables['prep_time'] = False, 0, 0  # reset game parameters
    variables['game_start'] = time.time()  # get start time
    while not variables['end']:  # loop over trials until end time has passed
        trial_prep = time.time()  # time between trials
        set_target()  # get next target
        freefield.calibrate_sensor(led_feedback=False, button_control=False)  # calibrate (wait for button)
        variables['game_start'] += variables['game_start'] + trial_prep  # count time only while playing
        play_trial()  # start trial
    print('Final score: %i points' % variables['score'])

def play_trial():
    freefield.play(1, 'RP2')  # start pulse train
    variables['trial_start'] = time.time()  # get trial start time
    count_down = False  # condition for counting time on target
    while True:
        # within trial loop: continuously update headpose and monitor time
        update_headpose()  # read headpose from sensor and send to dsp
        distance = freefield.read('distance', 'RP2')  # get headpose - target distance
        if distance < settings['target_size']:
            if not count_down:  # start counting down time as longs as pose matches target
                time_on_target, count_down = time.time(), True
        else:
            time_on_target, count_down = time.time(), False  # reset timer if pose no longer matches target
        # end trial if goal conditions are met
        if time.time() > time_on_target + settings['target_time']:
            if time.time() - variables['trial_start'] <= 3:
                points = 2
                play_end_sound('coins')
            else:
                points = 1
                play_end_sound('coin')
            variables['score'] += points
            print('Score! %i' % points)
            break
        # end trial after 10 seconds
        if time.time() > variables['trial_start'] + settings['trial_time']:
            play_end_sound('buzzer')
            print('Time out')
            break
        # end training sequence if game time is up
        if time.time() > variables['game_start'] + settings['game_time']:
            variables['end'] = True
            play_end_sound('buzzer')
            break
        else:
            continue

def set_target(min_dist=45):
    target = (numpy.random.randint(settings['az_range'][0], settings['az_range'][1]),
              numpy.random.randint(settings['ele_range'][0], settings['ele_range'][1]))
    if variables['target']:  # check if target is at least min_dist away from previous target
        diff = numpy.diff((target, variables['target']), axis=0)[0]
        euclidean_dist = numpy.sqrt(diff[0] ** 2 + diff[1] ** 2)
        while euclidean_dist < min_dist:
            target = (numpy.random.randint(settings['az_range'][0], settings['az_range'][1]),
                      numpy.random.randint(settings['ele_range'][0], settings['ele_range'][1]))
            diff = numpy.diff((target, variables['target']), axis=0)[0]
            euclidean_dist = numpy.sqrt(diff[0] ** 2 + diff[1] ** 2)
    freefield.write('target_az', target[0], 'RP2')
    freefield.write('target_ele', target[1], 'RP2')
    variables['target'] = target

def update_headpose():
    # get headpose from sensor and write to processor
    pose = freefield.get_head_pose()
    freefield.write('head_az', pose[0], 'RP2')
    freefield.write('head_ele', pose[1], 'RP2')

def play_end_sound(sound='buzzer'):
    goal_sound = sounds[sound]
    freefield.write('n_goal', goal_sound.n_samples, 'RP2')
    freefield.write('goal', goal_sound, 'RP2')
    freefield.play(2, 'RP2')  # stop pulse train and play buzzer sound
    freefield.wait_to_finish_playing('RP2', 'goal_play')

