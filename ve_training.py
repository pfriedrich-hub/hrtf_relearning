import matplotlib
import freefield
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from RM1 import connect_RM1, array2RM1
from pathlib import Path
import time
import numpy
import slab
data_path = Path.cwd() / 'data'


def training():
    global dsp, settings, sounds, variables

    # game settings, variables and sounds
    settings = {'target_size': .5, 'target_time': .5,
                 'game_time': 90, 'trial_time': 10}
    sounds = {'coins': slab.Sound(data=data_path / 'sounds' / 'coins.wav'),
              'coin': slab.Sound(data=data_path / 'sounds' / 'coin.wav'),
              'buzzer': slab.Sound(data_path / 'sounds' / 'buzzer.wav')}
    sounds['coins'].level, sounds['coin'].level, sounds['buzzer'].level = 70, 70, 75
    game_variables = {'target': None, 'score': 0, 'game_start': 0, 'prep_time': 0, 'end': False}

    # init proc
    dsp = connect_RM1(rcx_path=data_path / 'rcx' / 've_training.rcx')
    freefield.calibrate_sensor(led_feedback=False, button_control=False)

    # set target sound location
    target = (numpy.random.randint(-60, 60), numpy.random.randint(-60, 60))
    dsp.SetTagVal('target_az', target[0])
    dsp.SetTagVal('target_ele', target[1])

    game_start = time.time()  # start counting time
    end, score, prep_time = False, 0, 0  # reset trial parameters
    score = 0
    while not end:  # loop over trials
        play_trial()  # play trial

        # pick next target 45° away from previous
        [next_speaker] = speaker_choices[
                             numpy.where(speaker_choices[:, 0] == int(numpy.random.choice(speaker_choices[:, 0],
                                                                                          p=speaker_choices[:, 3])))][
                         :3]
        diff = numpy.diff((speaker[1:], next_speaker[1:]), axis=0)
        euclidean_dist = numpy.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
        while euclidean_dist < 45:
            [next_speaker] = speaker_choices[
                                 numpy.where(speaker_choices[:, 0] == int(numpy.random.choice(speaker_choices[:, 0],
                                                                                              p=speaker_choices[:,
                                                                                                3])))][:3]
            diff = numpy.diff((speaker[1:], next_speaker[1:]), axis=0)
            euclidean_dist = numpy.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
        speaker_choices = numpy.delete(speaker_choices, numpy.where(speaker_choices[:, 0] == next_speaker[0]), axis=0)
        speaker_choices[:, 3] = speaker_choices[:, 3] / speaker_choices[:, 3].sum()  # update probabilities
        speaker = next_speaker

    play_trial()

def play_trial():
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



        else: continue
def update_headpose():
    # read out headpose from sensor
    pose = freefield.get_head_pose()
    # send headpose to dsp
    dsp.SetTagVal('head_az', pose[0])
    dsp.SetTagVal('head_ele', pose[1])




    count_down = False  # condition for counting time on target
    trial_start = time.time()
    prep_time += trial_start - trial_prep  # count time only while playing
    while True:
        if distance <= 0:  # check if head pose is within target window
            if not count_down:  # start counting down time as longs as pose matches target
                start_time, count_down = time.time(), True
        else:
            start_time, count_down = time.time(), False  # reset timer if pose no longer matches target
        if time.time() > start_time + goal_attr['target_time']:  # end trial if goal conditions are met
            if time.time() - trial_start <= 3:
                points = 2
                freefield.write(tag='goal_data', value=coins.data, processors=['RX81', 'RX82'])
            else:
                points = 1
                freefield.write(tag='goal_data', value=coin.data, processors=['RX81', 'RX82'])
            score += points
            print('Score! %i' % points)
            freefield.write(tag='source', value=0, processors=['RX81', 'RX82'])  # set speaker input to goal sound
            freefield.play(kind='zBusB', proc='all')  # play from goal sound buffer
            break
        if time.time() > trial_start + goal_attr['trial_time']:  # end trial after 10 seconds
            freefield.play(kind='zBusB', proc='all')  # interrupt pulse train
            break
        if time.time() > game_start + prep_time + goal_attr['game_time']:  # end training sequence if time is up
            end = True
            freefield.write(tag='source', value=0, processors=['RX81', 'RX82'])  # set speaker input to goal sound
            freefield.write(tag='goal_data', value=buzzer.data, processors=['RX81', 'RX82'])   # write buzzer to
            freefield.write(tag='goal_len', value=buzzer.n_samples, processors=['RX81', 'RX82'])  # goal sound buffer
            freefield.play(kind='zBusB', proc='all')  # play from goal sound buffer
            print('Final score: %i points' % score)
            break
        else:
            continue
    while freefield.read('goal_playback', processor='RX81', n_samples=1):
        time.sleep(0.1)

