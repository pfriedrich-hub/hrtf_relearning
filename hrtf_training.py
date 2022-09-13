import freefield
import slab
import numpy
from numpy import linalg as la
from pathlib import Path
import time
import head_tracking.meta_motion.mm_pose as motion_sensor
data_dir = Path.cwd() / 'data'
fs = 48828
slab.set_default_samplerate(fs)

# t_max: maximal pulse interval in ms
# target_window: target window as euclidean distance of head pose from target speaker
# target_time: time matching head direction required to finish a trial
# test

def hrtf_training(time_limit=90, t_max=500, target_size=2, target_time=0.5, trial_time=10):
    global proc_list, speakers, sensor, game_time, buzzer, end, pulse_attr, goal_attr, \
           offset, score

    # initialize sensor
    try:
        sensor
    except NameError:
        sensor = motion_sensor.start_sensor()

    # initialize processors
    if not freefield.PROCESSORS.mode:
        proc_list = [['RX81', 'RX8', data_dir / 'rcx' / 'play_buf_pulse.rcx'],
                     ['RX82', 'RX8', data_dir / 'rcx' / 'play_buf_pulse.rcx'],
                     ['RP2', 'RP2', data_dir / 'rcx' / 'arduino_analog.rcx']]
        freefield.initialize('dome', device=proc_list)
        freefield.set_logger('warning')

    # generate sounds
    stim = slab.Sound.pinknoise(duration=10.0)
    freefield.write(tag='playbuflen', value=stim.n_samples, processors=['RX81', 'RX82'])
    freefield.write(tag='data', value=stim.data, processors=['RX81', 'RX82'])
    coin = slab.Sound(data=data_dir / 'sounds' / 'Mario_Coin_Retro.wav')  # load goal sound to buffer
    coin.level = 67
    freefield.write(tag='goal_data', value=coin.data, processors=['RX81', 'RX82'])
    freefield.write(tag='goal_len', value=coin.n_samples, processors=['RX81', 'RX82'])
    buzzer = slab.Sound(data_dir / 'sounds' / 'Buzzer1.wav')
    buzzer.level = 70

    # set variables to control pulse train and goal condition
    table_file = freefield.DIR / 'data' / 'tables' / Path(f'speakertable_dome.txt')
    speakers = numpy.loadtxt(table_file, skiprows=1, usecols=(0, 3, 4), delimiter=",", dtype=float)

    # todo test this - look out for slab or freefield error
    speakers = numpy.delete(speakers, [19, 23, 27], axis=0)

    pulse_attr = {'max_distance': la.norm(numpy.min(speakers[:, 1:], axis=0) - [0, 0]), 'max_pulse_interval': t_max}
    goal_attr = {'target_size': target_size, 'target_time': target_time,
                 'time_limit': time_limit, 'trial_time': trial_time}

    # create sequence of speakers to play from, without direct repetition of azimuth or elevation
    print('Setting target sequence...')
    sequence = numpy.random.permutation(numpy.tile(list(range(len(speakers))), 1))
    az_dist, ele_dist = numpy.diff(speakers[sequence, 1]), numpy.diff(speakers[sequence, 2])
    while numpy.min(numpy.abs(az_dist)) <= 1.0 and numpy.min(numpy.abs(ele_dist)) <= 1.0:
        sequence = numpy.random.permutation(numpy.tile(list(range(len(speakers))), 1))
        az_dist, ele_dist = numpy.diff(speakers[sequence, 1]), numpy.diff(speakers[sequence, 2])

    # sequence = numpy.delete(sequence, numpy.where(sequence == 23))  # remove 0, 0 target
    # sequence = numpy.delete(sequence, numpy.where(sequence == 27))  # remove 0, -50 target
    # sequence = numpy.delete(sequence, numpy.where(sequence == 19))  # remove 0, 50 target
    # offset = motion_sensor.calibrate_pose(sensor)  # get head pose offset

    trial_sequence = slab.Trialsequence(trials=sequence)
    end = False  # set end condition for training sequence
    score = 0
    game_time = time.time()  # start counting time
    for index, speaker_id in enumerate(trial_sequence):  # loop over trials
        if not end:
            play_trial(speaker_id)  # play trial
        else:  # end training sequence
            # print('score: %i trials completed in 1:30 minutes!' % (index+1))
            break
    # motion_sensor.disconnect(sensor)
    return

def play_trial(speaker_id):
    global offset, target, end
    offset = motion_sensor.calibrate_pose(sensor)  # get head pose offset
    freefield.write(tag='source', value=1, processors=['RX81', 'RX82'])  # set speaker input to pulse train buffer
    freefield.write(tag='chan', value=freefield.pick_speakers(speaker_id)[0].analog_channel,
                    processors=freefield.pick_speakers(speaker_id)[0].analog_proc)  # set channel for target speaker
    other_proc = [proc_list[1][0], proc_list[0][0]]
    other_proc.remove(freefield.pick_speakers(speaker_id)[0].analog_proc)
    freefield.write(tag='chan', value=99, processors=other_proc)
    target = speakers[speaker_id, 1:]  # get target coordinates
    print('\n TARGET| azimuth: %.1f, elevation %.1f\n' % (target[0], target[1]))
    set_pulse_train()  # set initial pulse train interval
    freefield.play(kind='zBusA', proc='all')  # start playing pulse train
    count_down = False  # condition for counting time on target
    trial_start = time.time()
    while True:
        distance = set_pulse_train()
        if time.time() > trial_start + 10:  # end trial after 10 seconds
            break
        if distance <= 0:  # check if head pose is within target window
            if not count_down:  # start counting down time as longs as pose matches target
                start_time, count_down = time.time(), True
        else:
            start_time, count_down = time.time(), False  # reset timer if pose no longer matches target
        if time.time() > start_time + goal_attr['target_time']:  # end trial if goal conditions are met
            # todo test if score is counted correctly
            score += 1
            break
        if time.time() > game_time + goal_attr['time_limit']:  # end training sequence if time is up
            end = True
            freefield.write(tag='goal_data', value=buzzer.data, processors=['RX81', 'RX82'])   # write buzzer to
            freefield.write(tag='goal_len', value=buzzer.n_samples, processors=['RX81', 'RX82'])  # goal sound buffer
            print('score: %i trials completed in 1:30 minutes!' % score)
            break
        else:
            continue
    freefield.write(tag='source', value=0, processors=['RX81', 'RX82'])  # set speaker input to goal sound
    freefield.play(kind='zBusB', proc='all')  # play from goal sound buffer
    while freefield.read('goal_playback', processor='RX81', n_samples=1):
        time.sleep(0.1)

def set_pulse_train():
    pose = get_pose()
    if all(pose):

        # todo use rectangular target window - test this
        total_distance = numpy.sqrt(((target[0] - pose[0]) ** 2) - 16 + (target[1] - pose[1]) ** 2)

        # todo test new scaling of pulse train (continues within target window)
        total_distance = la.norm(pose - target) - goal_attr['target_size']
        window_distance = total_distance - goal_attr['target_size']
        # distance of current head pose from target window
        # scale ISI with deviation of pose from sound source
        interval_scale = ((total_distance-2) + 1e-9) / pulse_attr['max_distance']  # scale factor for pulse interval duration
        interval = pulse_attr['max_pulse_interval'] * (numpy.log(interval_scale + 0.05) + 3) / 3  # log scaling
        print('head pose: azimuth: %.1f, elevation: %.1f' % (pose[0], pose[1]), end="\r", flush=True)
    else:  # if no pose is detected, set maximal pulse interval
        distance = pulse_attr['max_distance']
        interval = pulse_attr['max_pulse_interval']
        print('no marker detected', end="\r", flush=True)
    freefield.write('interval', interval, processors=['RX81', 'RX82'])  # write isi to processors
    return distance

def get_pose():
    global pose
    pose = motion_sensor.get_pose(sensor)
    if all(pose):
        pose = pose - offset
    return pose

if __name__ == "__main__":
    hrtf_training()
