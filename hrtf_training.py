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
from threading import Timer

# max_pulse_interval: maximal pulse interval in ms
# target_window: target window as euclidean distance of head pose from target speaker
# target_time: time matching head direction required to finish a trial
# test

def hrtf_training(max_pulse_interval=500, target_size=3, target_time=0.5, trial_time=10, game_time=90):
    global proc_list, speakers, sensor, game_start, buzzer, end, pulse_attr, goal_attr, \
           offset, prep_time, score
    # initialize sensor
    try:
        sensor
        if not sensor.device.is_connected:
            sensor = motion_sensor.start_sensor()
    except NameError:
        sensor = motion_sensor.start_sensor()
    # initialize processors
    if not freefield.PROCESSORS.mode:
        freefield.set_logger('warning')
        proc_list = [['RX81', 'RX8', data_dir / 'rcx' / 'play_buf_pulse.rcx'],
                     ['RX82', 'RX8', data_dir / 'rcx' / 'play_buf_pulse.rcx'],
                     ['RP2', 'RP2', data_dir / 'rcx' / 'arduino_analog.rcx']]
        freefield.initialize('dome', device=proc_list)

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
    pulse_attr = {'max_distance': la.norm(numpy.min(speakers[:, 1:], axis=0) - [0, 0]),
                  'max_pulse_interval': max_pulse_interval}
    goal_attr = {'target_size': target_size, 'target_time': target_time,
                 'game_time': game_time, 'trial_time': trial_time}

    # create sequence of speakers to play from, without direct repetition of azimuth or elevation
    print('Setting target sequence...')
    sequence = numpy.random.permutation(numpy.tile(list(range(len(speakers))), 1))
    az_dist, ele_dist = numpy.diff(speakers[sequence, 1]), numpy.diff(speakers[sequence, 2])
    while numpy.min(numpy.abs(az_dist)) <= 1.0 and numpy.min(numpy.abs(ele_dist)) <= 1.0:
        sequence = numpy.random.permutation(numpy.tile(list(range(len(speakers))), 1))
        az_dist, ele_dist = numpy.diff(speakers[sequence, 1]), numpy.diff(speakers[sequence, 2])
    sequence = numpy.delete(sequence, [numpy.where(sequence == 19),
               numpy.where(sequence == 23), numpy.where(sequence == 27)], 0)  # remove redundant speakers
    end, score, prep_time = False, 0, 0  # set end condition for training sequence
    game_start = time.time()  # start counting time
    for speaker_id in sequence:  # loop over trials
        if not end:
            play_trial(speaker_id)  # play trial
        else:  # end training sequence
            break
    return

def play_trial(speaker_id):
    global offset, target, end, score, prep_time
    trial_prep = time.time()
    freefield.write(tag='source', value=1, processors=['RX81', 'RX82'])  # set speaker input to pulse train buffer
    freefield.write(tag='chan', value=freefield.pick_speakers(speaker_id)[0].analog_channel,
                    processors=freefield.pick_speakers(speaker_id)[0].analog_proc)  # set channel for target speaker
    other_proc = [proc_list[1][0], proc_list[0][0]]
    other_proc.remove(freefield.pick_speakers(speaker_id)[0].analog_proc)
    freefield.write(tag='chan', value=99, processors=other_proc)
    offset = motion_sensor.calibrate_pose(sensor)  # get head pose offset
    target = speakers[speaker_id, 1:]  # get target coordinates
    print('\n TARGET| azimuth: %.1f, elevation %.1f\n' % (target[0], target[1]))
    set_pulse_train()  # set initial pulse train interval
    freefield.play(kind='zBusA', proc='all')  # start playing pulse train
    count_down = False  # condition for counting time on target
    trial_start = time.time()
    prep_time += trial_start - trial_prep  # count time only while playing
    while True:
        distance = set_pulse_train()
        if distance <= 0:  # check if head pose is within target window
            if not count_down:  # start counting down time as longs as pose matches target
                start_time, count_down = time.time(), True
        else:
            start_time, count_down = time.time(), False  # reset timer if pose no longer matches target
        if time.time() > start_time + goal_attr['target_time']:  # end trial if goal conditions are met
            score += 1
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
            print('Score: %i sources found in %i seconds!' % (score, goal_attr['game_time']))
            break
        else:
            continue
    while freefield.read('goal_playback', processor='RX81', n_samples=1):
        time.sleep(0.1)

def set_pulse_train():
    pose = get_pose()
    if all(pose):
        az_dist = numpy.abs(target[0] - pose[0])
        if az_dist > 3:
            az_dist = az_dist - 3
        else:
            az_dist = 0
        ele_dist = numpy.abs(target[1] - pose[1])
        # total distance of head pose from target
        total_distance = numpy.sqrt(az_dist ** 2 + ele_dist ** 2)
        # distance of current head pose from target window
        window_distance = total_distance - goal_attr['target_size']
        # scale ISI with total distance; use scale factor for pulse interval duration
        interval_scale = (total_distance - 2 + 1e-9) / pulse_attr['max_distance']
        interval = pulse_attr['max_pulse_interval'] * (numpy.log(interval_scale + 0.05) + 3) / 3  # log scaling
        print('head pose: azimuth: %.1f, elevation: %.1f, total distance: %.2f'
              % (pose[0], pose[1], total_distance), end="\r", flush=True)
    else:  # if no pose is detected, set maximal pulse interval
        window_distance = pulse_attr['max_distance']
        interval = pulse_attr['max_pulse_interval']
        print('no marker detected', end="\r", flush=True)
    freefield.write('interval', interval, processors=['RX81', 'RX82'])  # write isi to processors
    return window_distance

def get_pose():
    global pose
    pose = motion_sensor.get_pose(sensor)
    if all(pose):
        pose = pose - offset
    return pose

if __name__ == "__main__":
    repeat = True
    while repeat:
        hrtf_training()
        print('Press button to play again.')
        freefield.wait_for_button()  # start calibration after button press
