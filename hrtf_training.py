import freefield
import slab
import numpy
from numpy import linalg as la
from pathlib import Path
import time
import analysis.localization_analysis as localization
import head_tracking.meta_motion.mm_pose as motion_sensor

data_dir = Path.cwd() / 'data'
fs = 48828
slab.set_default_samplerate(fs)

# get probabilities for target speakers, depending on previous localisation performance
subject_id = 'svm'
condition = 'Earmolds Week 2'
subject_dir = data_dir / 'experiment' / 'bracket_3' / subject_id / condition
try:
    sequence = localization.load_latest(subject_dir)
    target_p = localization.get_target_proabilities(sequence, show=False)
except:
    print('Could not load localization data. Using equal target probabilities.')
    target_p = None

# max_pulse_interval: maximal pulse interval in ms
# target_window: target window as euclidean distance of head pose from target speaker
# target_time: time matching head direction required to finish a trial
# target_p: target speaker probabilities based on previous localization accuracy

def hrtf_training(max_pulse_interval=500, target_size=3, target_time=0.5, trial_time=10, game_time=90, target_p=None):
    global proc_list, speakers, sensor, game_start, buzzer, end, pulse_attr, goal_attr, \
           offset, prep_time, score, coin, coins
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
        freefield.load_equalization(data_dir / 'calibration' / 'calibration_dome_23.05')
    # generate sounds, set experiment parameters
    stim = slab.Sound.pinknoise(duration=10.0)
    freefield.write(tag='playbuflen', value=stim.n_samples, processors=['RX81', 'RX82'])
    freefield.write(tag='data', value=stim.data, processors=['RX81', 'RX82'])
    coin = slab.Sound(data=data_dir / 'sounds' / 'coin.wav')  # load goal sound to buffer
    coins = slab.Sound(data=data_dir / 'sounds' / 'coins.wav')  # load goal sound to buffer
    coin.level, coins.level = 70, 70
    freefield.write(tag='goal_len', value=coin.n_samples, processors=['RX81', 'RX82'])
    buzzer = slab.Sound(data_dir / 'sounds' / 'buzzer.wav')
    buzzer.level = 75
    # set variables to control pulse train and goal condition
    table_file = freefield.DIR / 'data' / 'tables' / Path(f'speakertable_dome.txt')
    speakers = numpy.loadtxt(table_file, skiprows=1, usecols=(0, 3, 4), delimiter=",", dtype=float)
    speakers = numpy.delete(speakers, [19, 23, 27], axis=0)
    pulse_attr = {'max_distance': la.norm(numpy.min(speakers[:, 1:], axis=0) - [0, 0]),
                  'max_pulse_interval': max_pulse_interval}
    goal_attr = {'target_size': target_size, 'target_time': target_time,
                 'game_time': game_time, 'trial_time': trial_time}
    if target_p is None:
        target_p = numpy.expand_dims(numpy.ones(len(speakers)), axis=1) / len(speakers)
    else:
        target_p = numpy.expand_dims(target_p[:, 3], axis=1)
    while True:  # loop over blocks
        # get list of speakers to play from
        speaker_choices = speakers
        speaker_choices = numpy.hstack((speaker_choices, target_p))
        [speaker] = speaker_choices[numpy.where(speaker_choices[:, 0] == int(numpy.random.choice(speaker_choices[:, 0],
                                                                    p=speaker_choices[:, 3])))][:3]
        # remove target speaker from speaker_choices to avoid repetition
        speaker_choices = numpy.delete(speaker_choices, numpy.where(speaker_choices[:, 0] == speaker[0]), axis=0)
        speaker_choices[:, 3] = speaker_choices[:, 3] / speaker_choices[:, 3].sum()  # update probabilities
        print('Starting...')
        game_start = time.time()  # start counting time
        end, score, prep_time = False, 0, 0  # reset trial parameters
        while not end:  # loop over trials
            play_trial(int(speaker[0]))  # play trial
            # pick next target 45° away from previous
            [next_speaker] = speaker_choices[numpy.where(speaker_choices[:, 0] == int(numpy.random.choice(speaker_choices[:, 0],
                                                                    p=speaker_choices[:, 3])))][:3]
            diff = numpy.diff((speaker[1:], next_speaker[1:]), axis=0)
            euclidean_dist = numpy.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
            while euclidean_dist < 45:
                [next_speaker] = speaker_choices[numpy.where(speaker_choices[:, 0] == int(numpy.random.choice(speaker_choices[:, 0],
                                                                    p=speaker_choices[:, 3])))][:3]
                diff = numpy.diff((speaker[1:], next_speaker[1:]), axis=0)
                euclidean_dist = numpy.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
            speaker_choices = numpy.delete(speaker_choices, numpy.where(speaker_choices[:, 0] == next_speaker[0]), axis=0)
            speaker_choices[:, 3] = speaker_choices[:, 3] / speaker_choices[:, 3].sum()  # update probabilities
            speaker = next_speaker
        print('Press button to play again.')
        freefield.wait_for_button()
        time.sleep(1)

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
    target = speakers[numpy.where(speakers[:, 0] == speaker_id), 1:][0][0]   # get target coordinates
    print('\n TARGET| azimuth: %.1f, elevation %.1f' % (target[0], target[1]))
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
    hrtf_training(target_p=target_p)
