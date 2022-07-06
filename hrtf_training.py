import freefield
import slab
import numpy
from numpy import linalg as la
from pathlib import Path
import time
import head_tracking.cam_tracking.aruco_pose as aruco
# import head_tracking.sensor_tracking.sensor_pose as sensor
data_dir = Path.cwd() / 'data'

fs = 48828
slab.set_default_samplerate(fs)

# t_max: maximal pulse interval in ms
# target_window: target window as euclidean distance of head pose from target speaker
# time_on_target: time matching head direction required to finish a trial
# n_trials = 15
time_limit = 90

def hrtf_training(time_limit, t_max=500, target_size=5, target_time=0.5):
    global speakers, offset, _target_size, _target_time, _max_distance, _max_pulse_interval,\
        game_time, end, buzzer, stim, proc_list
    # initialize processors and cameras
    proc_list = [['RX81', 'RX8', data_dir / 'rcx' / 'play_buf_pulse.rcx'],
                 ['RX82', 'RX8', data_dir / 'rcx' / 'play_buf_pulse.rcx'],
                 ['RP2', 'RP2', data_dir / 'rcx' / 'arduino_analog.rcx']]
    # if not freefield.PROCESSORS.mode:
    freefield.initialize('dome', device=proc_list)
    freefield.set_logger('warning')
    aruco.init_cams()
    table_file = freefield.DIR / 'data' / 'tables' / Path(f'speakertable_dome.txt')
    speakers = numpy.loadtxt(table_file, skiprows=1, usecols=(0, 3, 4), delimiter=",", dtype=float)

    # generate sounds
    stim = slab.Sound.pinknoise(duration=10.0)
    freefield.write(tag='playbuflen', value=stim.n_samples, processors=['RX81', 'RX82'])
    freefield.write(tag='data', value=stim.data, processors=['RX81', 'RX82'])
    # noise = slab.Sound.pinknoise(duration=0.025, level=90)
    # noise = noise.ramp(when='both', duration=0.01)
    # silence = slab.Sound.silence(duration=0.025)
    # stim = slab.Sound.sequence(noise, silence, noise, silence, noise,
    #                            silence, noise, silence, noise)
    # stim=slab.Sound.sequence(stim, stim, stim, stim)
    coin = slab.Sound(data=data_dir / 'sounds' / 'Mario_Coin_Retro.wav')  # load goal sound to buffer
    coin.level = 70
    freefield.write(tag='goal_data', value=coin.data, processors=['RX81', 'RX82'])
    freefield.write(tag='goal_len', value=coin.n_samples, processors=['RX81', 'RX82'])
    buzzer = slab.Sound(data_dir / 'sounds' / 'Buzzer1.wav')
    buzzer.level = 70

    # pulse train parameters
    _max_distance = la.norm(numpy.min(speakers[:, 1:], axis=0) - [0, 0])  # maximal distance from center speaker
    _target_time, _target_size, _max_pulse_interval = target_time, target_size, t_max

    # create sequence of speakers to play from, without direct repetition of azimuth or elevation
    sequence = numpy.random.permutation(numpy.tile(list(range(len(speakers))), 1))
    az_dist, ele_dist = numpy.diff(speakers[sequence, 1]), numpy.diff(speakers[sequence, 2])
    while numpy.min(numpy.abs(az_dist)) == 0.0 or numpy.min(numpy.abs(ele_dist)) == 0.0:
        sequence = numpy.random.permutation(numpy.tile(list(range(len(speakers))), 1))
        az_dist, ele_dist = numpy.diff(speakers[sequence, 1]), numpy.diff(speakers[sequence, 2])
    sequence = numpy.delete(sequence, numpy.where(sequence == 23))  # remove 0, 0 target
    trial_sequence = slab.Trialsequence(trials=sequence)

    offset = aruco.calibrate_pose(report=True)
    # loop over trials
    end = False
    game_time = time.time()
    for index, speaker_id in enumerate(trial_sequence):
        if not end:
            play_trial(speaker_id)  # play n trials
        else:
            print('score: %i trials completed in under 3 minutes!' % (index+1))
            break
    freefield.halt()
    aruco.deinit_cams()
    print('end')
    return


def play_trial(speaker_id):
    global end
    # set channel for target speaker
    freefield.write(tag='source', value=1, processors=['RX81', 'RX82'])
    freefield.write(tag='chan', value=freefield.pick_speakers(speaker_id)[0].analog_channel,
                    processors=freefield.pick_speakers(speaker_id)[0].analog_proc)
    other_proc = [proc_list[1][0], proc_list[0][0]]
    other_proc.remove(freefield.pick_speakers(speaker_id)[0].analog_proc)
    freefield.write(tag='chan', value=99, processors=other_proc)

    target = speakers[speaker_id, 1:]
    print('\n TARGET| azimuth: %.1f, elevation %.1f\n' % (target[0], target[1]))
    compare_pose(target, offset)  # set initial isi based on pose-target difference
    freefield.play(kind='zBusA', proc='all')   # start playing pulse train
    count_down = False
    while True:
        dist, pose = compare_pose(target, offset)  # set isi and return pose-target distance
        if all(pose):
            print('head pose: azimuth: %.1f, elevation: %.1f' % (pose[0], pose[1]), end="\r", flush=True)
        else:
            print('no head pose detected', end="\r", flush=True)
        if dist <= 0:  # check if head pose is within target window
            if not count_down:  # start counting down time as longs as pose matches target
                start_time = time.time()
                count_down = True
        else:
            start_time, count_down = time.time(), False  # reset timer if pose no longer matches target
        if time.time() > start_time + _target_time:
            break  # end trial if goal conditions are met or time runs out
        if time.time() > game_time + time_limit:
            end = True
            freefield.write(tag='goal_data', value=buzzer.data, processors=['RX81', 'RX82'])
            freefield.write(tag='goal_len', value=buzzer.n_samples, processors=['RX81', 'RX82'])
            break
        else:
            print('no head pose detected', end="\r", flush=True)
            continue
    freefield.write(tag='source', value=0, processors=['RX81', 'RX82'])
    freefield.play(kind='zBusB', proc='all')
    while freefield.read('goal_playback', processor='RX81', n_samples=1):
        time.sleep(0.1)

def compare_pose(target, offset):
    pose = aruco.get_pose()
    if all(pose):
        pose = pose - offset
        dist = la.norm(pose - target) - _target_size  # distance of current head pose from target window
        # scale ISI with deviation of pose from sound source
        interval_scale = (dist+1e-9) / _max_distance  # scale factor for pulse interval duration
        # scale ISI with deviation of pose from sound source
        interval = _max_pulse_interval * (numpy.log(interval_scale + 0.05) + 3) / 3  # log scaling
        # interval = _max_pulse_interval * numpy.sqrt((dist+1e-9)/_max_distance)  # square scaling
        # interval = _max_pulse_interval * interval_scale  # linear scaling
        print('head pose: azimuth: %.1f, elevation: %.1f' % (pose[0], pose[1]), end="\r", flush=True)
    else:
        dist = _max_distance
        interval = _max_pulse_interval
        print('no marker detected', end="\r", flush=True)
    freefield.write('interval', interval, processors=['RX81', 'RX82'])  # write isi to processors
    return dist, pose

if __name__ == "__main__":
    hrtf_training(time_limit=time_limit)


import head_tracking.meta_motion.mm_pose as motion_sensor

device = motion_sensor.start_sensor()

while True:
    print('roll %2.f, pitch %2.f, yaw %2.f'
          % (device.pose.roll, device.pose.pitch, device.pose.yaw),
          end="\r", flush=True)
    time.sleep(0.01)

motion_sensor.disconnect(device)
