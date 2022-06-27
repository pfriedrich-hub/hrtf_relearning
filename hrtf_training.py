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

def hrtf_training(n_trials, t_max=500, target_size=5, target_time=0.5):
    global speakers, offset, _target_size, _target_time, _max_distance, _max_pulse_interval
    # initialize processors and cameras
    proc_list = [['RX81', 'RX8', data_dir / 'rcx' / 'play_buf_pulse.rcx'],
                 ['RX82', 'RX8', data_dir / 'rcx' / 'play_buf_pulse.rcx'],
                 ['RP2', 'RP2', data_dir / 'rcx' / 'arduino_analog.rcx']]
    if not freefield.PROCESSORS.mode:
        freefield.initialize('dome', device=proc_list)
    freefield.set_logger('warning')
    aruco.init_cams()
    # load goal sound to buffer
    coin = slab.Sound(data=data_dir / 'sounds' / 'Mario_Coin_Retro.wav')
    coin.level = 70
    freefield.write(tag='goal_data', value=coin.data, processors=['RX81', 'RX82'])
    freefield.write(tag='goal_len', value=coin.n_samples, processors=['RX81', 'RX82'])
    # read list of speaker locations
    table_file = freefield.DIR / 'data' / 'tables' / Path(f'speakertable_dome.txt')
    speakers = numpy.loadtxt(table_file, skiprows=1, usecols=(0, 3, 4), delimiter=",", dtype=float)
    # pulse train parameters
    _max_distance = la.norm(numpy.min(speakers[:, 1:], axis=0) - [0, 0])  # maximal distance from center speaker
    _target_time, _target_size, _max_pulse_interval = target_time, target_size, t_max
    # generate trial sequence with target speaker locations
    trial_sequence = slab.Trialsequence(conditions=speakers[:, 0].astype(int), n_reps=1)
    offset = aruco.calibrate_pose(report=True)
    # loop over trials
    for index, speaker_id in enumerate(trial_sequence):
        if index < n_trials:
            play_trial(speaker_id)  # play n trials
    freefield.halt()
    aruco.deinit_cams()
    print('end')
    return

def play_trial(speaker_id):
    # generate stimuli and load to buffer
    freefield.write(tag='source', value=1, processors=['RX81', 'RX82'])
    stim = slab.Sound.pinknoise(duration=10.0)
    freefield.set_signal_and_speaker(signal=stim, speaker=speaker_id, equalize=True)
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
        if time.time() > start_time + _target_time:  # end trial if goal conditions are met
            break
        else:
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
    else:
        dist = _max_distance
        interval = _max_pulse_interval
    freefield.write('interval', interval, processors=['RX81', 'RX82'])  # write isi to processors
    return dist, pose

if __name__ == "__main__":
    hrtf_training(n_trials=15)
