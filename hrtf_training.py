import freefield
import slab
import numpy
from numpy import linalg as la
from pathlib import Path
import time
import head_tracking.cam_tracking.aruco_pose as headpose
import head_tracking.sensor_tracking.sensor_pose as headpose
data_dir = Path.cwd() / 'data'

fs = 48828
slab.set_default_samplerate(fs)

# t_min: minimal pulse interval in ms
# t_max: max pulse interval in ms
# target_window: target window as euclidean distance of head pose from target speaker
# time_on_target: time matching head direction required to finish a trial

def hrtf_training(n_trials=5, t_min=0, t_max=600, target_window=6, target_time=1):
    global speakers, pulse_train
    # initialize processors and cameras
    proc_list = [['RX81', 'RX8', data_dir / 'play_buf_pulse.rcx'],
                 ['RX82', 'RX8', data_dir / 'play_buf_pulse.rcx'],
                 ['RP2', 'RP2', data_dir / 'arduino_analog.rcx']]
    if not freefield.PROCESSORS.mode:
        freefield.initialize('dome', device=proc_list)
    freefield.set_logger('warning')
    headpose.init_cams()
    # load goal sound to buffer
    coin = slab.Sound(data=data_dir / 'sounds' / 'Mario_Coin.wav')
    coin.level = 70
    freefield.write(tag='goal_data', value=coin.data, processors=['RX81', 'RX82'])
    freefield.write(tag='goal_len', value=coin.n_samples, processors=['RX81', 'RX82'])
    # read list of speaker locations
    table_file = freefield.DIR / 'data' / 'tables' / Path(f'speakertable_dome.txt')
    speakers = numpy.loadtxt(table_file, skiprows=1, usecols=(0, 3, 4),
                                     delimiter=",", dtype=float)
    # write parameters for pulse train
    pulse_train = {'t_min': t_min, 't_max': t_max, 't_range': t_max - t_min,
                    'max_dst': la.norm(numpy.min(speakers[:, 1:], axis=0) - [0, 0]),
                    'target_window': target_window, 'target_time': target_time}
    # generate trial sequence with target speaker locations
    trial_sequence = slab.Trialsequence(conditions=speakers[:, 0].astype(int), n_reps=1)
    # loop over trials
    for index, speaker_id in enumerate(trial_sequence):
        if index < n_trials:
            play_trial(speaker_id)  # play n trials
    freefield.halt()
    headpose.deinit_cams()
    print('end')
    return

def play_trial(speaker_id):
    # generate stimuli and load to buffer
    freefield.write(tag='source', value=1, processors=['RX81', 'RX82'])
    stim = slab.Sound.pinknoise(duration=10.0)
    freefield.set_signal_and_speaker(signal=stim, speaker=speaker_id, equalize=True)
    target = speakers[speaker_id, 1:]
    # get offset head pose at 0 az, 0 ele
    offset = headpose.calibrate_aruco(limit=0.5, report=True)
    # start trial
    print('STARTING..\n TARGET| azimuth: %.1f, elevation %.1f' % (target[0], target[1]))
    time.sleep(2)
    compare_pose(target, offset)  # set initial isi based on pose-target difference
    freefield.play(kind='zBusA', proc='all')   # start playing pulse train
    count_down = False
    while True:
        diff = compare_pose(target, offset)  # set isi and return pose-target difference
        if diff - pulse_train['target_window'] < 0:  # check if head pose is within target window
            if not count_down:  # start counting down time as longs as pose matches target
                start_time = time.time()
                count_down = True
            print('ON TARGET for %i sec' % (time.time() - start_time), end="\r", flush=True)
        else:
            start_time, count_down = time.time(), False  # reset timer if pose no longer matches target
        if time.time() > start_time + pulse_train['target_time']:  # end trial if goal conditions are met
            break
        else:
            continue
    freefield.write(tag='source', value=0, processors=['RX81', 'RX82'])
    freefield.play(kind='zBusB', proc='all')
    while freefield.read('goal_playback', processor='RX81', n_samples=1):
        time.sleep(0.1)

def compare_pose(target, offset):
    pose = headpose.get_pose()
    if pose[0] != None and pose[1] != None:
        pose = pose - offset
        diff = la.norm(pose - target)
        # isi = isi_params['tmin'] + isi_params['trange'] * (numpy.log(diff/isi_params['max_dst']+0.05)+3)/3
        # scale ISI with deviation of pose from sound source
        interval = pulse_train['t_min'] + pulse_train['t_range'] *\
                   (diff - pulse_train['target_window']) / pulse_train['max_dst']
        print('head pose: azimuth: %.1f, elevation: %.1f' % (pose[0], pose[1]), end="\r", flush=True)
    else:
        diff = pulse_train['max_dst']
        interval = pulse_train['t_max']
        print('no marker detected', end="\r", flush=True)
    freefield.write('interval', interval, processors=['RX81', 'RX82'])  # write isi to processors
    return diff

if __name__ == "__main__":
    hrtf_training(n_trials=5)
