import freefield
import slab
import numpy
from numpy import linalg as la
from pathlib import Path
import time
import PySpin
from aruco_pose import get_pose, calibrate_aruco
DIR = Path.cwd()  # path for sound and rcx files

# t_min: minimal pulse interval in ms
# t_max: max pulse interval in ms
# target_window: target window as euclidean distance of head pose from target speaker
# time_on_target: time matching head direction required to finish a trial


def hrtf_relearning(n_trials=5, t_min=0, t_max=600, target_window=4, target_time=1):
    global speakers, cams, pulse_train
    # initialize processors
    proc_list = [['RX81', 'RX8', DIR / 'data' / 'rcx' / 'play_buf_pulse.rcx'],
                 ['RX82', 'RX8', DIR / 'data' / 'rcx' / 'play_buf_pulse.rcx'],
                 ['RP2', 'RP2', DIR / 'data' / 'rcx' / 'arduino_analog.rcx']]
    freefield.initialize('dome', device=proc_list)
    freefield.set_logger('warning')

    # load goal sound to buffer
    coin = slab.Sound(data=DIR / 'data' / 'sounds' / 'Mario_Coin.wav').resample(48828)
    coin.ramp(when='both', )
    coin.level = 70
    freefield.write(tag='goal_sound', value=coin.data, processors=['RX81', 'RX82'])
    freefield.write(tag='goal_len', value=coin.n_samples, processors=['RX81', 'RX82'])

    # initiate cameras
    system = PySpin.System.GetInstance()
    cams = system.GetCameras()
    for cam in cams:  # initialize cameras
        cam.Init()
        cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)  # disable auto exposure time
        cam.ExposureTime.SetValue(10000.0)
        cam.BeginAcquisition()

    # read list of speaker locations
    table_file = freefield.DIR / 'data' / 'tables' / Path(f'speakertable_dome.txt')
    speakers = numpy.loadtxt(table_file, skiprows=1, usecols=(0, 3, 4),
                                     delimiter=",", dtype=float)
    # write parameters for pulse interval
    pulse_train = {'t_min': t_min, 't_max': t_max, 't_range': t_max - t_min,
                    'max_dst': la.norm(numpy.min(speakers[:, 1:], axis=0) - [0, 0]),
                    'target_window': target_window, 'target_time': target_time}
    # generate trial sequence with target speaker locations
    trial_sequence = slab.Trialsequence(conditions=speakers[:, 0].astype(int), n_reps=1)
    proceed = True
    # loop over trials
    for index, speaker_id in enumerate(trial_sequence):
        if index < n_trials and proceed:
            proceed = play_trial(speaker_id)  # play n trials
    freefield.halt()
    for cam in cams:
        if cam.IsInitialized():
            cam.EndAcquisition()
            cam.DeInit()
    del cam
    cams.Clear()
    print('end')
    return

def play_trial(speaker_id):
    #generate stimuli and load to buffer
    stim = slab.Sound.pinknoise(duration=10.0)
    freefield.set_signal_and_speaker(signal=stim, speaker=speaker_id, equalize=True)
    freefield.write(tag='source', value=1, processors=['RX81', 'RX82'])
    target = speakers[speaker_id, 1:]
    # get offset head pose at 0 az, 0 ele
    offset = calibrate_aruco(cams, limit=0.5, report=False)
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
            print('ON TARGET for %i sec' % (time.time() - start_time))
        else:
            start_time, count_down = time.time(), False  # reset timer if pose no longer matches target
        if time.time() > start_time + pulse_train['target_time']:  # end trial if goal conditions are met
            break
        else:
            continue
    freefield.write(tag='source', value=0, processors=['RX81', 'RX82'])
    freefield.play(kind='zBusB', proc='all')
    # if input('Goal! Conintue? (y/n)') == 'y':
    #     proceed = True
    # else:
    #     proceed = False
    return True

def compare_pose(target, offset):
    azimuth = get_pose(cams[1])
    elevation = get_pose(cams[0])
    if azimuth != None and elevation != None:
        pose = numpy.array((azimuth, elevation)) - offset
        diff = la.norm(pose - target)
        # isi = isi_params['tmin'] + isi_params['trange'] * (numpy.log(diff/isi_params['max_dst']+0.05)+3)/3
        # scale ISI with deviation of pose from sound source
        interval = pulse_train['t_min'] + pulse_train['t_range'] *\
                   (diff - pulse_train['target_window']) / pulse_train['max_dst']
        print('head pose: azimuth: %.1f, elevation: %.1f' % (pose[0], pose[1]))
    else:
        diff = pulse_train['max_dst']
        interval = pulse_train['t_max']
        print('no marker detected')
    freefield.write('interval', interval, processors=['RX81', 'RX82'])  # write isi to processors
    return diff


if __name__ == "__main__":
    hrtf_relearning(n_trials=5)
