import freefield
import slab
import numpy
from numpy import linalg as la
from pathlib import Path
import time
import PySpin
import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import aruco_pose

DIR = Path.cwd()# path for sound and rcx files
tmin = 0  # minimal inter stim interval in ms(if pose and sound source match)
tmax = 600  # max ISI in ms

target_window = 3  # target window height and width in degree
time_on_target = 2  # time matching head direction required to finish a trial
pose, offset_pose = numpy.zeros(2), numpy.zeros(2) # offset_pose is used to normalize head position to 0, 0
isi_params = dict()  # holds parameters to scale isi with distance to target
n_trials = 5

def hrtf_relearning(n_trials=5):
    global isi_params, target_window, speakers
    proceed = True

    freefield.set_logger('INFO')
    # initialize processors
    proc_list = [['RX81', 'RX8', DIR / 'data' / 'rcx' / 'play_buf_pulse.rcx'],
                 ['RX82', 'RX8', DIR / 'data' / 'rcx' / 'play_buf_pulse.rcx'],
                 ['RP2', 'RP2', DIR / 'data' / 'rcx' / 'arduino_analog.rcx']]
    freefield.initialize('dome', device=proc_list)
    # freefield.load_equalization()

    # read list of speaker locations
    table_file = freefield.DIR / 'data' / 'tables' / Path(f'speakertable_dome.txt')
    speakers = numpy.loadtxt(table_file, skiprows=1, usecols=(0, 3, 4),
                                     delimiter=",", dtype=float)
    # write parameters for isi #todo set reasonable max range
    isi_params = {'tmin': tmin, 'tmax': tmax, 'trange': tmax - tmin,
                  'max_dst': la.norm(numpy.min(speakers[:, 1:], axis=0) - [0, 0])} #np.max(speakers, axis=0))}
    # generate trial sequence with target speaker locations
    trial_sequence = slab.Trialsequence(conditions=speakers[:, 0].astype(int), n_reps=1) #for now only use positive az#

    # loop over trials
    for index, speaker_id in enumerate(trial_sequence):
        if index < n_trials and proceed:
            print('TARGET: azimuth: %i, elevation: %i'
                  % (speakers[speaker_id, 1], speakers[speaker_id, 2]))
            proceed = play_trial(speaker_id)  # play n trials
    freefield.halt()
    print('end')
    return

def play_trial(speaker_id):
    global pose, cams
    # # sensor calibration
    freefield.set_logger('WARNING')
    [led_speaker] = freefield.pick_speakers((0, 0))  # get object for center speaker LED
    freefield.write(tag='bitmask', value=led_speaker.digital_channel, processors=led_speaker.digital_proc) # illuminate LED
    print('rest at center speaker and press button to start calibration...')
    freefield.wait_for_button() # start calibration after button press
    # offset_pose = calibrate_sensor()
    offet_pose = calibrate_aruco()
    freefield.write(tag='bitmask', value=0, processors=led_speaker.digital_proc) # turn off LED
    # print('calibration complete, thank you!')
    # freefield.set_logger('info')

    # initiate cameras
    system = PySpin.System.GetInstance()
    cams = system.GetCameras()
    for cam in cams:  # initialize cameras
        cam.Init()
        cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)  # disable auto exposure time
        cam.ExposureTime.SetValue(10000.0)
        cam.BeginAcquisition()

    target = speakers[speaker_id, 1:]
    #generate stimuli
    stim = slab.Sound.pinknoise(duration=10.0)
    freefield.write('playbuflen', len(stim.data), processors=['RX81', 'RX82'])
    freefield.set_signal_and_speaker(signal=stim, speaker=speaker_id, equalize=False)
    offset_pose = numpy.zeros(2)  # offset_pose is used to normalize head position to 0, 0
    print('STARTING..\n TARGET| azimuth: %.1f, elevation %.1f' % (target[0], target[1]))
    time.sleep(3)
    count_down = False
    isi, pose = compare_pose(target)  # get pose and return initial isi from la.norm(pose - target)
    freefield.write('isi', isi, processors=['RX81', 'RX82'])  # write isi to processors
    freefield.play(kind='zBusA', proc='all')
    # start to play stimuli (zBus)
    freefield.set_logger('warning')

    while True:  # loop over trials
        if pose.all() is not None:
            print('head pose: azimuth: %.1f, elevation: %.1f' % (pose[0], pose[1]))
        isi, pose = compare_pose(target)  #.astype('float16')  # get pose and return isi from la.norm(pose - target)
        freefield.write('isi', isi, processors=['RX81', 'RX82'])  # write isi to pulse generator (rcx file)
        if match_pose(target, limit=3):  # see if pose matches target speaker
            if not count_down:  # start counting down time as longs as pose matches target
                start_time = time.time()
                count_down = True
            print('ON TARGET for %i sec'%(time.time() - start_time))
        else:
            start_time, count_down = time.time(), False  # reset timer if pose no longer matches target
        if time.time() > start_time + time_on_target:  # end trial if goal conditions are met
            for cam in cams:
                if cam.IsInitialized():
                    cam.EndAcquisition()
                    cam.DeInit()
            del cam
            cams.Clear()
            break
        else:
            continue

    freefield.play(kind='zBusB', proc='all')
    if input('Goal! Conintue? (y/n)') == 'y':
        proceed = True
    else:
        proceed = False
    return proceed

def compare_pose(target):
    # azimuth, elevation = pose_from_sensor()
    azimuth = aruco_pose.get_pose(cams[1], show=False)
    elevation = aruco_pose.get_pose(cams[0], show=False)
    pose = numpy.array((azimuth, elevation))
    if not azimuth and not elevation:
        isi = isi_params['tmax']
    elif azimuth != None or elevation != None:
        diff = la.norm(pose[numpy.where(pose != None)] - target[numpy.where(pose != None)])
        # todo see how you can use nonlinear function of isi over distance
        # isi = isi_params['tmin'] + isi_params['trange'] * (numpy.log(diff/isi_params['max_dst']+0.05)+3)/3
        # scale ISI with deviation of pose from sound source
        isi = isi_params['tmin'] + isi_params['trange'] * ((diff-4) / isi_params['max_dst'])
        #todo change this (for now use 5 to set isi to 0 as soon as target window (3°) is hit
    return isi, pose

def match_pose(target, limit):  # criteria to end experiment (pose matches sound source)
    match = False
    if pose.all() is not None:
        if all(numpy.abs(target - pose) <= limit):  # set target window # todo see how this feels
            match = True
    return match

def calibrate_aruco(cams):
    while True:
        pose = [aruco_pose.get_pose(cams[1], show=False), aruco_pose.get_pose(cams[0], show=False)]

if __name__ == "__main__":
    hrtf_relearning(n_trials)