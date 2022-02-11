import torch
import freefield
import slab
import numpy as np
from numpy import linalg as la
import pathlib
import os
import re
from pathlib import Path
import time
import serial as srl
DIR = pathlib.Path(os.getcwd()) # path for sound and rcx files
port = 'COM6' # set port for arduino serial read
tmin = 0  # minimal inter stim interval in ms(if pose and sound source match)
tmax = 600  # max ISI in ms

target_window = 3 # target window height and width in degree
time_on_target = 2 # time matching head direction required to finish a trial
pose, offset_pose = np.zeros(2), np.zeros(2) # offset_pose is used to normalize head position to 0, 0
isi_params = dict() # holds parameters to scale isi with distance to target

def hrtf_relearning(n_trials=5):
    global isi_params, target_window
    proceed = True

    init_serial(port)
    freefield.set_logger('INFO')

    # load or create speaker equalization file
    # try:
    #     freefield.SETUP = 'dome'
    #     freefield.load_equalization()
    # except IOError:
    #     freefield.initialize('dome', default='play_rec')
    #     freefield.equalize_speakers()

    # initialize processors
    proc_list = [['RX81', 'RX8', DIR / 'data' / 'play_buf_pulse.rcx'],
                 ['RX82', 'RX8', DIR / 'data' / 'play_buf_pulse.rcx'],
                 ['RP2', 'RP2', DIR / 'data' / 'button.rcx']]
    freefield.initialize('dome', device=proc_list)
    freefield.load_equalization()

    # read list of speaker locations
    table_file = freefield.DIR / 'data' / 'tables' / Path(f'speakertable_dome.txt')
    speakers = np.loadtxt(table_file, skiprows=1, usecols=(3, 4), delimiter=",", dtype=float)
    # write parameters for isi #todo set reasonable max range
    isi_params = {'tmin': tmin, 'tmax': tmax, 'trange': tmax - tmin,
                  'max_dst': la.norm(np.min(speakers, axis=0) - [0,0])} #np.max(speakers, axis=0))}
    # generate trial sequence with target speaker locations
    trial_sequence = slab.Trialsequence(conditions=speakers[19:,:], n_reps=1) #for now only use positive az#

    # loop over trials
    for index, speaker_position in enumerate(trial_sequence):
        if index < n_trials and proceed:
            print('TARGET: azimuth: %i, elevation: %i'
                  % (speaker_position[0], speaker_position[1]))
            target = speaker_position # get target speaker coordinates
            proceed = play_trial(target)  # play n trials

    ser.close()
    freefield.halt()
    print('end')
    return

def play_trial(target):
    global offset_pose, pose
    # sensor calibration
    freefield.set_logger('WARNING')
    [led_speaker] = freefield.pick_speakers((0, 0)) # get object for center speaker LED
    freefield.write(tag='bitmask', value=led_speaker.digital_channel, processors=led_speaker.digital_proc) # illuminate LED
    print('rest at center speaker and press button to start calibration...')
    freefield.wait_for_button() # start calibration after button press
    offset_pose = calibrate_sensor()
    freefield.write(tag='bitmask', value=0, processors=led_speaker.digital_proc) # turn off LED
    print('calibration complete, thank you!')
    freefield.set_logger('info')

    #generate stimuli
    stim = slab.Sound.pinknoise(duration=10.0)
    freefield.write('playbuflen', len(stim.data), 'all')
    freefield.set_signal_and_speaker(stim, tuple((target[0], target[1])))

    # starting loop over stimuli
    print('STARTING..\n TARGET| azimuth: %i, elevation %i'%(target[0], target[1]))
    time.sleep(3)
    count_down = False
    isi = compare_pose(target)#.astype('float16')  # get pose and return initial isi from la.norm(pose - target)
    freefield.write('isi', isi, processors=['RX81', 'RX82'])  # write isi to processors
    freefield.play(kind='zBusA', proc='all')
    # start to play stimuli (zBus)
    freefield.set_logger('warning')

    while True:  # loop over trials
        if pose is not None:
            print('head pose: azimuth: %f, elevation: %f' %(pose[0], pose[1]))
        isi = compare_pose(target)#.astype('float16')  # get pose and return isi from la.norm(pose - target)
        freefield.write('isi', isi, processors=['RX81', 'RX82'])  # write isi to pulse generator (rcx file)
        if match_pose(target): # see if pose matches target speaker
            if not count_down: # start counting down time as longs as pose matches target
                start_time = time.time()
                count_down = True
            print('ON TARGET for %i sec'%(time.time() - start_time))
        else: start_time, count_down = time.time(), False # reset timer if pose no longer matches target
        if time.time() > start_time + time_on_target: # if goal conditions are met, break from stim loop
            break
        else: continue

    freefield.play(kind='zBusB', proc='all')
    if input('Goal! Conintue? (y/n)') == 'y':
        proceed = True
    else:
        proceed = False
    return proceed

def compare_pose(target):
    global pose
    pose = pose_from_sensor()
    if pose is None:
        isi = isi_params['tmax']
    else:
        diff = la.norm(pose - target) #todo see how you can use nonlinear function of isi over distance
        # isi = isi_params['tmin'] + isi_params['trange'] * (np.log(diff/isi_params['max_dst']+0.05)+3)/3
        # scale ISI with deviation of pose from sound source
        isi = isi_params['tmin'] + isi_params['trange'] * ((diff-4) / isi_params['max_dst'])
        #todo change this (for now use 5 to set isi to 0 as soon as target window (3°) is hit
    return isi

def match_pose(target):  # criteria to end experiment (pose matches sound source)
    if pose is not None:
        if all(np.abs(target - pose) <= 3): # set target window # todo see how this feels
            return 1
        else:
            return 0
    else: return 0

def pose_from_sensor(): #todo convert to interaural polar
    raw_pose = read_serial()
    if raw_pose is not None:
        pose = raw_pose - offset_pose  # get offset corrected sensor orientation
    else:
        pose = None
    return pose

def init_serial(port):
    global ser
    try:
        ser = srl.Serial(port, baudrate=115200, bytesize=srl.EIGHTBITS)
        print("Open serial communication with Arduino")
        print(ser)
        print("======================================")
    except srl.serialutil.SerialException:
        raise ValueError('Arduino serial port busy!')
    return ser.isOpen()

def read_serial():
    serial_read = str(ser.readline())
    if len(re.findall("\d+\.\d+", serial_read)) == 2:
        raw_pose = np.array((re.findall(r"[-+]?\d*\.\d+|\d+", serial_read))).astype(float)
        # if raw_pose[0] > 180:  # refine raw sensor orientation values
        #     raw_pose[0] = raw_pose[0] - 360
        raw_pose[0] *= -1
    else:
        raw_pose = None
        print('not enough values for pose estimation')
    return raw_pose

def calibrate_sensor(accuracy_limit=0.3): # smaller accuracy limits will result in higher precision take longer to calibrate
    # get offset pose (az 0, ele 0)
    pose_log = np.zeros(2)
    while True:  # wait in loop for sensor to stabilize
        read = read_serial()
        if read is not None:
            pose_log = np.vstack([pose_log, read])
            if len(pose_log>100):
                az_acc = np.mean(np.abs(np.diff(pose_log[-100:, 0]))).astype('float16')
                ele_acc = np.mean(np.abs(np.diff(pose_log[-100:, 1]))).astype('float16')
                print('az acc: %f,  ele acc: %f'%(az_acc, ele_acc))
        # check if differential is stable for at least 100 data points,
        # set accuracy limit to max 0.1 for stable measurements
        if len(pose_log) > 100 and np.mean(np.abs(np.diff(pose_log[-100:, 0]))) < accuracy_limit and\
                np.mean(np.abs(np.diff(pose_log[-100:, 1]))) < accuracy_limit:
            break
    offset_pose = np.mean(pose_log[-50:, :], axis=0)
    return offset_pose

if __name__ == "__main__":
    hrtf_relearning()

