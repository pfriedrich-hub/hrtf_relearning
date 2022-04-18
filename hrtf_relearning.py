import freefield
import slab
import numpy
from numpy import linalg as la
from pathlib import Path
import time
import PySpin
import cv2
import math
import PIL
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

DIR = Path.cwd()# path for sound and rcx files
tmin = 0  # minimal inter stim interval in ms(if pose and sound source match)
tmax = 600  # max ISI in ms

target_window = 3  # target window height and width in degree
time_on_target = 2  # time matching head direction required to finish a trial
pose, offset_pose = numpy.zeros(2), numpy.zeros(2) # offset_pose is used to normalize head position to 0, 0
isi_params = dict() # holds parameters to scale isi with distance to target
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
    for cam in cams:
        if cam.IsInitialized():
            cam.EndAcquisition()
            cam.DeInit()
    del cam
    cams.Clear()
    print('end')
    return

def play_trial(speaker_id):
    global pose, arucoDict, arucoParams, mtx, dist, cams
    # # sensor calibration
    # freefield.set_logger('WARNING')
    # [led_speaker] = freefield.pick_speakers((0, 0)) # get object for center speaker LED
    # freefield.write(tag='bitmask', value=led_speaker.digital_channel, processors=led_speaker.digital_proc) # illuminate LED
    # print('rest at center speaker and press button to start calibration...')
    # freefield.wait_for_button() # start calibration after button press
    # offset_pose = calibrate_sensor()
    # freefield.write(tag='bitmask', value=0, processors=led_speaker.digital_proc) # turn off LED
    # print('calibration complete, thank you!')
    # freefield.set_logger('info')
    target = speakers[speaker_id, 1:]
    # load aruco dict and calibrate camera
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
    arucoParams = cv2.aruco.DetectorParameters_create()
    system = PySpin.System.GetInstance()
    cams = system.GetCameras()
    for cam in cams:  # initialize cameras
        cam.Init()
        cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)  # disable auto exposure time
        cam.ExposureTime.SetValue(100000.0)
        cam.BeginAcquisition()
    image = get_image(cam)
    size = image.shape
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    mtx = numpy.array([[focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1]], dtype="double")
    dist = numpy.zeros((4, 1))  # Assuming no lens distortion

    #generate stimuli
    stim = slab.Sound.pinknoise(duration=10.0)
    freefield.write('playbuflen', len(stim.data), processors=['RX81', 'RX82'])
    freefield.set_signal_and_speaker(signal=stim, speaker=speaker_id, equalize=False)
    offset_pose = numpy.zeros(2)  # offset_pose is used to normalize head position to 0, 0
    print('STARTING..\n TARGET| azimuth: %.1f, elevation %.1f' % (target[0], target[1]))
    time.sleep(3)
    count_down = False
    isi, pose = compare_pose(target, show=False)  # get pose and return initial isi from la.norm(pose - target)
    freefield.write('isi', isi, processors=['RX81', 'RX82'])  # write isi to processors
    freefield.play(kind='zBusA', proc='all')
    # start to play stimuli (zBus)
    freefield.set_logger('warning')

    while True:  # loop over trials
        if pose.all() is not None:
            print('head pose: azimuth: %.1f, elevation: %.1f' % (pose[0], pose[1]))
        isi, pose = compare_pose(target, show=True)  #.astype('float16')  # get pose and return isi from la.norm(pose - target)
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

def compare_pose(target, show):
    # pose = pose_from_sensor()
    azimuth = pose_from_image(cams[1], show=show)
    elevation = pose_from_image(cams[0], show=show)
    pose = numpy.array((azimuth, elevation))
    if not azimuth and not elevation:
        isi = isi_params['tmax']
    elif not azimuth or not elevation:
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

def pose_from_image(cam, show=False):
    image = get_image(cam)
    # detect ArUco marker
    (corners, id, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
    # verify *at least* one ArUco marker was detected
    if len(corners) > 0:
        marker_len = 0.05
        rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(corners, marker_len, mtx, dist)
        imaxis = cv2.aruco.drawDetectedMarkers(image.copy(), corners, id)
        image = cv2.aruco.drawAxis(imaxis, mtx, dist, rvecs, tvecs, marker_len)
        # calculate euler angles (radians)
        rmat = cv2.Rodrigues(rvecs)[0]  # Convert rvec to a 3x3 matrix (cv2.Rodrigues)
        # Create the projection matrix P = [ R | t ]:
        P = numpy.hstack((rmat, tvecs.T))
        # retrieve euler angles from projection matrix (radians)
        euler_angles_radians = -cv2.decomposeProjectionMatrix(P)[6]
        # convert to degrees
        pitch, yaw, roll = [math.radians(_) for _ in euler_angles_radians]
        # pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        # yaw = math.degrees(math.asin(math.sin(yaw)))
        # show orientation on image
        bottomLeftCornerOfText = (int(corners[0][2][0]), int(corners[0][2][1]))
        cv2.putText(image, 'yaw: %f roll: %f pitch: %f' % (yaw, roll, pitch),
                    bottomLeftCornerOfText, cv2.FONT_HERSHEY_PLAIN, fontScale=0.7, color=(0, 0, 225), lineType=1,
                    thickness=1)
        return roll
    else:
        pass
    if show:
        cv2.imshow(cam.DeviceID(), image)

def get_image(cam):
    image_result = cam.GetNextImage()
    image = image_result.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)
    # image = image_result.Convert(PySpin.PixelFormat_RGB8, PySpin.HQ_LINEAR)
    image = image.GetNDArray()
    image = change_image_res(image, 0.7)
    image.setflags(write=1)
    image_result.Release()
    return image

def change_image_res(image, resolution):
    data = PIL.Image.fromarray(image)
    width = int(data.size[0] * resolution)
    height = int(data.size[1] * resolution)
    image = data.resize((width, height), PIL.Image.ANTIALIAS)
    return numpy.asarray(image)

def pose_from_sensor(): #todo convert to interaural polar
    pose[0] = freefield.read('elevation', 'RP2')
    pose[1] = freefield.read('azimuth', 'RP2')
    return pose

def calibrate_sensor(accuracy_limit=0.3): # smaller accuracy limits will result in higher precision take longer to calibrate
    # get offset pose (az 0, ele 0)
    pose_log = numpy.zeros(2)
    while True:  # wait in loop for sensor to stabilize
        read = pose_from_sensor()
        if read is not None:
            pose_log = numpy.vstack([pose_log, read])
            if len(pose_log>100):
                az_acc = numpy.mean(numpy.abs(numpy.diff(pose_log[-100:, 0]))).astype('float16')
                ele_acc = numpy.mean(numpy.abs(numpy.diff(pose_log[-100:, 1]))).astype('float16')
                print('az acc: %f,  ele acc: %f'%(az_acc, ele_acc))
        # check if differential is stable for at least 100 data points,
        # set accuracy limit to max 0.1 for stable measurements
        if len(pose_log) > 100 and numpy.mean(numpy.abs(numpy.diff(pose_log[-100:, 0]))) < accuracy_limit and\
                numpy.mean(numpy.abs(numpy.diff(pose_log[-100:, 1]))) < accuracy_limit:
            break
    offset_pose = numpy.mean(pose_log[-50:, :], axis=0)
    return offset_pose

# if __name__ == "__main__":
#     hrtf_relearning(n_trials)

