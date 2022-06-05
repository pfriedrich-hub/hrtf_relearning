import numpy
import cv2
import freefield
import PIL
from PIL import Image
arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
params = cv2.aruco.DetectorParameters_create()
import PySpin
freefield.set_logger('warning')

def show_fast(cam):
    image = get_image(cam)
    # image = change_res(image, 0.5)  # bad idea
    pose, info = pose_from_image(image)
    #cv2.imshow('camera %s' % cam.DeviceID(), image)
    print(pose)
    cv2.waitKey(1) & 0xFF

def get_pose(cam, show=False, scale=False):
    image = get_image(cam)
    pose, info = pose_from_image(image)
    if show:
        if scale:
            image = change_res(image, 0.5)
        image = draw_markers(image, pose, info)
        cv2.imshow('camera %s' % cam.DeviceID(), image)
        cv2.waitKey(1) & 0xFF
    else:
        cv2.waitKey(0)
    if pose:
        pose = numpy.mean(numpy.asarray(pose)[:, 2]).astype('float16')
    return pose

def get_image(cam):
    image_result = cam.GetNextImage()
    image = image_result.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)
    #image = image_result.Convert(PySpin.PixelFormat_RGB8, PySpin.HQ_LINEAR)
    image = image.GetNDArray()
    image.setflags(write=1)
    image_result.Release()
    return image

def pose_from_image(image): # get pose
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=params)
    if len(corners) == 0:
        return None, [0, 0, 0, 0]
    else:
        size = image.shape
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = numpy.array([[focal_length, 0, center[0]],
                                  [0, focal_length, center[1]],
                                  [0, 0, 1]], dtype="double")
        dist_coeffs = numpy.zeros((4, 1))  # Assuming no lens distortion
        #camera_matrix = numpy.loadtxt('mtx.txt')
        #dist_coeffs = numpy.loadtxt('dist.txt')
        rotation_vec, translation_vec, _objPoints = \
            cv2.aruco.estimatePoseSingleMarkers(corners, .05, camera_matrix, dist_coeffs)
        pose = [] #numpy.zeros([len(translation_vec), 2])
        info = [] #numpy.zeros([len(translation_vec), 4])
        for i in range(len(translation_vec)):
            rotation_mat = -cv2.Rodrigues(rotation_vec[i])[0]
            pose_mat = cv2.hconcat((rotation_mat, translation_vec[i].T))
            _, _, _, _, _, _, angles = cv2.decomposeProjectionMatrix(pose_mat)
            angles[1, 0] = numpy.radians(angles[1, 0])
            angles[1, 0] = numpy.degrees(numpy.arcsin(numpy.sin(numpy.radians(angles[1, 0]))))
            angles[0, 0] = -angles[0, 0]
            info.append([camera_matrix, dist_coeffs, rotation_vec[i], translation_vec[i]])
            pose.append([angles[1, 0], angles[0, 0], angles[2, 0]])
        return pose, info

def draw_markers(image, pose, info):
    marker_len = .05
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict)
    if len(corners) > 0:
        for i in range(len(corners)):
            Imaxis = cv2.aruco.drawDetectedMarkers(image.copy(), corners, ids=None)
            image = cv2.aruco.drawAxis(Imaxis, info[i][0], info[i][1], info[i][2], info[i][3], marker_len)
            # info: list of arrays [camera_matrix, dist_coeffs, rotation_vec, translation_vec]
            bottomLeftCornerOfText = (20, 20+(20*i))
            cv2.putText(image, 'roll: %f' % (pose[i][2]),  # display heade pose
                bottomLeftCornerOfText, cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(225, 225, 225),
                        lineType=1, thickness=1)
    return(image)

def change_res(image, resolution):
    data = PIL.Image.fromarray(image)
    width = int(data.size[0] * resolution)
    height = int(data.size[1] * resolution)
    image = data.resize((width, height), PIL.Image.ANTIALIAS)
    return numpy.asarray(image)

def calibrate_aruco(cams, limit=0.5, report=False):
    [led_speaker] = freefield.pick_speakers(23)  # get object for center speaker LED
    freefield.write(tag='bitmask', value=led_speaker.digital_channel,
                    processors=led_speaker.digital_proc)  # illuminate LED
    print('rest at center speaker and press button to start calibration...')
    freefield.wait_for_button()  # start calibration after button press
    log = numpy.zeros(2)
    while True:  # wait in loop for sensor to stabilize
        pose = [get_pose(cams[1]), get_pose(cams[0])]
        # print(pose)
        log = numpy.vstack((log, pose))
        if log[-1, 0] == None or log[-1, 1] == None:
            print('no marker detected')
        # check if orientation is stable for at least 30 data points
        if len(log) > 30 and all(log[-20:, 0] != None) and all(log[-20:, 1] != None):
            diff = numpy.mean(numpy.abs(numpy.diff(log[-20:], axis=0)), axis=0).astype('float16')
            if report:
                print('az diff: %f,  ele diff: %f' % (diff[0], diff[1]))
            if diff[0] < limit and diff[1] < limit:  # limit in degree
                break
    freefield.write(tag='bitmask', value=0, processors=led_speaker.digital_proc)  # turn off LED
    pose_offset = numpy.around(numpy.mean(log[-20:].astype('float16'), axis=0), decimals=2)
    print('calibration complete, thank you!')
    return pose_offset
