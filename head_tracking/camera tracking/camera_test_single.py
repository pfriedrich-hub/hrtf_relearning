import numpy
import cv2
import PySpin
import PIL
import logging
from PIL import Image
from scipy.spatial.transform import Rotation as R
import os

def aruco_test():
    print('starting..')
    images = []
    system = PySpin.System.GetInstance()
    cams = system.GetCameras()
    for cam in cams:  # initialize cameras
        cam.Init()
        cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off) # disable auto exposure time
        cam.ExposureTime.SetValue(30000)  # (100000.0)
        cam.BeginAcquisition()

    while True: # get pose and draw on marker orientation on image for each camera
        for i_cam, cam in enumerate(cams):
            image = get_image(cam)
            image = change_res(image, 0.5)
            pose, info = pose_from_image(image)
            image = draw_markers(image, pose, info)
            cv2.imshow('camera %i' % i_cam, image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            images.append(image)
        if key == ord("q"):  # if the `q` key was pressed, break
            for cam in cams:
                if cam.IsInitialized():
                    cam.EndAcquisition()
                    cam.DeInit()
                del cam
            cams.Clear()
            system.ReleaseInstance()
            break
    return images

def get_image(cam):
    image_result = cam.GetNextImage()
    image = image_result.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)
    #image = image_result.Convert(PySpin.PixelFormat_RGB8, PySpin.HQ_LINEAR)
    image = image.GetNDArray()
    image.setflags(write=1)
    image_result.Release()
    return image

def pose_from_image(image): # get pose
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters_create()
    # params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # set parameters
    # params.adaptiveThreshWinSizeMin=5
    # params.adaptiveThreshWinSizeMax=5
    # params.adaptiveThreshWinSizeStep=100

    # params.cornerRefinementWinSize=10
    # params.cornerRefinementMinAccuracy=0.001
    # params.cornerRefinementMaxIterations=50

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

            # #alternative with solve pnp
            # obj_points = numpy.zeros((2*2,3), numpy.float32)
            # obj_points[:, :2] = numpy.mgrid[0:2, 0:2].T.reshape(-1, 2)
            # ret, rotation_vec, translation_vec = cv2.solvePnP(obj_points, corners[i], camera_matrix, dist_coeffs)
            # rotation_mat = cv2.Rodrigues(rotation_vec)[0]
            # pose_mat = cv2.hconcat((rotation_mat, translation_vec))
            # _, _, _, _, _, _, angles = cv2.decomposeProjectionMatrix(pose_mat)

            #messinng around with vectors
            # r = R.from_euler('xz', [180,90], degrees=True)
            # rotation_mat=r.apply(rotation_mat)
            # #r = R.from_matrix(rotation_mat)
            # #r2 = r.inv()
            # #rotation_mat = r2.as_matrix()
            # rotation_vec = cv2.Rodrigues(-rotation_mat)[0]

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
    marker_len = .1
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict)
    if len(corners) > 0:
        for i in range(len(corners)):
            Imaxis = cv2.aruco.drawDetectedMarkers(image.copy(), corners, ids)
            image = cv2.aruco.drawAxis(Imaxis, info[i][0], info[i][1], info[i][2], info[i][3], marker_len)
            # info: list of arrays [camera_matrix, dist_coeffs, rotation_vec, translation_vec]
            bottomLeftCornerOfText = (int(corners[0][0][2][0]), int(corners[0][0][2][1]))
            cv2.putText(image, 'azimuth: %f elevation: %f roll: %f' % (pose[i][0], pose[i][1], pose[i][2]), # display heade pose
            bottomLeftCornerOfText, cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(225, 225, 225),
                        lineType=1, thickness=1)
    return(image)

def change_res(image, resolution):
    data = PIL.Image.fromarray(image)
    width = int(data.size[0] * resolution)
    height = int(data.size[1] * resolution)
    image = data.resize((width, height), PIL.Image.ANTIALIAS)
    return numpy.asarray(image)

if __name__ == "__main__":
    images = aruco_test()