import numpy
import cv2
import PySpin
import PIL
import logging
from PIL import Image
from scipy.spatial.transform import Rotation as R
import os
arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
params = cv2.aruco.DetectorParameters_create()

def get_pose(cam, show):
    image = get_image(cam)
    image = change_res(image, 0.5)
    pose, info = pose_from_image(image)
    image = draw_markers(image, pose, info)
    if show:
        cv2.imshow('camera %s' % cam.DeviceID(), image)
    cv2.waitKey(1) & 0xFF
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

# if __name__ == "__main__":
#     pose = aruco_pose(cams, show=False)