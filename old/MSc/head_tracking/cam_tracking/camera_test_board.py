import numpy as np
import cv2
import PySpin
import PIL
import logging
from PIL import Image
from scipy.spatial.transform import Rotation as R
def aruco_test():
    print('starting..')
    system = PySpin.System.GetInstance()
    cams = system.GetCameras()
    for cam in cams: # initialize cameras
        cam.Init()
        cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off) # disable auto exposure time
        cam.ExposureTime.SetValue(100000.0)
        cam.BeginAcquisition()
    n_cams = cams.GetSize()


    while True: # get pose and draw on marker orientation on image for each camera
        for i_cam, cam in enumerate(cams):
            image = get_image(cam)
            # image = change_image_res(image, 0.5)
            image = pose_from_image(image)
            image = change_image_res(image, 0.5)
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

def change_image_res(image, resolution):
    data = PIL.Image.fromarray(image)
    width = int(data.size[0] * resolution)
    height = int(data.size[1] * resolution)
    image = data.resize((width, height), PIL.Image.ANTIALIAS)
    return np.asarray(image)

def pose_from_image(image): # get pose
    pose = []
    info = []
    # try params
    params = cv2.aruco.DetectorParameters_create()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # set parameters
    # params.adaptiveThreshWinSizeMin=5
    # params.adaptiveThreshWinSizeMax=5
    # params.adaptiveThreshWinSizeStep=100
    # params.cornerRefinementWinSize=10
    # params.cornerRefinementMinAccuracy=0.001
    # params.cornerRefinementMaxIterations=50

    #arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    #get corners
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=params)
    # use aruco board for pose estimation
    board = cv2.aruco.GridBoard_create(
        markersX=3,
        markersY=2,
        markerLength=0.04,
        markerSeparation=0.01,
        dictionary=arucoDict)
    cv2.aruco.refineDetectedMarkers(image, board, corners, ids, rejected)
    size = image.shape
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    if len(corners) > 0:
        image = cv2.aruco.drawDetectedMarkers(image, corners, ids)
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[0], .05, camera_matrix, dist_coeffs)
        rval, rotation_vec, translation_vec = cv2.aruco.estimatePoseBoard(corners, ids, board, camera_matrix,
                                                                       dist_coeffs, rvec, tvec)
        if rval != None: # draw board
            image = cv2.aruco.drawAxis(image.copy(), camera_matrix, dist_coeffs, rotation_vec, translation_vec, .3)
            rotation_mat = -cv2.Rodrigues(rotation_vec)[0]
            pose_mat = cv2.hconcat((rotation_mat, translation_vec))
            _, _, _, _, _, _, angles = cv2.decomposeProjectionMatrix(pose_mat)
            angles[1, 0] = np.radians(angles[1, 0])
            angles[1, 0] = np.degrees(np.arcsin(np.sin(np.radians(angles[1, 0]))))
            angles[0, 0] = -angles[0, 0]
            bottomLeftCornerOfText = (int(corners[0][0][2][0]), int(corners[0][0][2][1]))
            cv2.putText(image, 'azimuth: %f elevation: %f roll: %f' % (angles[1], angles[0], angles[2]),
                        # display heade pose
                        bottomLeftCornerOfText, cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(225, 225, 225),
                        lineType=1, thickness=1)

    return image

def change_res(image, imsize, resolution):
    image = PIL.Image.fromarray(image)
    width = int(imsize[1] * resolution)
    height = int(imsize[0] * resolution)
    image = image.resize((width, height), PIL.Image.ANTIALIAS)
    return np.asarray(image)

if __name__ == "__main__":
    images = aruco_test()
