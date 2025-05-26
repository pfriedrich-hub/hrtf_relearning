import numpy as np
import cv2
import time
import os
from pathlib import Path

#generate small board for pose estimation
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
board = cv2.aruco.GridBoard_create(
    markersX=7,
    markersY=4,
    markerLength=0.015,
    markerSeparation=0.007,
    dictionary=aruco_dict)

im=board.draw([1080,720])
cv2.imwrite(str(Path.cwd() / 'head_tracking' / 'cam_tracking' / 'aruco_markers' / 'board_7_4x4_100_ts.png'), im)
cv2.imshow('im',im)

# calibration
# ChArUco board for calibration
cv2.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.CharucoBoard_create(7, 5, 1, .8, cv2.aruco_dict)

# get some images for cam calibration (single cam for now)
system = PySpin.System.GetInstance()
cams = system.GetCameras()
for cam in cams:
    cam.Init()
    cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)  # disable auto exposure time
    cam.ExposureTime.SetValue(100000.0)
    cam.BeginAcquisition()

images=[]
time.sleep(10)
for i in range(12):
    image_result = cam.GetNextImage()
    image = image_result.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)
    # image = image_result.Convert(PySpin.PixelFormat_RGB8, PySpin.HQ_LINEAR)
    image = image.GetNDArray()
    image.setflags(write=1)
    image_result.Release()
    print('done! next picture in 5sec')
    images.append(image)
    time.sleep(5)

images=np.asarray(images)
cv2.imshow('im', image)
cv2.waitKey(0)

for i in range(len(images)):
    cv2.imwrite('image %i'%(i), images[i])


datadir = os.getcwd() + '\\cam_calibr\\'
images = np.array([datadir + f for f in os.listdir(datadir) if f.endswith(".png") ])
# order = np.argsort([int(p.split(".")[-2].split("_")[-1]) for p in images])
# images = images[order]
images

def read_chessboards(images):
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    print("POSE ESTIMATION STARTS:")
    allCorners = []
    allIds = []
    decimator = 0
    # SUB PIXEL CORNER DETECTION CRITERION
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
    for im in images:
        print("=> Processing image {0}".format(im))
        frame = cv2.imread(im)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)
        if len(corners)>0:
            # SUB PIXEL DETECTION
            for corner in corners:
                cv2.cornerSubPix(gray, corner, winSize = (3,3), zeroZone = (-1,-1),criteria = criteria)
            res2 = cv2.aruco.interpolateCornersCharuco(corners,ids,gray,board)
            if res2[1] is not None and res2[2] is not None and len(res2[1])>3 and decimator%1==0:
                allCorners.append(res2[1])
                allIds.append(res2[2])
        decimator+=1
    imsize = gray.shape
    return allCorners,allIds,imsize

def calibrate_camera(allCorners,allIds,imsize):
    print("CAMERA CALIBRATION")
    cameraMatrixInit = np.array([[ 1000.,    0., imsize[0]/2.],
                                 [    0., 1000., imsize[1]/2.],
                                 [    0.,    0.,           1.]])
    distCoeffsInit = np.zeros((5,1))
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
    #flags = (cv2.CALIB_RATIONAL_MODEL)
    (ret, camera_matrix, distortion_coefficients0,
     rotation_vectors, translation_vectors,
     stdDeviationsIntrinsics, stdDeviationsExtrinsics,
     perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(charucoCorners=allCorners, charucoIds=allIds,
                      board=board,imageSize=imsize, cameraMatrix=cameraMatrixInit, distCoeffs=distCoeffsInit,
                      flags=flags,criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))
    return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors

allCorners,allIds,imsize=read_chessboards(images)
ret, mtx, dist, rvecs, tvecs = calibrate_camera(allCorners,allIds,imsize)

np.savetxt('mtx.txt', mtx, fmt='%d')
np.savetxt('dist.txt', dist, fmt='%d')





