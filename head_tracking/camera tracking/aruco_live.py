#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 17:13:58 2021

@author: paulfriedrich

#eg run via terminal: python aruco_live.py --type DICT_5X5_100
"""

# import the necessary packages
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import sys
import numpy as np
import math
import PySpin
from PIL import Image
import PIL

# construct the argument parser and parse the arguments
# define names of each possible ArUco tag OpenCV supports
ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

def change_image_res(image, resolution):
    data = PIL.Image.fromarray(image)
    width = int(data.size[0] * resolution)
    height = int(data.size[1] * resolution)
    image = data.resize((width, height), PIL.Image.ANTIALIAS)
    return np.asarray(image)

def get_image(cam):
    image_result = cam.GetNextImage()
    image = image_result.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)
    #image = image_result.Convert(PySpin.PixelFormat_RGB8, PySpin.HQ_LINEAR)
    image = image.GetNDArray()
    image.setflags(write=1)
    image_result.Release()
    return image

# verify that the supplied ArUCo tag exists and is supported by

# load the ArUCo dictionary and grab the ArUCo parameters
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT["DICT_5X5_100"])
arucoParams = cv2.aruco.DetectorParameters_create()

#load cam calibration parameters

system = PySpin.System.GetInstance()
cams = system.GetCameras()
for cam in cams: # initialize cameras
    cam.Init()
    cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off) # disable auto exposure time
    cam.ExposureTime.SetValue(100000.0)
    cam.BeginAcquisition()

image = get_image(cam)

size = image.shape
focal_length = size[1]
center = (size[1] / 2, size[0] / 2)
mtx = np.array([[focal_length, 0, center[0]],
                          [0, focal_length, center[1]],
                          [0, 0, 1]], dtype="double")
dist = np.zeros((4, 1))  # Assuming no lens distortion


# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
	# to have a maximum width of 1000 pixels
    image = get_image(cam)
    # frame = change_image_res(image, 0.5)
    #frame = vs.read()
	#frame = imutils.resize(frame, width=1000)
	# detect ArUco markers in the input frame
    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame,arucoDict, parameters=arucoParams)
    	# verify *at least* one ArUco marker was detected
    if len(corners) > 0:
        marker_len=0.05
        rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(corners, marker_len, mtx, dist)
        imaxis = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, ids)
        for i in range(len(tvecs)):
            frame = cv2.aruco.drawAxis(imaxis, mtx, dist, rvecs[i], tvecs[i], marker_len)
            # calculate euler angles (radians)
            #Convert rvec to a 3x3 matrix using cv2.Rodrigues():
            rmat = cv2.Rodrigues(rvecs[i])[0]
            #camera position expressed in the world frame (OXYZ):
            #cam_pos = -np.matrix(rmat).T * np.matrix(tvec)
            #Create the projection matrix P = [ R | t ]:
            P = np.hstack((rmat,tvecs[i].T))
            # retrieve euler angles from projection matrix (radians)
            euler_angles_radians = -cv2.decomposeProjectionMatrix(P)[6]
            # convert to degrees
            pitch, yaw, roll = [math.radians(_) for _ in euler_angles_radians]
            pitch = math.degrees(math.asin(math.sin(pitch)))
            roll = -math.degrees(math.asin(math.sin(roll)))
            yaw = math.degrees(math.asin(math.sin(yaw)))

            # show orientation on frame
            font = cv2.FONT_HERSHEY_PLAIN
            bottomLeftCornerOfText = (int(corners[i][0][2][0]),int(corners[i][0][2][1]))
            cv2.putText(frame,'yaw: %f roll: %f pitch: %f'%(yaw,roll,pitch),
            bottomLeftCornerOfText, cv2.FONT_HERSHEY_PLAIN, fontScale=0.7, color=(0,0,225), lineType=1, thickness=1)
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        for cam in cams:
            if cam.IsInitialized():
                cam.EndAcquisition()
                cam.DeInit()
        del cam
        cams.Clear()
        break

# cleanup
cv2.destroyAllWindows()

"""

# source: https://www.pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/
            # https://mecaruco2.readthedocs.io/en/latest/notebooks_rst/Aruco/sandbox/ludovic/aruco_calibration_rotation.html
            """