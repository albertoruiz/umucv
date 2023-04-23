#!/usr/bin/env python

# reproyectamos los puntos del tablero con la pose estimada.

# Es equivalente a hacer htrans(M, pts) precorrigiendo
# la distorsión radial. 

import numpy as np
import cv2   as cv
from umucv.stream import autoStream

square_size = 1
pattern_size = (9, 6)
pattern_points = np.zeros( (np.prod(pattern_size), 3), np.float32 )
pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size


calibdata = np.loadtxt("calib.txt")

K = calibdata[:9].reshape(3,3)
D = calibdata[9:]

orig = False

for key,frame in autoStream():
    img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    h, w = img.shape
    found, corners = cv.findChessboardCorners(img, pattern_size)
    if found:
        term = ( cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1 )
        cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
        corners = corners.reshape(-1,2)
        
        ok,rvec,tvec = cv.solvePnP(pattern_points, corners, K, D, flags=1*cv.SOLVEPNP_ITERATIVE)
        if ok:
            repro = cv.projectPoints(pattern_points,rvec,tvec,K,D)[0].reshape(-1,2)
            for x,y in repro:
                cv.circle(frame,(int(x),int(y)),4,(0,255,255),-1,cv.LINE_AA)
                cv.circle(frame,(int(x),int(y)),2,(0,0,0),-1,cv.LINE_AA)

    cv.imshow("source",frame)

