#!/usr/bin/env python

import numpy as np
import cv2   as cv
from umucv.stream import autoStream

square_size = 1
pattern_size = (9, 6)
pattern_points = np.zeros( (np.prod(pattern_size), 3), np.float32 )
pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size

obj_points = []
img_points = []

for _, frame in autoStream():
    img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    h, w = img.shape
    found, corners = cv.findChessboardCorners(img, pattern_size)
    if found:
        term = ( cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1 )
        cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
    if not found:
        print('chessboard not found')
        continue
    img_points.append(corners.reshape(-1, 2))
    obj_points.append(pattern_points)

    print('ok')

print(len(obj_points))
rms, camera_matrix, dist_coefs, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, (w, h), None, None)

print("RMS:", rms)
print("camera matrix:\n", np.round((camera_matrix)))
print("distortion coefficients: ", np.round(dist_coefs.ravel(),3))

np.savetxt('calib.txt', np.array([x for x in camera_matrix.flatten()]+[x for x in dist_coefs.flatten()]))

