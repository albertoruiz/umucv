#!/usr/bin/env python

# Triangulamos la posición 3D de los puntos de calibración
# Elegimos las vistas pulsando 1 y 2

import numpy as np
import cv2   as cv
from umucv.stream import autoStream
from umucv.htrans import htrans, jc, inhomog
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from umucv.util import cameraOutline


def undistort(pts, K, D):
    # mantenemos la K para tener luego M completas
    # (si quitamos htrans(K, ...) aquí, debemos quitar K @ al formar M)
    criteria = (cv.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    return htrans(K, cv.undistortPointsIter(pts.reshape(-1,1,2), K, D,None,None,criteria).reshape(-1,2))


def plot3(ax,c,color):
    ax.plot(c[:,0],c[:,1],c[:,2],color)


square_size = 0.5
pattern_size = (9, 6)
pattern_points = np.zeros( (np.prod(pattern_size), 3), np.float32 )
pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size

calibdata = np.loadtxt("calib.txt")

K = calibdata[:9].reshape(3,3)
D = calibdata[9:]

pts1 = None
pts2 = None

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
            R,_ = cv.Rodrigues(rvec)
            M = K @ jc(R,tvec)

            repro = cv.projectPoints(pattern_points,rvec,tvec,K,D)[0].reshape(-1,2)
            for x,y in repro:
                cv.circle(frame,(int(x),int(y)),4,(0,255,255),-1,cv.LINE_AA)
                cv.circle(frame,(int(x),int(y)),2,(0,0,0),-1,cv.LINE_AA)

            if key==ord('1'):
                pts1 = corners
                M1 = M
            if key==ord('2'):
                pts2 = corners
                M2 = M

    cv.imshow("source",frame)
    
    if pts1 is not None and pts2 is not None:
        
        rec1 = undistort(pts1, K, D)
        rec2 = undistort(pts2, K, D)

        p3d = inhomog(cv.triangulatePoints(M1,M2,rec1.T,rec2.T).T)
        print(p3d)
        pts2 = None
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection='3d')
        plot3(ax,cameraOutline(M1),'blue');
        plot3(ax,cameraOutline(M2),'red');
        plot3(ax,p3d,'.g')
        ax.set_aspect('equal')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        plt.show()


