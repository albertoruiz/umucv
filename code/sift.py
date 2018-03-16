#!/usr/bin/env python

import cv2 as cv
import time

from umucv.stream import autoStream
from umucv.util import putText

sift = cv.xfeatures2d.SIFT_create(nfeatures=0, contrastThreshold=0.05)

matcher = cv.BFMatcher()  # buscador de coincidencias por fuerza bruta

x0 = None

for key, x in autoStream():

    if key == ord('x'):
        x0 = None

    t0 = time.time()
    keypoints , descriptors = sift.detectAndCompute(x, mask=None)
    t1 = time.time()
    putText(x, '{}  {:.0f}ms'.format(len(keypoints), 1000*(t1-t0)))

    if key == ord('c'):
        # guardamos una imagen de referencia, con sus puntos y descriptores
        k0, d0, x0 = keypoints, descriptors, x


    if x0 is None:
        flag = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        cv.drawKeypoints(x,keypoints,x, color=(100,150,255), flags=flag)
        cv.imshow('SIFT', x)
    else:
        t2 = time.time()
        matches = matcher.knnMatch(descriptors, d0, k=2)  # dame las dos mejores coincidencias
        t3 = time.time()

        # ratio test
        good = []
        for m in matches:
            if len(m) >= 2:
                best,second = m
                if best.distance < 0.75*second.distance:
                    good.append(best)
        
        imgm = cv.drawMatches(x, keypoints, x0, k0, good,
                              flags=0,
                              matchColor=(128,255,128),
                              singlePointColor = (128,128,128),
                              outImg=None)

        putText(imgm ,'{} {:.0f}ms'.format(len(good),1000*(t3-t2)), (150,16), (200,255,200))            
        cv.imshow("SIFT",imgm)

