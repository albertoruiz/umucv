#!/usr/bin/env python

# eliminamos muchas coincidencias erróneas mediante el "ratio test"

import cv2 as cv
import time

from umucv.stream import autoStream
from umucv.util import putText

sift = cv.SIFT_create(nfeatures=500)

matcher = cv.BFMatcher()

x0 = None

for key, frame in autoStream():

    if key == ord('x'):
        x0 = None

    t0 = time.time()
    keypoints , descriptors = sift.detectAndCompute(frame, mask=None)
    t1 = time.time()
    putText(frame, f'{len(keypoints)} pts  {1000*(t1-t0):.0f} ms')

    if key == ord('c'):
        k0, d0, x0 = keypoints, descriptors, frame

    if x0 is None:
        flag = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        cv.drawKeypoints(frame, keypoints, frame, color=(100,150,255), flags=flag)
        cv.imshow('SIFT', frame)
    else:
        t2 = time.time()
        # solicitamos las dos mejores coincidencias de cada punto, no solo la mejor
        matches = matcher.knnMatch(descriptors, d0, k=2)
        t3 = time.time()

        # ratio test
        # nos quedamos solo con las coincidencias que son mucho mejores que
        # que la "segunda opción". Es decir, si un punto se parece más o menos lo mismo
        # a dos puntos diferentes del modelo lo eliminamos.
        good = []
        for m in matches:
            if len(m) >= 2:
                best,second = m
                if best.distance < 0.75*second.distance:
                    good.append(best)
        
        imgm = cv.drawMatches(frame, keypoints, x0, k0, good,
                              flags=0,
                              matchColor=(128,255,128),
                              singlePointColor = (128,128,128),
                              outImg=None)

        putText(imgm ,f'{len(good)} matches  {1000*(t3-t2):.0f} ms', 
                      orig=(5,36), color=(200,255,200))           
        cv.imshow("SIFT",imgm)


