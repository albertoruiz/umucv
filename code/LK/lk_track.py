#!/usr/bin/env python

# La solución anterior (lk_track1.py) tiene el problema de que las asociaciones de puntos
# a veces son incorrectas y el número de trayectorias crece indefinidamente
# repitiendo puntos. Esto se soluciona con dos mejoras.

import cv2 as cv
import numpy as np
from umucv.stream import autoStream, sourceArgs
from umucv.util import putText
from collections import deque
import time

tracks = []
track_len = 20
detect_interval = 5

corners_params = dict( maxCorners = 500,
                       qualityLevel= 0.1,
                       minDistance = 10,
                       blockSize = 7)

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))


for n, (key, frame) in enumerate(autoStream()):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    t0 = time.time()
    if tracks:

        # el criterio para considerar bueno un punto siguiente es que si lo proyectamos
        # hacia el pasado, vuelva muy cerca del punto incial, es decir:
        # "back-tracking for match verification between frames"
        p0 = np.float32( [t[-1] for t in tracks] )
        p1,  _, _ =  cv.calcOpticalFlowPyrLK(prevgray, gray, p0, None, **lk_params)
        p0r, _, _ =  cv.calcOpticalFlowPyrLK(gray, prevgray, p1, None, **lk_params)
        d = abs(p0-p0r).reshape(-1,2).max(axis=1)
        good = d < 1

        new_tracks = []
        for t, point, ok in zip(tracks, p1.reshape(-1,2), good):
            if not ok:
                continue
            t.append( point )
            new_tracks.append(t)

        tracks = new_tracks

        cv.polylines(frame, [ np.int32(t) for t in tracks ], isClosed=False, color=(0,0,255))
        for t in tracks:
            point = np.int32(t[-1])
            cv.circle(frame, center=point, radius=2, color=(0, 0, 255), thickness=-1)

    t1 = time.time()

    if n % detect_interval == 0:
        # Creamos una máscara para indicar al detector de puntos nuevos las zona
        # permitida, que es toda la imagen, quitando círculos alrededor de los puntos
        # existentes (los últimos de las trayectorias).
        mask = np.zeros_like(gray)
        mask[:] = 255
        for x,y in [np.int32(t[-1]) for t in tracks]:
            cv.circle(mask, (x,y), 5, 0, -1)
        #cv.imshow("mask",mask)
        corners = cv.goodFeaturesToTrack(gray, mask=mask, **corners_params)
        if corners is not None:
            for [pt] in np.float32(corners):
                tracks.append( deque([pt], maxlen=track_len) )

    putText(frame, f'{len(tracks)} corners, {(t1-t0)*1000:.0f}ms' )
    cv.imshow('input', frame)
    prevgray = gray

