#!/usr/bin/env python

# Calculamos una lista de posiciones de cada trayectoria.
# Cada trayectoria tiene una longitud máxima.
# Se añaden nuevos puntos iniciales cada "detect_interval" frames.


import cv2 as cv
import numpy as np
from umucv.stream import autoStream, sourceArgs
from umucv.util import putText
import time


tracks = []
track_len = 20
detect_interval = 5


corners_params = dict( maxCorners = 500,
                       qualityLevel= 0.1,
                       minDistance = 10,
                       blockSize = 7,
                       mask = None)

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))



for n, (key, frame) in enumerate(autoStream()):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    t0 = time.time()
    if len(tracks):

       
        # sacamos los últimos puntos de cada trayectoria
        p0 = np.float32( [t[-1] for t in tracks] )
        
        # calculamos su posición siguiente
        p1, good, _ = cv.calcOpticalFlowPyrLK(prevgray, gray, p0, None, **lk_params)
        
        new_tracks = []
        for t, (x,y), ok in zip(tracks, p1.reshape(-1,2), good):
            # si no se encuentra, abandonamos esta trayectoria
            if not ok:
                continue
            # añadimos el punto siguiente encontrado
            t.append( [x,y] )
            # y si se ha alcanzado el tamaño máximo quitamos el primero
            if len(t) > track_len:
                del t[0]
            new_tracks.append(t)

        tracks = new_tracks
    
        # dibujamos las trayectorias
        cv.polylines(frame, [ np.int32(t) for t in tracks ], isClosed=False, color=(0,0,255))
    
    t1 = time.time()
    
    # Añadimos nuevos principios de trayectoria con puntos nuevos
    if n % detect_interval == 0:
        corners = cv.goodFeaturesToTrack(gray, **corners_params).reshape(-1,2)
        if corners is not None:
            for x, y in np.float32(corners):
                tracks.append( [  [ x,y ]  ] )

    putText(frame, f'{len(tracks)} corners, {(t1-t0)*1000:.0f}ms' )
    cv.imshow('input', frame)
    prevgray = gray

