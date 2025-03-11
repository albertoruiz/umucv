#!/usr/bin/env python

# Añadimos argumento en la línea de órdenes y medida del tiempo de cálculo

import cv2 as cv
import numpy as np
from umucv.stream import autoStream
from umucv.util import putText, read_arguments
import time
import argparse

# recordamos la forma de añadir argumentos en la línea de órdenes
def my_arguments(parser):
    parser.add_argument('--quality', type=float, default=0.1, help='intensidad de los puntos (0-1)')

args = read_arguments(my_arguments)


# parámetros del detector
corners_params = dict( maxCorners = 500,
                       qualityLevel= args.quality,
                       minDistance = 10,
                       blockSize = 7 )

for key, frame in autoStream():
    # pasamos a monocromo
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    t0 = time.time()    
    corners = cv.goodFeaturesToTrack(gray, **corners_params).reshape(-1,2)
    t1 = time.time()
    
    #print(corners)
    for x,y in corners:
        cv.circle(frame, (int(x), int(y)) , radius=3 , color=(0,0,255), thickness=-1, lineType=cv.LINE_AA )
    
    putText(frame, f'{len(corners)} corners, {(t1-t0)*1000:.0f}ms' )
    cv.imshow('input', frame)

