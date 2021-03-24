#!/usr/bin/env python

# Calculamos y mostramos los puntos SIFT

import cv2 as cv
import time
from umucv.stream import autoStream
from umucv.util import putText

# inicializamos el detector con los parámetros de trabajo deseados
# mira en la documentación su significado y prueba distintos valores
# https://docs.opencv.org/3.4/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html

sift = cv.xfeatures2d.SIFT_create(nfeatures=0, contrastThreshold=0.1, edgeThreshold=8)
# sift = cv.AKAZE_create()

for key, frame in autoStream():

    t0 = time.time()
    # invocamos al detector (por ahora no usamos los descriptores)
    keypoints , descriptors = sift.detectAndCompute(frame, mask=None)
    t1 = time.time()

    putText(frame, '{} keypoints  {:.0f} ms'.format(len(keypoints), 1000*(t1-t0)))

    # dibujamos los puntos encontrados, con un círculo que indica su tamaño y un radio
    # que indica su orientación.
    # al mover la cámara el tamaño y orientación debe mantenerse coherente con la imagen
    flag = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    cv.drawKeypoints(frame, keypoints, frame, color=(100,150,255), flags=flag)
    
    cv.imshow('SIFT', frame)

