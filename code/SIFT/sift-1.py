#!/usr/bin/env python

# calculamos la mejor coincidencia de cada punto con una imagen de referencia

import cv2 as cv
import time

from umucv.stream import autoStream
from umucv.util import putText

sift = cv.SIFT_create(nfeatures=500)

# añadimos un algoritmo para encontrar asociaciones por fuerza bruta :(
matcher = cv.BFMatcher()

# variable donde guardaremos una imagen de referencia
x0 = None

for key, frame in autoStream():

    # para borrar la imagen de referencia
    if key == ord('x'):
        x0 = None

    t0 = time.time()
    keypoints , descriptors = sift.detectAndCompute(frame, mask=None)
    t1 = time.time()
    putText(frame, f'{len(keypoints)} pts  {1000*(t1-t0):.0f} ms')

    if key == ord('c'):
        # guardamos una imagen de referencia, con sus puntos y descriptores
        k0, d0, x0 = keypoints, descriptors, frame


    # si no hay imagen de referencia simplemente mostramos los puntos como antes
    if x0 is None:
        flag = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        cv.drawKeypoints(frame, keypoints, frame, color=(100,150,255), flags=flag)
        cv.imshow('SIFT', frame)
    else:
        
        # calculamos la mejor coincidencia de cada punto de la imagen actual
        # con la del modelo. Usamos una técnica de vecino más próximo, con un solo vecino
        t2 = time.time()
        matches = matcher.knnMatch(descriptors, d0, k=1)
        t3 = time.time()
        
        # sacamos la mejor (y en este caso única coincidencia) de cada uno
        best_match = [m[0] for m in matches]

        # y las dibujamos
        imgm = cv.drawMatches(frame, keypoints, x0, k0, best_match,
                              flags=0,
                              matchColor=(128,255,128),
                              singlePointColor = (128,128,128),
                              outImg=None)

        putText(imgm ,f'{len(best_match)} matches  {1000*(t3-t2):.0f} ms', 
                      orig=(5,36), color=(200,255,200))            
        cv.imshow("SIFT",imgm)


# habrá muchas coincidencias malas. Algunas se puede eliminar en base al grado de coincidencia,
# pero en el siguiente ejemplo de código haremos algo mejor

