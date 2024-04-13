#!/usr/bin/env python

#https://mediapipe.readthedocs.io/en/latest/solutions/hands.html

import cv2 as cv
import numpy as np

from umucv.stream import autoStream
from umucv.util import putText

# usar el detector de manos de mediapipe
import mediapipe as mp
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# recorrer todos los fotogramas del flujo de entrada
for _, frame in autoStream():
    H, W, _ = frame.shape
    # es mejor trabajar con imagen espejo
    imagecv = cv.flip(frame, 1)
    # el detector necesita tener los canales de color en el orden correcto
    image = cv.cvtColor(imagecv, cv.COLOR_BGR2RGB)
    
    # lanzamos el proceso de detección principal mágico que hace mediapipe
    results = hands.process(image)
    
    points = []
    # tengo detección?
    if results.multi_hand_landmarks:
        # para cada mano detectada
        for hand_landmarks in results.multi_hand_landmarks:
            # meter los "landmarks" en un array de numpy
            for k in range(21):
                x = hand_landmarks.landmark[k].x # entre 0 y 1 !
                y = hand_landmarks.landmark[k].y            
                points.append([int(x*W),int(y*H)])  # int para dibujar en cv
            break

        points = np.array(points) # mejor un array para poder operar matemáticamente
        #print(points)

        # dibujar un segmento de recta en el dedo índice
        cv.line(imagecv, points[5], points[8], color=(0,255,255), thickness=3)

        # dibujo un círculo centrado en la palma de la mano

        center = np.mean( points[ [5,0,17] ] , axis=0 )
        # (extraigo los 3 puntos de la palma y hago la media por columnas)
        radio = np.linalg.norm(center - points[5])
        # (el radio calculado así solo es aceptable con la mano vertical)
        putText(imagecv, F"radio = {radio:.0f}")
        
        cv.circle(imagecv, center.astype(int), int(radio), color=(0,255,255), thickness=3)

    cv.imshow("mirror", imagecv)

