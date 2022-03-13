#!/usr/bin/env python

# Paso 1: extracción de contornos.

# puedes probarlo con la webcam, o con esta imagen de prueba
# ./trebol1.py --dev=dir:../../images/cards.png


import cv2          as cv
from umucv.stream import autoStream
import numpy as np

def binarize(gray):
    # Tres métodos alternativos. Elige uno:

    # 1) umbral fijo, implementado con numpy
    # r = (gray > 128).astype(np.uint8)*255

    # 2) umbral automático, método de Otsu
    _, r = cv.threshold(gray, 128, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

    # 3) umbral automático local
    # r = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 101, -10)
    return r
    

def extractContours(g):  
    # extraemos todos los contornos, incluidos los internos, y no reducimos los vértices
    # (llamamos a la función de forma que funcione en diferentes versiones de opencv)
    contours, _ = cv.findContours(g.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)[-2:]
    # quitamos la dimensión redundante de siempre
    contours = [ c.reshape(-1,2) for c in contours ]
    # ordenamos los contornos de mayor a menor área
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    return contours


# para invertir la máscara cuando buscamos objetos oscuros en fondo claro
black = True


# en el bucle de captura mostramos la imagen original, la binarizada, y los contornos
for (key,frame) in autoStream():

    cv.imshow('input',frame)

    g = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    if black:
        g = 255-g
    
    b = binarize(g)
    cv.imshow('binary', b)

    contours = extractContours(b)

    result = np.zeros_like(frame)
    cv.drawContours(result, contours, contourIdx=-1, color=(0,255,255), thickness=1, lineType=cv.LINE_AA)
    cv.imshow('result',result)

cv.destroyAllWindows()


# Puedes añadir parámetros a binarize para elegir el método y controlar el umbral fijo con un trackbar

