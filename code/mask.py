#!/usr/bin/env python

# Dos formas de copiar regiones de imagen definidas por máscaras
# (arrays de True/False ó 0/1)
# 1) multiplicando directamente por la matriz de 0/1
# 2) usando copyto

# En ambos casos mask debe tener los mismos canales que la imagen
# (1 en monocromo y 3 en color)


import cv2 as cv
import numpy as np
from umucv.stream import autoStream

polygon = np.array([(50,70), (120,90), (60,200)])


for key, frame in autoStream():        
    cv.imshow('input',frame)
    
    mask = np.zeros_like(frame)
    todos = -1
    on = (1,1,1)
    relleno = -1
    cv.drawContours(mask, [polygon], todos, on, relleno)
    cv.imshow('mask',mask*255)
    
    result = mask*frame
    cv.imshow('result',result)
    
    result2 = np.zeros_like(frame)
    np.copyto(result2, frame, where= mask != 1)
    cv.imshow('result2',result2)

