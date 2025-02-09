#!/usr/bin/env python

# Igual que mask pero con una sola ventana y seleccionando
# la salida con el teclado

import cv2 as cv
import numpy as np
from umucv.stream import autoStream


modes = "source mask1 masked1a masked1b mask2 masked2".split(' ')
current_mode = 0

for key, frame in autoStream():
    if key==ord('+'):
        current_mode = (current_mode + 1) % len(modes)
    mode = modes[current_mode]

    if mode == 'source':
        cv.imshow('mask',frame)
    
    polygon = np.array([(50,70), (120,90), (60,200)])
    mask = np.zeros_like(frame)
    todos = -1
    on = (1,1,1)
    relleno = -1
    cv.drawContours(mask, [polygon], todos, on, relleno)
    
    result = mask*frame
    
    result2 = np.zeros_like(frame)
    np.copyto(result2, frame, where= mask != 1)
    
    
    if mode == 'mask1':    
        cv.imshow('mask',mask*255)

    if mode == 'masked1a':        
        cv.imshow('mask',result)

    if mode == 'masked1b':
        cv.imshow('mask',result2)

    h,w,_= frame.shape
    r = np.arange(h).reshape(-1,1)
    c = np.arange(w).reshape(1,-1)
    
    other_mask = (r+c>100) & (r+c < 200)
    result3 = np.expand_dims(other_mask,2) * frame
    
    if mode == 'mask2':
        cv.imshow('mask',other_mask.astype(float))
    
    if mode == 'masked2':
        cv.imshow("mask", result3)
    
