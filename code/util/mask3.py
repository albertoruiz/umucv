#!/usr/bin/env python

# Igual que mask2 pero con un interfaz más informativo

import cv2 as cv
import numpy as np
from umucv.stream import autoStream
from umucv.util import putText

modes = {'fotograma original': lambda : frame,
         'máscara rgb':        lambda : mask*255,
         'fotograma enmascarado con multiplicación': lambda : result,
         'fotograma enmascarado con copyto': lambda : result2,
         'máscara lógica':     lambda : other_mask.astype(float),
         'fotograma enmascarado con máscara lógica y np.expand_dims': lambda : result3} 

current_mode = 0

for key, frame in autoStream():
    if key==ord('+'):
        current_mode = (current_mode + 1) % len(modes)
    mode = list(modes.keys())[current_mode]

    polygon = np.array([(50,70), (120,90), (60,200)])
    mask = np.zeros_like(frame)
    todos = -1
    on = (1,1,1)
    relleno = -1
    cv.drawContours(mask, [polygon], todos, on, relleno)
    
    result = mask*frame
    
    result2 = np.zeros_like(frame)
    np.copyto(result2, frame, where= mask != 1)
    
    h,w,_= frame.shape
    r = np.arange(h).reshape(-1,1)
    c = np.arange(w).reshape(1,-1)
    
    other_mask = (r+c>100) & (r+c < 200)
    result3 = np.expand_dims(other_mask,2) * frame
 
    output = modes[mode]()
    putText(output, mode)
    cv.imshow('mask',output)
    
