#!/usr/bin/env python


import cv2   as cv
import numpy as np
from umucv.stream import autoStream
from umucv.util import Help

help = Help(
"""
HELP WINDOW DEMO

c: color
m: monocromo

i: invert on/off

SPC: pausa

h: show/hide help
""")

color = False
invert = False


for key,frame in autoStream():
    help.show_if(key, ord('h'))
    
    if key == ord('c'):
        color = True
    if key == ord('m'):
        color = False
    if key == ord('i'):
        invert = not invert
    
    if not color:
        result = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    else:
        result = frame
        
    if invert:
        result = 255-result    
    
    cv.imshow('input',result)


