#!/usr/bin/env python

# Otro ejemplo del utilidad de grabación de vídeo.
# En este caso se guardan los frames que cumplan una condición 

import cv2          as cv
from umucv.stream import autoStream
from umucv.util import Video


video = Video(fps=20)
video.ON = True

for key,frame in autoStream():

    cv.imshow('input',frame)
    
    # la condición es que pulsemos la tecla g
    # (pero lo interesante es guardar frames que cumplan alguna
    #  condición más interesante)
    # También podemos activar o desactivar la grabación con video.ON
    
    if key == ord('g'):
        video.write(frame)

cv.destroyAllWindows()
video.release()

