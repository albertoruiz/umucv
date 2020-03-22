#!/usr/bin/env python

# ejemplo del utilidad de grabación de vídeo.
# Simplemente graba la fuente de imágenes que
# se muestra en la ventana.

import cv2          as cv
from umucv.stream import autoStream
from umucv.util import Video

# se puede indicar fps automático con Video()
video = Video(fps=15)

for key,frame in autoStream():

    
    cv.imshow('input',frame)

    # la tecla v inicia y detiene la grabación
    # no debe cambiar de tamaño durante la grabación
    # debe ser una imagen en formato BGR
    video.write(frame, key, ord('v'))

cv.destroyAllWindows()
video.release()

