#!/usr/bin/env python

# ejemplo del utilidad de grabación de vídeo.
# Simplemente graba la fuente de imágenes que
# se muestra en la ventana.

import cv2 as cv
from umucv.stream import autoStream
from umucv.util import Video

# Los argumentos por omisión son video h264 y fps automático
# video = Video()

# Si no está disponible el codec usamos formato mjpg
video = Video(fps=15, codec="MJPG",ext="avi")

# Si queremos que empiece a grabar desde el primer frame
# video.ON = True


for key,frame in autoStream():

    cv.imshow('input',frame)
 
    # la tecla v inicia y detiene la grabación
    # la imagen no debe cambiar de tamaño
    # y debe estar en formato BGR
    video.write(frame, key, ord('v'))

cv.destroyAllWindows()
video.release()


# Puedes usar simplemente
# video.write(frame)
# y controlar la grabación con
# video.ON = True o False dependiendo de las condiciones que te interesen.

