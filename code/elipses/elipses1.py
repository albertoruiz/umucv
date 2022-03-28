#!/usr/bin/env python

# Detección del marcador de 4 círculos
# pruébalo con:

# ./elipses1.py --dev=dir:*.png


from umucv.stream   import autoStream
import cv2 as cv
import numpy as np
from umucv.util import mkParam, putText

# Toda la maquinaria anterior de detección de elipses está disponible en umucv
from umucv.contours import extractContours, detectEllipses

cv.namedWindow("elipses")
param = mkParam("elipses")
param.addParam("err",30,50)


for key,frame in autoStream():
    g = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    cs = extractContours(g, minarea=5)

    els = detectEllipses(cs, tol=param.err)

    # para cada elipse detectada
    for e in els:
        cv.ellipse(frame,e, color=(0,0,255))
        
        # sacamos el centro
        cx,cy = int(e[0][0]), int(e[0][1])
        # y escribimos el nivel de gris que tiene el centro de la elipse
        info = '{}'.format(g[cy,cx])
        putText(frame, info, (cx,cy),color=(255,255,255))
        # Lo necesitamos para detectar el círculo especial que no está
        # relleno, situado en el origen de coordenadas.
        
    # Cuando hemos encontrado 4 elipses tenemos que ordenarlas correctamente,
    # empezando por el círculo especial y en sentido de las agujas del reloj
    # (El detector puede devolverlas revueltas y la homografía se calcularía mal)
    # Para hacer esto calculamos la envolvente convexa de los 4 centros. La figura
    # es la misma, pero la función devuelve una polilínea bien ordenada, en la que
    # no se cruzan las diagonales
    # finalmente rotamos el polígono para que el círculo hueco quede el primero
    if len(els)==4:    
        hull = cv.convexHull(np.array([(e[0][0], e[0][1]) for e in els]).astype(np.float32)).reshape(-1,2)
        
        # extraemos los valores de gris del centro de cada elipse
        vals = np.array([ g[int(y), int(x)] for x,y in hull])
        
        # vemos en qué posición está el más claro
        orig = np.where(vals == max(vals))[0][0]

        # lo ponemos el primero
        hull = hull[ np.roll(np.arange(4), -orig) ]

        # dibujamos líneas para verificar que la ordenación es correcta
        cv.polylines(frame, [hull.astype(int)], False, (128,0,0), 1, lineType=cv.LINE_AA)

    cv.imshow('elipses',frame)

