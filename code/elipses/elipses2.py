#!/usr/bin/env python

# Rectificación del marcador de 4 círculos
# pruébalo con:

# ./elipses2.py --dev=dir:*.png


from umucv.stream   import autoStream
import cv2 as cv
import numpy as np
from umucv.contours import extractContours, detectEllipses
from umucv.util import mkParam, putText
from umucv.htrans import htrans, desp, rot3, scale

cv.namedWindow("ellipses")
param = mkParam("ellipses")
param.addParam("err",30,50)
param.addParam("thres",0,255)


# coordenadas de los centros de los 4 círculos de referencia
# (en unidades arbitrarias). Si lo imprimes puedes medir las dimensiones
# reales que quedan en el papel.
ref = np.array(
      [[0,0],
       [0,8],
       [5,8],
       [5,0]])


for key,frame in autoStream():
    g = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    cs = extractContours(g, minarea=5)

    els = detectEllipses(cs, tol=param.err)

    for e in els:            
        cx,cy = int(e[0][0]), int(e[0][1])

        cv.ellipse(frame,e, color=(0,0,255))

        info = '{}'.format(g[cy,cx])
        putText(frame, info, (cx,cy),color=(255,255,255))
    
    if len(els)==4:    
        hull = cv.convexHull(np.array([(e[0][0], e[0][1]) for e in els]).astype(np.float32)).reshape(-1,2)
        vals = np.array([ g[int(y), int(x)] for x,y in hull])
        orig = np.where(vals == max(vals))[0][0]
        hull = hull[ np.roll(np.arange(4), -orig) ]
        cv.polylines(frame, [hull.astype(int)], False, (128,0,0), 1, lineType=cv.LINE_AA)

        # hacemos los pasos anteriores igual que antes para encontrar las
        # posiciones de los centros de los círculos en la imagen y ordenarlos
        # igual que la referencia
    
        
        # Calculamos la homografía de rectificación, haciendo que una unidad
        # ocupe 100 pixels.
        H,_ = cv.findHomography(hull, ref*100)

        # corregimos la imagen con esa homografía
        rectif = cv.warpPerspective(frame, H, (500,800))
        
        # obtenemos una imagen en la que el rectángulo interior a los círculos
        # ocupa todo el espacio disponible. Hemos elegido las dimensiones del
        # resultado para que encaje perfectamente.
        cv.imshow('rectified', rectif)
        
        # en otra ventana podemos mostrar otra rectificación en la que ahora
        # la referencia queda más pequeña (30 pixels por unidad), y está
        # desplazada para que se vea el resto de la imagen que hay alrededor
        H,_ = cv.findHomography(hull, ref*30 + (100,100))
        rectif = cv.warpPerspective(frame, H, (500,500))
        
        cv.imshow('rectified 2', rectif)

    cv.imshow('source',frame)


# La ventaja de usar este marcador basadço en círculos es que los puntos
# de referencia se calculan como un promedio y por tanto deben ser robustos
# frente al ruido de imagen, y que deja libre el espacio central de rectificación
# para situar los objetos que queramos estudiar.

