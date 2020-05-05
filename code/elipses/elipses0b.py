#!/usr/bin/env python

# Detección de elipses (segundo método)
# pruébalo con:

# ./elipses0b.py --dev=dir:*.png


from umucv.stream   import autoStream
import cv2 as cv
import numpy as np
from umucv.contours import extractContours
from umucv.util import mkParam, putText
from umucv.htrans import htrans, desp, rot3, scale

# La idea es la siguiente: ajustando adecuadamente la escala, si una figura
# tiene forma de elipse debe coincidir con su elipse de incertidumbre.
# (Está definido por la media y la matriz de covarianza. Recuerda el tema de
# flujo óptico y el notebook covarianza.ipynb.)
# Para comparar un contorno denso más o menos irregular con una elipse
# lo que hacemos es calcular la transformación afín que convierte la elipse
# en un círculo unidad y transformamos con ella el contorno.
# Si todos los puntos quedan aproximadamente a distancia 1 del origen
# el contorno es una elipse aceptable.


# tamaño de los ejes de la elipse y orientación
def eig22(cxx,cyy,cxy):
    l,v = np.linalg.eigh(np.array([[cxx,cxy],[cxy,cyy]]))
    return l[1], l[0], np.arctan2(v[1][1],v[1][0])

# calculamos media, varianzas y covarianza de la figura
def moments_2(c):
    m = cv.moments(c.astype(np.float32))  # int32, float32, but not float64!
    s = m['m00']
    return (m['m10']/s, m['m01']/s, m['mu20']/s, m['mu02']/s, m['mu11']/s)

# Para las elipses de incertidumbre dentro de los límites de tamaño deseados
# calculamos la transformación que la convierte en círculo unidad, luego
# transformarmos el contorno y vemos cuánto se aleja de la circunferencia.
def errorEllipse(e,c, mindiam = 20, minratio = 0.2):
    (cx,cy), (A,B), ang = e
    if A < mindiam: return 2000
    if B/A < minratio: return 1000
    T = np.linalg.inv(desp((cx,cy)) @ rot3(np.radians(ang)) @ scale((A/2,B/2)))
    cc = htrans(T,c)
    r = np.sqrt((cc*cc).sum(axis=1))
    return abs(r-1).max()


# Aquí hacemos todo el proceso. Nos quedamos con los contornos que
# se aproximan mucho a una elipse y devolvemos la representación compacta que
# usa opencv: centro, tamaño de los ejes, y ángulo.
# Las ordenamos de mayor a menor tamaño.
def detectEllipses(contours, mindiam = 20, minratio = 0.2, tol = 5):
    res = []
    for c in contours:
        mx,my,cxx,cyy,cxy = moments_2(c)
        v1,v2,a = eig22(cxx,cyy,cxy)
        p = mx,my
        s1, s2 = np.sqrt(v1), np.sqrt(v2)
        e = p, (4*s1,4*s2), a/np.pi*180
        err = errorEllipse(e, c, mindiam=mindiam, minratio=minratio)
        if err < tol/100:
            res.append(e)
    return sorted(res, key = lambda x: - x[1][0] * x[1][1])



cv.namedWindow("elipses")
param = mkParam("elipses")
param.addParam("err",30,50)
param.addParam("area",5,20)




for key,frame in autoStream():
    g = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    cs = extractContours(g, minarea= max(1,param.area) ) 

    els = detectEllipses(cs, tol=param.err)

    for e in els:            

        # podemos dibujar directamente las elipses con opencv
        cv.ellipse(frame,e, color=(0,0,255), thickness=2, lineType=cv.LINE_AA)

    cv.imshow('elipses',frame)


