#!/usr/bin/env python

# Paso 4: Definimos la función que calcula el invariante
#         de forma basado en las frecuencias dominantes
          

# ./trebol4.py --dev=dir:../../images/cards.png

import cv2          as cv
from umucv.stream import autoStream
import numpy as np

# la fft de numpy
from numpy.fft import fft


# esta función recibe un contorno y produce descriptor de tamaño wmax*2+1
def invar(c, wmax=10):
    x,y = c.T                       # separamos las coordenadas x e y
    z = x+y*1j                      # convertimos los puntos en números complejos
    f  = fft(z)                     # calculamos la transformada de Fourier discreta
    fa = abs(f)                     # tomamos el módulo para conseguir invarianza a rotación
                                    # y punto de partida
    s = fa[1] + fa[-1]              # La amplitud de la frecuencia 1 nos da el tamaño global de la figura
                                    # y servirá para normalizar la escala
    v = np.zeros(2*wmax+1)          # preparamos espacio para el resultado
    v[:wmax] = fa[2:wmax+2];        # cogemos las componentes de baja frecuencia, positivas
    v[wmax:] = fa[-wmax-1:];        # y las negativas.
                                    # Añadimos también la frecuencia -1, que tiene
                                    # que ver con la "redondez" global de la figura
   
    if fa[-1] > fa[1]:              # normalizamos el sentido de recorrido
        v[:-1] = v[-2::-1]          # (El círculo dominante debe moverse en sentido positivo)
        v[-1] = fa[1]
    
    return v / s                    # normalizamos el tamaño




def binarize(gray):
    _, r = cv.threshold(gray, 128, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    return r

def extractContours(image):
    g = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    if black:
        g = 255-g
    b = binarize(g)  
    contours, _ = cv.findContours(b.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)[-2:]
    contours = [ c.reshape(-1,2) for c in contours ]
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    return contours

def razonable(c):
    return 100**2 >= cv.contourArea(c) >= 10**2

def orientation(x):
    return cv.contourArea(x.astype(np.float32),oriented=True) >= 0


black = True

shcont = False

model = extractContours(cv.imread('../../images/shapes/trebol.png'))[0]

modelaux = np.zeros([200,200], np.uint8)
cv.drawContours(modelaux, [model], -1, 255, cv.FILLED)
cv.imshow('model', modelaux)

invmodel = invar(model)

# El modelo tiene 346 puntos, pero queda descrito con un vector invariante de 21.
print(len(model))
print(invmodel)


for (key,frame) in autoStream():

    if key == ord('c'):
        shcont = not shcont

    contours = extractContours(frame)
    
    ok = [c for c in contours if razonable(c) and not orientation(c)]

    if shcont:
        result = np.zeros_like(frame)
        cp = [c for c in contours if orientation(c) ]
        cn = [c for c in contours if not orientation(c) ]
        
        cv.drawContours(result, cp, contourIdx=-1, color=(255,128,128), thickness=1, lineType=cv.LINE_AA)
        cv.drawContours(result, cn, -1, (128,128,255), 1)
    else:
        result = frame
        cv.drawContours(result, ok, -1, (255,0,0), cv.FILLED)
    
    cv.imshow('shape recognition',result)

cv.destroyAllWindows()

