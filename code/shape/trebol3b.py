#!/usr/bin/env python

# Paso 3b: Mostramos gráficamente las componentes frecuenciales
#          de los contornos
          

# ./trebol3b.py --dev=dir:../../images/cards.png

import cv2          as cv
from umucv.stream import autoStream
import numpy as np

# la fft de numpy
from numpy.fft import fft, ifft



def componentes(c):
    x,y = c.T                       # separamos las coordenadas x e y
    z = x+y*1j                      # convertimos los puntos en números complejos
    f  = fft(z)                     # calculamos la transformada de Fourier discreta
    return f


# calcula el contorno de la elipse de frecuencia k
def elipse(f,k):                    # dada una transformada de fourier y una frecuencia deseada
    s = np.zeros_like(f)            # preparamos espacio para una una trasformada
    s[[k,-k,0]] = f[[k,-k,0]]       # copiamos la frecuencia cero (para centrar la elipse en su sitio)
                                    # y las frecuencias k y -k, que son círculos que se combinan en una elipse
    r = ifft(s)                     # Deshacemos la transformación
    x = np.real(r)                  # convertimos los números complejos en puntos del plano
    y = np.imag(r)
    return np.array([x,y]).T        # y devolvemos el contorno de la elipse

# preparamos las n elipses principales de forma que se puedan dibujar con drawContours
def elipses(cont, n):
    f = componentes(cont)
    return [elipse(f, k).astype(int) for k in range(1,n+1)]

# lo mismo, pero quitando la dominante y ampliando el tamaño
def elipses2(cont, n):
    f = componentes(cont)
    return [(elipse(f,0) + (elipse(f, k)-elipse(f,0))*5).astype(int) for k in range(2,n+1)]



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
    p = cv.arcLength(c.astype(np.float32),closed=True)
    a = cv.contourArea(c)
    r = 100*4*np.pi*abs(a)/p**2 if p> 0 else 0
    #print(p,a,r)
    return (100**2 >= a >= 10**2) and r > 5

def orientation(x):
    return cv.contourArea(x.astype(np.float32),oriented=True) >= 0


black = True


for (key,frame) in autoStream():

    contours = extractContours(frame)
    
    ok = [c for c in contours if razonable(c) and not orientation(c)]
        
    result = frame.copy()

    for c in ok:
        # todas las componentes
        #cv.drawContours(result, elipses(c, 4) , -1, (0,255,255), 1, cv.LINE_AA)

        # las más discriminantes ampliadas
        cv.drawContours(result, elipses2(c, 4) , -1, (0,255,255), 1, cv.LINE_AA)
    
    cv.imshow('shape recognition',result)

cv.destroyAllWindows()

