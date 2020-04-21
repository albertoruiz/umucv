#!/usr/bin/env python

# 1) Buscamos manchas de tinta mediante componentes conexas.


# nuestros paquetes habituales
import cv2 as cv
import numpy as np
from umucv.stream import autoStream



# usaremos dos funciones auxiliares

# binarización con umbral automático
def binarize(gray):
    _, r = cv.threshold(gray, 128, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    cv.imshow('bin',r)
    return r

# cálculo de las componentes conexas. Devuelve en una lista con las que tienen
# un tamaño intermedio (se debería controlar el intervalo de áreas permitidas con trackbar)
# Se devuelven los recortes de cada componente conexa obtenidos con los bounding box.
# Pero ojo: en cada recorte quitamos los pixels que de otras componentes.
# (no suele ocurrir, pero un bounding box puede abarcar un trozo de otro número).
# Devolvemos también el centro para escribir luego en esa posición de la imagen el resultado de la clasificación.
def ccs(mask):
    n, cc, st, cen = cv.connectedComponentsWithStats(mask)
    trozos = []
    LMIN=10
    LMAX=100
    for x in range(n):
        if LMIN**2 < st[x][cv.CC_STAT_AREA] < LMAX**2:
            x1 = st[x][cv.CC_STAT_LEFT]
            y1 = st[x][cv.CC_STAT_TOP]
            x2 = st[x][cv.CC_STAT_WIDTH] + x1
            y2 = st[x][cv.CC_STAT_HEIGHT] + y1
            trozos.append( ( cen[x], (cc[y1:y2+1, x1:x2+1]==x).astype(np.uint8)*255 ) )
    return trozos


# La función de alto nivel que busca manchas de tinta
def extractThings(image):
    g = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    if black:
        g = 255-g
    b = binarize(g)  
    return ccs(b)

# buscamos números escritos con tinta oscura sobre papel claro
black = True


# mostramos en una ventana cada uno de los objetos encontrados
# que podemos cambiar con las teclas +/-
p = 0

for key, frame in autoStream():
    cv.imshow('input', frame)
    
    if key == ord('+'):
        p = p+1
    if key == ord('-'):
        p = p-1 

    cosas = extractThings(frame)

    if cosas:
        s = cosas[p%len(cosas)][1]
        cv.imshow('cosas', s)

cv.destroyAllWindows()

