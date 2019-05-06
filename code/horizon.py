#!/usr/bin/env python

# intentamos combinar ventanas de opencv y de matplotlib

import cv2          as cv
import numpy as np
from umucv.stream import autoStream

import matplotlib.pyplot as plt


# muestra un polígono cuyos nodos son las filas de un array 2D
def shcont(c, color='blue', nodes=True):
    x = c[:,0]
    y = c[:,1]
    x = np.append(x,x[0])
    y = np.append(y,y[0])
    plot(x,y,color)
    if nodes: plot(x,y,'.',color=color, markerSize=11)

def shpoint(p, color='blue'):
    plot(p[0],p[1],'.',color=color, markerSize=15)        
        
# dibuja una recta "infinita"
def shline(l,xmin=-2000,xmax=2000, color='red'):
    a,b,c = l / np.linalg.norm(l)
    if abs(b) < 1e-6:
        x = -c/a
        r = np.array([[x,-2000],[x,2000]])
    else:
        y0 = (-a*xmin - c) / b
        y1 = (-a*xmax - c) / b
        r = np.array([[xmin,y0],[xmax,y1]])
    return r


# convierte un conjunto de puntos ordinarios (almacenados como filas de la matriz de entrada)
# en coordenas homogéneas (añadimos una columna de 1)
def homog(x):
    ax = np.array(x)
    uc = np.ones(ax.shape[:-1]+(1,))
    return np.append(ax,uc,axis=-1)

# convierte en coordenadas tradicionales
def inhomog(x):
    ax = np.array(x)
    return ax[..., :-1] / ax[...,[-1]]


# producto vectorial (normalizando el resultado)
def cross(u,v):
    r = np.cross(u,v)
    return r / np.linalg.norm(r)

# matplotlib en modo interactivo
plt.ion()

# preparamos una figura, con tamaño deseado
fig = plt.figure(figsize=(4,4))
# creamos un espacio de dibujo, que ocupa todo
ax = fig.add_axes([0,0,1,1])
# inicializamos un objeto de tipo imagen
im = ax.imshow(np.zeros((480,640,3)))
[line1] = ax.plot([],[],'-', color='green')
[line2] = ax.plot([],[],'-', color='green')
[pline1] = ax.plot([],[],'o', color='red')

[line3] = ax.plot([],[],'-', color='green')
[line4] = ax.plot([],[],'-', color='green')
[pline2] = ax.plot([],[],'o', color='red')

[lineh] = ax.plot([],[],'-', color='blue')

#[line2] = ax.plot([],[],'o', color='green')

#quitamos los ejes
ax.set_axis_off()
#          x1  x2  y1  y2      las coordenadas de las esquinas del ax
ax.axis([-300,1000,500,-500])





for key,frame in autoStream():
    
    # conversión a monocromo
    g = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # binarización automática
    _, gt = cv.threshold(g,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)
    # extracción de contornos
    cs = cv.findContours(gt.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)[-2]
    # eliminación de contornos muy pequeños
    cs = [c for c in cs if len(c) > 50]
    # simplificación de nodos (aproximación poligonal)
    reduced = [ cv.approxPolyDP(c,3,True) for c in cs ]
    # selección de polígonos con 6 vértices (y eliminación de dimensión extra)
    hexagons = [ c.reshape(-1,2) for c in reduced if len(c) == 6 ]
    # los ordenamos por área
    hexagons = sorted(hexagons, key=cv.contourArea, reverse=True)

    # dibujamos el más grande
    cv.drawContours(frame, hexagons[:1], -1,(0,255,255), 3, cv.LINE_AA)
    
    # si hay marcadores
    for x in hexagons[:1]:
        # calculamos la envolvente convexa, devolviendo los índices de los puntos
        hull = cv.convexHull(x, returnPoints = False)
        # calculamos las concavidades (puede haber un bug en opencv 3.1 de mempo,
        # es mejor usar el nuevo entorno)
        defects = cv.convexityDefects(x, hull)
        
        if defects is not None and len(defects)>0:
            [(s,e,f,d)] = defects[-1]
            x = np.roll(x,-f,axis=0)
            
            p0,p1,p2,p3,p4,p5 = x
            
            l1 = cross(homog(p2), homog(p3))
            l2 = cross(homog(p4), homog(p5))
            line1.set_data(*shline(l1).T)
            line2.set_data(*shline(l2).T)
            
            h1h = cross(l1,l2)
            h1 = inhomog(h1h)
            pline1.set_data(*h1)
            
            l3 = cross(homog(p3), homog(p4))
            l4 = cross(homog(p1), homog(p2))
            line3.set_data(*shline(l3).T)
            line4.set_data(*shline(l4).T)
            
            h2h = cross(l3,l4)
            h2 = inhomog(h2h)
            pline2.set_data(*h2)
            
            horiz = cross(h1h, h2h)
            lineh.set_data(*shline(horiz).T)
            
        
        # dibujo en rojo el primer vértice
        cv.circle(frame, tuple(x[0]), 3, (0,0,255), -1, cv.LINE_AA)
        
    
    

    cv.imshow('input',frame)
    im.set_data(cv.cvtColor(frame,cv.COLOR_BGR2RGB))

cv.destroyAllWindows()




