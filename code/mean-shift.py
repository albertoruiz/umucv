#!/usr/bin/env python

# Tracking de un objeto de colores destacados
# mediante reproyección de histograma.
# Implementamos una versión simple de "mean shift"
# y comparamos con cv.CamShift

# El objeto a seguir se marca con el ratón y
# se pulsa 't'. Entonces se calcula su histograma
# color y a partir de ahí se obtiene la reproyección
# de dicho histograma en la imagen de entrada.
# La nueva posición del objeto sera el valor medio
# de la reproyección, pero restringido a la zona
# cercana a la posición anterior.

import cv2          as cv
import numpy        as np
from umucv.stream import autoStream
from umucv.util   import ROI

# calcula el histograma multidimensional de la imagen
def hist(x, bins=16):
    return cv.calcHist([x],     # lista de imágenes
                       [0,1,2], # canales deseados
                       None,    # posible máscara
                       [bins, bins, bins],  # número de cajas
                       [0,256] + [0,256] + [0,256]) # intervalos en cada canal


# selección de región de interés más cómoda que cv.selectROI
cv.namedWindow('input')
roi = ROI('input')

H = None

bins = 16

for key,frame in autoStream():
    
    if H is not None:
        # reproyección de histograma

        # 1) separamos los canales y calculamos la posición de sus
        #    valores en el histograma
        b,g,r = (frame // (256//bins) ).transpose(2,0,1)

        # 2) obtenemos el valor del histograma en toda la imagen
        #    aprovechando características de numpy
        L = H[b,g,r]

        # las imágenes de float se muestran en la escala 0:negro 1:blanco
        cv.imshow('likelihood', L/L.max())
        
        if True:
            # tracking mediante camshift de opencv
            term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
            elipse, track_window = cv.CamShift(L, track_window, term_crit )
            cv.ellipse(frame, elipse, (0,128,255), 2)
            (c,r,w,h) = track_window
            cv.rectangle(frame, (c,r), (c+w, r+h), (0,128,255), 1)
        

        # implementación propia de mean shift

        # mostramos la posición anterior 
        cv.rectangle(frame, (cm-szx, rm-szy), (cm+szx, rm+szy), (255,128,255), 2)
        
        # preparamos una máscara de tamaño doble para calcular la media
        # en un entorno del objeto
        mask = np.zeros_like(L, np.uint8)
        s = 2
        # thickness = -1 rellena el rectángulo
        cv.rectangle(mask,  (cm-2*szx, rm-2*szy), (cm+2*szx, rm+2*szy), color=255, thickness=-1)
        cv.imshow('mask', mask)
        
        # restringimos la reproyección a esa zona 
        L = L*mask
       
        # calculamos el valor medio ponderado aprovechando de nuevo numpy
        h,w = L.shape
        rs = np.arange(h).reshape(-1,1)
        cs = np.arange(w).reshape(1,-1)
        sL = np.sum(L)
        if sL == 0:
            H = None
            continue
        rm = int(np.sum(L*rs) / sL)
        cm = int(np.sum(L*cs) / sL)
        #print(rm,cm)


    
    # seleccionamos una región
    if roi.roi:
        [x1,y1,x2,y2] = roi.roi
        
        if key == ord('t'):
            trozo = frame[y1:y2+1, x1:x2+1].copy()
            cv.imshow('trozo', trozo)
            H = hist(trozo, bins)
            # suavizado opcional
            H = cv.GaussianBlur(H,(0,0),1)
            #print(H.shape)
            
            # región de búsqueda inicial para cv.CamShift            
            c = x1
            r = y1
            w = x2-x1
            h = y2-y1
            track_window = (c,r,w,h)
            
            # reción de búsqueda para nuestro mean shift
            cm, rm   = (x1+x2)//2 , (y1+y2)//2
            szx,szy  = (x2-x1)//2 , (y2-y1)//2
            
            # quitar el roi cuando se selecciona la región
            roi.roi = []
        
        cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,255), thickness=2)
        
    #print(roi.roi)
    
    cv.imshow('input',frame)

cv.destroyAllWindows()

