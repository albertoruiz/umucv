#!/usr/bin/env python

# lector de código de barras con la cámara

# paquetes habituales
import cv2 as cv
from umucv.stream import autoStream
from umucv.util import putText
import time

# acceso a zbar
from pyzbar.pyzbar import decode



for key, frame in autoStream():

    # medimos el tiempo de búsqueda y decodifiación de los códigos
    t0=time.time()
    results = decode(frame)
    t1=time.time()
    putText(frame, f'{(t1-t0)*1000:.0f} ms', orig = (5,40) )

    if results:
    
        # imprimimos tadas las decodificaciones encontradas
        for result in results:
            # By default zbar returns barcode data as byte array, so decode byte array as ascii
            print(result.type, result.data.decode("ascii"))

        # la primera de ellas se muestra también en la ventana
        r = results[0]
        putText(frame, f'{r.type}: {r.data.decode("ascii")}')

        # marcamos las 4 esquinas de un QR
        for x,y in r.polygon:
            cv.circle(frame, (x,y), 4, (0,0,255), -1, cv.LINE_AA)

    cv.imshow('zbar', frame)

