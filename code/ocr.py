#!/usr/bin/env python

# OCR en vivo de una línea de texto marcada con un ROI


# paquetes usuales
import cv2 as cv
import numpy as np
from umucv.stream import autoStream
from umucv.util import ROI, putText
import time

# usamos el interfaz tesserocr, que accede directamete al API de tesseract
import tesserocr
# para conversión al formato requerido
from PIL import Image

# para verificar nuestra instalación 
print(tesserocr.tesseract_version())  # print tesseract-ocr version
print(tesserocr.get_languages())      # prints tessdata path and list of available languages

# establecemos la configuración de trabajo
tesseract = tesserocr.PyTessBaseAPI(lang='eng', psm=tesserocr.PSM.SINGLE_LINE, oem=tesserocr.OEM.DEFAULT)
# lo importante aquí es la opción de SINGLE_LINE (y que en el ROI haya una sola línea realmente)

# FIXME: no he conseguido que estas opciones funcionen
# sería importante limitar los caracteres que pueden aparecer en el texto
#tesseract.SetVariable('tessedit_char_whitelist', '123456789')
#tesseract.SetVariable('tessedit_char_blacklist', 'aeiou')

# usaremos la selección de ROI de umucv
cv.namedWindow("OCR")
roi = ROI('OCR')

for key, frame in autoStream():
    if roi.roi:
        [x1,y1,x2,y2] = roi.roi
        # si la región es muy pequeña no hacemos nada
        if abs(y2-y1) < 10: continue
        
        # extraemos la región y la marcamos
        region = frame[y1:y2,x1:x2].copy()
        cv.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 1)
        # cv.imshow('ROI', region)
        
        # medimos el tiempo de proceso
        t0 = time.time()
        # binarizamos la imagen con umbral automático (opcional)
        #_, region = cv.threshold(region[:,:,1], 160, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        # llamamos al OCR
        tesseract.SetImage(Image.fromarray(region))
        ocr_result = tesseract.GetUTF8Text()
        t1 = time.time()
        
        print(ocr_result)
        
        # mostramos el resultado en la ventana junto con el tamaño del ROI y el tiempo de cómputo
        h,w = region.shape[:2]
        putText(frame, f'{ocr_result[:-1]}   ({w}x{h}, {1000*(t1-t0):.0f}ms)', orig=(x1+5,y1-8))
        
    cv.imshow('OCR', frame)

