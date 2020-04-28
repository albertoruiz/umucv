#!/usr/bin/env python

# Reconocimiento de caras
# Se apoya en el detector de caras y landmarks de dlib
# Utiliza un "encoding" precomputado (basado en deep learning)
# que transforma la cara, una vez alineada a un marco común 
# mediante los landmarks, en un vector de propiedades de dimensión 128.

# pip install face_recognition

# adaptado de
# https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam.py
# https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78
# https://towardsdatascience.com/real-time-face-recognition-with-cpu-983d35cc3ec5

import face_recognition
import cv2 as cv
import numpy as np
import time
from umucv.util import putText
from umucv.stream import autoStream
import glob

def readrgb(filename):
    return cv.cvtColor(cv.imread(filename), cv.COLOR_BGR2RGB) 

def readModels(path):
    fmods = sorted([name for name in glob.glob(path+'/*.*') if name[-3:] != 'txt'])
    models = [ readrgb(f) for f in fmods ]
    return fmods, models

filenames, models = readModels('gente')
names = [ x.split('/')[-1].split('.')[0].split('-')[0] for x in filenames ]
encodings = [ face_recognition.face_encodings(x)[0] for x in models ]


#print(names)
#print(encodings)
print(encodings[0].shape)

for key, frame in autoStream():

    t0 = time.time()

    face_locations = face_recognition.face_locations(frame)
    t1 = time.time()

    face_encodings = face_recognition.face_encodings(frame, face_locations)
    t2 = time.time()

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        match = face_recognition.compare_faces( encodings, face_encoding)
        #print(match)

        name = "Unknown"
        for n, m in zip(names, match):
            if m:
                name = n

        cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)
        putText(frame, name, orig=(left+3,bottom+16))

    putText(frame, f'{(t1-t0)*1000:.0f} ms {(t2-t1)*1000:.0f} ms')

    cv.imshow('Video', frame)

