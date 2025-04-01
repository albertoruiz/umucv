#!/usr/bin/env python

import cv2 as cv
from umucv.stream import autoStream
from umucv.util import putText
import numpy as np

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# cv.namedWindow('MediaPipe FaceMesh', cv.WINDOW_NORMAL| cv.WINDOW_GUI_NORMAL)

face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

N = 0

for k, (key,frame) in enumerate(autoStream()):
    frame = cv.flip(frame,1);
    h,w = frame.shape[:2]

    if key==ord("s"):
        SAVE = True

    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    if results.multi_face_landmarks:
        if k%30 == 29:
            N += 1
            cv.imwrite(f"train/images/{N:03}.jpg", frame)
        for face_landmarks in results.multi_face_landmarks:

            selected = [(int(lan.x*w), int(lan.y*h)) for k in [0,18,57,287] for lan in [face_landmarks.landmark[k]] ]

            x0,y0 = np.min(selected,axis=0)
            x1,y1 = np.max(selected,axis=0)

            cv.rectangle(frame,(x0,y0),(x1,y1),color=(255,255,255), thickness=1)

            for x,y in selected:
                cv.circle(frame, (x,y), 3, color=(255,255,255), thickness=-1)

            if k%30 == 29:
                xc = (x0+x1)/2
                yc = (y0+y1)/2
                bw = x1-x0
                bh = y1-y0
                with open(f"train/labels/{N:03}.txt","w") as f:
                    f.write(f"0 {xc/w:.5f} {yc/h:.5f} {bw/w:.5f} {bh/h:.5f}")


    cv.imshow('MediaPipe FaceMesh', frame)
    SAVE = False

