#!/usr/bin/env python

import cv2 as cv
from umucv.stream import autoStream
from umucv.util import putText

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# cv.namedWindow('MediaPipe FaceMesh', cv.WINDOW_NORMAL| cv.WINDOW_GUI_NORMAL)

POINTS = False

for key,frame in autoStream():
    frame = cv.flip(frame,1);
    h,w = frame.shape[:2]
    
    if key==ord("p"):
        POINTS = not POINTS
    
    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as face_mesh:
        
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                
                if POINTS:
                    for p, lan in enumerate(face_landmarks.landmark):
                        x=int(lan.x*w)
                        y=int(lan.y*h)
                        putText(frame,str(p), orig=(x,y), div=1, scale=0.5)

                else:                
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec)
    
    cv.imshow('MediaPipe FaceMesh', frame)

cv.destroyAllWindows()

