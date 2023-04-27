#!/usr/bin/env python

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

import cv2 as cv
from umucv.stream import autoStream

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

    for key,frame in autoStream():
        frame = cv.flip(frame,1);
        h,w = frame.shape[:2]

        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
          for hand_landmarks in results.multi_hand_landmarks:
            
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        cv.imshow('MediaPipe Hands', frame)

