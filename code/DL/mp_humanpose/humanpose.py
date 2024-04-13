#!/usr/bin/env python

# ./humanpose.py --dev dir:/home/brutus/repos/umucv/images/madelman.png

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

import cv2 as cv
from umucv.stream import autoStream

with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:

    for key,frame in autoStream():
        frame = cv.flip(frame,1);
        h,w = frame.shape[:2]

        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = pose.process(image)

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        cv.imshow('MediaPipe Pose', frame)

