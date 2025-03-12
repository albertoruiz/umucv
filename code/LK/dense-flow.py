#!/usr/bin/env python

# adaptado de # https://docs.opencv.org/4.10.0/d4/dee/tutorial_optical_flow.html

import cv2 as cv
import numpy as np

from umucv.stream import autoStream

for k,(key,frame) in enumerate(autoStream()):
    if k == 0:
        # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
        prev_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Creates an image filled with zero intensities with the same dimensions as the frame
        mask = np.zeros_like(frame)
        # Sets image saturation to maximum
        mask[..., 1] = 255
        continue

    cv.imshow('input',frame)
    # Converts each frame to grayscale - we previously only converted the first frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Calculates dense optical flow by Farneback method
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
    # Sets image hue according to the optical flow direction
    mask[..., 0] = angle * 180 / np.pi / 2
    # Sets image value according to the optical flow magnitude (normalized)
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    # Converts HSV to RGB (BGR) color representation
    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
    # Opens a new window and displays the output frame
    cv.imshow("dense optical flow", rgb)
    # Updates previous frame
    prev_gray = gray

