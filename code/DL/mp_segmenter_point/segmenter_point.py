#!/usr/bin/env python

# https://github.com/googlesamples/mediapipe/blob/main/examples/interactive_segmentation/python/interactive_segmenter.ipynb
# wget -O model.tflite -q https://storage.googleapis.com/mediapipe-models/interactive_segmenter/magic_touch/float32/1/magic_touch.tflite


import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components import containers


BG_COLOR = (192, 192, 192) # gray
MASK_COLOR = (255, 255, 255) # white

RegionOfInterest = vision.InteractiveSegmenterRegionOfInterest
NormalizedKeypoint = containers.keypoint.NormalizedKeypoint

# Create the options that will be used for InteractiveSegmenter
base_options = python.BaseOptions(model_asset_path='model.tflite')
options = vision.ImageSegmenterOptions(base_options=base_options,
                                       output_category_mask=True)


segmenter = vision.InteractiveSegmenter.create_from_options(options)

point = [None]

def manejador(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        point[0] = x,y

cv.namedWindow("input")
cv.setMouseCallback("input", manejador)

from umucv.stream import autoStream
from umucv.util import putText

for key, frame in autoStream():
    H, W, _ = frame.shape
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mpimage = mpimage = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    if point[0] is not None:
        x,y = point[0]
        roi = RegionOfInterest(format=RegionOfInterest.Format.KEYPOINT,
                               keypoint=NormalizedKeypoint(x/W, y/H))
        result = segmenter.segment(mpimage, roi)
        mask = result.category_mask.numpy_view()

        #cv.imshow("object", mask)
        mask3 = np.expand_dims(mask,2)
        bg = np.zeros_like(frame)
        bg[:,:] = (0,128,128)
        final = np.where(mask3,bg,frame)
        cv.imshow("cosa", final)

        point[0] = None
    
    cv.imshow("input", frame)


