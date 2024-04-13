#!/usr/bin/env python

# https://github.com/googlesamples/mediapipe/blob/main/examples/image_segmentation/python/image_segmentation.ipynb
# !wget -O deeplabv3.tflite -q https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Create the options that will be used for ImageSegmenter
base_options = python.BaseOptions(
    model_asset_path='deeplabv3.tflite')

options = vision.ImageSegmenterOptions(
    base_options=base_options,
    output_category_mask=True)

segmenter = vision.ImageSegmenter.create_from_options(options)

from umucv.stream import autoStream

for key, frame in autoStream():
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mpimage = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    segmentation_result = segmenter.segment(mpimage)
    category_mask = segmentation_result.category_mask.numpy_view()
    mask = np.expand_dims(category_mask>0,2)
    
    cv.imshow("foreground", frame * mask)

    

