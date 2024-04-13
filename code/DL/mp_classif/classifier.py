#!/usr/bin/env python

# https://github.com/googlesamples/mediapipe/blob/main/examples/image_classification/python/image_classifier.ipynb
# wget -O classifier.tflite -q https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite


import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision

# Create an ImageClassifier object.
base_options = python.BaseOptions(model_asset_path='classifier.tflite')
options = vision.ImageClassifierOptions(base_options=base_options, max_results=4)
classifier = vision.ImageClassifier.create_from_options(options)


from umucv.stream import autoStream
from umucv.util import putText

for key, frame in autoStream():
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mpimage = mpimage = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    classification_result = classifier.classify(mpimage)

    cats = classification_result.classifications[0]
    cats = [cats.categories[k].category_name for k in range(4)]
    putText(frame, str(cats))
    
    cv.imshow("classifier", frame)


