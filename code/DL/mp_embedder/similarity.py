#!/usr/bin/env python

# https://github.com/googlesamples/mediapipe/blob/main/examples/image_embedder/python/image_embedder.ipynb


import cv2 as cv
from umucv.stream import autoStream
from umucv.util import check_and_download
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

check_and_download("embedder.tflite","https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite")

options = vision.ImageEmbedderOptions(
    base_options = python.BaseOptions(model_asset_path='embedder.tflite'),
    l2_normalize = True, quantize = True)

embedder = vision.ImageEmbedder.create_from_options(options)

model = None

for key,frame in autoStream():
    
    mpimage = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv.cvtColor(frame, cv.COLOR_BGR2RGB))

    descriptor = embedder.embed(mpimage).embeddings[0]
    
    if key==ord('c'):
        model = descriptor
    
    if model is not None:
        similarity = vision.ImageEmbedder.cosine_similarity(descriptor, model)
        W = frame.shape[1]
        cv.rectangle(frame,(0,0),(int(similarity*W), 20), color=(0,255,0), thickness=-1)
        
    cv.imshow("similarity", frame)

