#!/usr/bin/env python

# Extraemos los vectores y calculamos la similitud con nuestra propia función

import cv2 as cv
from umucv.stream import autoStream
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import numpy as np

def my_cosine_sim(u,v):
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    return u @ v / (nu*nv)


options = vision.ImageEmbedderOptions(
    base_options = python.BaseOptions(model_asset_path='embedder.tflite'),
    l2_normalize = False, quantize = False)

embedder = vision.ImageEmbedder.create_from_options(options)

model = None

for key,frame in autoStream():
    
    mpimage = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv.cvtColor(frame, cv.COLOR_BGR2RGB))

    descriptor = embedder.embed(mpimage).embeddings[0].embedding
    
    if key==ord('c'):
        model = descriptor
    
    if model is not None:
        similarity = my_cosine_sim(descriptor, model)
        W = frame.shape[1]
        cv.rectangle(frame,(0,0),(int(similarity*W), 20), color=(0,255,0), thickness=-1)
        
    cv.imshow("similarity", frame)

