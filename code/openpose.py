#!/usr/bin/env python

# necesita el paquete https://github.com/ildoonet/tf-pose-estimation

# ./openpose.py --dev=dir:../images/madelman.png
# ./openpose.py --size=400x300


import cv2   as cv
import numpy as np
from threading import Thread
import time
from umucv.util import putText
from umucv.stream import autoStream

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

e = TfPoseEstimator(get_graph_path('mobilenet_thin'))


def detect(image):    
    humans = e.inference(image, resize_to_default=False, upsample_size=4)
    #print(humans)
    if humans:
        print(list(humans[0].body_parts.keys()))

    # FIXME    
    try:
        print(humans[0].body_parts[7].x, humans[0].body_parts[7].y)
    except:
        pass

    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    return image


frame = None
goon = True
result  = None

def work():
    global result
    while goon:
        if frame is not None:
            t0 = time.time()
            result = detect(frame)    
            t1 = time.time()
            putText(result, '{:.0f}ms'.format(1000*(t1-t0)))

t = Thread(target=work,args=())
t.start()


for key, frame in autoStream():
    cv.imshow('cam',frame)
    if result is not None:
            cv.imshow('openpose', result)
            result = None

goon = False

