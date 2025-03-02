#!/usr/bin/env python

from ultralytics import YOLO

model = YOLO('yolo11n-seg.pt')

import numpy as np
import cv2 as cv
from umucv.stream import autoStream

for key,frame in autoStream():
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    [result] = model(rgb)

    if result.masks is not None:
        mask = 0
        for m in result.masks.data:
            mask += np.expand_dims(m.cpu().numpy(),-1)*[[[1,1,1]]]

        #print(mask.shape, frame.shape)
        cv.imshow('mask',mask)

        h,w,_ = frame.shape
        mask = cv.resize(mask,(w,h))
        frame[mask<0.5] = 0

    cv.imshow('input', frame)
    cv.imshow("result", cv.cvtColor(result.plot(), cv.COLOR_RGB2BGR))

