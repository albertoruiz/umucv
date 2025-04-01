#!/usr/bin/env python

from ultralytics import YOLO

model = YOLO("boca.pt")

import cv2 as cv
from umucv.stream import autoStream

for key,frame in autoStream():
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    [result] = model(rgb)

    cv.imshow("YOLO 11", cv.cvtColor(result.plot(), cv.COLOR_RGB2BGR))

