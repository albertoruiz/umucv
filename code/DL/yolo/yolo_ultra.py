#!/usr/bin/env python

from ultralytics import YOLO

# class labels:
# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml

model = YOLO("yolov8n.pt")

import numpy as np
import cv2 as cv
from umucv.stream import autoStream

for key,frame in autoStream():
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    [result] = model(rgb)

    for b in result.boxes:
        print(np.array(b.xywh.cpu()))
        print(b.cls.cpu())
        [[x1,y1,x2,y2]] = np.array(b.xyxy.cpu())
        cv.rectangle(frame, (int(x1),int(y1)), (int(x2), int(y2)), color=(0,0,255))
    
    cv.imshow("result", cv.cvtColor(result.plot(), cv.COLOR_RGB2BGR))

    cv.imshow('input',frame)
