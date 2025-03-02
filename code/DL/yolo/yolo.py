#!/usr/bin/env python

from ultralytics import YOLO

# class labels:
# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml

import yaml, requests
url = "https://raw.githubusercontent.com/ultralytics/ultralytics/refs/heads/main/ultralytics/cfg/datasets/coco.yaml"
response = requests.get(url)
response.raise_for_status()
labels = yaml.safe_load(response.text)['names']

model = YOLO("yolo11n.pt")

import numpy as np
import cv2 as cv
from umucv.stream import autoStream
from umucv.util import putText, Slider

C = Slider("conf","YOLO 11",0.8,0,1,0.01)

for key,frame in autoStream():
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    [result] = model(rgb, verbose=False)

    for b in result.boxes:
        conf = b.conf.cpu().numpy()[0]
        if conf < C.value:
            continue
        [[x1,y1,x2,y2]] = np.array(b.xyxy.cpu()).astype(int)
        cv.rectangle(frame, (x1,y1), (x2, y2), color=(0,0,255))
        idx = round(b.cls.cpu().numpy()[0])
        putText(frame,f"{labels[idx]} {conf:.2f}", (x1+4,y1+15))

    #cv.imshow("result", cv.cvtColor(result.plot(), cv.COLOR_RGB2BGR))

    cv.imshow('YOLO 11',frame)
