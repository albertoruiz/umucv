#!/usr/bin/env python

import numpy as np
import cv2 as cv
from umucv.stream import autoStream
from umucv.util import Slider, check_and_download, read_arguments

from ultralytics import YOLO
import yaml

from collections import defaultdict, deque


model = YOLO("yolo11n.pt")

# class labels:
url = "https://raw.githubusercontent.com/ultralytics/ultralytics/refs/heads/main/ultralytics/cfg/datasets/coco.yaml"
check_and_download("coco.yaml", url)
labels = yaml.safe_load(open("coco.yaml"))['names']

C = Slider("conf","YOLO 11 Tracking",0.5,0,1,0.01)

def my_arguments(parser):
    parser.add_argument('--clases', help='tipo de objeto (default = person)', type=str, default='person')
args = read_arguments(my_arguments)

selected = args.clases.split(',')
print(selected)

# Store the track history
track_history = defaultdict(lambda: deque(maxlen=50))


# Loop through the video frames
for key, frame in autoStream():

    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    [results] = model.track(rgb, persist=True, verbose=False)

    # Get the boxes and track IDs
    boxes = results.boxes.xywh.cpu()
    track_ids = results.boxes.id.int().cpu().tolist()
    confs     = results.boxes.conf.cpu().tolist()
    clases    = results.boxes.cls.int().cpu().tolist()

    # Visualize the results on the frame
    #annotated_frame = results[0].plot()

    # Plot the tracks
    for box, track_id, conf, cls in zip(boxes, track_ids, confs, clases):
        #print(box)
        #print(track_id)
        #print(conf)
        #print(cls, labels[round(cls)])
        x, y, w, h = box
        idx = round(cls)
        if conf < C.value or labels[idx] not in selected:
            continue

        track = track_history[track_id]
        track.append((float(x), float(y)))  # x, y center point

        cv.rectangle(frame, (int(x-w/2),int(y-h/2)), (int(x+w/2), int(y+h/2)), color=(0,0,255))

        #putText(frame,f"{labels[idx]} {conf:.2f}", (int(x),int(y)))

        # Draw the tracking lines
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv.polylines(frame, [points], isClosed=False, color=(0,0,255), thickness=3)

    # Display the annotated frame
    cv.imshow("YOLO 11 Tracking", frame)
