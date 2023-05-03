#!/usr/bin/env python

# https://gilberttanner.com/blog/yolo-object-detection-with-opencv/
# adapted for umucv

# captura asíncrona

import numpy as np
import cv2 as cv
import os
import time
from umucv.stream import autoStream
from umucv.util import putText
from threading import Thread

def extract_boxes_confidences_classids(outputs, confidence, width, height):
    boxes = []
    confidences = []
    classIDs = []

    for output in outputs:
        for detection in output:            
            # Extract the scores, classid, and the confidence of the prediction
            scores = detection[5:]
            classID = np.argmax(scores)
            conf = scores[classID]
            
            # Consider only the predictions that are above the confidence threshold
            if conf > confidence:
                # Scale the bounding box back to the size of the image
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, w, h = box.astype('int')

                # Use the center coordinates, width and height to get the coordinates of the top left corner
                x = int(centerX - (w / 2))
                y = int(centerY - (h / 2))

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(conf))
                classIDs.append(classID)

    return boxes, confidences, classIDs


def draw_bounding_boxes(image, boxes, confidences, classIDs, idxs, colors):
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            color = [int(c) for c in colors[classIDs[i]]]
            cv.rectangle(image, (x, y), (x + w, y + h), color, 2)
            label = labels[classIDs[i]]
            confi = confidences[i]
            putText(image, f"{label}: {confi:.2f}", (x,y-7))


def make_prediction(net, layer_names, labels, image, confidence, threshold):
    height, width = image.shape[:2]
    
    # Create a blob and pass it through the model
    blob = cv.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(layer_names)

    # Extract bounding boxes, confidences and classIDs
    boxes, confidences, classIDs = extract_boxes_confidences_classids(outputs, confidence, width, height)

    # Apply Non-Max Suppression
    idxs = cv.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

    return boxes, confidences, classIDs, idxs




# Minimum confidence for a box to be detected
confidence = 0.5

# Threshold for Non-Max Suppression
threshold  = 0.3

# Get the labels
labels = open('model/coco.names').read().strip().split('\n')

# Create a list of colors for the labels
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

# Load weights using OpenCV
net = cv.dnn.readNetFromDarknet('model/yolov3.cfg', 'model/yolov3.weights')

if False:
    print('Using GPU')
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

# Get the ouput layer names
layer_names = net.getLayerNames()
layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

frame = None
goon  = True
ready = False

def GUI():
    global frame, goon, ready
    for key, frame in autoStream():
        cv.imshow('cam',frame)
        if ready:
            cv.imshow('result', result)
            ready = False
    goon = False

t = Thread(target=GUI,args=())
t.start()

while frame is None: pass

while goon:    
    
    t0 = time.time()
    boxes, confidences, classIDs, idxs = make_prediction(net, layer_names, labels, frame, confidence, threshold)

    result = frame.copy()
    draw_bounding_boxes(result, boxes, confidences, classIDs, idxs, colors)
    t1 = time.time()

    putText(result, '{:.0f}ms'.format(1000*(t1-t0)))
    ready = True


