#!/usr/bin/env python

from umucv.stream import autoStream
from umucv.util   import putText, read_arguments
import cv2 as cv
import numpy as np

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

model_id = "IDEA-Research/grounding-dino-tiny"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

def my_arguments(parser):
    parser.add_argument('--text', help='categories of objects to detect', type=str, default='person')

args = read_arguments(my_arguments)



for key, frame in autoStream():
    image = Image.fromarray(frame)

    inputs = processor(images=image, text=args.text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    [results] = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )

    for lab, box in zip(results['labels'], results['boxes']):
        x1,y1,x2,y2 = np.array(box).astype(int)
        putText(frame, lab, orig=(x1+3,y1-6))
        cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,0,255))
        
        
    cv.imshow('grounding dino', frame)
