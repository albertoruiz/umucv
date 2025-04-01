#!/usr/bin/env python

#pip install open_clip_torch

# ./runclip.py --classes=girl,car,beach --dev dir:/path/to/repos/umucv/images/palm*.* [--resize=640x0]


import numpy as np
import cv2 as cv
from umucv.stream import autoStream
from umucv.util import putText, read_arguments

import torch
from PIL import Image
import open_clip
import time
import numpy as np
import argparse, sys

#import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def my_arguments(parser):
    parser.add_argument('--classes', type=str, help="classes separated by commas", default = "car,beach")

args = read_arguments(my_arguments)


classes = [x.strip() for x in args.classes.split(',')]
print(classes)

start_time = time.time()
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer('ViT-B-32')
end_time = time.time()
print(f"Done! CLIP model initialization time: {end_time - start_time:.1f}s\n")

text = tokenizer(classes)
text_features = model.encode_text(text)
text_features /= text_features.norm(dim=-1, keepdim=True)
#print(text_features.shape)

for k, (key,frame) in enumerate(autoStream()):

    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
   
    start_time = time.time()
    image = preprocess(Image.fromarray(rgb)).unsqueeze(0)
    with torch.no_grad(): #, torch.cuda.amp.autocast():
        image_features = model.encode_image(image)        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        #print(image_features.shape)
        probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)    
    probs = probs.cpu().numpy().flatten()
    end_time = time.time()
       
    #print("  Label probs:", list(zip(classes, probs)))
    #print()
    
    kmax = np.argmax(probs)
    
    #print(" ".join(map(lambda p: f"{100*p:3.0f}", probs)))
    shprobs = f"{classes[kmax]} {100*probs[kmax]:.0f}%   ({(end_time - start_time)*1000:.0f} ms)"
    putText(frame,shprobs)
    cv.imshow('input',frame)

