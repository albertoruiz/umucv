#!/usr/bin/env python

import cv2 as cv
import numpy as np
from umucv.stream import autoStream
from umucv.util import putText
import time
import torch

from argu import parser, parse
parser.add_argument('--model', help="name of model to use", type=str, default='caras.torch')
args = parse()

sdev = 'cuda' if torch.cuda.is_available() else 'cpu'
print(sdev)
device = torch.device(sdev)

from myUNET import *
model = torch.load(args.model, map_location=device)

for key, orig in autoStream():
    frame = cv.cvtColor(orig, cv.COLOR_BGR2GRAY)
    cv.imshow('input',frame)

    H,W = frame.shape
    inputframes = np.array([frame]).reshape(1,1,H,W).astype(np.float32)
    
    t0 = time.time()
    inputframes = torch.from_numpy(inputframes).to(device)
    
    [r] = model(inputframes)
    r = np.clip(r[0].detach().cpu().numpy(),0,255).astype(np.uint8)
    #cv.imshow('mask', r)
    t1 = time.time()

    r = np.expand_dims(r,2)/255    
    mix = (orig * r).astype(np.uint8)
    putText(mix, F"{(t1-t0)*1000:5.0f} ms")
    
    cv.imshow('UNET', mix )
